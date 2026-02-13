"""Plumise Agent -- main orchestrator for distributed LLM inference.

Replaces the Petals-based ``PlumiseServer`` with a purpose-built node
that manages:
1. Chain authentication and on-chain registration
2. Oracle registration and dynamic layer assignment
3. Partial or full model loading based on the assignment
4. gRPC server for inter-node pipeline communication
5. HTTP API for the InferenceAPI gateway
6. Periodic metrics reporting (heartbeat and reward claiming are Oracle-sponsored)
7. Live topology refresh for node join/leave events
"""

from __future__ import annotations

import asyncio
import logging
import signal
import threading
import time
from typing import TYPE_CHECKING, Optional

from plumise_agent.chain.agent import ChainAgent
from plumise_agent.chain.auth import PlumiseAuth
from plumise_agent.chain.config import AgentConfig
from plumise_agent.chain.proof import InferenceProofGenerator, ProofData
from plumise_agent.chain.reporter import OracleReporter
from plumise_agent.chain.rewards import RewardTracker
from plumise_agent.model.layer_range import LayerRange
from plumise_agent.node.metrics import MetricsCollector
from plumise_agent.node.registry import OracleRegistry

if TYPE_CHECKING:
    from plumise_agent.model.engine import InferenceEngine
    from plumise_agent.pipeline.executor import PipelineExecutor
    from plumise_agent.pipeline.topology import PipelineTopology

logger = logging.getLogger(__name__)


class PlumiseAgent:
    """Distributed LLM inference agent with Plumise chain integration.

    Orchestrates the full node lifecycle from startup to graceful
    shutdown, including chain handshakes, model loading, server
    startup, and background maintenance loops.

    Usage::

        config = AgentConfig(plumise_private_key="0x...")
        agent = PlumiseAgent(config)

        loop = asyncio.get_event_loop()
        agent.install_signal_handlers(loop)
        loop.run_until_complete(agent.start())

    Args:
        config: Fully-populated agent configuration.
    """

    def __init__(self, config: AgentConfig) -> None:
        self.config = config

        # Chain components
        self.auth = PlumiseAuth(config)
        self.chain_agent = ChainAgent(
            config=config,
            w3=self.auth.w3,
            account=self.auth.account,
        )
        self.reporter = OracleReporter(
            auth=self.auth,
            oracle_url=config.oracle_api_url,
            interval=config.report_interval,
        )
        self.rewards = RewardTracker(
            config=config,
            w3=self.auth.w3,
            account=self.auth.account,
        )
        self.proof_generator = InferenceProofGenerator(
            model_name=config.model_name,
            agent_address=self.auth.address,
        )
        self.metrics = MetricsCollector()

        # Oracle registration
        self.registry = OracleRegistry(auth=self.auth, config=config)

        # Inference pipeline (populated during startup)
        self.engine: InferenceEngine | None = None
        self.layer_range: LayerRange | None = None
        self.topology: PipelineTopology | None = None
        self.executor: PipelineExecutor | None = None
        self.grpc_server: object | None = None

        # State
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._start_time: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the Plumise Agent.

        This is the single entry point that performs all setup steps in
        order and then blocks until a shutdown signal is received.
        """
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(self._asyncio_exception_handler)

        self._start_time = time.time()

        logger.info("=" * 60)
        logger.info("  Plumise Agent")
        logger.info("  Address: %s", self.auth.address)
        logger.info("  Model:   %s", self.config.model_name)
        logger.info("  Chain:   %s (ID %d)", self.config.plumise_rpc_url, self.config.plumise_chain_id)
        logger.info("  Oracle:  %s", self.config.oracle_api_url)
        logger.info("  Device:  %s", self.config.device)
        logger.info("  gRPC:    %s:%d", self.config.grpc_host, self.config.grpc_port)
        logger.info("  API:     %s:%d", self.config.api_host, self.config.api_port)
        logger.info("=" * 60)

        # Step 1: Verify chain connectivity and agent registration
        await self._preflight_checks()

        # Step 2: Register on-chain via precompile 0x21 (must precede Oracle)
        # Oracle verifies on-chain registration, so this must happen first.
        await self._register_on_chain()

        # Step 3: Register with Oracle, get layer assignment
        await self._register_with_oracle()

        # Step 4: Load assigned model layers
        await self._load_model()

        # Step 5: Initialize pipeline executor
        self._init_executor()

        # Step 6: Start gRPC server for inter-node communication
        await self._start_grpc_server()

        # Step 7: Start HTTP API
        self._start_api_server()

        # Step 8: Report ready to Oracle
        await self.registry.report_ready()

        # Step 9: Start background tasks
        self._running = True
        await self.reporter.start(self.metrics)

        tasks = [
            asyncio.create_task(self._topology_refresh_loop(), name="topology-refresh"),
        ]

        logger.info("Agent is running. Press Ctrl+C to stop.")
        try:
            await self._shutdown_event.wait()
        finally:
            for t in tasks:
                t.cancel()
            await self._shutdown()

    # ------------------------------------------------------------------
    # Startup steps
    # ------------------------------------------------------------------

    async def _preflight_checks(self) -> None:
        """Verify chain connectivity and agent registration status."""
        if self.auth.is_chain_connected():
            logger.info("Chain connection OK")
            balance = self.auth.get_balance()
            logger.info("Agent balance: %.6f PLM", balance / 10**18)
        else:
            logger.warning(
                "Cannot reach chain at %s; continuing in offline mode",
                self.config.plumise_rpc_url,
            )

        # Check AgentRegistry if deployed
        if self.config.agent_registry_address:
            if self.auth.verify_registration():
                logger.info("Agent registration verified (AgentRegistry)")
            else:
                logger.warning(
                    "Agent %s is NOT registered in AgentRegistry; "
                    "will attempt precompile registration",
                    self.auth.address,
                )
            if self.auth.is_active():
                logger.info("Agent is ACTIVE")
            else:
                logger.warning("Agent is not in Active status")
        else:
            logger.info("AgentRegistry not deployed yet; using precompile-only mode")

    async def _register_with_oracle(self) -> None:
        """Register with the Oracle and obtain a layer assignment.

        If manual layer boundaries are configured, those are used
        instead of Oracle assignment.  If the Oracle is unreachable,
        the node falls back to single-node mode (all layers).
        """
        if self.config.layer_start is not None and self.config.layer_end is not None:
            # Manual layer override
            from plumise_agent.model.loader import ModelLoader

            loader = ModelLoader(
                self.config.model_name, self.config.device, self.config.hf_token,
                expected_hash=self.config.model_hash,
            )
            total = loader.get_total_layers()
            self.layer_range = LayerRange(
                self.config.layer_start, self.config.layer_end, total
            )
            logger.info(
                "Using manual layer assignment: [%d, %d) of %d",
                self.layer_range.start,
                self.layer_range.end,
                self.layer_range.total,
            )
        else:
            # Dynamic Oracle assignment
            self.layer_range = await self.registry.register()
            if self.layer_range is None:
                # Fallback: serve all layers in single-node mode
                from plumise_agent.model.loader import ModelLoader

                loader = ModelLoader(
                    self.config.model_name, self.config.device, self.config.hf_token
                )
                total = loader.get_total_layers()
                self.layer_range = LayerRange(0, total, total)
                logger.warning(
                    "Oracle unavailable; running in single-node mode (all %d layers)",
                    total,
                )

        # Fetch the full pipeline layout
        self.topology = await self.registry.get_topology()

    async def _load_model(self) -> None:
        """Load model layers based on the current assignment.

        The heavy loading work runs in a thread pool so we do not block
        the event loop.
        """
        from plumise_agent.model.engine import InferenceEngine
        from plumise_agent.model.loader import ModelLoader

        assert self.layer_range is not None

        loader = ModelLoader(
            self.config.model_name, self.config.device, self.config.hf_token
        )

        logger.info(
            "Loading model layers [%d, %d) of %d ...",
            self.layer_range.start,
            self.layer_range.end,
            self.layer_range.total,
        )

        loop = asyncio.get_running_loop()

        # load_partial handles the is_full case internally (returns all parts)
        model_parts, tokenizer = await loop.run_in_executor(
            None, lambda: loader.load_partial(self.layer_range)
        )

        self.engine = InferenceEngine(model_parts, tokenizer, loader.device)
        logger.info("Model loaded successfully")

    def _init_executor(self) -> None:
        """Initialize the pipeline executor for distributed inference."""
        from plumise_agent.pipeline.executor import PipelineExecutor
        from plumise_agent.pipeline.topology import PipelineTopology

        assert self.engine is not None
        assert self.layer_range is not None

        if self.topology is None:
            # Single-node fallback topology
            self.topology = PipelineTopology(
                model_name=self.config.model_name,
                total_layers=self.layer_range.total,
                nodes=[],
            )

        self.executor = PipelineExecutor(
            engine=self.engine,
            topology=self.topology,
            layer_range=self.layer_range,
            my_address=self.auth.address,
        )
        logger.info("Pipeline executor initialized")

    async def _start_grpc_server(self) -> None:
        """Start the gRPC server for inter-node pipeline communication.

        If the topology shows other nodes, a ``PipelineClient`` is
        created to forward activations to the next node in the chain.
        """
        from plumise_agent.grpc_.client import PipelineClient
        from plumise_agent.grpc_.server import (
            InferencePipelineServicer,
            start_grpc_server,
        )

        assert self.engine is not None
        assert self.layer_range is not None

        # Create client to the next node if one exists in the topology
        grpc_client: PipelineClient | None = None
        if self.topology and not self.topology.is_single_node:
            my_slot = self.topology.find_by_address(self.auth.address)
            if my_slot:
                slot_index, _ = my_slot
                next_node = self.topology.get_next_node(slot_index)
                if next_node:
                    grpc_client = PipelineClient(
                        next_node.grpc_endpoint,
                        tls_ca=self.config.grpc_tls_ca,
                        tls_cert=self.config.grpc_tls_cert,
                        tls_key=self.config.grpc_tls_key,
                    )
                    logger.info(
                        "Pipeline: forwarding to next node %s (%s)",
                        next_node.address[:10] + "...",
                        next_node.grpc_endpoint,
                    )

        servicer = InferencePipelineServicer(
            engine=self.engine,
            layer_range=self.layer_range,
            grpc_client=grpc_client,
            metrics=self.metrics,
        )

        self.grpc_server = await start_grpc_server(
            servicer,
            self.config.grpc_host,
            self.config.grpc_port,
            tls_cert=self.config.grpc_tls_cert,
            tls_key=self.config.grpc_tls_key,
            tls_ca=self.config.grpc_tls_ca,
        )
        logger.info(
            "gRPC server started on %s:%d",
            self.config.grpc_host,
            self.config.grpc_port,
        )

    def _start_api_server(self) -> None:
        """Start the HTTP API server in a daemon thread."""
        from plumise_agent.api.server import create_app, run_api_server

        app = create_app(agent=self)
        thread = threading.Thread(
            target=run_api_server,
            args=(app, self.config.api_host, self.config.api_port),
            name="api-server",
            daemon=True,
        )
        thread.start()
        logger.info(
            "HTTP API started on %s:%d",
            self.config.api_host,
            self.config.api_port,
        )

    async def _register_on_chain(self) -> None:
        """Register the agent on-chain via precompile 0x21."""
        if self.chain_agent.is_registered:
            logger.info("Agent already registered on-chain")
            return

        # Derive agent name from model + address suffix
        model_short = self.config.model_name.split("/")[-1][:16]
        agent_name = f"{model_short}-{self.auth.address[-8:]}"

        logger.info("Registering agent on-chain: %s", agent_name)

        loop = asyncio.get_running_loop()
        success = await loop.run_in_executor(
            None,
            lambda: self.chain_agent.register(
                name=agent_name,
                model_hash=b"\x00" * 32,
                capabilities=[],
            ),
        )

        if success:
            logger.info("Agent on-chain registration complete")
        else:
            logger.warning("Agent on-chain registration failed; will retry on next start")

    # ------------------------------------------------------------------
    # Background maintenance loops
    # ------------------------------------------------------------------

    async def _topology_refresh_loop(self) -> None:
        """Periodically refresh the pipeline topology from the Oracle.

        Runs every 30 seconds so the node can detect peers joining or
        leaving the pipeline and update its gRPC forwarding accordingly.
        """
        interval = 30  # seconds
        while self._running:
            try:
                await asyncio.sleep(interval)
                new_topology = await self.registry.get_topology()
                if new_topology is not None:
                    old_count = len(self.topology.nodes) if self.topology else 0
                    self.topology = new_topology
                    if self.executor:
                        self.executor.update_topology(new_topology)
                    new_count = len(new_topology.nodes)
                    if new_count != old_count:
                        logger.info(
                            "Topology changed: %d → %d nodes",
                            old_count,
                            new_count,
                        )
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error refreshing topology")

    # ------------------------------------------------------------------
    # Inference proof integration
    # ------------------------------------------------------------------

    def record_inference(
        self,
        input_data: str | bytes,
        output_data: str | bytes,
        token_count: int,
        latency_ms: float,
    ) -> Optional[ProofData]:
        """Record an inference and generate a proof.

        Called from the HTTP API handler after each generation.
        Records metrics, generates a cryptographic proof, and buffers
        it for the next Oracle report.  If ``verify_on_chain`` is
        enabled the proof is also queued for on-chain verification.

        Args:
            input_data: Raw input (prompt text or bytes).
            output_data: Raw output (generated text or bytes).
            token_count: Number of tokens generated.
            latency_ms: End-to-end latency in milliseconds.

        Returns:
            The generated ``ProofData``, or ``None`` on error.
        """
        # 1. Record metrics (always)
        self.metrics.record_inference(token_count, latency_ms)

        # 2. Generate proof
        try:
            proof = self.proof_generator.generate_proof(
                input_data=input_data,
                output_data=output_data,
                token_count=token_count,
            )
        except Exception:
            logger.exception("Failed to generate inference proof")
            return None

        # 3. Buffer proof for Oracle reporting
        self.metrics.record_proof(proof)

        # 4. Queue on-chain verification (if enabled)
        if self.config.verify_on_chain:
            try:
                loop = asyncio.get_event_loop()
                loop.call_soon_threadsafe(self._schedule_on_chain_verify, proof)
            except RuntimeError:
                # No running event loop (e.g. called from a sync context)
                logger.debug("Skipping on-chain verify: no running event loop")

        return proof

    def _schedule_on_chain_verify(self, proof: ProofData) -> None:
        """Schedule on-chain verification as a background coroutine."""
        asyncio.ensure_future(self._verify_on_chain(proof))

    async def _verify_on_chain(self, proof: ProofData) -> None:
        """Run on-chain verification in a thread pool."""
        try:
            loop = asyncio.get_running_loop()
            tx_hash = await loop.run_in_executor(
                None,
                self.chain_agent.verify_inference,
                proof,
            )
            if tx_hash:
                logger.info(
                    "On-chain verification complete: tx=%s proofHash=%s",
                    tx_hash,
                    "0x" + proof.proof_hash.hex()[:16] + "...",
                )
        except Exception:
            logger.exception("On-chain verification error")

    # ------------------------------------------------------------------
    # Status helpers (consumed by API)
    # ------------------------------------------------------------------

    @property
    def uptime_seconds(self) -> int:
        """Seconds elapsed since the agent started."""
        if self._start_time == 0.0:
            return 0
        return int(time.time() - self._start_time)

    @property
    def is_ready(self) -> bool:
        """True when the model engine is loaded and serving."""
        return self.engine is not None

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def _shutdown(self) -> None:
        """Graceful shutdown: final report, metrics log, key cleanup."""
        logger.info("Shutting down...")
        self._running = False

        # Final metrics report
        await self.reporter.send_final_report(self.metrics)
        await self.reporter.stop()

        # Stop gRPC server
        if self.grpc_server is not None:
            try:
                await self.grpc_server.stop(grace=5)  # type: ignore[union-attr]
            except Exception:
                logger.debug("gRPC server stop error (non-critical)")

        # Log final metrics
        final = self.metrics.get_snapshot()
        logger.info("Final metrics: %s", final.to_dict())

        # Reward summary
        summary = self.rewards.summary()
        logger.info("Reward summary: %s", summary)

        # Clear private key from config memory (best effort)
        try:
            self.config.plumise_private_key = ""  # type: ignore[misc]
        except Exception:
            pass

        logger.info("Shutdown complete")

    def request_shutdown(self) -> None:
        """Request a graceful shutdown from any thread."""
        self._shutdown_event.set()

    # ------------------------------------------------------------------
    # Exception handling
    # ------------------------------------------------------------------

    def _asyncio_exception_handler(
        self, loop: asyncio.AbstractEventLoop, context: dict
    ) -> None:
        """Handle uncaught asyncio exceptions without crashing the process."""
        exception = context.get("exception")
        message = context.get("message", "Unhandled asyncio exception")
        if exception:
            logger.error(
                "Unhandled asyncio exception: %s - %s",
                message,
                exception,
                exc_info=exception,
            )
        else:
            logger.error("Unhandled asyncio error: %s", message)

    # ------------------------------------------------------------------
    # Signal handling
    # ------------------------------------------------------------------

    def install_signal_handlers(self, loop: asyncio.AbstractEventLoop) -> None:
        """Install SIGINT/SIGTERM handlers for graceful shutdown."""
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self.request_shutdown)
            except (NotImplementedError, RuntimeError):
                logger.debug("Signal handlers not supported on this platform")
                break

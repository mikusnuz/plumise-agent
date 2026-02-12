"""Pipeline executor for orchestrating distributed inference.

``PipelineExecutor`` is the high-level entry point for running an
inference request. It decides whether to run locally (single-node) or
to initiate a distributed pipeline across multiple gRPC-connected nodes.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

from plumise_agent.grpc_.client import PipelineClient
from plumise_agent.grpc_.serializer import serialize_tensor
from plumise_agent.model.layer_range import LayerRange
from plumise_agent.pipeline.topology import PipelineTopology

# gRPC generated stubs may not exist yet.
try:
    from plumise_agent.grpc_.generated import inference_pb2
except ImportError:
    inference_pb2 = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from plumise_agent.model.engine import InferenceEngine

logger = logging.getLogger(__name__)


class PipelineExecutor:
    """Orchestrates distributed forward passes across pipeline nodes.

    For single-node deployments the executor simply calls
    ``engine.forward_full()``. For multi-node pipelines starting on the
    first node, it embeds the prompt locally, then chains gRPC
    ``ForwardPass`` calls through the remaining nodes.

    Args:
        engine: Local inference engine.
        topology: Current pipeline topology from the Oracle.
        layer_range: Layer range assigned to this node.
        my_address: This node's wallet address (for topology lookup).
    """

    def __init__(
        self,
        engine: InferenceEngine,
        topology: PipelineTopology,
        layer_range: LayerRange,
        my_address: str = "",
    ) -> None:
        self._engine = engine
        self._topology = topology
        self._layer_range = layer_range
        self._my_address = my_address

        # Determine our position in the pipeline
        self._my_order: int = 0
        found = topology.find_by_address(my_address) if my_address else None
        if found is not None:
            self._my_order = found[1].pipeline_order

        logger.info(
            "PipelineExecutor initialized: order=%d single_node=%s layers=[%d,%d)",
            self._my_order,
            topology.is_single_node,
            layer_range.start,
            layer_range.end,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        params: dict | None = None,
        request_id: str | None = None,
    ) -> tuple[str, int]:
        """Run inference on the given prompt.

        Automatically chooses between single-node and distributed execution
        based on the topology.

        Args:
            prompt: Input text for generation.
            params: Generation parameters (``max_new_tokens``, ``temperature``,
                ``top_p``, ``repetition_penalty``, ``do_sample``).
            request_id: Optional unique request identifier. If ``None``,
                a UUID4 is generated.

        Returns:
            ``(generated_text, num_tokens)`` tuple.

        Raises:
            RuntimeError: If pipeline execution fails.
        """
        if request_id is None:
            request_id = uuid.uuid4().hex

        if params is None:
            params = {}

        if self._topology.is_single_node or self._layer_range.is_full:
            return self._execute_local(prompt, params)

        if self._layer_range.is_first:
            return await self._execute_pipeline(prompt, params, request_id)

        raise RuntimeError(
            "PipelineExecutor.generate() should only be called on the "
            "first node or a single-node deployment. Middle/last nodes "
            "receive work via the gRPC ForwardPass RPC."
        )

    # ------------------------------------------------------------------
    # Single-node execution
    # ------------------------------------------------------------------

    def _execute_local(
        self,
        prompt: str,
        params: dict,
    ) -> tuple[str, int]:
        """Run inference entirely on this node."""
        logger.info("Executing local (single-node) inference")
        return self._engine.forward_full(
            prompt=prompt,
            max_new_tokens=params.get("max_new_tokens", 128),
            temperature=params.get("temperature", 0.7),
            top_p=params.get("top_p", 0.9),
            repetition_penalty=params.get("repetition_penalty", 1.2),
            do_sample=params.get("do_sample", True),
        )

    # ------------------------------------------------------------------
    # Multi-node pipeline execution
    # ------------------------------------------------------------------

    async def _execute_pipeline(
        self,
        prompt: str,
        params: dict,
        request_id: str,
    ) -> tuple[str, int]:
        """Execute the full distributed pipeline starting from this (first) node.

        Steps:
          1. Run ``forward_first()`` to embed and process through local layers.
          2. Serialize the resulting hidden states.
          3. Build the gRPC ``ForwardRequest`` with generation params and
             pipeline metadata.
          4. Send to the next node via ``PipelineClient``.
          5. If the response contains ``generated_text``, return it.
             Otherwise propagate to the next node (handled server-side).
        """
        logger.info(
            "Executing pipeline inference: request_id=%s nodes=%d",
            request_id,
            len(self._topology.nodes),
        )

        # Step 1: First-node forward pass
        hidden_states = self._engine.forward_first(prompt)

        # Step 2: Serialize
        hs_bytes, hs_shape, hs_dtype = serialize_tensor(hidden_states)

        # Step 3: Find the next node
        next_node = self._topology.get_next_node(self._my_order)
        if next_node is None:
            raise RuntimeError(
                "Pipeline has no next node after order "
                f"{self._my_order}, but this is not the last node."
            )

        # Build generation params protobuf
        gen_params = self._build_gen_params(params)

        # Build pipeline metadata
        metadata = self._build_metadata(prompt, request_id)

        # Step 4: Forward to next node
        client = PipelineClient(
            target=next_node.grpc_endpoint,
            timeout=min(120.0, max(60.0, params.get("max_new_tokens", 128) * 0.5)),
        )

        try:
            response = await client.forward(
                request_id=request_id,
                hidden_states=hs_bytes,
                shape=hs_shape,
                dtype=hs_dtype,
                params=gen_params,
                metadata=metadata,
            )
        finally:
            await client.close()

        # Step 5: Check response
        if not response.success:
            raise RuntimeError(
                f"Pipeline forward failed at node {next_node.address}: "
                f"{response.error}"
            )

        if response.generated_text:
            logger.info(
                "Pipeline inference complete: request_id=%s tokens=%d",
                request_id,
                response.num_tokens,
            )
            return response.generated_text, response.num_tokens

        # If the response contains hidden states but no text, something
        # went wrong in the pipeline chain (middle node didn't forward).
        raise RuntimeError(
            "Pipeline returned hidden states instead of generated text. "
            "This likely means the last node could not be reached."
        )

    # ------------------------------------------------------------------
    # Protobuf builders
    # ------------------------------------------------------------------

    def _build_gen_params(self, params: dict) -> inference_pb2.GenerationParams:
        """Build a ``GenerationParams`` protobuf from a plain dict."""
        if inference_pb2 is None:
            raise ImportError("gRPC generated stubs not available")

        return inference_pb2.GenerationParams(
            max_new_tokens=params.get("max_new_tokens", 128),
            temperature=params.get("temperature", 0.7),
            top_p=params.get("top_p", 0.9),
            repetition_penalty=params.get("repetition_penalty", 1.2),
            do_sample=params.get("do_sample", True),
        )

    def _build_metadata(
        self,
        prompt: str,
        request_id: str,
    ) -> inference_pb2.PipelineMetadata:
        """Build a ``PipelineMetadata`` protobuf for the pipeline chain."""
        if inference_pb2 is None:
            raise ImportError("gRPC generated stubs not available")

        total_nodes = len(self._topology.nodes)

        # Determine next-next node endpoint (so the next node knows where to forward)
        next_node = self._topology.get_next_node(self._my_order)
        next_next_node = self._topology.get_next_node(self._my_order + 1)
        next_next_endpoint = next_next_node.grpc_endpoint if next_next_node else ""

        return inference_pb2.PipelineMetadata(
            total_nodes=total_nodes,
            node_index=self._my_order + 1,  # Next node's index
            is_first=False,  # The receiver is not the first node
            is_last=(self._my_order + 1 == total_nodes - 1),
            requester_address=self._my_address,
            original_prompt=prompt,
            next_node_endpoint=next_next_endpoint,
        )

    # ------------------------------------------------------------------
    # Topology update
    # ------------------------------------------------------------------

    def update_topology(self, topology: PipelineTopology) -> None:
        """Hot-swap the pipeline topology (e.g. after an Oracle refresh).

        Args:
            topology: New topology from the Oracle.
        """
        self._topology = topology

        found = topology.find_by_address(self._my_address) if self._my_address else None
        if found is not None:
            self._my_order = found[1].pipeline_order

        logger.info(
            "Topology updated: %d nodes, my_order=%d",
            len(topology.nodes),
            self._my_order,
        )

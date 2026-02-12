"""gRPC server for the InferencePipeline service.

Receives hidden-state tensors from upstream nodes, processes them through
the local ``InferenceEngine``, and either forwards to the next node or
returns the final generated text.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import grpc
from grpc import aio as grpc_aio

from plumise_agent.grpc_.serializer import deserialize_tensor, serialize_tensor
from plumise_agent.model.layer_range import LayerRange

# gRPC generated stubs may not exist yet (generated from proto at build time).
try:
    from plumise_agent.grpc_.generated import inference_pb2, inference_pb2_grpc
except ImportError:
    inference_pb2 = None  # type: ignore[assignment]
    inference_pb2_grpc = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from plumise_agent.grpc_.client import PipelineClient
    from plumise_agent.model.engine import InferenceEngine
    from plumise_agent.node.metrics import MetricsCollector

logger = logging.getLogger(__name__)


class InferencePipelineServicer(inference_pb2_grpc.InferencePipelineServicer):
    """Implements the ``InferencePipeline`` gRPC service.

    Args:
        engine: Local inference engine with loaded model layers.
        layer_range: The layer range this node is serving.
        model_name: HuggingFace model identifier.
        agent_address: This node's wallet address (for ping responses).
        grpc_client: Optional client for forwarding to the next pipeline node.
        metrics: Optional metrics collector for recording inference stats.
    """

    def __init__(
        self,
        engine: InferenceEngine,
        layer_range: LayerRange,
        model_name: str = "",
        agent_address: str = "",
        grpc_client: PipelineClient | None = None,
        metrics: MetricsCollector | None = None,
    ) -> None:
        self._engine = engine
        self._layer_range = layer_range
        self._model_name = model_name
        self._agent_address = agent_address
        self._grpc_client = grpc_client
        self._metrics = metrics
        self._start_time = time.time()
        self._ready = True

    # ------------------------------------------------------------------
    # ForwardPass RPC
    # ------------------------------------------------------------------

    async def ForwardPass(
        self,
        request: inference_pb2.ForwardRequest,
        context: grpc.aio.ServicerContext,
    ) -> inference_pb2.ForwardResponse:
        """Process a forward-pass request.

        Workflow:
          1. Deserialize incoming hidden states.
          2. Run through local engine (``forward_middle`` or ``forward_last``).
          3. If last node: return ``generated_text``.
          4. If middle node with next endpoint: forward via ``grpc_client``.
          5. If middle node without next: return hidden states.
        """
        t0 = time.perf_counter()
        request_id = request.request_id

        try:
            # Deserialize hidden states
            hidden_states = deserialize_tensor(
                data=request.hidden_states,
                shape=list(request.shape),
                dtype=request.dtype,
                device=self._engine.device,
            )

            # Deserialize attention mask if provided
            attention_mask = None
            if request.attention_mask and len(request.attention_mask) > 0:
                attention_mask = deserialize_tensor(
                    data=request.attention_mask,
                    shape=list(request.attention_mask_shape),
                    dtype="float32",
                    device=self._engine.device,
                )

            metadata = request.metadata
            params = request.params

            # Determine if this node is the last in the pipeline
            is_last = self._layer_range.is_last
            if metadata and metadata.is_last:
                is_last = True

            if is_last:
                # Last node: generate text
                text, num_tokens = self._engine.forward_last(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    max_new_tokens=params.max_new_tokens or 128,
                    temperature=params.temperature or 0.7,
                    top_p=params.top_p or 0.9,
                    repetition_penalty=params.repetition_penalty or 1.2,
                    do_sample=params.do_sample,
                )

                latency_ms = (time.perf_counter() - t0) * 1000
                self._record_metrics(num_tokens, latency_ms)

                return inference_pb2.ForwardResponse(
                    request_id=request_id,
                    generated_text=text,
                    num_tokens=num_tokens,
                    latency_ms=latency_ms,
                    success=True,
                )
            else:
                # Middle node: process and potentially forward
                output_hidden = self._engine.forward_middle(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                )

                # Try to forward to next node
                next_endpoint = (
                    metadata.next_node_endpoint if metadata else ""
                )

                if next_endpoint and self._grpc_client:
                    # Forward to next node
                    out_bytes, out_shape, out_dtype = serialize_tensor(output_hidden)

                    response = await self._grpc_client.forward(
                        request_id=request_id,
                        hidden_states=out_bytes,
                        shape=out_shape,
                        dtype=out_dtype,
                        attention_mask=request.attention_mask,
                        params=params,
                        metadata=metadata,
                    )

                    latency_ms = (time.perf_counter() - t0) * 1000
                    self._record_metrics(0, latency_ms)
                    return response
                else:
                    # Return hidden states to caller
                    out_bytes, out_shape, out_dtype = serialize_tensor(output_hidden)

                    latency_ms = (time.perf_counter() - t0) * 1000
                    self._record_metrics(0, latency_ms)

                    return inference_pb2.ForwardResponse(
                        request_id=request_id,
                        hidden_states=out_bytes,
                        shape=out_shape,
                        dtype=out_dtype,
                        latency_ms=latency_ms,
                        success=True,
                    )

        except Exception as exc:
            logger.exception("ForwardPass failed for request %s", request_id)
            latency_ms = (time.perf_counter() - t0) * 1000
            return inference_pb2.ForwardResponse(
                request_id=request_id,
                latency_ms=latency_ms,
                success=False,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # GetLayerInfo RPC
    # ------------------------------------------------------------------

    async def GetLayerInfo(
        self,
        request: inference_pb2.Empty,
        context: grpc.aio.ServicerContext,
    ) -> inference_pb2.LayerInfoResponse:
        """Return information about the layers this node is serving."""
        return inference_pb2.LayerInfoResponse(
            model_name=self._model_name,
            layer_start=self._layer_range.start,
            layer_end=self._layer_range.end,
            total_layers=self._layer_range.total,
            device=self._engine.device,
            is_ready=self._ready,
        )

    # ------------------------------------------------------------------
    # Ping RPC
    # ------------------------------------------------------------------

    async def Ping(
        self,
        request: inference_pb2.Empty,
        context: grpc.aio.ServicerContext,
    ) -> inference_pb2.PingResponse:
        """Return liveness information."""
        uptime = int(time.time() - self._start_time)
        return inference_pb2.PingResponse(
            agent_address=self._agent_address,
            model_name=self._model_name,
            layer_start=self._layer_range.start,
            layer_end=self._layer_range.end,
            is_ready=self._ready,
            uptime_seconds=uptime,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _record_metrics(self, tokens: int, latency_ms: float) -> None:
        """Record inference metrics if a collector is configured."""
        if self._metrics is not None:
            self._metrics.record_inference(tokens, latency_ms)

    def set_ready(self, ready: bool) -> None:
        """Update the readiness flag (used during startup/shutdown)."""
        self._ready = ready


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------

async def start_grpc_server(
    servicer: InferencePipelineServicer,
    host: str = "0.0.0.0",
    port: int = 50051,
    max_workers: int = 4,
    max_message_length: int = 256 * 1024 * 1024,  # 256 MiB
) -> grpc_aio.Server:
    """Create, configure, and start an async gRPC server.

    Args:
        servicer: The ``InferencePipelineServicer`` instance.
        host: Listen address.
        port: Listen port.
        max_workers: Maximum number of concurrent RPC handlers.
        max_message_length: Maximum message size in bytes (default 256 MiB,
            needed for large hidden-state tensors).

    Returns:
        The running ``grpc.aio.Server`` instance.
    """
    if inference_pb2_grpc is None:
        raise ImportError(
            "gRPC generated stubs not found. "
            "Run `python -m grpc_tools.protoc` to generate them first."
        )

    options = [
        ("grpc.max_send_message_length", max_message_length),
        ("grpc.max_receive_message_length", max_message_length),
    ]

    server = grpc_aio.server(options=options)
    inference_pb2_grpc.add_InferencePipelineServicer_to_server(servicer, server)
    server.add_insecure_port(f"{host}:{port}")

    await server.start()
    logger.info("gRPC server started on %s:%d", host, port)

    return server

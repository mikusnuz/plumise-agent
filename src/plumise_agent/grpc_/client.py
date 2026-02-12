"""gRPC client for sending hidden states to the next pipeline node.

``PipelineClient`` wraps a single ``grpc.aio`` channel and provides typed
helper methods for the three ``InferencePipeline`` RPCs.
"""

from __future__ import annotations

import logging
from typing import Any

import grpc
from grpc import aio as grpc_aio

# gRPC generated stubs may not exist yet (generated from proto at build time).
try:
    from plumise_agent.grpc_.generated import inference_pb2, inference_pb2_grpc
except ImportError:
    inference_pb2 = None  # type: ignore[assignment]
    inference_pb2_grpc = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Default maximum message size: 256 MiB (matches server config).
_MAX_MESSAGE_LENGTH = 256 * 1024 * 1024


class PipelineClient:
    """Async gRPC client for forwarding hidden states to the next node.

    Args:
        target: gRPC endpoint in ``"host:port"`` format.
        timeout: Default RPC deadline in seconds.
    """

    def __init__(
        self,
        target: str,
        timeout: float = 60.0,
        tls_ca: str = "",
        tls_cert: str = "",
        tls_key: str = "",
    ) -> None:
        if inference_pb2_grpc is None:
            raise ImportError(
                "gRPC generated stubs not found. "
                "Run `python -m grpc_tools.protoc` to generate them first."
            )

        self.target = target
        self.timeout = timeout

        options = [
            ("grpc.max_send_message_length", _MAX_MESSAGE_LENGTH),
            ("grpc.max_receive_message_length", _MAX_MESSAGE_LENGTH),
        ]

        if tls_ca or tls_cert:
            from pathlib import Path

            root_ca = Path(tls_ca).read_bytes() if tls_ca else None
            cert = Path(tls_cert).read_bytes() if tls_cert else None
            key = Path(tls_key).read_bytes() if tls_key else None
            credentials = grpc.ssl_channel_credentials(
                root_certificates=root_ca,
                private_key=key,
                certificate_chain=cert,
            )
            self.channel: grpc_aio.Channel = grpc_aio.secure_channel(
                target, credentials, options=options,
            )
            logger.info("PipelineClient TLS for target=%s", target)
        else:
            self.channel = grpc_aio.insecure_channel(target, options=options)
            logger.info("PipelineClient insecure for target=%s", target)

        self.stub = inference_pb2_grpc.InferencePipelineStub(self.channel)

    # ------------------------------------------------------------------
    # ForwardPass
    # ------------------------------------------------------------------

    async def forward(
        self,
        request_id: str,
        hidden_states: bytes,
        shape: list[int],
        dtype: str,
        attention_mask: bytes | None = None,
        attention_mask_shape: list[int] | None = None,
        params: Any | None = None,
        metadata: Any | None = None,
    ) -> Any:
        """Send hidden states to the next node for processing.

        Args:
            request_id: Unique identifier for this inference request.
            hidden_states: Serialized hidden-state tensor bytes.
            shape: Shape of the hidden-state tensor.
            dtype: Dtype string (``"float32"``, ``"float16"``, ``"bfloat16"``).
            attention_mask: Optional serialized attention mask bytes.
            attention_mask_shape: Shape of the attention mask tensor.
            params: ``GenerationParams`` protobuf message (forwarded as-is).
            metadata: ``PipelineMetadata`` protobuf message (forwarded as-is).

        Returns:
            ``ForwardResponse`` protobuf message from the target node.

        Raises:
            grpc.RpcError: On gRPC communication failure.
        """
        request = inference_pb2.ForwardRequest(
            request_id=request_id,
            hidden_states=hidden_states,
            shape=shape,
            dtype=dtype,
        )

        if attention_mask is not None:
            request.attention_mask = attention_mask
            if attention_mask_shape:
                request.attention_mask_shape.extend(attention_mask_shape)

        if params is not None:
            request.params.CopyFrom(params)

        if metadata is not None:
            request.metadata.CopyFrom(metadata)

        logger.debug(
            "Forwarding request %s to %s (shape=%s, dtype=%s)",
            request_id,
            self.target,
            shape,
            dtype,
        )

        response = await self.stub.ForwardPass(
            request, timeout=self.timeout
        )

        if not response.success:
            logger.error(
                "ForwardPass to %s failed: %s", self.target, response.error
            )

        return response

    # ------------------------------------------------------------------
    # Ping
    # ------------------------------------------------------------------

    async def ping(self) -> Any:
        """Check if the target node is alive.

        Returns:
            ``PingResponse`` protobuf message.

        Raises:
            grpc.RpcError: If the target is unreachable.
        """
        request = inference_pb2.Empty()
        response = await self.stub.Ping(request, timeout=10.0)
        logger.debug(
            "Ping %s: ready=%s layers=[%d,%d) uptime=%ds",
            self.target,
            response.is_ready,
            response.layer_start,
            response.layer_end,
            response.uptime_seconds,
        )
        return response

    # ------------------------------------------------------------------
    # GetLayerInfo
    # ------------------------------------------------------------------

    async def get_layer_info(self) -> Any:
        """Query the target node's layer assignment.

        Returns:
            ``LayerInfoResponse`` protobuf message.

        Raises:
            grpc.RpcError: If the target is unreachable.
        """
        request = inference_pb2.Empty()
        response = await self.stub.GetLayerInfo(request, timeout=10.0)
        logger.debug(
            "LayerInfo %s: model=%s layers=[%d,%d) device=%s ready=%s",
            self.target,
            response.model_name,
            response.layer_start,
            response.layer_end,
            response.device,
            response.is_ready,
        )
        return response

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Gracefully close the underlying gRPC channel."""
        await self.channel.close()
        logger.info("PipelineClient channel to %s closed", self.target)

    async def __aenter__(self) -> PipelineClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

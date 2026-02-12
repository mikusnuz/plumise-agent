"""gRPC service layer for the inference pipeline."""

from plumise_agent.grpc_.client import PipelineClient
from plumise_agent.grpc_.serializer import deserialize_tensor, serialize_tensor
from plumise_agent.grpc_.server import InferencePipelineServicer, start_grpc_server

__all__ = [
    "InferencePipelineServicer",
    "PipelineClient",
    "deserialize_tensor",
    "serialize_tensor",
    "start_grpc_server",
]

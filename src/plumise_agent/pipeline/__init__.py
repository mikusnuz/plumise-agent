"""Pipeline orchestration for distributed inference."""

from plumise_agent.pipeline.executor import PipelineExecutor
from plumise_agent.pipeline.topology import NodeSlot, PipelineTopology

__all__ = [
    "NodeSlot",
    "PipelineExecutor",
    "PipelineTopology",
]

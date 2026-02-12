"""Pipeline topology management.

Fetches the current pipeline layout from the Plumise Oracle and provides
lookup helpers for navigating the node ordering.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import aiohttp

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NodeSlot:
    """A single node's position in the inference pipeline.

    Attributes:
        address: Wallet address of the node operator.
        grpc_endpoint: How to reach this node's gRPC server (``"host:port"``).
        http_endpoint: How to reach this node's HTTP API.
        layer_start: First assigned layer (inclusive).
        layer_end: Past-the-end assigned layer (exclusive).
        pipeline_order: Zero-based position in the pipeline.
        ready: Whether the node has reported itself ready.
    """

    address: str
    grpc_endpoint: str
    http_endpoint: str
    layer_start: int
    layer_end: int
    pipeline_order: int
    ready: bool


@dataclass
class PipelineTopology:
    """Describes the full multi-node pipeline for a given model.

    Attributes:
        model_name: HuggingFace model identifier.
        total_layers: Total number of transformer layers in the model.
        nodes: Ordered list of participating nodes.
    """

    model_name: str
    total_layers: int
    nodes: list[NodeSlot] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Fetch from Oracle
    # ------------------------------------------------------------------

    @classmethod
    async def fetch(
        cls,
        oracle_url: str,
        model_name: str,
        timeout: float = 15.0,
    ) -> PipelineTopology:
        """Fetch the pipeline topology from the Oracle API.

        Calls ``GET /api/v1/pipeline/topology?model=<model>``.

        Args:
            oracle_url: Base URL of the Oracle API.
            model_name: HuggingFace model identifier.
            timeout: HTTP request timeout in seconds.

        Returns:
            A populated ``PipelineTopology`` instance.

        Raises:
            aiohttp.ClientError: On network failure.
            ValueError: If the Oracle response is malformed.
        """
        url = f"{oracle_url.rstrip('/')}/api/v1/pipeline/topology"
        params = {"model": model_name}

        logger.info("Fetching pipeline topology from %s for model=%s", url, model_name)

        client_timeout = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise ValueError(
                        f"Oracle returned HTTP {resp.status} "
                        f"for topology request: {text[:500]}"
                    )
                data = await resp.json()

        if not isinstance(data, dict):
            raise ValueError(f"Expected dict from Oracle, got {type(data).__name__}")

        total_layers = data.get("totalLayers", data.get("total_layers", 0))
        raw_nodes = data.get("nodes", [])

        nodes: list[NodeSlot] = []
        for i, raw in enumerate(raw_nodes):
            node = NodeSlot(
                address=raw.get("address", ""),
                grpc_endpoint=raw.get("grpcEndpoint", raw.get("grpc_endpoint", "")),
                http_endpoint=raw.get("httpEndpoint", raw.get("http_endpoint", "")),
                layer_start=raw.get("layerStart", raw.get("layer_start", 0)),
                layer_end=raw.get("layerEnd", raw.get("layer_end", 0)),
                pipeline_order=raw.get("pipelineOrder", raw.get("pipeline_order", i)),
                ready=raw.get("ready", False),
            )
            nodes.append(node)

        # Ensure ordering by pipeline_order
        nodes.sort(key=lambda n: n.pipeline_order)

        topology = cls(
            model_name=data.get("modelName", data.get("model_name", model_name)),
            total_layers=total_layers,
            nodes=nodes,
        )

        logger.info(
            "Pipeline topology loaded: model=%s total_layers=%d nodes=%d",
            topology.model_name,
            topology.total_layers,
            len(topology.nodes),
        )
        return topology

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def get_next_node(self, current_order: int) -> NodeSlot | None:
        """Return the next node in the pipeline, or ``None`` if current is last.

        Args:
            current_order: ``pipeline_order`` of the current node.

        Returns:
            The ``NodeSlot`` with the next higher ``pipeline_order``, or ``None``.
        """
        for node in self.nodes:
            if node.pipeline_order == current_order + 1:
                return node
        return None

    def find_by_address(self, address: str) -> tuple[int, NodeSlot] | None:
        """Find a node by its wallet address (case-insensitive).

        Args:
            address: Ethereum wallet address to search for.

        Returns:
            ``(index, NodeSlot)`` tuple, or ``None`` if not found.
        """
        addr_lower = address.lower()
        for i, node in enumerate(self.nodes):
            if node.address.lower() == addr_lower:
                return i, node
        return None

    def get_node_by_order(self, order: int) -> NodeSlot | None:
        """Return the node with a given ``pipeline_order``."""
        for node in self.nodes:
            if node.pipeline_order == order:
                return node
        return None

    @property
    def is_single_node(self) -> bool:
        """True if the pipeline has at most one node."""
        return len(self.nodes) <= 1

    @property
    def first_node(self) -> NodeSlot | None:
        """Return the first node in the pipeline (order 0)."""
        return self.nodes[0] if self.nodes else None

    @property
    def last_node(self) -> NodeSlot | None:
        """Return the last node in the pipeline."""
        return self.nodes[-1] if self.nodes else None

    def to_dict(self) -> dict:
        """Serialize the topology to a plain dictionary."""
        return {
            "model": self.model_name,
            "totalLayers": self.total_layers,
            "nodes": [
                {
                    "address": n.address,
                    "grpcEndpoint": n.grpc_endpoint,
                    "httpEndpoint": n.http_endpoint,
                    "layerStart": n.layer_start,
                    "layerEnd": n.layer_end,
                    "pipelineOrder": n.pipeline_order,
                    "ready": n.ready,
                }
                for n in self.nodes
            ],
        }

    def __repr__(self) -> str:
        node_summary = ", ".join(
            f"[{n.pipeline_order}]={n.address[:10]}..({n.layer_start}-{n.layer_end})"
            for n in self.nodes
        )
        return (
            f"PipelineTopology(model={self.model_name}, "
            f"layers={self.total_layers}, nodes=[{node_summary}])"
        )

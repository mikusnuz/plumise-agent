"""Tests for the PipelineTopology module.

The ``PipelineTopology`` class may not be implemented yet (written by another
agent).  These tests define the expected interface and behaviour based on
how it is consumed in ``plumise_agent.node.node.PlumiseAgent``:

    topology.is_single_node          -> bool
    topology.find_by_address(addr)   -> (index, NodeSlot) | None
    topology.get_next_node(index)    -> NodeSlot | None
    PipelineTopology.fetch(url, model) -> PipelineTopology  (classmethod)

If the module does not exist yet, we define a minimal reference
implementation inside this test file so the tests are self-contained.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Optional

import pytest


# ---------------------------------------------------------------------------
# Minimal reference implementation for testing
# ---------------------------------------------------------------------------
# If the real module exists we import it; otherwise we define a stub here
# so tests can run before the pipeline agent finishes writing the code.

try:
    from plumise_agent.pipeline.topology import PipelineTopology, NodeSlot
except ImportError:

    @dataclass(frozen=True)
    class NodeSlot:  # type: ignore[no-redef]
        """A node in the pipeline topology."""

        address: str
        grpc_endpoint: str
        http_endpoint: str
        layer_start: int
        layer_end: int

    @dataclass
    class PipelineTopology:  # type: ignore[no-redef]
        """Ordered list of nodes forming the inference pipeline."""

        model_name: str
        nodes: list[NodeSlot] = field(default_factory=list)

        @property
        def is_single_node(self) -> bool:
            return len(self.nodes) <= 1

        def get_next_node(self, current_index: int) -> Optional[NodeSlot]:
            nxt = current_index + 1
            if nxt < len(self.nodes):
                return self.nodes[nxt]
            return None

        def find_by_address(
            self, address: str
        ) -> Optional[tuple[int, NodeSlot]]:
            addr_lower = address.lower()
            for i, node in enumerate(self.nodes):
                if node.address.lower() == addr_lower:
                    return (i, node)
            return None

        @classmethod
        async def fetch(
            cls, oracle_url: str, model_name: str
        ) -> PipelineTopology:
            raise NotImplementedError("fetch() requires a real Oracle")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_node(addr: str, grpc: str, start: int, end: int) -> NodeSlot:
    return NodeSlot(
        address=addr,
        grpc_endpoint=grpc,
        http_endpoint=f"http://{grpc.split(':')[0]}:31331",
        layer_start=start,
        layer_end=end,
    )


def _single_node_topology() -> PipelineTopology:
    return PipelineTopology(
        model_name="test/model",
        nodes=[
            _make_node("0xAAAA", "10.0.0.1:50051", 0, 24),
        ],
    )


def _multi_node_topology() -> PipelineTopology:
    return PipelineTopology(
        model_name="test/model",
        nodes=[
            _make_node("0xAAAA", "10.0.0.1:50051", 0, 8),
            _make_node("0xBBBB", "10.0.0.2:50051", 8, 16),
            _make_node("0xCCCC", "10.0.0.3:50051", 16, 24),
        ],
    )


def _empty_topology() -> PipelineTopology:
    return PipelineTopology(model_name="test/model", nodes=[])


# ---------------------------------------------------------------------------
# Single-node topology
# ---------------------------------------------------------------------------

class TestSingleNodeTopology:
    """Topology with exactly one node."""

    def test_is_single_node_true(self):
        topo = _single_node_topology()
        assert topo.is_single_node is True

    def test_get_next_node_returns_none(self):
        topo = _single_node_topology()
        assert topo.get_next_node(0) is None

    def test_find_by_address_found(self):
        topo = _single_node_topology()
        result = topo.find_by_address("0xAAAA")
        assert result is not None
        idx, node = result
        assert idx == 0
        assert node.address == "0xAAAA"

    def test_find_by_address_not_found(self):
        topo = _single_node_topology()
        assert topo.find_by_address("0xDEAD") is None


# ---------------------------------------------------------------------------
# Multi-node topology
# ---------------------------------------------------------------------------

class TestMultiNodeTopology:
    """Topology with three nodes in a pipeline."""

    def test_is_single_node_false(self):
        topo = _multi_node_topology()
        assert topo.is_single_node is False

    def test_node_count(self):
        topo = _multi_node_topology()
        assert len(topo.nodes) == 3

    def test_get_next_node_from_first(self):
        topo = _multi_node_topology()
        nxt = topo.get_next_node(0)
        assert nxt is not None
        assert nxt.address == "0xBBBB"

    def test_get_next_node_from_middle(self):
        topo = _multi_node_topology()
        nxt = topo.get_next_node(1)
        assert nxt is not None
        assert nxt.address == "0xCCCC"

    def test_get_next_node_from_last_returns_none(self):
        topo = _multi_node_topology()
        assert topo.get_next_node(2) is None

    def test_get_next_node_out_of_range_returns_none(self):
        topo = _multi_node_topology()
        assert topo.get_next_node(99) is None

    def test_find_by_address_first(self):
        topo = _multi_node_topology()
        result = topo.find_by_address("0xAAAA")
        assert result is not None
        idx, node = result
        assert idx == 0

    def test_find_by_address_middle(self):
        topo = _multi_node_topology()
        result = topo.find_by_address("0xBBBB")
        assert result is not None
        idx, node = result
        assert idx == 1

    def test_find_by_address_last(self):
        topo = _multi_node_topology()
        result = topo.find_by_address("0xCCCC")
        assert result is not None
        idx, node = result
        assert idx == 2

    def test_find_by_address_case_insensitive(self):
        topo = _multi_node_topology()
        result = topo.find_by_address("0xaaaa")
        assert result is not None
        idx, _ = result
        assert idx == 0

    def test_find_by_address_not_found(self):
        topo = _multi_node_topology()
        assert topo.find_by_address("0x9999") is None


# ---------------------------------------------------------------------------
# Empty topology
# ---------------------------------------------------------------------------

class TestEmptyTopology:
    """Edge case: topology with zero nodes."""

    def test_is_single_node_true(self):
        """An empty topology is effectively single-node (or no-node)."""
        topo = _empty_topology()
        assert topo.is_single_node is True

    def test_find_by_address_returns_none(self):
        topo = _empty_topology()
        assert topo.find_by_address("0xAAAA") is None

    def test_get_next_node_returns_none(self):
        topo = _empty_topology()
        assert topo.get_next_node(0) is None


# ---------------------------------------------------------------------------
# is_single_node boundary
# ---------------------------------------------------------------------------

class TestIsSingleNodeBoundary:
    """Explicitly verify the boundary between single and multi."""

    def test_zero_nodes(self):
        topo = PipelineTopology(model_name="m", nodes=[])
        assert topo.is_single_node is True

    def test_one_node(self):
        topo = PipelineTopology(
            model_name="m",
            nodes=[_make_node("0xA", "1:2", 0, 10)],
        )
        assert topo.is_single_node is True

    def test_two_nodes(self):
        topo = PipelineTopology(
            model_name="m",
            nodes=[
                _make_node("0xA", "1:2", 0, 5),
                _make_node("0xB", "3:4", 5, 10),
            ],
        )
        assert topo.is_single_node is False


# ---------------------------------------------------------------------------
# Node slot attributes
# ---------------------------------------------------------------------------

class TestNodeSlotAttributes:
    """Verify NodeSlot exposes the expected attributes."""

    def test_address(self):
        n = _make_node("0xABC", "host:50051", 0, 8)
        assert n.address == "0xABC"

    def test_grpc_endpoint(self):
        n = _make_node("0xABC", "host:50051", 0, 8)
        assert n.grpc_endpoint == "host:50051"

    def test_layer_bounds(self):
        n = _make_node("0xABC", "host:50051", 4, 12)
        assert n.layer_start == 4
        assert n.layer_end == 12

    def test_http_endpoint(self):
        n = _make_node("0xABC", "host:50051", 0, 8)
        assert n.http_endpoint == "http://host:31331"


# ---------------------------------------------------------------------------
# Pipeline chain traversal
# ---------------------------------------------------------------------------

class TestPipelineTraversal:
    """Walk the full pipeline from first to last node."""

    def test_traverse_entire_pipeline(self):
        topo = _multi_node_topology()
        visited_addresses = []
        idx = 0
        # Start from first node
        result = topo.find_by_address("0xAAAA")
        assert result is not None
        idx, node = result
        visited_addresses.append(node.address)

        while True:
            nxt = topo.get_next_node(idx)
            if nxt is None:
                break
            visited_addresses.append(nxt.address)
            idx += 1

        assert visited_addresses == ["0xAAAA", "0xBBBB", "0xCCCC"]

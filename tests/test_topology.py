"""Tests for plumise_agent.pipeline.topology -- PipelineTopology and NodeSlot."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pytest

from plumise_agent.pipeline.topology import PipelineTopology, NodeSlot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_node(
    addr: str,
    grpc: str,
    start: int,
    end: int,
    order: int = 0,
    ready: bool = True,
) -> NodeSlot:
    return NodeSlot(
        address=addr,
        grpc_endpoint=grpc,
        http_endpoint=f"http://{grpc.split(':')[0]}:31331",
        layer_start=start,
        layer_end=end,
        pipeline_order=order,
        ready=ready,
    )


def _single_node_topology() -> PipelineTopology:
    return PipelineTopology(
        model_name="test/model",
        total_layers=24,
        nodes=[
            _make_node("0xAAAA", "10.0.0.1:50051", 0, 24, order=0),
        ],
    )


def _multi_node_topology() -> PipelineTopology:
    return PipelineTopology(
        model_name="test/model",
        total_layers=24,
        nodes=[
            _make_node("0xAAAA", "10.0.0.1:50051", 0, 8, order=0),
            _make_node("0xBBBB", "10.0.0.2:50051", 8, 16, order=1),
            _make_node("0xCCCC", "10.0.0.3:50051", 16, 24, order=2),
        ],
    )


def _empty_topology() -> PipelineTopology:
    return PipelineTopology(model_name="test/model", total_layers=24, nodes=[])


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

    def test_get_node_by_order(self):
        topo = _multi_node_topology()
        node = topo.get_node_by_order(1)
        assert node is not None
        assert node.address == "0xBBBB"

    def test_get_node_by_order_not_found(self):
        topo = _multi_node_topology()
        assert topo.get_node_by_order(99) is None


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

    def test_first_node_none(self):
        topo = _empty_topology()
        assert topo.first_node is None

    def test_last_node_none(self):
        topo = _empty_topology()
        assert topo.last_node is None


# ---------------------------------------------------------------------------
# is_single_node boundary
# ---------------------------------------------------------------------------

class TestIsSingleNodeBoundary:
    """Explicitly verify the boundary between single and multi."""

    def test_zero_nodes(self):
        topo = PipelineTopology(model_name="m", total_layers=10, nodes=[])
        assert topo.is_single_node is True

    def test_one_node(self):
        topo = PipelineTopology(
            model_name="m",
            total_layers=10,
            nodes=[_make_node("0xA", "1:2", 0, 10, order=0)],
        )
        assert topo.is_single_node is True

    def test_two_nodes(self):
        topo = PipelineTopology(
            model_name="m",
            total_layers=10,
            nodes=[
                _make_node("0xA", "1:2", 0, 5, order=0),
                _make_node("0xB", "3:4", 5, 10, order=1),
            ],
        )
        assert topo.is_single_node is False


# ---------------------------------------------------------------------------
# Node slot attributes
# ---------------------------------------------------------------------------

class TestNodeSlotAttributes:
    """Verify NodeSlot exposes the expected attributes."""

    def test_address(self):
        n = _make_node("0xABC", "host:50051", 0, 8, order=0)
        assert n.address == "0xABC"

    def test_grpc_endpoint(self):
        n = _make_node("0xABC", "host:50051", 0, 8, order=0)
        assert n.grpc_endpoint == "host:50051"

    def test_layer_bounds(self):
        n = _make_node("0xABC", "host:50051", 4, 12, order=0)
        assert n.layer_start == 4
        assert n.layer_end == 12

    def test_http_endpoint(self):
        n = _make_node("0xABC", "host:50051", 0, 8, order=0)
        assert n.http_endpoint == "http://host:31331"

    def test_pipeline_order(self):
        n = _make_node("0xABC", "host:50051", 0, 8, order=5)
        assert n.pipeline_order == 5

    def test_ready_default_true(self):
        n = _make_node("0xABC", "host:50051", 0, 8, order=0)
        assert n.ready is True

    def test_ready_false(self):
        n = _make_node("0xABC", "host:50051", 0, 8, order=0, ready=False)
        assert n.ready is False

    def test_frozen(self):
        n = _make_node("0xABC", "host:50051", 0, 8, order=0)
        with pytest.raises(AttributeError):
            n.address = "0xDEF"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# first_node / last_node
# ---------------------------------------------------------------------------

class TestFirstLastNode:
    """Test the first_node and last_node properties."""

    def test_first_node_single(self):
        topo = _single_node_topology()
        assert topo.first_node is not None
        assert topo.first_node.address == "0xAAAA"

    def test_last_node_single(self):
        topo = _single_node_topology()
        assert topo.last_node is not None
        assert topo.last_node.address == "0xAAAA"

    def test_first_node_multi(self):
        topo = _multi_node_topology()
        assert topo.first_node is not None
        assert topo.first_node.pipeline_order == 0

    def test_last_node_multi(self):
        topo = _multi_node_topology()
        assert topo.last_node is not None
        assert topo.last_node.pipeline_order == 2


# ---------------------------------------------------------------------------
# to_dict serialization
# ---------------------------------------------------------------------------

class TestToDict:
    """Test the to_dict serialization."""

    def test_to_dict_keys(self):
        topo = _multi_node_topology()
        d = topo.to_dict()
        assert "model" in d
        assert "totalLayers" in d
        assert "nodes" in d

    def test_to_dict_node_count(self):
        topo = _multi_node_topology()
        d = topo.to_dict()
        assert len(d["nodes"]) == 3

    def test_to_dict_node_fields(self):
        topo = _single_node_topology()
        d = topo.to_dict()
        node = d["nodes"][0]
        assert "address" in node
        assert "grpcEndpoint" in node
        assert "httpEndpoint" in node
        assert "layerStart" in node
        assert "layerEnd" in node
        assert "pipelineOrder" in node
        assert "ready" in node


# ---------------------------------------------------------------------------
# Pipeline chain traversal
# ---------------------------------------------------------------------------

class TestPipelineTraversal:
    """Walk the full pipeline from first to last node."""

    def test_traverse_entire_pipeline(self):
        topo = _multi_node_topology()
        visited_addresses = []
        # Start from first node
        result = topo.find_by_address("0xAAAA")
        assert result is not None
        idx, node = result
        visited_addresses.append(node.address)

        current_order = node.pipeline_order
        while True:
            nxt = topo.get_next_node(current_order)
            if nxt is None:
                break
            visited_addresses.append(nxt.address)
            current_order = nxt.pipeline_order

        assert visited_addresses == ["0xAAAA", "0xBBBB", "0xCCCC"]


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------

class TestRepr:
    """Ensure repr is informative."""

    def test_repr_contains_model(self):
        topo = _single_node_topology()
        r = repr(topo)
        assert "test/model" in r

    def test_repr_contains_layer_count(self):
        topo = _single_node_topology()
        r = repr(topo)
        assert "24" in r

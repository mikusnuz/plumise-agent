"""Shared fixtures for plumise-agent tests.

Provides mock configurations, sample tensors, and topology helpers
so individual test modules can stay focused on their own assertions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
import torch

from plumise_agent.model.layer_range import LayerRange


# ---------------------------------------------------------------------------
# AgentConfig fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_config():
    """Return an AgentConfig populated with safe test defaults.

    The private key is a well-known Hardhat/Foundry test key so no real
    funds are ever at risk.
    """
    # Prevent pydantic-settings from reading a real .env file
    with patch.dict(os.environ, {}, clear=False):
        from plumise_agent.chain.config import AgentConfig

        return AgentConfig(
            plumise_rpc_url="http://localhost:26902",
            plumise_chain_id=41956,
            plumise_private_key="0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
            agent_registry_address=None,
            reward_pool_address=None,
            oracle_api_url="http://localhost:3100",
            report_interval=60,
            model_name="test-org/test-model",
            device="cpu",
            hf_token="",
            layer_start=None,
            layer_end=None,
            grpc_host="0.0.0.0",
            grpc_port=50051,
            api_host="0.0.0.0",
            api_port=31331,
            node_endpoint="",
            grpc_endpoint="",
            verify_on_chain=False,
            claim_threshold_wei=10**18,
        )


# ---------------------------------------------------------------------------
# Tensor fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_tensor_f32() -> torch.Tensor:
    """Small float32 tensor for serialization tests."""
    return torch.randn(2, 4, 8, dtype=torch.float32)


@pytest.fixture()
def sample_tensor_f16() -> torch.Tensor:
    """Small float16 tensor for serialization tests."""
    return torch.randn(2, 4, 8, dtype=torch.float16)


@pytest.fixture()
def sample_tensor_bf16() -> torch.Tensor:
    """Small bfloat16 tensor for serialization tests."""
    return torch.randn(2, 4, 8, dtype=torch.bfloat16)


@pytest.fixture()
def sample_tensor() -> torch.Tensor:
    """Generic sample tensor (float32)."""
    return torch.randn(2, 4, 8, dtype=torch.float32)


# ---------------------------------------------------------------------------
# LayerRange fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def full_layer_range() -> LayerRange:
    """A LayerRange covering all 24 layers (single-node)."""
    return LayerRange(start=0, end=24, total=24)


@pytest.fixture()
def first_layer_range() -> LayerRange:
    """A LayerRange for the first node: layers [0, 8) of 24."""
    return LayerRange(start=0, end=8, total=24)


@pytest.fixture()
def middle_layer_range() -> LayerRange:
    """A LayerRange for a middle node: layers [8, 16) of 24."""
    return LayerRange(start=8, end=16, total=24)


@pytest.fixture()
def last_layer_range() -> LayerRange:
    """A LayerRange for the last node: layers [16, 24) of 24."""
    return LayerRange(start=16, end=24, total=24)


# ---------------------------------------------------------------------------
# Topology helpers
# ---------------------------------------------------------------------------

@dataclass
class MockNodeSlot:
    """Lightweight stand-in for PipelineTopology node slots."""

    address: str
    grpc_endpoint: str
    http_endpoint: str
    layer_start: int
    layer_end: int


@pytest.fixture()
def mock_topology_nodes() -> list[MockNodeSlot]:
    """Three nodes splitting 24 layers evenly."""
    return [
        MockNodeSlot(
            address="0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
            grpc_endpoint="10.0.0.1:50051",
            http_endpoint="http://10.0.0.1:31331",
            layer_start=0,
            layer_end=8,
        ),
        MockNodeSlot(
            address="0xBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB",
            grpc_endpoint="10.0.0.2:50051",
            http_endpoint="http://10.0.0.2:31331",
            layer_start=8,
            layer_end=16,
        ),
        MockNodeSlot(
            address="0xCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",
            grpc_endpoint="10.0.0.3:50051",
            http_endpoint="http://10.0.0.3:31331",
            layer_start=16,
            layer_end=24,
        ),
    ]

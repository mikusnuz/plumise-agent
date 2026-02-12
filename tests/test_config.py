"""Tests for plumise_agent.chain.config.AgentConfig."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from plumise_agent.chain.config import AgentConfig, _CONTRACTS_DIR


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------

class TestAgentConfigDefaults:
    """Verify that AgentConfig has sensible defaults for every field."""

    def test_default_rpc_url(self):
        cfg = AgentConfig(plumise_private_key="0x" + "ab" * 32)
        assert cfg.plumise_rpc_url == "http://localhost:26902"

    def test_default_chain_id(self):
        cfg = AgentConfig(plumise_private_key="0x" + "ab" * 32)
        assert cfg.plumise_chain_id == 41956

    def test_default_model_name(self):
        cfg = AgentConfig(plumise_private_key="0x" + "ab" * 32)
        assert cfg.model_name == "bigscience/bloom-560m"

    def test_default_device(self):
        cfg = AgentConfig(plumise_private_key="0x" + "ab" * 32)
        assert cfg.device == "auto"

    def test_default_grpc_port(self):
        cfg = AgentConfig(plumise_private_key="0x" + "ab" * 32)
        assert cfg.grpc_port == 50051

    def test_default_api_port(self):
        cfg = AgentConfig(plumise_private_key="0x" + "ab" * 32)
        assert cfg.api_port == 31331

    def test_default_report_interval(self):
        cfg = AgentConfig(plumise_private_key="0x" + "ab" * 32)
        assert cfg.report_interval == 60

    def test_default_verify_on_chain_false(self):
        cfg = AgentConfig(plumise_private_key="0x" + "ab" * 32)
        assert cfg.verify_on_chain is False

    def test_default_claim_threshold(self):
        cfg = AgentConfig(plumise_private_key="0x" + "ab" * 32)
        assert cfg.claim_threshold_wei == 10**18

    def test_default_agent_registry_none(self):
        cfg = AgentConfig(plumise_private_key="0x" + "ab" * 32)
        assert cfg.agent_registry_address is None

    def test_default_reward_pool_none(self):
        cfg = AgentConfig(plumise_private_key="0x" + "ab" * 32)
        assert cfg.reward_pool_address is None

    def test_default_layer_start_none(self):
        cfg = AgentConfig(plumise_private_key="0x" + "ab" * 32)
        assert cfg.layer_start is None

    def test_default_layer_end_none(self):
        cfg = AgentConfig(plumise_private_key="0x" + "ab" * 32)
        assert cfg.layer_end is None


# ---------------------------------------------------------------------------
# Environment variable overrides
# ---------------------------------------------------------------------------

class TestAgentConfigEnvOverride:
    """Verify that environment variables override defaults."""

    def test_rpc_url_from_env(self):
        with patch.dict(os.environ, {"PLUMISE_RPC_URL": "http://custom:8545"}):
            cfg = AgentConfig(plumise_private_key="0x" + "ab" * 32)
            assert cfg.plumise_rpc_url == "http://custom:8545"

    def test_chain_id_from_env(self):
        with patch.dict(os.environ, {"PLUMISE_CHAIN_ID": "12345"}):
            cfg = AgentConfig(plumise_private_key="0x" + "ab" * 32)
            assert cfg.plumise_chain_id == 12345

    def test_model_name_from_env(self):
        with patch.dict(os.environ, {"MODEL_NAME": "custom/model"}):
            cfg = AgentConfig(plumise_private_key="0x" + "ab" * 32)
            assert cfg.model_name == "custom/model"

    def test_grpc_port_from_env(self):
        with patch.dict(os.environ, {"GRPC_PORT": "55555"}):
            cfg = AgentConfig(plumise_private_key="0x" + "ab" * 32)
            assert cfg.grpc_port == 55555

    def test_report_interval_from_env(self):
        with patch.dict(os.environ, {"REPORT_INTERVAL": "120"}):
            cfg = AgentConfig(plumise_private_key="0x" + "ab" * 32)
            assert cfg.report_interval == 120

    def test_private_key_from_env(self):
        key = "ab" * 32
        with patch.dict(os.environ, {"PLUMISE_PRIVATE_KEY": key}):
            cfg = AgentConfig()
            # Should be normalized to have 0x prefix
            assert cfg.plumise_private_key == "0x" + key


# ---------------------------------------------------------------------------
# Private key normalization
# ---------------------------------------------------------------------------

class TestPrivateKeyNormalization:
    """Verify the field_validator on plumise_private_key."""

    def test_with_0x_prefix_unchanged(self):
        key = "0x" + "ab" * 32
        cfg = AgentConfig(plumise_private_key=key)
        assert cfg.plumise_private_key == key

    def test_without_0x_prefix_adds_it(self):
        raw = "ab" * 32
        cfg = AgentConfig(plumise_private_key=raw)
        assert cfg.plumise_private_key == "0x" + raw

    def test_empty_key_stays_empty(self):
        cfg = AgentConfig(plumise_private_key="")
        assert cfg.plumise_private_key == ""

    def test_whitespace_stripped(self):
        raw = "  " + "ab" * 32 + "  "
        cfg = AgentConfig(plumise_private_key=raw)
        assert cfg.plumise_private_key == "0x" + "ab" * 32

    def test_0x_prefix_with_whitespace(self):
        raw = "  0x" + "ab" * 32 + "  "
        cfg = AgentConfig(plumise_private_key=raw)
        assert cfg.plumise_private_key == "0x" + "ab" * 32


# ---------------------------------------------------------------------------
# Validation: port ranges and intervals
# ---------------------------------------------------------------------------

class TestAgentConfigValidation:
    """Test pydantic field constraints (ge, le)."""

    def test_grpc_port_min(self):
        with pytest.raises(Exception):  # ValidationError
            AgentConfig(plumise_private_key="0x" + "ab" * 32, grpc_port=0)

    def test_grpc_port_max(self):
        with pytest.raises(Exception):
            AgentConfig(plumise_private_key="0x" + "ab" * 32, grpc_port=70000)

    def test_grpc_port_valid_boundary_low(self):
        cfg = AgentConfig(plumise_private_key="0x" + "ab" * 32, grpc_port=1)
        assert cfg.grpc_port == 1

    def test_grpc_port_valid_boundary_high(self):
        cfg = AgentConfig(plumise_private_key="0x" + "ab" * 32, grpc_port=65535)
        assert cfg.grpc_port == 65535

    def test_api_port_min(self):
        with pytest.raises(Exception):
            AgentConfig(plumise_private_key="0x" + "ab" * 32, api_port=0)

    def test_api_port_max(self):
        with pytest.raises(Exception):
            AgentConfig(plumise_private_key="0x" + "ab" * 32, api_port=70000)

    def test_report_interval_too_low(self):
        with pytest.raises(Exception):
            AgentConfig(plumise_private_key="0x" + "ab" * 32, report_interval=5)

    def test_report_interval_minimum_valid(self):
        cfg = AgentConfig(plumise_private_key="0x" + "ab" * 32, report_interval=10)
        assert cfg.report_interval == 10

    def test_claim_threshold_zero_allowed(self):
        cfg = AgentConfig(plumise_private_key="0x" + "ab" * 32, claim_threshold_wei=0)
        assert cfg.claim_threshold_wei == 0

    def test_claim_threshold_negative_rejected(self):
        with pytest.raises(Exception):
            AgentConfig(plumise_private_key="0x" + "ab" * 32, claim_threshold_wei=-1)


# ---------------------------------------------------------------------------
# load_abi
# ---------------------------------------------------------------------------

class TestLoadAbi:
    """Test the static load_abi helper."""

    def test_load_abi_missing_file(self):
        """FileNotFoundError when the ABI JSON does not exist."""
        with pytest.raises(FileNotFoundError):
            AgentConfig.load_abi("NonExistentContract")

    def test_load_abi_raw_array(self, tmp_path: Path):
        """When the JSON file is a raw ABI list, return it directly."""
        abi_list = [{"type": "function", "name": "foo", "inputs": []}]
        abi_file = tmp_path / "TestContract.json"
        abi_file.write_text(json.dumps(abi_list))

        import plumise_agent.chain.config as config_mod
        original = config_mod._CONTRACTS_DIR
        config_mod._CONTRACTS_DIR = tmp_path
        try:
            result = AgentConfig.load_abi("TestContract")
            assert result == abi_list
        finally:
            config_mod._CONTRACTS_DIR = original

    def test_load_abi_wrapper_object(self, tmp_path: Path):
        """When the JSON has an ``abi`` key, extract that."""
        inner_abi = [{"type": "function", "name": "bar", "inputs": []}]
        wrapper = {"abi": inner_abi, "bytecode": "0x1234"}
        abi_file = tmp_path / "Wrapped.json"
        abi_file.write_text(json.dumps(wrapper))

        import plumise_agent.chain.config as config_mod
        original = config_mod._CONTRACTS_DIR
        config_mod._CONTRACTS_DIR = tmp_path
        try:
            result = AgentConfig.load_abi("Wrapped")
            assert result == inner_abi
        finally:
            config_mod._CONTRACTS_DIR = original

    def test_load_abi_wrapper_without_abi_key(self, tmp_path: Path):
        """When the JSON is a dict but has no ``abi`` key, return the dict."""
        data = {"name": "SomeContract", "functions": []}
        abi_file = tmp_path / "NoAbiKey.json"
        abi_file.write_text(json.dumps(data))

        import plumise_agent.chain.config as config_mod
        original = config_mod._CONTRACTS_DIR
        config_mod._CONTRACTS_DIR = tmp_path
        try:
            result = AgentConfig.load_abi("NoAbiKey")
            assert result == data
        finally:
            config_mod._CONTRACTS_DIR = original

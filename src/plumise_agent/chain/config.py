"""Configuration management for Plumise Agent.

Loads settings from environment variables and/or .env file using
pydantic-settings (Pydantic v2). All Petals/DHT fields have been removed;
this config is purpose-built for the distributed gRPC inference pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Contract ABI directory: <project_root>/contracts/
_CONTRACTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "contracts"


class AgentConfig(BaseSettings):
    """Plumise Agent configuration.

    Values are loaded in priority order:
      1. Explicit constructor arguments
      2. Environment variables
      3. ``.env`` file
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- Chain ----
    plumise_rpc_url: str = Field(
        default="http://localhost:26902",
        description="Plumise chain JSON-RPC endpoint",
    )
    plumise_chain_id: int = Field(
        default=41956,
        description="Plumise chain ID",
    )
    plumise_private_key: str = Field(
        default="",
        description="Hex-encoded private key for the agent wallet",
    )
    agent_registry_address: str | None = Field(
        default=None,
        description="AgentRegistry contract address (deployed post-genesis)",
    )
    reward_pool_address: str | None = Field(
        default=None,
        description="RewardPool contract address (deployed post-genesis)",
    )

    # ---- Oracle ----
    oracle_api_url: str = Field(
        default="http://localhost:3100",
        description="Plumise Oracle API base URL",
    )
    report_interval: int = Field(
        default=60,
        ge=10,
        description="Metrics report interval in seconds",
    )

    # ---- Model ----
    model_name: str = Field(
        default="bigscience/bloom-560m",
        description="HuggingFace model identifier to serve",
    )
    device: str = Field(
        default="auto",
        description="Device: auto, cpu, cuda, cuda:0, mps, etc.",
    )
    hf_token: str = Field(
        default="",
        description="Optional HuggingFace token for gated models",
    )

    # ---- Layer assignment (None = Oracle auto-assigns) ----
    layer_start: int | None = Field(
        default=None,
        description="First transformer layer index (inclusive). None lets Oracle decide.",
    )
    layer_end: int | None = Field(
        default=None,
        description="Last transformer layer index (exclusive). None lets Oracle decide.",
    )

    # ---- gRPC ----
    grpc_host: str = Field(
        default="0.0.0.0",
        description="gRPC server listen address",
    )
    grpc_port: int = Field(
        default=50051,
        ge=1,
        le=65535,
        description="gRPC server listen port",
    )

    # ---- HTTP API ----
    api_host: str = Field(
        default="0.0.0.0",
        description="HTTP API server listen address",
    )
    api_port: int = Field(
        default=31331,
        ge=1,
        le=65535,
        description="HTTP API server port for inference requests",
    )

    # ---- Public endpoints (how other nodes reach this one) ----
    node_endpoint: str = Field(
        default="",
        description="Public HTTP endpoint for this node (e.g. http://1.2.3.4:31331)",
    )
    grpc_endpoint: str = Field(
        default="",
        description="Public gRPC endpoint for this node (e.g. 1.2.3.4:50051)",
    )

    # ---- Inference proof ----
    verify_on_chain: bool = Field(
        default=False,
        description="Enable on-chain proof verification via precompile 0x20",
    )
    claim_threshold_wei: int = Field(
        default=10**18,  # 1 PLM
        ge=0,
        description="Minimum pending reward (wei) to trigger auto-claim",
    )

    # ---- Validators ----

    @field_validator("plumise_private_key")
    @classmethod
    def _normalize_private_key(cls, v: str) -> str:
        """Ensure the private key has a ``0x`` prefix."""
        if not v:
            return v
        v = v.strip()
        if not v.startswith("0x"):
            v = "0x" + v
        return v

    # ---- ABI loader ----

    @staticmethod
    def load_abi(name: str) -> list:
        """Load a contract ABI from the ``contracts/`` directory.

        Args:
            name: Contract name without extension, e.g. ``"AgentRegistry"``.

        Returns:
            Parsed ABI as a list of dicts.

        Raises:
            FileNotFoundError: If the ABI file does not exist.
        """
        path = _CONTRACTS_DIR / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"ABI file not found: {path}")
        with open(path) as f:
            data = json.load(f)
        # Support both raw ABI arrays and {abi: [...]} wrappers
        if isinstance(data, list):
            return data
        return data.get("abi", data)


# Backward-compatible alias used by chain.auth, chain.rewards, chain.agent, etc.
PlumiseConfig = AgentConfig

"""Agent authentication with Plumise chain.

Provides wallet-based identity and contract verification for agents
participating in the distributed inference network.
"""

from __future__ import annotations

import json
import logging
from typing import Any, NamedTuple, Optional

from eth_account import Account
from eth_account.messages import encode_defunct
from web3 import Web3
from web3.contract import Contract

from plumise_agent.chain.config import PlumiseConfig

logger = logging.getLogger(__name__)


class AgentInfo(NamedTuple):
    """Parsed agent record from AgentRegistry."""

    address: str
    name: str
    metadata: str
    status: int  # 0=Inactive, 1=Active, 2=Suspended
    registered_at: int
    last_active: int
    total_tasks: int


class PlumiseAuth:
    """Wallet-based authentication and contract verification.

    Args:
        config: Plumise configuration instance.
    """

    def __init__(self, config: PlumiseConfig) -> None:
        self.config = config
        self.w3 = Web3(Web3.HTTPProvider(
            config.plumise_rpc_url,
            request_kwargs={"timeout": 30},
        ))
        self.account = Account.from_key(config.plumise_private_key)
        self.address: str = self.account.address
        self.chain_id: int = config.plumise_chain_id

        self._registry: Optional[Contract] = None
        if config.agent_registry_address:
            abi = PlumiseConfig.load_abi("AgentRegistry")
            self._registry = self.w3.eth.contract(
                address=Web3.to_checksum_address(config.agent_registry_address),
                abi=abi,
            )

        logger.info("PlumiseAuth initialized for %s (chain %d)", self.address, self.chain_id)

    # ------------------------------------------------------------------
    # Chain verification
    # ------------------------------------------------------------------

    def is_chain_connected(self) -> bool:
        """Return True if the RPC endpoint is reachable."""
        try:
            self.w3.eth.block_number
            return True
        except Exception:
            return False

    def verify_registration(self) -> bool:
        """Check whether this agent is registered in AgentRegistry.

        Returns ``True`` when the contract confirms the agent is registered.
        If the contract address is not configured yet, returns ``True``
        optimistically (pre-genesis mode).
        """
        if self._registry is None:
            logger.warning(
                "AgentRegistry address not configured; skipping registration check"
            )
            return True

        try:
            return self._registry.functions.isRegistered(self.address).call()
        except Exception as exc:
            logger.error("Failed to check registration: %s", exc)
            return False

    def is_active(self) -> bool:
        """Check whether the agent is in *Active* status."""
        if self._registry is None:
            return True

        try:
            return self._registry.functions.isActive(self.address).call()
        except Exception as exc:
            logger.error("Failed to check active status: %s", exc)
            return False

    def get_agent_info(self) -> Optional[AgentInfo]:
        """Fetch full agent record from the registry."""
        if self._registry is None:
            return None

        try:
            raw = self._registry.functions.getAgent(self.address).call()
            return AgentInfo(*raw)
        except Exception as exc:
            logger.error("Failed to fetch agent info: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Message signing
    # ------------------------------------------------------------------

    def sign_message(self, message: str) -> str:
        """Sign an arbitrary message using ``personal_sign``.

        Args:
            message: Plain-text message to sign.

        Returns:
            Hex-encoded signature string.
        """
        msg = encode_defunct(text=message)
        signed = self.account.sign_message(msg)
        return signed.signature.hex()

    def sign_payload(self, payload: dict[str, Any]) -> str:
        """Deterministically sign a JSON payload.

        The payload is serialized with sorted keys to guarantee a
        canonical representation before signing.

        Args:
            payload: Dictionary to sign.

        Returns:
            Hex-encoded signature string.
        """
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return self.sign_message(canonical)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_balance(self) -> int:
        """Return the native token balance in wei."""
        return self.w3.eth.get_balance(self.address)

    def __repr__(self) -> str:
        return f"PlumiseAuth(address={self.address}, chain_id={self.chain_id})"

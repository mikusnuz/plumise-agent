"""On-chain agent lifecycle management via precompiled contracts.

Plumise chain provides three precompiled contracts for agent coordination:
- 0x20: Verify inference proof
- 0x21: Agent registration
- 0x22: Heartbeat ping

This module encapsulates the registration, heartbeat, and verification logic.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from eth_account.account import Account
from web3 import Web3

if TYPE_CHECKING:
    from plumise_agent.chain.config import PlumiseConfig
    from plumise_agent.chain.proof import ProofData

logger = logging.getLogger(__name__)

# Precompiled contract addresses
PRECOMPILE_VERIFY_INFERENCE = "0x0000000000000000000000000000000000000020"
PRECOMPILE_AGENT_REGISTER = "0x0000000000000000000000000000000000000021"
PRECOMPILE_AGENT_HEARTBEAT = "0x0000000000000000000000000000000000000022"


class ChainAgent:
    """Manages on-chain agent registration and heartbeat via precompiled contracts.

    Args:
        config: Plumise configuration instance.
        w3: Web3 instance connected to Plumise chain.
        account: Agent's eth_account.Account instance.
    """

    def __init__(
        self,
        config: PlumiseConfig,
        w3: Web3,
        account: Account,
    ) -> None:
        self.config = config
        self.w3 = w3
        self.account = account
        self.address = account.address

        self._registered = False
        self._registration_attempted = False

        logger.info("ChainAgent initialized for %s", self.address)

    # ------------------------------------------------------------------
    # Agent Registration (precompile 0x21)
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        model_hash: bytes = b"\x00" * 32,
        capabilities: Optional[list[bytes]] = None,
    ) -> bool:
        """Register this agent on-chain via the 0x21 precompile.

        Args:
            name: Agent name (max 32 bytes, will be padded).
            model_hash: Model identifier hash (32 bytes).
            capabilities: List of 32-byte capability hashes.

        Returns:
            True if registration succeeded, False otherwise.
        """
        if self._registration_attempted:
            logger.warning("Agent registration already attempted")
            return self._registered

        # Normalize name to 32 bytes
        name_bytes = name.encode("utf-8")[:32].ljust(32, b"\x00")

        # Build capabilities list
        if capabilities is None:
            capabilities = []

        cap_count = len(capabilities)
        cap_count_bytes = cap_count.to_bytes(32, "big")

        # Encode input: name(32B) + modelHash(32B) + capCount(32B) + capabilities(32B×N)
        input_data = name_bytes + model_hash + cap_count_bytes
        for cap in capabilities:
            if len(cap) != 32:
                raise ValueError("Each capability must be exactly 32 bytes")
            input_data += cap

        try:
            # Build transaction
            nonce = self.w3.eth.get_transaction_count(self.address)
            gas_price = self.w3.eth.gas_price

            tx = {
                "from": self.address,
                "to": Web3.to_checksum_address(PRECOMPILE_AGENT_REGISTER),
                "value": 0,
                "gas": 300000,
                "gasPrice": gas_price,
                "nonce": nonce,
                "data": "0x" + input_data.hex(),
                "chainId": self.config.plumise_chain_id,
            }

            # Sign and send
            signed = self.account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed.rawTransaction)

            logger.info("Agent registration tx sent: %s", tx_hash.hex())

            # Wait for receipt
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=30)

            if receipt["status"] == 1:
                logger.info("Agent registration successful")
                self._registered = True
                self._registration_attempted = True
                return True
            else:
                logger.error("Agent registration failed (status=0)")
                self._registration_attempted = True
                return False

        except Exception as exc:
            logger.error("Failed to register agent: %s", exc)
            self._registration_attempted = True
            return False

    # ------------------------------------------------------------------
    # Heartbeat (precompile 0x22)
    # ------------------------------------------------------------------

    def heartbeat(self) -> bool:
        """Send a heartbeat ping via the 0x22 precompile.

        The precompile uses msg.sender as the agent address, so no input data is needed.

        Returns:
            True if heartbeat succeeded, False otherwise.
        """
        try:
            # Build transaction
            nonce = self.w3.eth.get_transaction_count(self.address)
            gas_price = self.w3.eth.gas_price

            tx = {
                "from": self.address,
                "to": Web3.to_checksum_address(PRECOMPILE_AGENT_HEARTBEAT),
                "value": 0,
                "gas": 100000,
                "gasPrice": gas_price,
                "nonce": nonce,
                "data": "0x",
                "chainId": self.config.plumise_chain_id,
            }

            # Sign and send
            signed = self.account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed.rawTransaction)

            logger.debug("Heartbeat tx sent: %s", tx_hash.hex())

            # Wait for receipt
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=30)

            if receipt["status"] == 1:
                logger.debug("Heartbeat successful")
                return True
            else:
                logger.warning("Heartbeat failed (status=0)")
                return False

        except Exception as exc:
            logger.warning("Heartbeat transaction failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Inference Verification (precompile 0x20)
    # ------------------------------------------------------------------

    def verify_inference(self, proof: ProofData) -> Optional[str]:
        """Submit an inference proof to the verifyInference precompile (0x20).

        This is an optional on-chain verification step. The precompile
        expects 160 bytes of ABI-encoded proof data.

        Args:
            proof: ``ProofData`` instance from ``InferenceProofGenerator``.

        Returns:
            Transaction hash hex string on success, or ``None`` on failure.
        """
        try:
            input_data = proof.encode_precompile_input()

            nonce = self.w3.eth.get_transaction_count(self.address)
            gas_price = self.w3.eth.gas_price

            tx = {
                "from": self.address,
                "to": Web3.to_checksum_address(PRECOMPILE_VERIFY_INFERENCE),
                "value": 0,
                "gas": 200_000,
                "gasPrice": gas_price,
                "nonce": nonce,
                "data": "0x" + input_data.hex(),
                "chainId": self.config.plumise_chain_id,
            }

            signed = self.account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed.rawTransaction)

            logger.info(
                "verifyInference tx sent: %s (tokens=%d)",
                tx_hash.hex(),
                proof.token_count,
            )

            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=30)

            if receipt["status"] == 1:
                logger.info("verifyInference successful: %s", tx_hash.hex())
                return tx_hash.hex()
            else:
                logger.warning("verifyInference failed (status=0): %s", tx_hash.hex())
                return None

        except Exception as exc:
            logger.error("Failed to verify inference on-chain: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def is_registered(self) -> bool:
        """Return True if registration was successful."""
        return self._registered

"""Reward tracking and claiming from the Plumise RewardPool contract.

Monitors pending rewards and triggers claims when a configurable
threshold is met. Also exposes contribution/earning history.
"""

from __future__ import annotations

import logging
from typing import NamedTuple, Optional

from web3 import Web3
from web3.contract import Contract

from plumise_agent.chain.config import PlumiseConfig

logger = logging.getLogger(__name__)


class Contribution(NamedTuple):
    """Agent contribution record from RewardPool."""

    task_count: int
    uptime_seconds: int
    response_score: int
    last_updated: int


class RewardTracker:
    """Track and claim rewards from the RewardPool contract.

    Args:
        config: Plumise configuration.
        w3: An existing Web3 instance (shared with auth).
        account: The agent's eth_account.Account.
    """

    def __init__(
        self,
        config: PlumiseConfig,
        w3: Web3,
        account: object,
    ) -> None:
        self.config = config
        self.w3 = w3
        self.account = account
        self.address: str = account.address
        self.chain_id: int = config.plumise_chain_id

        self._pool: Optional[Contract] = None
        if config.reward_pool_address:
            abi = PlumiseConfig.load_abi("RewardPool")
            self._pool = self.w3.eth.contract(
                address=Web3.to_checksum_address(config.reward_pool_address),
                abi=abi,
            )
            logger.info("RewardTracker connected to RewardPool at %s", config.reward_pool_address)
        else:
            logger.warning("RewardPool address not configured; reward tracking disabled")

    # ------------------------------------------------------------------
    # Read-only queries
    # ------------------------------------------------------------------

    def get_pending_reward(self) -> int:
        """Return pending (unclaimed) reward in wei.

        Returns 0 if the contract is not configured.
        """
        if self._pool is None:
            return 0
        try:
            return self._pool.functions.getPendingReward(self.address).call()
        except Exception as exc:
            logger.error("Failed to query pending reward: %s", exc)
            return 0

    def get_contribution(self) -> Optional[Contribution]:
        """Fetch the agent's contribution record."""
        if self._pool is None:
            return None
        try:
            raw = self._pool.functions.getContribution(self.address).call()
            return Contribution(*raw)
        except Exception as exc:
            logger.error("Failed to query contribution: %s", exc)
            return None

    def get_current_epoch(self) -> int:
        """Return the current reward epoch number."""
        if self._pool is None:
            return 0
        try:
            return self._pool.functions.getCurrentEpoch().call()
        except Exception as exc:
            logger.error("Failed to query current epoch: %s", exc)
            return 0

    # ------------------------------------------------------------------
    # Claim
    # ------------------------------------------------------------------

    def should_claim(self) -> bool:
        """Return True if pending reward exceeds the configured threshold."""
        pending = self.get_pending_reward()
        threshold = self.config.claim_threshold_wei
        if pending >= threshold:
            logger.info(
                "Pending reward %s wei >= threshold %s wei; claim recommended",
                pending,
                threshold,
            )
            return True
        return False

    def claim_reward(self) -> Optional[str]:
        """Submit a ``claimReward()`` transaction.

        Returns the transaction hash on success, or ``None`` on failure.
        """
        if self._pool is None:
            logger.warning("Cannot claim: RewardPool not configured")
            return None

        try:
            nonce = self.w3.eth.get_transaction_count(self.address)
            tx = self._pool.functions.claimReward().build_transaction(
                {
                    "from": self.address,
                    "nonce": nonce,
                    "chainId": self.chain_id,
                    "gas": 200_000,
                    "gasPrice": self.w3.eth.gas_price,
                }
            )
            signed = self.account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
            hex_hash = tx_hash.hex()
            logger.info("Claim transaction sent: %s", hex_hash)
            return hex_hash
        except Exception as exc:
            logger.error("Failed to claim reward: %s", exc)
            return None

    def claim_if_ready(self) -> Optional[str]:
        """Convenience: claim only if threshold is met.

        Returns tx hash if claimed, else ``None``.
        """
        if self.should_claim():
            return self.claim_reward()
        return None

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return a human-friendly summary of reward state."""
        pending = self.get_pending_reward()
        contribution = self.get_contribution()
        epoch = self.get_current_epoch()
        return {
            "address": self.address,
            "pending_reward_wei": pending,
            "pending_reward_plm": pending / 10**18 if pending else 0.0,
            "current_epoch": epoch,
            "contribution": contribution._asdict() if contribution else None,
        }

"""Inference proof generation for verifiable computation.

Generates cryptographic proofs for each inference request by hashing
the model identifier, input data, output data, agent address, and
token count. Proofs can be reported to the Oracle and optionally
verified on-chain via the ``verifyInference`` precompile (0x20).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Union

from web3 import Web3

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProofData:
    """Immutable inference proof record.

    Attributes:
        model_hash: keccak256 of the model identifier.
        input_hash: keccak256 of the raw input data.
        output_hash: keccak256 of the raw output data.
        agent_address: Checksummed agent address.
        token_count: Number of tokens produced.
        proof_hash: keccak256(modelHash || inputHash || outputHash || agent).
    """

    model_hash: bytes
    input_hash: bytes
    output_hash: bytes
    agent_address: str
    token_count: int
    proof_hash: bytes

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "modelHash": "0x" + self.model_hash.hex(),
            "inputHash": "0x" + self.input_hash.hex(),
            "outputHash": "0x" + self.output_hash.hex(),
            "agentAddress": self.agent_address,
            "tokenCount": self.token_count,
            "proofHash": "0x" + self.proof_hash.hex(),
        }

    def encode_precompile_input(self) -> bytes:
        """Encode proof data for the verifyInference precompile (0x20).

        Layout (160 bytes):
            bytes  0..31  : bytes32 modelHash
            bytes 32..63  : bytes32 inputHash
            bytes 64..95  : bytes32 outputHash
            bytes 96..127 : address agent (left-padded to 32 bytes)
            bytes 128..159: uint256 tokenCount
        """
        # Agent address -> 20 bytes, left-padded to 32
        addr_bytes = bytes.fromhex(self.agent_address[2:])  # strip '0x'
        addr_padded = addr_bytes.rjust(32, b"\x00")

        token_bytes = self.token_count.to_bytes(32, "big")

        return (
            self.model_hash
            + self.input_hash
            + self.output_hash
            + addr_padded
            + token_bytes
        )


class InferenceProofGenerator:
    """Generates keccak256 proofs for inference requests.

    Args:
        model_name: HuggingFace model identifier (e.g. ``"bigscience/bloom-560m"``).
        agent_address: Checksummed Ethereum address of the agent.
    """

    def __init__(self, model_name: str, agent_address: str) -> None:
        self.model_name = model_name
        self.agent_address = Web3.to_checksum_address(agent_address)

        # Pre-compute the model hash (constant for the lifetime of this generator)
        self.model_hash: bytes = Web3.keccak(text=model_name)

        logger.info(
            "InferenceProofGenerator initialized: model=%s modelHash=%s agent=%s",
            model_name,
            "0x" + self.model_hash.hex()[:16] + "...",
            agent_address,
        )

    def generate_proof(
        self,
        input_data: Union[str, bytes],
        output_data: Union[str, bytes],
        token_count: int,
    ) -> ProofData:
        """Generate a proof for a single inference request.

        Args:
            input_data: Raw input (prompt text or bytes).
            output_data: Raw output (generated text or bytes).
            token_count: Number of tokens generated.

        Returns:
            An immutable ``ProofData`` instance.
        """
        # Hash input
        if isinstance(input_data, str):
            input_hash = Web3.keccak(text=input_data)
        else:
            input_hash = Web3.keccak(primitive=input_data)

        # Hash output
        if isinstance(output_data, str):
            output_hash = Web3.keccak(text=output_data)
        else:
            output_hash = Web3.keccak(primitive=output_data)

        # Compute composite proof hash:
        # keccak256(modelHash || inputHash || outputHash || agent)
        addr_bytes = bytes.fromhex(self.agent_address[2:])  # 20 bytes
        addr_padded = addr_bytes.rjust(32, b"\x00")

        proof_hash = Web3.keccak(
            primitive=(
                self.model_hash
                + input_hash
                + output_hash
                + addr_padded
            )
        )

        return ProofData(
            model_hash=self.model_hash,
            input_hash=input_hash,
            output_hash=output_hash,
            agent_address=self.agent_address,
            token_count=token_count,
            proof_hash=proof_hash,
        )

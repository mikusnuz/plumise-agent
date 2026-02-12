"""Inference engine for running forward passes through loaded model layers.

``InferenceEngine`` supports four operational modes depending on where
this node sits in the distributed pipeline:

- **full**: Complete model inference (single-node deployment).
- **first**: Embedding + partial layers -> hidden states (sent to next node).
- **middle**: Hidden states -> partial layers -> hidden states.
- **last**: Hidden states -> remaining layers + lm_head -> generated text.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import torch
import torch.nn.functional as F

from plumise_agent.model.loader import ModelParts

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Runs forward passes through loaded model layers.

    Thread-safe: a ``threading.Lock`` guards all forward operations to
    prevent concurrent use of the model weights.

    Args:
        model_parts: Loaded model components from ``ModelLoader``.
        tokenizer: HuggingFace tokenizer instance.
        device: Device string where tensors live (``"cpu"``, ``"cuda"``, etc.).
    """

    def __init__(
        self,
        model_parts: ModelParts,
        tokenizer: Any,
        device: str,
    ) -> None:
        self.parts = model_parts
        self.tokenizer = tokenizer
        self.device = device
        self._lock = threading.Lock()

        logger.info(
            "InferenceEngine ready: layers=[%d,%d) of %d, device=%s",
            model_parts.layer_range.start,
            model_parts.layer_range.end,
            model_parts.layer_range.total,
            device,
        )

    # ------------------------------------------------------------------
    # Full model inference (single-node)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward_full(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        do_sample: bool = True,
    ) -> tuple[str, int]:
        """Run complete autoregressive generation on a single node.

        Args:
            prompt: Input text.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            repetition_penalty: Penalty for repeating tokens.
            do_sample: Whether to sample (``False`` for greedy).

        Returns:
            ``(generated_text, num_tokens)`` tuple.
        """
        with self._lock:
            t0 = time.perf_counter()

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_ids = inputs["input_ids"]

            generated_ids = self._autoregressive_loop(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
            )

            # Decode only the newly generated tokens
            new_ids = generated_ids[:, input_ids.shape[1]:]
            text = self.tokenizer.decode(new_ids[0], skip_special_tokens=True)
            num_tokens = new_ids.shape[1]

            elapsed = (time.perf_counter() - t0) * 1000
            logger.debug(
                "forward_full: %d tokens in %.1f ms (%.1f tok/s)",
                num_tokens,
                elapsed,
                num_tokens / (elapsed / 1000) if elapsed > 0 else 0,
            )
            return text, num_tokens

    # ------------------------------------------------------------------
    # First node: embed + layers -> hidden states
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward_first(self, prompt: str) -> torch.Tensor:
        """First-node forward: tokenize, embed, run through assigned layers.

        Args:
            prompt: Input text to process.

        Returns:
            Hidden-states tensor of shape ``(batch, seq_len, hidden_dim)``.
        """
        with self._lock:
            t0 = time.perf_counter()

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_ids = inputs["input_ids"]

            if self.parts.embedding is None:
                raise RuntimeError("forward_first requires an embedding module")

            hidden_states = self.parts.embedding(input_ids)

            # Do NOT pass the tokenizer's 2D attention_mask to layers;
            # transformer layers generate their own causal mask internally.
            hidden_states = self._run_through_layers(hidden_states)

            elapsed = (time.perf_counter() - t0) * 1000
            logger.debug(
                "forward_first: shape=%s in %.1f ms", list(hidden_states.shape), elapsed
            )
            return hidden_states

    # ------------------------------------------------------------------
    # First node: forward from pre-tokenized IDs (pipeline autoregressive)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward_first_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """First-node forward from pre-tokenized input IDs.

        Used by the pipeline autoregressive loop where the executor
        manages tokenization and appends generated tokens each iteration.

        Args:
            input_ids: Token ID tensor of shape ``(batch, seq_len)``.

        Returns:
            Hidden-states tensor of shape ``(batch, seq_len, hidden_dim)``.
        """
        with self._lock:
            t0 = time.perf_counter()

            input_ids = input_ids.to(self.device)

            if self.parts.embedding is None:
                raise RuntimeError("forward_first_ids requires an embedding module")

            hidden_states = self.parts.embedding(input_ids)
            hidden_states = self._run_through_layers(hidden_states)

            elapsed = (time.perf_counter() - t0) * 1000
            logger.debug(
                "forward_first_ids: shape=%s in %.1f ms",
                list(hidden_states.shape),
                elapsed,
            )
            return hidden_states

    # ------------------------------------------------------------------
    # Middle node: hidden states -> layers -> hidden states
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward_middle(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Middle-node forward: process hidden states through assigned layers.

        Args:
            hidden_states: Input tensor from the previous node.

        Returns:
            Processed hidden-states tensor.
        """
        with self._lock:
            t0 = time.perf_counter()

            hidden_states = hidden_states.to(self.device)

            hidden_states = self._run_through_layers(hidden_states)

            elapsed = (time.perf_counter() - t0) * 1000
            logger.debug(
                "forward_middle: shape=%s in %.1f ms",
                list(hidden_states.shape),
                elapsed,
            )
            return hidden_states

    # ------------------------------------------------------------------
    # Last node: hidden states -> layers + norm + lm_head -> text
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward_last(
        self,
        hidden_states: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        do_sample: bool = True,
    ) -> tuple[str, int]:
        """Last-node forward: remaining layers + lm_head + autoregressive sampling.

        Args:
            hidden_states: Input tensor from the previous node.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            repetition_penalty: Penalty for repeating tokens.
            do_sample: Whether to sample (``False`` for greedy).

        Returns:
            ``(generated_text, num_tokens)`` tuple.
        """
        with self._lock:
            t0 = time.perf_counter()

            hidden_states = hidden_states.to(self.device)

            if self.parts.lm_head is None:
                raise RuntimeError("forward_last requires a lm_head module")

            # Run through remaining layers
            hidden_states = self._run_through_layers(hidden_states)

            # Apply final layer norm
            if self.parts.norm is not None:
                hidden_states = self.parts.norm(hidden_states)

            # Autoregressive generation from the last hidden state
            generated_tokens: list[int] = []
            eos_token_id = self.tokenizer.eos_token_id

            # Start with the logits from the last position
            current_hidden = hidden_states

            for _ in range(max_new_tokens):
                # Get logits for the last position
                logits = self.parts.lm_head(current_hidden[:, -1:, :])
                logits = logits[:, -1, :]  # (batch, vocab)

                # Apply repetition penalty
                if repetition_penalty != 1.0 and generated_tokens:
                    logits = self._apply_repetition_penalty(
                        logits, generated_tokens, repetition_penalty
                    )

                # Sample next token
                next_token_id = self._sample_token(
                    logits,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                )

                if next_token_id == eos_token_id:
                    break

                generated_tokens.append(next_token_id)

                # For subsequent tokens we would need embedding + full layers,
                # but in pipeline mode the last node only has final layers.
                # In practice, distributed autoregressive generation requires
                # the full pipeline for each new token. For the initial
                # implementation we generate one "pass" worth of output from
                # the final hidden states.
                break

            text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            num_tokens = len(generated_tokens)

            elapsed = (time.perf_counter() - t0) * 1000
            logger.debug(
                "forward_last: %d tokens in %.1f ms", num_tokens, elapsed
            )
            return text, num_tokens

    # ------------------------------------------------------------------
    # Last node: single-token sampling (pipeline autoregressive)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward_last_token(
        self,
        hidden_states: torch.Tensor,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        do_sample: bool = True,
        past_token_ids: list[int] | None = None,
    ) -> tuple[int, bool]:
        """Last-node forward: sample exactly one token from hidden states.

        Used by the pipeline autoregressive loop. Runs hidden states through
        the remaining layers, applies norm, then samples a single token from
        the lm_head logits.

        Args:
            hidden_states: Input tensor from the previous node.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            repetition_penalty: Penalty for repeating tokens.
            do_sample: Whether to sample (``False`` for greedy).
            past_token_ids: Previously generated token IDs for repetition penalty.

        Returns:
            ``(token_id, is_eos)`` tuple.
        """
        with self._lock:
            hidden_states = hidden_states.to(self.device)

            if self.parts.lm_head is None:
                raise RuntimeError("forward_last_token requires a lm_head module")

            # Run through remaining layers
            hidden_states = self._run_through_layers(hidden_states)

            # Apply final layer norm
            if self.parts.norm is not None:
                hidden_states = self.parts.norm(hidden_states)

            # Get logits for the last position only
            logits = self.parts.lm_head(hidden_states[:, -1:, :])
            logits = logits[:, -1, :]  # (batch, vocab)

            # Apply repetition penalty
            if repetition_penalty != 1.0 and past_token_ids:
                logits = self._apply_repetition_penalty(
                    logits, past_token_ids, repetition_penalty
                )

            # Sample next token
            token_id = self._sample_token(
                logits,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            )

            is_eos = token_id == self.tokenizer.eos_token_id
            return token_id, is_eos

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_through_layers(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Pass hidden states through all layers in ``self.parts.layers``.

        Computes rotary position embeddings if available and passes them
        to each layer, supporting modern transformer architectures (Llama,
        Mistral, GPT-OSS, etc.) that require pre-computed (cos, sin) tensors.

        Note: We do NOT pass an explicit attention_mask. The tokenizer
        produces a 2D mask (batch, seq_len), but transformer layers expect
        a 4D causal mask. By omitting it, each layer constructs its own
        causal mask internally (which is correct for decoder-only models).
        """
        # Compute position embeddings once for all layers
        position_embeddings = None
        position_ids = None
        seq_len = hidden_states.shape[1]

        if self.parts.rotary_emb is not None:
            position_ids = torch.arange(
                seq_len, dtype=torch.long, device=hidden_states.device
            ).unsqueeze(0)
            position_embeddings = self.parts.rotary_emb(
                hidden_states, position_ids
            )

        for layer in self.parts.layers:
            kwargs: dict[str, Any] = {}
            if position_embeddings is not None:
                kwargs["position_embeddings"] = position_embeddings
            if position_ids is not None:
                kwargs["position_ids"] = position_ids

            layer_output = layer(hidden_states, **kwargs)
            # Most layers return a tuple (hidden_states, ...) or just a tensor
            if isinstance(layer_output, tuple):
                hidden_states = layer_output[0]
            else:
                hidden_states = layer_output
        return hidden_states

    def _autoregressive_loop(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        do_sample: bool,
    ) -> torch.Tensor:
        """Full-model autoregressive generation loop.

        Processes one token at a time through the complete model
        (embedding -> layers -> norm -> lm_head -> sample).
        """
        eos_token_id = self.tokenizer.eos_token_id
        generated_ids = input_ids.clone()

        for _ in range(max_new_tokens):
            # Embed
            hidden_states = self.parts.embedding(generated_ids)

            # Run through all layers
            hidden_states = self._run_through_layers(hidden_states)

            # Final norm
            if self.parts.norm is not None:
                hidden_states = self.parts.norm(hidden_states)

            # LM head
            logits = self.parts.lm_head(hidden_states[:, -1:, :])
            logits = logits[:, -1, :]  # (batch, vocab)

            # Repetition penalty
            if repetition_penalty != 1.0:
                past_tokens = generated_ids[0].tolist()
                logits = self._apply_repetition_penalty(
                    logits, past_tokens, repetition_penalty
                )

            # Sample
            next_token_id = self._sample_token(
                logits,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            )

            if next_token_id == eos_token_id:
                break

            next_token = torch.tensor(
                [[next_token_id]], dtype=torch.long, device=self.device
            )
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

        return generated_ids

    @staticmethod
    def _apply_repetition_penalty(
        logits: torch.Tensor,
        past_token_ids: list[int],
        penalty: float,
    ) -> torch.Tensor:
        """Penalise logits for tokens that have already appeared."""
        if not past_token_ids:
            return logits
        unique_ids = list(set(past_token_ids))
        score = logits[:, unique_ids]
        # If score > 0 divide by penalty, if score < 0 multiply by penalty
        score = torch.where(score > 0, score / penalty, score * penalty)
        logits[:, unique_ids] = score
        return logits

    @staticmethod
    def _sample_token(
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
        do_sample: bool,
    ) -> int:
        """Sample a single token from logits.

        Applies temperature scaling and top-p (nucleus) filtering before
        sampling. Falls back to greedy argmax when ``do_sample=False``.
        """
        if not do_sample:
            return int(torch.argmax(logits, dim=-1).item())

        # Temperature scaling
        if temperature > 0 and temperature != 1.0:
            logits = logits / temperature

        # Top-p filtering
        if 0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[sorted_mask] = float("-inf")

            # Scatter back
            logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

        # Sample from the distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return int(next_token.item())

"""Model loading with optional layer splitting for distributed inference.

``ModelLoader`` wraps HuggingFace ``transformers`` to load a causal-LM
model either in full or partially (keeping only the transformer layers
assigned to this node).
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from plumise_agent.model.layer_range import LayerRange

logger = logging.getLogger(__name__)


@dataclass
class ModelParts:
    """Holds the loaded model components after layer splitting.

    Attributes:
        embedding: Token embedding module (first node only, else ``None``).
        layers: The transformer layers assigned to this node.
        norm: Final layer-norm module (last node only, else ``None``).
        lm_head: Language-model head (last node only, else ``None``).
        config: The model's ``PretrainedConfig`` object.
        layer_range: Which layers this node is responsible for.
    """

    embedding: torch.nn.Module | None
    layers: torch.nn.ModuleList
    norm: torch.nn.Module | None
    lm_head: torch.nn.Module | None
    config: Any
    layer_range: LayerRange


# ---------------------------------------------------------------------------
# Architecture helpers
# ---------------------------------------------------------------------------

def _get_model_body(model: torch.nn.Module) -> torch.nn.Module:
    """Return the inner model body (e.g. ``model.model`` for LLaMA/Mistral).

    Supported attribute names (checked in order):
    ``model``, ``transformer``, ``gpt_neox``.
    """
    for attr in ("model", "transformer", "gpt_neox"):
        if hasattr(model, attr):
            return getattr(model, attr)
    raise AttributeError(
        f"Cannot locate model body on {type(model).__name__}. "
        "Supported wrappers: model.model, model.transformer, model.gpt_neox"
    )


def _get_layers(body: torch.nn.Module) -> torch.nn.ModuleList:
    """Return the ``ModuleList`` of transformer layers from the model body."""
    for attr in ("layers", "h", "block"):
        if hasattr(body, attr):
            return getattr(body, attr)
    raise AttributeError(
        f"Cannot locate transformer layers on {type(body).__name__}. "
        "Supported attributes: layers, h, block"
    )


def _get_embedding(body: torch.nn.Module) -> torch.nn.Module:
    """Return the token embedding module."""
    for attr in ("embed_tokens", "wte", "word_embeddings", "embed_in"):
        if hasattr(body, attr):
            return getattr(body, attr)
    raise AttributeError(
        f"Cannot locate embedding on {type(body).__name__}. "
        "Supported attributes: embed_tokens, wte, word_embeddings, embed_in"
    )


def _get_norm(body: torch.nn.Module) -> torch.nn.Module | None:
    """Return the final layer-norm module, or ``None`` if absent."""
    for attr in ("norm", "ln_f", "final_layer_norm", "norm_f"):
        if hasattr(body, attr):
            return getattr(body, attr)
    return None


def _get_lm_head(model: torch.nn.Module) -> torch.nn.Module | None:
    """Return the language-model head, or ``None`` if absent."""
    for attr in ("lm_head", "embed_out"):
        if hasattr(model, attr):
            return getattr(model, attr)
    return None


# ---------------------------------------------------------------------------
# ModelLoader
# ---------------------------------------------------------------------------

class ModelLoader:
    """Loads transformer models with optional layer splitting.

    Args:
        model_name: HuggingFace model identifier (e.g. ``"bigscience/bloom-560m"``).
        device: Target device string (``"auto"``, ``"cpu"``, ``"cuda"``, ``"mps"``).
        hf_token: Optional HuggingFace authentication token.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        hf_token: str = "",
    ) -> None:
        self.model_name = model_name
        self.hf_token = hf_token or None  # transformers treats None as "no token"
        self._device = self._resolve_device(device)

        self._model_config = AutoConfig.from_pretrained(
            model_name,
            token=self.hf_token,
        )

        logger.info(
            "ModelLoader initialized: model=%s device=%s total_layers=%d",
            model_name,
            self._device,
            self.get_total_layers(),
        )

    # ---- Public API ----

    def get_total_layers(self) -> int:
        """Return the total number of transformer layers in the model."""
        for attr in ("num_hidden_layers", "n_layer", "num_layers"):
            if hasattr(self._model_config, attr):
                return getattr(self._model_config, attr)
        raise AttributeError(
            f"Cannot determine layer count from config: {type(self._model_config).__name__}"
        )

    def load_full(self) -> tuple[torch.nn.Module, Any]:
        """Load the complete model and tokenizer.

        Returns:
            ``(model, tokenizer)`` tuple.
        """
        logger.info("Loading full model: %s", self.model_name)

        dtype = self._infer_dtype()
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map=self._device if self._device != "cpu" else None,
            token=self.hf_token,
        )
        if self._device == "cpu":
            model = model.to("cpu")

        tokenizer = self._load_tokenizer()

        model.eval()
        logger.info("Full model loaded: dtype=%s params=%s", dtype, _param_count(model))
        return model, tokenizer

    def load_partial(self, layer_range: LayerRange) -> tuple[ModelParts, Any]:
        """Load the model then discard layers outside ``layer_range``.

        For memory efficiency the full model is loaded first, then
        unneeded components are deleted and garbage-collected.

        - **First node**: keeps embedding + layers[start:end].
        - **Last node**: keeps layers[start:end] + norm + lm_head.
        - **Middle node**: keeps only layers[start:end].

        Args:
            layer_range: Which layers this node owns.

        Returns:
            ``(ModelParts, tokenizer)`` tuple.
        """
        if layer_range.is_full:
            logger.info("layer_range covers all layers; delegating to load_full()")
            model, tokenizer = self.load_full()
            body = _get_model_body(model)
            parts = ModelParts(
                embedding=_get_embedding(body),
                layers=_get_layers(body),
                norm=_get_norm(body),
                lm_head=_get_lm_head(model),
                config=self._model_config,
                layer_range=layer_range,
            )
            return parts, tokenizer

        logger.info(
            "Loading partial model: layers [%d, %d) of %d",
            layer_range.start,
            layer_range.end,
            layer_range.total,
        )

        dtype = self._infer_dtype()
        # Load the full model on CPU first, then split and move
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            token=self.hf_token,
        )
        model.eval()

        body = _get_model_body(model)
        all_layers = _get_layers(body)

        # Extract the slice of layers we need
        kept_layers = torch.nn.ModuleList(
            [all_layers[i] for i in range(layer_range.start, layer_range.end)]
        )

        # Decide which components to keep
        embedding = _get_embedding(body) if layer_range.is_first else None
        norm = _get_norm(body) if layer_range.is_last else None
        lm_head = _get_lm_head(model) if layer_range.is_last else None

        parts = ModelParts(
            embedding=embedding,
            layers=kept_layers,
            norm=norm,
            lm_head=lm_head,
            config=self._model_config,
            layer_range=layer_range,
        )

        # Move kept parts to target device
        device = self._device
        if device not in ("cpu", "auto"):
            if parts.embedding is not None:
                parts.embedding = parts.embedding.to(device)
            parts.layers = parts.layers.to(device)
            if parts.norm is not None:
                parts.norm = parts.norm.to(device)
            if parts.lm_head is not None:
                parts.lm_head = parts.lm_head.to(device)

        # Delete the original model to free memory
        del model, body, all_layers
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(
            "Partial model loaded: kept %d layers, embedding=%s, norm=%s, lm_head=%s",
            layer_range.count,
            parts.embedding is not None,
            parts.norm is not None,
            parts.lm_head is not None,
        )
        return parts, self._load_tokenizer()

    # ---- Private helpers ----

    def _load_tokenizer(self) -> Any:
        """Load the tokenizer for the configured model."""
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=self.hf_token,
        )
        # Ensure pad_token is set (many models lack it)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _infer_dtype(self) -> torch.dtype:
        """Choose the best dtype for the target device."""
        if self._device == "cpu":
            return torch.float32
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if torch.cuda.is_available():
            return torch.float16
        # MPS or other accelerators
        return torch.float32

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve ``"auto"`` to the best available device string."""
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @property
    def device(self) -> str:
        """Resolved device string."""
        return self._device


def _param_count(model: torch.nn.Module) -> str:
    """Return a human-readable parameter count string."""
    total = sum(p.numel() for p in model.parameters())
    if total >= 1e9:
        return f"{total / 1e9:.1f}B"
    if total >= 1e6:
        return f"{total / 1e6:.1f}M"
    return f"{total / 1e3:.1f}K"

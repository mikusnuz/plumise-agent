"""Transformer layer range representation.

A ``LayerRange`` describes a contiguous slice of transformer layers
assigned to one node in the distributed inference pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayerRange:
    """Represents a contiguous range of transformer layers assigned to a node.

    Attributes:
        start: First layer index (inclusive).
        end: Past-the-end layer index (exclusive).
        total: Total number of transformer layers in the model.
    """

    start: int
    end: int
    total: int

    # ---- Positional queries ----

    @property
    def is_first(self) -> bool:
        """True if this range includes the very first layer."""
        return self.start == 0

    @property
    def is_last(self) -> bool:
        """True if this range includes the very last layer."""
        return self.end == self.total

    @property
    def is_full(self) -> bool:
        """True if this range covers all layers (single-node mode)."""
        return self.start == 0 and self.end == self.total

    @property
    def count(self) -> int:
        """Number of layers in the range."""
        return self.end - self.start

    # ---- Validation ----

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(
                f"Invalid layer range: start ({self.start}) must be >= 0"
            )
        if self.end <= self.start:
            raise ValueError(
                f"Invalid layer range: end ({self.end}) must be > start ({self.start})"
            )
        if self.end > self.total:
            raise ValueError(
                f"Invalid layer range: end ({self.end}) exceeds total ({self.total})"
            )

    # ---- Display ----

    def __repr__(self) -> str:
        return f"LayerRange([{self.start}, {self.end}) of {self.total})"

"""Tests for plumise_agent.model.layer_range.LayerRange."""

from __future__ import annotations

import pytest

from plumise_agent.model.layer_range import LayerRange


# ---------------------------------------------------------------------------
# Valid ranges
# ---------------------------------------------------------------------------

class TestLayerRangeValid:
    """Constructing valid LayerRange instances and reading properties."""

    def test_basic_construction(self):
        lr = LayerRange(start=0, end=8, total=24)
        assert lr.start == 0
        assert lr.end == 8
        assert lr.total == 24

    def test_full_range(self, full_layer_range: LayerRange):
        assert full_layer_range.start == 0
        assert full_layer_range.end == 24
        assert full_layer_range.total == 24

    def test_single_layer(self):
        lr = LayerRange(start=5, end=6, total=10)
        assert lr.count == 1


# ---------------------------------------------------------------------------
# is_first / is_last / is_full properties
# ---------------------------------------------------------------------------

class TestLayerRangePositionProperties:
    """Test the boolean position helpers."""

    def test_is_first_true(self, first_layer_range: LayerRange):
        assert first_layer_range.is_first is True

    def test_is_first_false(self, middle_layer_range: LayerRange):
        assert middle_layer_range.is_first is False

    def test_is_last_true(self, last_layer_range: LayerRange):
        assert last_layer_range.is_last is True

    def test_is_last_false(self, first_layer_range: LayerRange):
        assert first_layer_range.is_last is False

    def test_is_full_true(self, full_layer_range: LayerRange):
        assert full_layer_range.is_full is True

    def test_is_full_false_first(self, first_layer_range: LayerRange):
        assert first_layer_range.is_full is False

    def test_is_full_false_middle(self, middle_layer_range: LayerRange):
        assert middle_layer_range.is_full is False

    def test_is_full_false_last(self, last_layer_range: LayerRange):
        assert last_layer_range.is_full is False

    def test_first_and_last_not_full(self):
        """A range that is both first and last but not all layers is not full."""
        # This cannot happen in practice (if start=0 and end=total then is_full),
        # so let's just verify the full case again.
        lr = LayerRange(start=0, end=10, total=10)
        assert lr.is_first is True
        assert lr.is_last is True
        assert lr.is_full is True


# ---------------------------------------------------------------------------
# count property
# ---------------------------------------------------------------------------

class TestLayerRangeCount:
    """Test the count property."""

    def test_count_full(self, full_layer_range: LayerRange):
        assert full_layer_range.count == 24

    def test_count_partial(self, first_layer_range: LayerRange):
        assert first_layer_range.count == 8

    def test_count_middle(self, middle_layer_range: LayerRange):
        assert middle_layer_range.count == 8

    def test_count_single_layer(self):
        lr = LayerRange(start=3, end=4, total=10)
        assert lr.count == 1

    def test_count_matches_end_minus_start(self):
        lr = LayerRange(start=2, end=7, total=20)
        assert lr.count == lr.end - lr.start == 5


# ---------------------------------------------------------------------------
# Invalid ranges raise ValueError
# ---------------------------------------------------------------------------

class TestLayerRangeInvalid:
    """Invalid parameters must raise ValueError."""

    def test_negative_start(self):
        with pytest.raises(ValueError, match="start.*must be >= 0"):
            LayerRange(start=-1, end=5, total=10)

    def test_end_equals_start(self):
        with pytest.raises(ValueError, match="end.*must be > start"):
            LayerRange(start=5, end=5, total=10)

    def test_end_less_than_start(self):
        with pytest.raises(ValueError, match="end.*must be > start"):
            LayerRange(start=5, end=3, total=10)

    def test_end_exceeds_total(self):
        with pytest.raises(ValueError, match="end.*exceeds total"):
            LayerRange(start=0, end=11, total=10)

    def test_zero_total_end_positive(self):
        with pytest.raises(ValueError):
            LayerRange(start=0, end=1, total=0)


# ---------------------------------------------------------------------------
# Frozen dataclass
# ---------------------------------------------------------------------------

class TestLayerRangeFrozen:
    """LayerRange is frozen -- attributes cannot be mutated."""

    def test_cannot_set_start(self, full_layer_range: LayerRange):
        with pytest.raises(AttributeError):
            full_layer_range.start = 5  # type: ignore[misc]

    def test_cannot_set_end(self, full_layer_range: LayerRange):
        with pytest.raises(AttributeError):
            full_layer_range.end = 5  # type: ignore[misc]

    def test_cannot_set_total(self, full_layer_range: LayerRange):
        with pytest.raises(AttributeError):
            full_layer_range.total = 5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------

class TestLayerRangeRepr:
    """Ensure a human-readable representation."""

    def test_repr_format(self):
        lr = LayerRange(start=8, end=16, total=24)
        s = repr(lr)
        assert "8" in s
        assert "16" in s
        assert "24" in s

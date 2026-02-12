"""Tests for plumise_agent.grpc_.serializer -- tensor round-trip serialization."""

from __future__ import annotations

import pytest
import torch

from plumise_agent.grpc_.serializer import (
    deserialize_tensor,
    serialize_tensor,
    _torch_dtype_to_str,
)


# ---------------------------------------------------------------------------
# Round-trip: float32
# ---------------------------------------------------------------------------

class TestRoundTripFloat32:
    """Serialize then deserialize float32 tensors."""

    def test_roundtrip_values(self, sample_tensor_f32: torch.Tensor):
        raw, shape, dtype = serialize_tensor(sample_tensor_f32)
        result = deserialize_tensor(raw, shape, dtype)
        assert torch.allclose(result, sample_tensor_f32)

    def test_roundtrip_dtype_string(self, sample_tensor_f32: torch.Tensor):
        _, _, dtype = serialize_tensor(sample_tensor_f32)
        assert dtype == "float32"

    def test_roundtrip_shape(self, sample_tensor_f32: torch.Tensor):
        _, shape, _ = serialize_tensor(sample_tensor_f32)
        assert shape == list(sample_tensor_f32.shape)

    def test_result_dtype_matches(self, sample_tensor_f32: torch.Tensor):
        raw, shape, dtype = serialize_tensor(sample_tensor_f32)
        result = deserialize_tensor(raw, shape, dtype)
        assert result.dtype == torch.float32

    def test_various_shapes(self):
        for s in [(1,), (3, 5), (2, 3, 4), (1, 1, 1, 1)]:
            t = torch.randn(*s, dtype=torch.float32)
            raw, shape, dtype = serialize_tensor(t)
            result = deserialize_tensor(raw, shape, dtype)
            assert torch.allclose(result, t)
            assert result.shape == t.shape


# ---------------------------------------------------------------------------
# Round-trip: float16
# ---------------------------------------------------------------------------

class TestRoundTripFloat16:
    """Serialize then deserialize float16 tensors."""

    def test_roundtrip_values(self, sample_tensor_f16: torch.Tensor):
        raw, shape, dtype = serialize_tensor(sample_tensor_f16)
        result = deserialize_tensor(raw, shape, dtype)
        assert torch.allclose(result, sample_tensor_f16)

    def test_roundtrip_dtype_string(self, sample_tensor_f16: torch.Tensor):
        _, _, dtype = serialize_tensor(sample_tensor_f16)
        assert dtype == "float16"

    def test_result_dtype_matches(self, sample_tensor_f16: torch.Tensor):
        raw, shape, dtype = serialize_tensor(sample_tensor_f16)
        result = deserialize_tensor(raw, shape, dtype)
        assert result.dtype == torch.float16


# ---------------------------------------------------------------------------
# Round-trip: bfloat16
# ---------------------------------------------------------------------------

class TestRoundTripBfloat16:
    """Serialize then deserialize bfloat16 tensors.

    bfloat16 uses torch.save/load internally because numpy has no native
    bfloat16 support.
    """

    def test_roundtrip_values(self, sample_tensor_bf16: torch.Tensor):
        raw, shape, dtype = serialize_tensor(sample_tensor_bf16)
        result = deserialize_tensor(raw, shape, dtype)
        assert torch.allclose(result, sample_tensor_bf16)

    def test_roundtrip_dtype_string(self, sample_tensor_bf16: torch.Tensor):
        _, _, dtype = serialize_tensor(sample_tensor_bf16)
        assert dtype == "bfloat16"

    def test_result_dtype_matches(self, sample_tensor_bf16: torch.Tensor):
        raw, shape, dtype = serialize_tensor(sample_tensor_bf16)
        result = deserialize_tensor(raw, shape, dtype)
        assert result.dtype == torch.bfloat16

    def test_bfloat16_uses_torch_save(self):
        """The bfloat16 path produces bytes that are NOT just raw numpy."""
        t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
        raw, shape, dtype = serialize_tensor(t)
        # torch.save output is always larger than 3*2 = 6 bytes of raw data
        assert len(raw) > t.numel() * 2


# ---------------------------------------------------------------------------
# Shape preservation
# ---------------------------------------------------------------------------

class TestShapePreservation:
    """Ensure shapes survive the round-trip correctly."""

    def test_1d(self):
        t = torch.randn(10, dtype=torch.float32)
        raw, shape, dtype = serialize_tensor(t)
        result = deserialize_tensor(raw, shape, dtype)
        assert result.shape == (10,)

    def test_2d(self):
        t = torch.randn(3, 7, dtype=torch.float32)
        raw, shape, dtype = serialize_tensor(t)
        result = deserialize_tensor(raw, shape, dtype)
        assert result.shape == (3, 7)

    def test_3d(self):
        t = torch.randn(2, 4, 8, dtype=torch.float32)
        raw, shape, dtype = serialize_tensor(t)
        result = deserialize_tensor(raw, shape, dtype)
        assert result.shape == (2, 4, 8)

    def test_4d(self):
        t = torch.randn(1, 2, 3, 4, dtype=torch.float32)
        raw, shape, dtype = serialize_tensor(t)
        result = deserialize_tensor(raw, shape, dtype)
        assert result.shape == (1, 2, 3, 4)

    def test_shape_list_type(self):
        t = torch.randn(2, 3, dtype=torch.float32)
        _, shape, _ = serialize_tensor(t)
        assert isinstance(shape, list)
        assert all(isinstance(s, int) for s in shape)


# ---------------------------------------------------------------------------
# Empty tensor
# ---------------------------------------------------------------------------

class TestEmptyTensor:
    """Edge case: tensors with zero elements."""

    def test_empty_float32(self):
        t = torch.empty(0, dtype=torch.float32)
        raw, shape, dtype = serialize_tensor(t)
        result = deserialize_tensor(raw, shape, dtype)
        assert result.shape == (0,)
        assert result.numel() == 0

    def test_empty_float16(self):
        t = torch.empty(0, dtype=torch.float16)
        raw, shape, dtype = serialize_tensor(t)
        result = deserialize_tensor(raw, shape, dtype)
        assert result.shape == (0,)
        assert result.numel() == 0

    def test_empty_2d(self):
        t = torch.empty(0, 5, dtype=torch.float32)
        raw, shape, dtype = serialize_tensor(t)
        result = deserialize_tensor(raw, shape, dtype)
        assert result.shape == (0, 5)

    def test_empty_bfloat16(self):
        t = torch.empty(0, dtype=torch.bfloat16)
        raw, shape, dtype = serialize_tensor(t)
        result = deserialize_tensor(raw, shape, dtype)
        assert result.numel() == 0


# ---------------------------------------------------------------------------
# Unsupported dtype
# ---------------------------------------------------------------------------

class TestUnsupportedDtype:
    """Verify that unknown dtypes raise errors."""

    def test_serialize_int32_raises(self):
        t = torch.tensor([1, 2, 3], dtype=torch.int32)
        with pytest.raises(ValueError, match="Unsupported"):
            serialize_tensor(t)

    def test_deserialize_unknown_dtype_raises(self):
        with pytest.raises(ValueError, match="Unsupported dtype"):
            deserialize_tensor(b"\x00\x00", [1], "int64")

    def test_torch_dtype_to_str_unsupported(self):
        with pytest.raises(ValueError, match="Unsupported"):
            _torch_dtype_to_str(torch.int64)


# ---------------------------------------------------------------------------
# Device placement
# ---------------------------------------------------------------------------

class TestDevicePlacement:
    """Deserialized tensor should land on the requested device."""

    def test_cpu_device(self, sample_tensor_f32: torch.Tensor):
        raw, shape, dtype = serialize_tensor(sample_tensor_f32)
        result = deserialize_tensor(raw, shape, dtype, device="cpu")
        assert result.device == torch.device("cpu")

    def test_noncontiguous_input(self):
        """Non-contiguous tensors should still serialize correctly."""
        t = torch.randn(4, 4, dtype=torch.float32)
        t_nc = t.T  # transpose makes it non-contiguous
        assert not t_nc.is_contiguous()
        raw, shape, dtype = serialize_tensor(t_nc)
        result = deserialize_tensor(raw, shape, dtype)
        assert torch.allclose(result, t_nc.contiguous())

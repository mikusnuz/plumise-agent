"""Tensor serialization utilities for gRPC transfer.

Provides efficient conversion between ``torch.Tensor`` and raw bytes for
transport over the ``InferencePipeline`` gRPC service. Hidden-state tensors
are serialized as contiguous numpy byte buffers (no pickle overhead).

bfloat16 special handling: NumPy has no native ``bfloat16`` dtype, so
tensors in this format are round-tripped through ``torch.save``/``torch.load``
on a ``BytesIO`` buffer.
"""

from __future__ import annotations

import io
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Mapping from dtype string to numpy/torch types
_NUMPY_DTYPES: dict[str, np.dtype] = {
    "float32": np.dtype("float32"),
    "float16": np.dtype("float16"),
}

_TORCH_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def serialize_tensor(tensor: torch.Tensor) -> tuple[bytes, list[int], str]:
    """Serialize a tensor to raw bytes, shape list, and dtype string.

    For ``float32`` and ``float16`` tensors the data is copied through
    numpy for zero-overhead serialization. ``bfloat16`` tensors use
    ``torch.save`` into a ``BytesIO`` buffer since numpy lacks native
    bfloat16 support.

    Args:
        tensor: The tensor to serialize.

    Returns:
        ``(raw_bytes, shape_list, dtype_string)`` tuple.
    """
    # Ensure contiguous and on CPU
    t = tensor.detach().contiguous().cpu()
    dtype_str = _torch_dtype_to_str(t.dtype)
    shape = list(t.shape)

    if dtype_str == "bfloat16":
        # bfloat16 has no numpy equivalent; use torch serialization
        buf = io.BytesIO()
        torch.save(t, buf)
        raw_bytes = buf.getvalue()
    else:
        raw_bytes = t.numpy().tobytes()

    logger.debug(
        "serialize_tensor: shape=%s dtype=%s size=%d bytes",
        shape,
        dtype_str,
        len(raw_bytes),
    )
    return raw_bytes, shape, dtype_str


def deserialize_tensor(
    data: bytes,
    shape: list[int],
    dtype: str,
    device: str = "cpu",
) -> torch.Tensor:
    """Deserialize raw bytes back into a tensor.

    Args:
        data: Raw byte buffer from ``serialize_tensor``.
        shape: Original tensor shape.
        dtype: Dtype string (``"float32"``, ``"float16"``, ``"bfloat16"``).
        device: Target device for the reconstructed tensor.

    Returns:
        Reconstructed tensor on the specified device.

    Raises:
        ValueError: If ``dtype`` is not supported.
    """
    if dtype not in _TORCH_DTYPES:
        raise ValueError(
            f"Unsupported dtype '{dtype}'. "
            f"Supported: {sorted(_TORCH_DTYPES.keys())}"
        )

    if dtype == "bfloat16":
        buf = io.BytesIO(data)
        tensor = torch.load(buf, map_location="cpu", weights_only=True)
    else:
        np_dtype = _NUMPY_DTYPES[dtype]
        arr = np.frombuffer(data, dtype=np_dtype).reshape(shape)
        tensor = torch.from_numpy(arr.copy())

    tensor = tensor.to(device)

    logger.debug(
        "deserialize_tensor: shape=%s dtype=%s device=%s",
        shape,
        dtype,
        device,
    )
    return tensor


def _torch_dtype_to_str(dtype: torch.dtype) -> str:
    """Convert a torch dtype to our canonical string representation."""
    mapping = {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
    }
    s = mapping.get(dtype)
    if s is None:
        raise ValueError(
            f"Unsupported tensor dtype: {dtype}. "
            f"Supported: {list(mapping.values())}"
        )
    return s

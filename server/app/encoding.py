"""Wire encoding for numeric arrays.

Numeric data never travels as a JSON array of numbers. It travels as base64
typed arrays with an explicit dtype/shape/transform header, decoded by
`client/src/app/core/util/encoding.ts`. See docs/api-contract.md section 6.
"""

from __future__ import annotations

import base64
import math
from typing import Any, Literal

import torch

Dtype = Literal["f32", "u8", "u16", "i32"]
Layout = Literal["dense", "causal_lower"]
Transform = Literal["linear", "sqrt"]

EncodedArray = dict[str, Any]


def _b64(buffer: bytes) -> str:
    return base64.b64encode(buffer).decode("ascii")


def _to_cpu(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to("cpu").contiguous()


def encode_f32(values: torch.Tensor, shape: list[int] | None = None) -> EncodedArray:
    """Dense float32, no quantization. For residuals, stats and small matrices."""
    t = _to_cpu(values).to(torch.float32)
    return {
        "dtype": "f32",
        "shape": shape if shape is not None else list(t.shape),
        "layout": "dense",
        "transform": "linear",
        "scale": 1.0,
        "offset": 0.0,
        "encoding": "base64",
        # numpy().tobytes() is little-endian on every platform this runs on;
        # the contract requires little-endian explicitly.
        "data": _b64(t.flatten().numpy().astype("<f4").tobytes()),
    }


def encode_u8_exact(values: torch.Tensor, shape: list[int] | None = None) -> EncodedArray:
    """Dense uint8 with scale 1 / offset 0, so decoded values are exact integers.

    Used for the KV occupancy grid, whose 0/1/2 states are LUT indices rather
    than measurements -- quantizing them would be meaningless.
    """
    t = _to_cpu(values).to(torch.uint8)
    return {
        "dtype": "u8",
        "shape": shape if shape is not None else list(t.shape),
        "layout": "dense",
        "transform": "linear",
        "scale": 1.0,
        "offset": 0.0,
        "encoding": "base64",
        "data": _b64(t.flatten().numpy().tobytes()),
    }


def pack_attention_u8(weights: torch.Tensor) -> bytes:
    """Quantize one [T, T] causal attention matrix to packed lower-triangular u8.

    Two decisions, both required by the contract rather than chosen here:

    `sqrt` -- attention is extremely peaked, and linear u8 has a step of
    1/255 ~= 0.0039, which floors every weight below it to zero. That is
    exactly the low-probability structure the heatmap exists to show. Storing
    round(255*sqrt(p)) and squaring on decode gives roughly 16x the resolution
    near zero for the same bytes.

    `causal_lower` -- the upper triangle is structurally zero, so packing row i
    as its first i+1 entries halves the payload for free.
    """
    t = _to_cpu(weights).to(torch.float32)
    n = t.shape[-1]
    rows, cols = torch.tril_indices(n, n)
    packed = t[rows, cols]
    quantized = (packed.clamp_min(0.0).sqrt() * 255.0).round().clamp(0, 255).to(torch.uint8)
    return quantized.numpy().tobytes()


def encoded_attention(packed: bytes, sequence_length: int) -> EncodedArray:
    """Wraps bytes from `pack_attention_u8` in the EncodedArray envelope."""
    return {
        "dtype": "u8",
        "shape": [sequence_length, sequence_length],
        "layout": "causal_lower",
        "transform": "sqrt",
        "scale": 1.0 / 255.0,
        "offset": 0.0,
        "encoding": "base64",
        "data": _b64(packed),
    }


def encode_attention(weights: torch.Tensor) -> EncodedArray:
    """Convenience: quantize and wrap in one step."""
    return encoded_attention(pack_attention_u8(weights), int(weights.shape[-1]))


def vector_stats(values: torch.Tensor) -> dict[str, float]:
    """The summary that rides along with a vector so the UI needn't scan it."""
    t = _to_cpu(values).to(torch.float32).flatten()
    if t.numel() == 0:
        return {"l2": 0.0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "l2": float(torch.linalg.vector_norm(t)),
        "mean": float(t.mean()),
        # Population std, matching the client's statsOf().
        "std": float(t.std(unbiased=False)),
        "min": float(t.min()),
        "max": float(t.max()),
    }


def row_entropy(probabilities: torch.Tensor) -> torch.Tensor:
    """Shannon entropy in nats along the last dimension, ignoring zeros."""
    p = probabilities.clamp_min(0.0)
    safe = torch.where(p > 0, p, torch.ones_like(p))
    return -(p * safe.log()).sum(dim=-1)


def max_entropy(sequence_length: int) -> float:
    """The ceiling the client normalizes head entropies against."""
    return math.log(max(2, sequence_length))

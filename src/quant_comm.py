"""
quant_comm.py — INT8 Quantization Utilities for ATLAS Communication

Provides symmetric per-channel INT8 quantization of LoRA adapter weights
and split-learning activations for communication-efficient federated learning.

Why symmetric per-channel?
  - Per-channel scale captures weight magnitude variation across output
    features, which is critical for preserving gradient alignment after
    dequantization (Nagel et al., 2021; GPTQ; LLM.int8).
  - Symmetric (zero-point = 0) avoids zero-point biasing in the Laplacian
    update: W_k ← W_k − η Σ a_kl(W_k − W_l).  If both sides have the
    same zero-point the difference is unbiased.
  - INT8 gives a 4× lossless communication reduction vs FP32, with
    quantization noise ≈ 0.5 LSB RMS on the dequantized tensor
    (SNR ≈ 48 dB) — negligible for LoRA rank ≥ 4.

Byte accounting:
  INT8 tensor      → 1 byte per element
  + scale factors  → float32 scale per output channel (p.shape[0])
  Total bytes     ≈ numel + 4 * shape[0]   (scale overhead is <1%)

  Compare to FP32: 4 * numel bytes → savings ≈ 4× (minus scale overhead).

Usage:
  q, scale = quantize_int8(tensor)          # quantize for transmission
  tensor_fp32 = dequantize_int8(q, scale)   # dequantize before computation

  # Byte cost helpers
  n = int8_bytes(tensor)                    # bytes to transmit (int8 + scales)
  n_fp32 = fp32_bytes(tensor)               # reference FP32 byte count
  ratio = compression_ratio(tensor)         # fp32 / int8  (≈ 4.0)
"""

import torch
from typing import Tuple, Dict


# ────────────────────────────────────────────────────────────────────────────
# Core quantization primitives
# ────────────────────────────────────────────────────────────────────────────

def quantize_int8(
    tensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric per-channel INT8 quantization.

    Channel = first dimension (output features).  For LoRA_A (r × d_in),
    the channel is the rank dimension; for LoRA_B (d_out × r) it is the
    output feature dimension.  Both cases are handled uniformly.

    Args:
        tensor: FP32 (or BF16) tensor to quantize.

    Returns:
        q      : INT8 tensor, same shape as input.
        scale  : FP32 per-channel scale, shape = (tensor.shape[0],).
                 Stored as FP32 for exact reconstruction.
    """
    t = tensor.detach().float()  # always work in f32 for scale computation

    # Per-channel absolute maximum (avoid zero-scale for all-zero channels)
    # abs_max shape: (C,) where C = t.shape[0]
    abs_max = t.abs().amax(dim=list(range(1, t.ndim)), keepdim=True)  # (C, 1, ...)
    abs_max = abs_max.clamp(min=1e-8)

    # Scale: map [-abs_max, abs_max]  →  [-127, 127]
    scale = abs_max / 127.0            # (C, 1, ...)
    q = (t / scale).round().clamp(-128, 127).to(torch.int8)

    return q, scale.squeeze()         # scale: (C,)


def dequantize_int8(
    q: torch.Tensor,
    scale: torch.Tensor,
    target_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Dequantize INT8 tensor back to floating-point.

    Args:
        q           : INT8 tensor, shape (C, *).
        scale       : FP32 per-channel scale, shape (C,).
        target_dtype: Output dtype (default float32).

    Returns:
        Dequantized tensor, same shape as q, dtype=target_dtype.
    """
    # Reshape scale to broadcast over all non-channel dims
    view_shape = (-1,) + (1,) * (q.ndim - 1)   # e.g. (C, 1) for 2-D
    s = scale.float().view(view_shape)
    return (q.float() * s).to(target_dtype)


# ────────────────────────────────────────────────────────────────────────────
# Byte-cost helpers
# ────────────────────────────────────────────────────────────────────────────

def int8_bytes(tensor: torch.Tensor) -> int:
    """
    Simulated transmission bytes for INT8-quantized tensor.

    Cost = int8 data (1 byte/element) + FP32 per-channel scales (4 bytes/channel).
    """
    data_bytes  = tensor.numel()                    # 1 byte per INT8 value
    scale_bytes = tensor.shape[0] * 4               # float32 scale per channel
    return data_bytes + scale_bytes


def fp32_bytes(tensor: torch.Tensor) -> int:
    """Naïve FP32 byte count (4 bytes / element)."""
    return tensor.numel() * 4


def compression_ratio(tensor: torch.Tensor) -> float:
    """FP32 / INT8+scale ratio (≈ 4.0 for large tensors)."""
    int8 = int8_bytes(tensor)
    fp32 = fp32_bytes(tensor)
    return fp32 / max(int8, 1)


# ────────────────────────────────────────────────────────────────────────────
# Named-parameter dict utilities (for model state-dicts)
# ────────────────────────────────────────────────────────────────────────────

LoraKeywords = ('lora_a', 'lora_b')


def is_lora_param(name: str) -> bool:
    """Return True if parameter name belongs to a LoRA adapter."""
    name_low = name.lower()
    return any(kw in name_low for kw in LoraKeywords)


def quantize_lora_state(
    state: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, Tuple[torch.Tensor, torch.Tensor]], Dict[str, torch.Tensor]]:
    """
    Quantize all LoRA parameters in a state-dict to INT8.

    Non-LoRA parameters (classifier head, etc.) are passed through
    unchanged (they are typically small and handled server-side).

    Args:
        state: {name: tensor} parameter dict.

    Returns:
        quant_lora  : {name: (int8_tensor, scale)} for LoRA params only.
        passthrough : {name: tensor} for non-LoRA params (unchanged).
    """
    quant_lora: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    passthrough: Dict[str, torch.Tensor] = {}
    for name, param in state.items():
        if is_lora_param(name):
            q, scale = quantize_int8(param)
            quant_lora[name] = (q, scale)
        else:
            passthrough[name] = param
    return quant_lora, passthrough


def dequantize_lora_state(
    quant_lora: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    passthrough: Dict[str, torch.Tensor],
    target_dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """
    Reconstruct a full state-dict from quantized LoRA + passthrough params.

    This is called server-side before the Laplacian regularization step so
    that W_k − W_ℓ is computed in FP32 (no INT8 arithmetic errors).

    Args:
        quant_lora  : {name: (int8, scale)} from quantize_lora_state.
        passthrough : {name: tensor} non-LoRA parameters.
        target_dtype: Target floating-point dtype.

    Returns:
        Reconstructed {name: tensor} state dict (FP32 LoRA + passthrough).
    """
    state: Dict[str, torch.Tensor] = {}
    for name, (q, scale) in quant_lora.items():
        state[name] = dequantize_int8(q, scale, target_dtype)
    state.update(passthrough)
    return state


def lora_state_int8_bytes(state: Dict[str, torch.Tensor]) -> int:
    """
    Total INT8 byte cost for all LoRA params in a state-dict.
    Non-LoRA params are excluded from the count (they are not re-transmitted
    as LoRA updates; they are broadcast by the server).
    """
    total = 0
    for name, param in state.items():
        if is_lora_param(name):
            total += int8_bytes(param)
    return total


def lora_state_fp32_bytes(state: Dict[str, torch.Tensor]) -> int:
    """FP32 byte cost for all LoRA params in a state-dict (reference)."""
    total = 0
    for name, param in state.items():
        if is_lora_param(name):
            total += fp32_bytes(param)
    return total

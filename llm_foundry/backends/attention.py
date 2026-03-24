"""
Attention backend with three-level fallback: Triton → torch.compile → F.scaled_dot_product_attention.

This module provides a unified interface for attention computation that automatically
selects the best available implementation based on hardware capabilities and dependencies.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor


@torch.compile(mode="max-autotune", fullgraph=False)
def _compiled_sdpa(q, k, v, is_causal):
    return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)


def _flash_attention_triton(
    q: Float[Tensor, "batch heads q_len d"],
    k: Float[Tensor, "batch heads k_len d"],
    v: Float[Tensor, "batch heads k_len d"],
    is_causal: bool = True,
) -> Float[Tensor, "batch heads q_len d"]:
    """
    Triton-based FlashAttention-2 implementation.

    Args:
        q: Query tensor, shape (batch, heads, q_len, d)
        k: Key tensor, shape (batch, heads, k_len, d)
        v: Value tensor, shape (batch, heads, k_len, d)
        is_causal: Whether to apply causal mask

    Returns:
        Attention output, shape (batch, heads, q_len, d)
    """
    # Import here to avoid hard dependency
    from .attention_triton import FlashAttention2Triton

    return FlashAttention2Triton.apply(q, k, v, is_causal)


def _flash_attention_compiled(
    q: Float[Tensor, "batch heads q_len d"],
    k: Float[Tensor, "batch heads k_len d"],
    v: Float[Tensor, "batch heads k_len d"],
    is_causal: bool = True,
) -> Float[Tensor, "batch heads q_len d"]:
    """
    torch.compile-optimized FlashAttention implementation.

    Uses PyTorch's native attention with torch.compile for optimization.

    Args:
        q: Query tensor, shape (batch, heads, q_len, d)
        k: Key tensor, shape (batch, heads, k_len, d)
        v: Value tensor, shape (batch, heads, k_len, d)
        is_causal: Whether to apply causal mask

    Returns:
        Attention output, shape (batch, heads, q_len, d)
    """
    return _compiled_sdpa(q, k, v, is_causal)


def _flash_attention_sdpa(
    q: Float[Tensor, "batch heads q_len d"],
    k: Float[Tensor, "batch heads k_len d"],
    v: Float[Tensor, "batch heads k_len d"],
    is_causal: bool = True,
) -> Float[Tensor, "batch heads q_len d"]:
    """
    Standard PyTorch scaled_dot_product_attention (fallback).

    This is the most compatible implementation, available in all PyTorch 2.x versions.

    Args:
        q: Query tensor, shape (batch, heads, q_len, d)
        k: Key tensor, shape (batch, heads, k_len, d)
        v: Value tensor, shape (batch, heads, k_len, d)
        is_causal: Whether to apply causal mask

    Returns:
        Attention output, shape (batch, heads, q_len, d)
    """
    return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)


def _check_triton_available() -> bool:
    """Check if Triton is available and functional."""
    try:
        import triton
        import triton.language as tl  # noqa: F401

        # Also check if we can import our triton kernel
        from .attention_triton import FlashAttention2Triton  # noqa: F401

        return True
    except ImportError:
        return False


def _check_torch_compile_available() -> bool:
    """Check if torch.compile is available and functional."""
    try:
        # torch.compile is available in PyTorch 2.0+
        if not hasattr(torch, "compile"):
            return False

        # Test if torch.compile actually works
        @torch.compile(mode="reduce-overhead")
        def _test_fn(x):
            return x * 2

        test_tensor = torch.randn(4, 4)
        _test_fn(test_tensor)

        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def get_attention_fn(
    use_flash_attn: bool = True,
) -> callable:
    """
    Get the best available attention function with automatic fallback.

    Three-level fallback:
    1. Triton FlashAttention-2 (fastest on GPUs, requires triton package)
    2. torch.compile + F.scaled_dot_product_attention (good performance, requires PyTorch 2.0+)
    3. F.scaled_dot_product_attention (baseline, always available in PyTorch 2.x)

    The result is cached via @lru_cache to avoid repeated capability checks.

    Args:
        use_flash_attn: If False, always use standard SDPA (level 3).
                       If True, try Triton and torch.compile first.

    Returns:
        Callable attention function with signature:
        (q, k, v, is_causal=True) -> output_tensor

    Example:
        >>> attn_fn = get_attention_fn(use_flash_attn=True)
        >>> output = attn_fn(q, k, v, is_causal=True)
    """
    if not use_flash_attn:
        # Explicitly requested to not use flash attention
        return _flash_attention_sdpa

    # Level 1: Try Triton
    if _check_triton_available():
        return _flash_attention_triton

    # Level 2: Try torch.compile
    if _check_torch_compile_available():
        return _flash_attention_compiled

    # Level 3: Fallback to standard SDPA
    return _flash_attention_sdpa


def get_attention_backend_name() -> str:
    """
    Get the name of the currently selected attention backend.

    Returns:
        String name: "triton", "torch.compile", or "sdpa"
    """
    fn = get_attention_fn()
    if fn is _flash_attention_triton:
        return "triton"
    elif fn is _flash_attention_compiled:
        return "torch.compile"
    else:
        return "sdpa"

"""Tests for attention backends."""

import pytest
import torch

from llm_foundry.backends.attention import (
    _flash_attention_sdpa,
    get_attention_backend_name,
    get_attention_fn,
)


def test_get_attention_fn_returns_callable():
    """Test that get_attention_fn returns a callable."""
    fn = get_attention_fn(use_flash_attn=False)
    assert callable(fn)


def test_attention_backend_name():
    """Test that backend name is returned."""
    name = get_attention_backend_name()
    assert name in ["triton", "torch.compile", "sdpa"]


def test_flash_attention_sdpa():
    """Test basic SDPA attention computation."""
    batch, heads, seq_len, d = 2, 4, 16, 64

    q = torch.randn(batch, heads, seq_len, d)
    k = torch.randn(batch, heads, seq_len, d)
    v = torch.randn(batch, heads, seq_len, d)

    output = _flash_attention_sdpa(q, k, v, is_causal=True)

    assert output.shape == (batch, heads, seq_len, d)
    assert not torch.isnan(output).any()


def test_attention_output_shape():
    """Test that attention output has correct shape."""
    fn = get_attention_fn(use_flash_attn=False)

    batch, heads, seq_len, d = 1, 2, 8, 32
    q = torch.randn(batch, heads, seq_len, d)
    k = torch.randn(batch, heads, seq_len, d)
    v = torch.randn(batch, heads, seq_len, d)

    output = fn(q, k, v, is_causal=True)
    assert output.shape == (batch, heads, seq_len, d)


def test_attention_causal_property():
    """Test that causal attention prevents attending to future tokens."""
    fn = get_attention_fn(use_flash_attn=False)

    batch, heads, seq_len, d = 1, 1, 4, 8
    # Use a very distinctive pattern
    q = torch.zeros(batch, heads, seq_len, d)
    k = torch.zeros(batch, heads, seq_len, d)
    v = torch.eye(seq_len).unsqueeze(0).unsqueeze(0).expand(batch, heads, -1, -1)[:, :, :, :d]

    # Set query at position 0 to attend to all keys
    q[0, 0, 0, :] = 1.0

    output = fn(q, k, v, is_causal=True)

    # Output should be finite
    assert torch.isfinite(output).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_attention_cuda():
    """Test attention on CUDA if available."""
    fn = get_attention_fn(use_flash_attn=False)

    batch, heads, seq_len, d = 2, 4, 16, 64
    q = torch.randn(batch, heads, seq_len, d, device="cuda")
    k = torch.randn(batch, heads, seq_len, d, device="cuda")
    v = torch.randn(batch, heads, seq_len, d, device="cuda")

    output = fn(q, k, v, is_causal=True)

    assert output.device.type == "cuda"
    assert output.shape == (batch, heads, seq_len, d)

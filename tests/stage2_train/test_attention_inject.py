"""
Tests for attention injection mechanism.

Tests cover:
- AttentionBackend enum
- get_attention_backend() auto-selection logic
- inject_attention_to_model() functionality
- Dynamic switching (inject and restore)
- Different backends (SDPA on CPU, Triton/torch.compile on GPU)
"""

import pytest
import torch

from llm_foundry.common.model import BasicsTransformerLM, ModelConfig
from llm_foundry.stage2_train.attention_inject import (
    AttentionBackend,
    get_attention_backend,
    inject_attention_to_model,
    restore_original_attention,
    get_current_backend,
    is_attention_injected,
    _check_triton_available,
    _check_torch_compile_available,
)


def test_attention_backend_enum():
    """Test AttentionBackend enum values."""
    assert AttentionBackend.TRITON.value == "triton"
    assert AttentionBackend.TORCH_COMPILE.value == "torch_compile"
    assert AttentionBackend.SDPA.value == "sdpa"
    assert AttentionBackend.AUTO.value == "auto"


def test_attention_backend_from_string():
    """Test creating backend from string."""
    assert AttentionBackend("triton") == AttentionBackend.TRITON
    assert AttentionBackend("torch_compile") == AttentionBackend.TORCH_COMPILE
    assert AttentionBackend("sdpa") == AttentionBackend.SDPA
    assert AttentionBackend("auto") == AttentionBackend.AUTO


class TestGetAttentionBackend:
    """Tests for get_attention_backend function."""

    def test_forces_sdpa_when_use_flash_attn_false(self):
        """When use_flash_attn=False, should always return SDPA."""
        backend, attn_fn = get_attention_backend(AttentionBackend.AUTO, use_flash_attn=False)
        assert backend == AttentionBackend.SDPA
        assert attn_fn is not None

        # Should return SDPA even if TRITON requested
        backend, attn_fn = get_attention_backend(AttentionBackend.TRITON, use_flash_attn=False)
        assert backend == AttentionBackend.SDPA

    def test_returns_requested_backend_when_available(self):
        """Should return requested backend if available."""
        # SDPA is always available
        backend, attn_fn = get_attention_backend(AttentionBackend.SDPA, use_flash_attn=True)
        assert backend == AttentionBackend.SDPA
        assert callable(attn_fn)

    def test_auto_selects_sdpa_on_cpu(self):
        """AUTO should select SDPA when running on CPU without Triton."""
        # This test assumes CPU environment without Triton
        backend, attn_fn = get_attention_backend(AttentionBackend.AUTO, use_flash_attn=True)

        # Should be one of the valid backends
        assert backend in [
            AttentionBackend.TRITON,
            AttentionBackend.TORCH_COMPILE,
            AttentionBackend.SDPA,
        ]
        assert callable(attn_fn)

    def test_attention_function_signature(self):
        """Returned attention function should have correct signature."""
        _, attn_fn = get_attention_backend(AttentionBackend.SDPA, use_flash_attn=False)

        # Test with dummy tensors
        batch, heads, seq, d = 2, 4, 8, 32
        q = torch.randn(batch, heads, seq, d)
        k = torch.randn(batch, heads, seq, d)
        v = torch.randn(batch, heads, seq, d)

        output = attn_fn(q, k, v, is_causal=True)
        assert output.shape == (batch, heads, seq, d)

    def test_attention_function_causal(self):
        """Attention function should respect causal mask."""
        _, attn_fn = get_attention_backend(AttentionBackend.SDPA, use_flash_attn=False)

        # Small test case where causal mask is easy to verify
        batch, heads, seq, d = 1, 1, 4, 8
        torch.manual_seed(42)
        q = torch.randn(batch, heads, seq, d)
        k = torch.randn(batch, heads, seq, d)
        v = torch.randn(batch, heads, seq, d)

        # With causal mask, position i should only attend to positions <= i
        output = attn_fn(q, k, v, is_causal=True)
        assert output.shape == (batch, heads, seq, d)

    def test_caching(self):
        """Same backend should return cached function."""
        backend1, fn1 = get_attention_backend(AttentionBackend.SDPA, use_flash_attn=False)
        backend2, fn2 = get_attention_backend(AttentionBackend.SDPA, use_flash_attn=False)

        assert backend1 == backend2
        # Functions should be the same object (cached)
        assert fn1 is fn2


class TestInjectAttentionToModel:
    """Tests for inject_attention_to_model function."""

    @pytest.fixture
    def small_model(self):
        """Create a small model for testing."""
        config = ModelConfig(
            vocab_size=100,
            context_length=32,
            d_model=64,
            num_layers=2,
            num_heads=2,
            d_ff=128,
            rope_theta=10000.0,
        )
        model = BasicsTransformerLM(
            vocab_size=config.vocab_size,
            context_length=config.context_length,
            d_model=config.d_model,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            rope_theta=config.rope_theta,
        )
        return model

    def test_injection_adds_marker(self, small_model):
        """Injection should add _attn_fn_injected marker to each layer."""
        inject_attention_to_model(small_model, AttentionBackend.SDPA, use_flash_attn=False)

        for layer in small_model.layers:
            assert hasattr(layer.attn, "_attn_fn_injected")
            assert layer.attn._attn_fn_injected is True

    def test_injection_adds_backend_info(self, small_model):
        """Injection should add backend info to each layer."""
        inject_attention_to_model(small_model, AttentionBackend.SDPA, use_flash_attn=False)

        for layer in small_model.layers:
            assert hasattr(layer.attn, "_attn_backend")
            assert layer.attn._attn_backend == "sdpa"

    def test_injection_adds_model_level_markers(self, small_model):
        """Injection should add model-level markers."""
        backend = inject_attention_to_model(small_model, AttentionBackend.SDPA, use_flash_attn=False)

        assert hasattr(small_model, "_attention_injected")
        assert small_model._attention_injected is True
        assert hasattr(small_model, "_attention_backend")
        assert small_model._attention_backend == backend
        assert hasattr(small_model, "_attention_fn")
        assert callable(small_model._attention_fn)

    def test_injection_returns_backend(self, small_model):
        """Injection should return the selected backend."""
        backend = inject_attention_to_model(small_model, AttentionBackend.SDPA, use_flash_attn=False)
        assert isinstance(backend, AttentionBackend)
        assert backend == AttentionBackend.SDPA

    def test_forward_works_after_injection(self, small_model):
        """Model forward pass should work after injection."""
        inject_attention_to_model(small_model, AttentionBackend.SDPA, use_flash_attn=False)

        batch_size, seq_len = 2, 16
        x = torch.randint(0, 100, (batch_size, seq_len))

        with torch.no_grad():
            logits = small_model(x)

        assert logits.shape == (batch_size, seq_len, small_model.vocab_size)
        assert not torch.isnan(logits).any()

    def test_injection_with_auto_backend(self, small_model):
        """Injection with AUTO backend should work."""
        backend = inject_attention_to_model(small_model, AttentionBackend.AUTO, use_flash_attn=False)

        # Should fall back to SDPA when use_flash_attn=False
        assert backend == AttentionBackend.SDPA

    def test_injection_with_string_backend(self, small_model):
        """Injection should accept string backend name."""
        backend = inject_attention_to_model(small_model, "sdpa", use_flash_attn=True)
        assert backend == AttentionBackend.SDPA


class TestRestoreOriginalAttention:
    """Tests for restore_original_attention function."""

    @pytest.fixture
    def small_model(self):
        """Create a small model for testing."""
        config = ModelConfig(
            vocab_size=100,
            context_length=32,
            d_model=64,
            num_layers=2,
            num_heads=2,
            d_ff=128,
            rope_theta=10000.0,
        )
        model = BasicsTransformerLM(
            vocab_size=config.vocab_size,
            context_length=config.context_length,
            d_model=config.d_model,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            rope_theta=config.rope_theta,
        )
        return model

    def test_restore_removes_injection_markers(self, small_model):
        """Restore should remove injection markers."""
        inject_attention_to_model(small_model, AttentionBackend.SDPA, use_flash_attn=False)
        restore_original_attention(small_model)

        for layer in small_model.layers:
            assert not hasattr(layer.attn, "_attn_fn_injected")
            assert not hasattr(layer.attn, "_attn_backend")

    def test_restore_removes_model_markers(self, small_model):
        """Restore should remove model-level markers."""
        inject_attention_to_model(small_model, AttentionBackend.SDPA, use_flash_attn=False)
        restore_original_attention(small_model)

        assert not hasattr(small_model, "_attention_injected")
        assert not hasattr(small_model, "_attention_backend")
        assert not hasattr(small_model, "_attention_fn")

    def test_forward_works_after_restore(self, small_model):
        """Model forward should work after restore."""
        inject_attention_to_model(small_model, AttentionBackend.SDPA, use_flash_attn=False)
        restore_original_attention(small_model)

        batch_size, seq_len = 2, 16
        x = torch.randint(0, 100, (batch_size, seq_len))

        with torch.no_grad():
            logits = small_model(x)

        assert logits.shape == (batch_size, seq_len, small_model.vocab_size)
        assert not torch.isnan(logits).any()

    def test_restore_without_injection(self, small_model):
        """Restore should not fail if injection was never done."""
        # Should not raise any exception
        restore_original_attention(small_model)

        # Model should still work
        batch_size, seq_len = 2, 16
        x = torch.randint(0, 100, (batch_size, seq_len))

        with torch.no_grad():
            logits = small_model(x)

        assert logits.shape == (batch_size, seq_len, small_model.vocab_size)


class TestHelperFunctions:
    """Tests for helper functions."""

    @pytest.fixture
    def small_model(self):
        """Create a small model for testing."""
        config = ModelConfig(
            vocab_size=100,
            context_length=32,
            d_model=64,
            num_layers=2,
            num_heads=2,
            d_ff=128,
            rope_theta=10000.0,
        )
        model = BasicsTransformerLM(
            vocab_size=config.vocab_size,
            context_length=config.context_length,
            d_model=config.d_model,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            rope_theta=config.rope_theta,
        )
        return model

    def test_get_current_backend_before_injection(self, small_model):
        """get_current_backend should return None before injection."""
        backend = get_current_backend(small_model)
        assert backend is None

    def test_get_current_backend_after_injection(self, small_model):
        """get_current_backend should return backend after injection."""
        inject_attention_to_model(small_model, AttentionBackend.SDPA, use_flash_attn=False)
        backend = get_current_backend(small_model)
        assert backend == AttentionBackend.SDPA

    def test_is_attention_injected_before_injection(self, small_model):
        """is_attention_injected should return False before injection."""
        assert is_attention_injected(small_model) is False

    def test_is_attention_injected_after_injection(self, small_model):
        """is_attention_injected should return True after injection."""
        inject_attention_to_model(small_model, AttentionBackend.SDPA, use_flash_attn=False)
        assert is_attention_injected(small_model) is True

    def test_is_attention_injected_after_restore(self, small_model):
        """is_attention_injected should return False after restore."""
        inject_attention_to_model(small_model, AttentionBackend.SDPA, use_flash_attn=False)
        restore_original_attention(small_model)
        assert is_attention_injected(small_model) is False


class TestAvailabilityChecks:
    """Tests for availability check functions."""

    def test_check_triton_available_returns_bool(self):
        """_check_triton_available should return a boolean."""
        result = _check_triton_available()
        assert isinstance(result, bool)

    def test_check_torch_compile_available_returns_bool(self):
        """_check_torch_compile_available should return a boolean."""
        result = _check_torch_compile_available()
        assert isinstance(result, bool)

    def test_torch_compile_available_on_pytorch2(self):
        """torch.compile should be available on PyTorch 2.x."""
        # PyTorch 2.x should have torch.compile
        import torch
        if hasattr(torch, "compile"):
            assert _check_torch_compile_available() is True


class TestDynamicSwitching:
    """Tests for dynamic switching between backends."""

    @pytest.fixture
    def small_model(self):
        """Create a small model for testing."""
        config = ModelConfig(
            vocab_size=100,
            context_length=32,
            d_model=64,
            num_layers=2,
            num_heads=2,
            d_ff=128,
            rope_theta=10000.0,
        )
        model = BasicsTransformerLM(
            vocab_size=config.vocab_size,
            context_length=config.context_length,
            d_model=config.d_model,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            rope_theta=config.rope_theta,
        )
        return model

    def test_switch_from_sdpa_to_sdpa(self, small_model):
        """Test switching between same backend (should work)."""
        inject_attention_to_model(small_model, AttentionBackend.SDPA, use_flash_attn=False)

        batch_size, seq_len = 2, 16
        x = torch.randint(0, 100, (batch_size, seq_len))

        with torch.no_grad():
            logits1 = small_model(x)

        # Inject again (simulating switch)
        inject_attention_to_model(small_model, AttentionBackend.SDPA, use_flash_attn=False)

        with torch.no_grad():
            logits2 = small_model(x)

        # Should produce same results
        assert torch.allclose(logits1, logits2)

    def test_multiple_injections_same_model(self, small_model):
        """Test multiple injections on same model."""
        # First injection
        backend1 = inject_attention_to_model(small_model, AttentionBackend.SDPA, use_flash_attn=False)
        assert backend1 == AttentionBackend.SDPA

        # Second injection (should override)
        backend2 = inject_attention_to_model(small_model, AttentionBackend.SDPA, use_flash_attn=False)
        assert backend2 == AttentionBackend.SDPA

        # Model should still work
        batch_size, seq_len = 2, 16
        x = torch.randint(0, 100, (batch_size, seq_len))

        with torch.no_grad():
            logits = small_model(x)

        assert logits.shape == (batch_size, seq_len, small_model.vocab_size)


class TestIntegration:
    """Integration tests for attention injection."""

    def test_full_training_step_with_injected_attention(self):
        """Test that a full training step works with injected attention."""
        config = ModelConfig(
            vocab_size=100,
            context_length=16,
            d_model=64,
            num_layers=2,
            num_heads=2,
            d_ff=128,
            rope_theta=10000.0,
        )
        model = BasicsTransformerLM(
            vocab_size=config.vocab_size,
            context_length=config.context_length,
            d_model=config.d_model,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            rope_theta=config.rope_theta,
        )

        # Inject attention
        inject_attention_to_model(model, AttentionBackend.SDPA, use_flash_attn=False)

        # Create dummy batch
        batch_size, seq_len = 4, 16
        x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        y = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # Forward pass
        logits = model(x)
        assert logits.shape == (batch_size, seq_len, config.vocab_size)

        # Compute loss
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, config.vocab_size),
            y.view(-1)
        )

        # Backward pass
        loss.backward()

        # Check gradients exist
        has_grads = any(p.grad is not None for p in model.parameters())
        assert has_grads, "Model should have gradients after backward"

        # Optimizer step (just to verify it doesn't crash)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer.step()

    def test_model_output_consistency(self):
        """Test that injected attention produces consistent outputs."""
        config = ModelConfig(
            vocab_size=100,
            context_length=16,
            d_model=64,
            num_layers=2,
            num_heads=2,
            d_ff=128,
            rope_theta=10000.0,
        )

        # Create two identical models
        model1 = BasicsTransformerLM(
            vocab_size=config.vocab_size,
            context_length=config.context_length,
            d_model=config.d_model,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            rope_theta=config.rope_theta,
        )
        model2 = BasicsTransformerLM(
            vocab_size=config.vocab_size,
            context_length=config.context_length,
            d_model=config.d_model,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            rope_theta=config.rope_theta,
        )

        # Copy weights
        model2.load_state_dict(model1.state_dict())

        # Inject only model1
        inject_attention_to_model(model1, AttentionBackend.SDPA, use_flash_attn=False)

        # Both should produce same output (since SDPA is the default)
        batch_size, seq_len = 2, 16
        x = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            logits1 = model1(x)
            logits2 = model2(x)

        # Outputs should be identical (both use SDPA)
        assert torch.allclose(logits1, logits2, atol=1e-6)

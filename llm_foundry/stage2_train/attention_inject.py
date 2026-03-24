"""
Attention injection mechanism for training stage.

Provides monkey-patch based attention backend injection into Transformer models,
supporting dynamic switching between different attention implementations.

Key features:
- Three-level fallback: Triton FlashAttention → torch.compile → F.scaled_dot_product_attention
- Monkey-patch injection without modifying original model code
- Dynamic switch support (inject for training, restore for evaluation)
- Automatic hardware capability detection
"""

from __future__ import annotations

import enum
import logging
from typing import TYPE_CHECKING, Callable

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from llm_foundry.common.model import BasicsTransformerLM

logger = logging.getLogger(__name__)


class AttentionBackend(enum.Enum):
    """Available attention computation backends.

    Attributes:
        TRITON: Triton-based FlashAttention-2 (fastest on GPUs)
        TORCH_COMPILE: torch.compile-optimized SDPA (good performance)
        SDPA: Standard PyTorch scaled_dot_product_attention (baseline)
        AUTO: Automatically select best available backend
    """

    TRITON = "triton"
    TORCH_COMPILE = "torch_compile"
    SDPA = "sdpa"
    AUTO = "auto"


# Type alias for attention function using string annotations
AttentionFn = Callable[..., torch.Tensor]


# Global cache for attention functions per backend
_attention_fn_cache: dict[AttentionBackend, AttentionFn | None] = {
    AttentionBackend.TRITON: None,
    AttentionBackend.TORCH_COMPILE: None,
    AttentionBackend.SDPA: None,
}


def _check_triton_available() -> bool:
    """Check if Triton FlashAttention is available and functional."""
    try:
        import triton
        import triton.language as tl  # noqa: F401

        # Check if we can import the triton kernel module
        from llm_foundry.backends.attention_triton import FlashAttention2Triton  # noqa: F401

        return True
    except ImportError:
        return False


def _check_torch_compile_available() -> bool:
    """Check if torch.compile is available and functional."""
    try:
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


def _get_triton_attention_fn() -> AttentionFn:
    """Get Triton FlashAttention-2 function."""
    from llm_foundry.backends.attention_triton import FlashAttention2Triton

    def _triton_attn(q, k, v, is_causal=True):
        return FlashAttention2Triton.apply(q, k, v, is_causal)

    return _triton_attn


def _get_torch_compile_attention_fn() -> AttentionFn:
    """Get torch.compile-optimized SDPA function."""

    @torch.compile(mode="max-autotune", fullgraph=False)
    def _compiled_sdpa(q, k, v, is_causal):
        return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

    def _torch_compile_attn(q, k, v, is_causal=True):
        return _compiled_sdpa(q, k, v, is_causal)

    return _torch_compile_attn


def _get_sdpa_attention_fn() -> AttentionFn:
    """Get standard PyTorch SDPA function."""

    def _sdpa_attn(q, k, v, is_causal=True):
        return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

    return _sdpa_attn


def get_attention_backend(
    backend: AttentionBackend | str = AttentionBackend.AUTO,
    use_flash_attn: bool = True,
) -> tuple[AttentionBackend, AttentionFn]:
    """Get the appropriate attention backend and function.

    Three-level fallback when backend=AUTO:
    1. Triton FlashAttention-2 (fastest on GPUs, requires triton)
    2. torch.compile + F.scaled_dot_product_attention (PyTorch 2.0+)
    3. F.scaled_dot_product_attention (baseline, always available)

    Args:
        backend: Which backend to use. If AUTO, automatically selects best available.
        use_flash_attn: If False, always use SDPA (level 3), ignoring backend setting.

    Returns:
        Tuple of (selected_backend, attention_function)

    Example:
        >>> backend, attn_fn = get_attention_backend(AttentionBackend.AUTO)
        >>> output = attn_fn(q, k, v, is_causal=True)
    """
    # Convert string to enum if needed
    if isinstance(backend, str):
        backend = AttentionBackend(backend)

    # If flash attention is disabled, force SDPA
    if not use_flash_attn:
        if _attention_fn_cache[AttentionBackend.SDPA] is None:
            _attention_fn_cache[AttentionBackend.SDPA] = _get_sdpa_attention_fn()
        return AttentionBackend.SDPA, _attention_fn_cache[AttentionBackend.SDPA]

    # Return cached if available
    if backend != AttentionBackend.AUTO and _attention_fn_cache[backend] is not None:
        return backend, _attention_fn_cache[backend]

    # Determine which backend to use
    selected_backend = backend

    if backend == AttentionBackend.AUTO:
        # Try each backend in order of preference
        if _check_triton_available():
            selected_backend = AttentionBackend.TRITON
        elif _check_torch_compile_available():
            selected_backend = AttentionBackend.TORCH_COMPILE
        else:
            selected_backend = AttentionBackend.SDPA
    elif backend == AttentionBackend.TRITON and not _check_triton_available():
        logger.warning("Triton backend requested but not available, falling back to AUTO")
        return get_attention_backend(AttentionBackend.AUTO, use_flash_attn)
    elif backend == AttentionBackend.TORCH_COMPILE and not _check_torch_compile_available():
        logger.warning("torch.compile backend requested but not available, falling back to AUTO")
        return get_attention_backend(AttentionBackend.AUTO, use_flash_attn)

    # Create and cache the attention function
    if _attention_fn_cache[selected_backend] is None:
        if selected_backend == AttentionBackend.TRITON:
            _attention_fn_cache[selected_backend] = _get_triton_attention_fn()
        elif selected_backend == AttentionBackend.TORCH_COMPILE:
            _attention_fn_cache[selected_backend] = _get_torch_compile_attention_fn()
        else:  # SDPA
            _attention_fn_cache[selected_backend] = _get_sdpa_attention_fn()

    return selected_backend, _attention_fn_cache[selected_backend]


def _make_patched_forward(original_attn, attn_fn: AttentionFn):
    """Factory function: create a patched forward method for attention module.

    Args:
        original_attn: The original CausalMultiHeadSelfAttention instance
        attn_fn: The attention function to inject

    Returns:
        Patched forward function
    """
    from einops import rearrange

    # Store original forward for restoration
    original_forward = original_attn.forward

    def patched_forward(x, token_positions=None):
        """Patched forward using injected attention function."""
        *b, sequence_length, d_model = x.size()
        assert d_model == original_attn.d_model

        # Project to Q/K/V
        Q = original_attn.q_proj(x)
        K = original_attn.k_proj(x)
        V = original_attn.v_proj(x)

        # Reshape to multi-head
        Q, K, V = (
            rearrange(X, "... seq (heads d) -> ... heads seq d", heads=original_attn.num_heads)
            for X in (Q, K, V)
        )

        # Handle token positions
        if token_positions is None:
            token_positions = torch.arange(sequence_length, device=x.device).unsqueeze(0)
            if len(b) > 0:
                token_positions = token_positions.expand(*b, -1)

        token_positions = rearrange(token_positions, "... seq -> ... 1 seq")

        # Apply RoPE
        Q = original_attn.positional_encoder(Q, token_positions)
        K = original_attn.positional_encoder(K, token_positions)

        # Use injected attention function
        attn_output = attn_fn(Q, K, V, is_causal=True)

        # Reshape and project output
        attn_output = rearrange(attn_output, "batch heads seq d_v -> batch seq (heads d_v)").contiguous()
        output = original_attn.output_proj(attn_output)
        return output

    # Attach reference to original for restoration
    patched_forward._original_forward = original_forward
    return patched_forward


def inject_attention_to_model(
    model: BasicsTransformerLM,
    backend: AttentionBackend | str = AttentionBackend.AUTO,
    use_flash_attn: bool = True,
) -> AttentionBackend:
    """Inject custom attention function into a Transformer model.

    Uses monkey-patching to replace the forward method of each attention module
    without modifying the original class definition.

    Args:
        model: The Transformer model to inject into
        backend: Which attention backend to use
        use_flash_attn: Whether to use flash attention (if False, uses SDPA)

    Returns:
        The selected backend that was injected

    Example:
        >>> model = Transformer(config)
        >>> backend = inject_attention_to_model(model, AttentionBackend.TRITON)
        >>> print(f"Injected {backend.value} attention")
    """
    # Get the appropriate attention function
    selected_backend, attn_fn = get_attention_backend(backend, use_flash_attn)

    # Inject into each transformer block
    injected_count = 0
    for layer in model.layers:
        attn = layer.attn

        # Create and apply patched forward
        patched_forward = _make_patched_forward(attn, attn_fn)
        attn.forward = patched_forward

        # Mark as injected
        attn._attn_fn_injected = True
        attn._attn_backend = selected_backend.value
        injected_count += 1

    # Store injection info on model
    model._attention_backend = selected_backend
    model._attention_fn = attn_fn
    model._attention_injected = True

    logger.info(f"Injected {selected_backend.value} attention into {injected_count} layers")
    return selected_backend


def restore_original_attention(model: BasicsTransformerLM) -> None:
    """Restore original attention forward methods.

    Reverses the effect of inject_attention_to_model by restoring the
    original forward methods that were saved during injection.

    Args:
        model: The Transformer model to restore

    Example:
        >>> inject_attention_to_model(model, AttentionBackend.TRITON)
        >>> # ... training ...
        >>> restore_original_attention(model)  # Restore for evaluation
    """
    restored_count = 0
    for layer in model.layers:
        attn = layer.attn

        # Check if this layer was injected
        if hasattr(attn, "forward") and hasattr(attn.forward, "_original_forward"):
            attn.forward = attn.forward._original_forward
            restored_count += 1

        # Remove injection markers
        if hasattr(attn, "_attn_fn_injected"):
            delattr(attn, "_attn_fn_injected")
        if hasattr(attn, "_attn_backend"):
            delattr(attn, "_attn_backend")

    # Remove model-level injection markers
    if hasattr(model, "_attention_injected"):
        delattr(model, "_attention_injected")
    if hasattr(model, "_attention_backend"):
        delattr(model, "_attention_backend")
    if hasattr(model, "_attention_fn"):
        delattr(model, "_attention_fn")

    logger.info(f"Restored original attention for {restored_count} layers")


def get_current_backend(model: BasicsTransformerLM) -> AttentionBackend | None:
    """Get the currently injected attention backend for a model.

    Args:
        model: The Transformer model to check

    Returns:
        The current backend if injected, None otherwise
    """
    if hasattr(model, "_attention_backend"):
        return model._attention_backend
    return None


def is_attention_injected(model: BasicsTransformerLM) -> bool:
    """Check if attention has been injected into the model.

    Args:
        model: The Transformer model to check

    Returns:
        True if attention is injected, False otherwise
    """
    return getattr(model, "_attention_injected", False)

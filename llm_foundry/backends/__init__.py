"""Hardware backend abstractions for attention and inference.

This module provides hardware-optimized implementations with automatic fallback:

Attention Backends (attention.py):
    - Triton FlashAttention-2 (fastest, requires triton package)
    - torch.compile + SDPA (good performance, PyTorch 2.0+)
    - F.scaled_dot_product_attention (baseline, always available)

Inference Backends (inference.py):
    - vLLM (highest throughput for serving)
    - HuggingFace generate (most compatible)

Example:
    >>> from llm_foundry.backends import get_attention_fn, get_inference_backend
    >>>
    >>> # Get best available attention function
    >>> attn_fn = get_attention_fn(use_flash_attn=True)
    >>> output = attn_fn(q, k, v, is_causal=True)
    >>>
    >>> # Get inference backend
    >>> backend = get_inference_backend("auto", model_name_or_path="gpt2")
    >>> generated = backend.generate("Hello, world!")
"""

from .attention import get_attention_fn
from .inference import (
    InferenceBackend,
    HFInferenceBackend,
    VLLMInferenceBackend,
    GenerationConfig,
    get_inference_backend,
    check_inference_backends,
)

__all__ = [
    # Attention
    "get_attention_fn",
    # Inference
    "InferenceBackend",
    "HFInferenceBackend",
    "VLLMInferenceBackend",
    "GenerationConfig",
    "get_inference_backend",
    "check_inference_backends",
]

"""
Inference backend with two-level fallback: vLLM → HuggingFace generate.

This module provides a unified interface for text generation that automatically
selects the best available backend based on dependencies and use case.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


@dataclass
class GenerationConfig:
    """Configuration for text generation.

    Attributes:
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (1.0 = no change, <1.0 = more focused, >1.0 = more random)
        top_k: Top-k sampling (0 = disabled)
        top_p: Nucleus sampling threshold (1.0 = disabled)
        repetition_penalty: Penalty for repeating tokens (1.0 = no penalty)
        do_sample: Whether to use sampling (False = greedy decoding)
    """

    max_new_tokens: int = 256
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    do_sample: bool = True


class InferenceBackend(ABC):
    """Abstract base class for inference backends."""

    @abstractmethod
    def generate(self, prompt: str, config: GenerationConfig | None = None) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Input text prompt
            config: Generation configuration (uses defaults if None)

        Returns:
            Generated text string
        """
        pass

    @abstractmethod
    def generate_batch(
        self, prompts: list[str], config: GenerationConfig | None = None
    ) -> list[str]:
        """Generate text for multiple prompts.

        Args:
            prompts: List of input text prompts
            config: Generation configuration (uses defaults if None)

        Returns:
            List of generated text strings
        """
        pass


class HFInferenceBackend(InferenceBackend):
    """HuggingFace transformers-based inference backend."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = "cuda",
    ):
        """Initialize HF backend.

        Args:
            model: HuggingFace model instance
            tokenizer: HuggingFace tokenizer instance
            device: Device to run inference on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate(self, prompt: str, config: GenerationConfig | None = None) -> str:
        """Generate text using HF model.generate()."""
        if config is None:
            config = GenerationConfig()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature if config.do_sample else None,
                top_k=config.top_k if config.do_sample else None,
                top_p=config.top_p if config.do_sample else None,
                repetition_penalty=config.repetition_penalty,
                do_sample=config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        input_length = inputs["input_ids"].shape[1]
        new_tokens = outputs[0][input_length:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def generate_batch(
        self, prompts: list[str], config: GenerationConfig | None = None
    ) -> list[str]:
        """Generate text for multiple prompts using batching."""
        if config is None:
            config = GenerationConfig()

        # Tokenize all prompts
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature if config.do_sample else None,
                top_k=config.top_k if config.do_sample else None,
                top_p=config.top_p if config.do_sample else None,
                repetition_penalty=config.repetition_penalty,
                do_sample=config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the new tokens for each prompt
        results = []
        for i, prompt in enumerate(prompts):
            input_length = inputs["attention_mask"][i].sum().item()
            new_tokens = outputs[i][input_length:]
            results.append(self.tokenizer.decode(new_tokens, skip_special_tokens=True))

        return results


class VLLMInferenceBackend(InferenceBackend):
    """vLLM-based inference backend for high-throughput serving."""

    def __init__(
        self,
        model_name_or_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
    ):
        """Initialize vLLM backend.

        Args:
            model_name_or_path: Model identifier or path
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use
            max_model_len: Maximum sequence length (None = use model config)
        """
        from vllm import LLM

        self.llm = LLM(
            model=model_name_or_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )

    def generate(self, prompt: str, config: GenerationConfig | None = None) -> str:
        """Generate text using vLLM."""
        from vllm import SamplingParams

        if config is None:
            config = GenerationConfig()

        sampling_params = SamplingParams(
            n=1,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            repetition_penalty=config.repetition_penalty,
            max_tokens=config.max_new_tokens,
        )

        outputs = self.llm.generate(prompt, sampling_params)
        return outputs[0].outputs[0].text

    def generate_batch(
        self, prompts: list[str], config: GenerationConfig | None = None
    ) -> list[str]:
        """Generate text for multiple prompts using vLLM batching."""
        from vllm import SamplingParams

        if config is None:
            config = GenerationConfig()

        sampling_params = SamplingParams(
            n=1,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            repetition_penalty=config.repetition_penalty,
            max_tokens=config.max_new_tokens,
        )

        outputs = self.llm.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]


def get_inference_backend(
    backend_type: str = "auto",
    model_name_or_path: str | None = None,
    model: PreTrainedModel | None = None,
    tokenizer: PreTrainedTokenizer | None = None,
    device: str = "cuda",
    **kwargs,
) -> InferenceBackend:
    """
    Get the best available inference backend with automatic fallback.

    Two-level fallback:
    1. vLLM (highest throughput for serving, requires vllm package)
    2. HuggingFace generate (most compatible, always available with transformers)

    Args:
        backend_type: "auto", "vllm", or "hf"
        model_name_or_path: Model identifier for vLLM backend
        model: HF model instance for HF backend
        tokenizer: HF tokenizer instance for HF backend
        device: Device for HF backend
        **kwargs: Additional arguments passed to backend constructor

    Returns:
        InferenceBackend instance

    Raises:
        ValueError: If backend_type is invalid or required args are missing
    """
    if backend_type == "auto":
        # Try vLLM first, fall back to HF
        try:
            import vllm  # noqa: F401

            if model_name_or_path is not None:
                return VLLMInferenceBackend(model_name_or_path, **kwargs)
            else:
                # Can't use vLLM without model path, fall back to HF
                backend_type = "hf"
        except ImportError:
            backend_type = "hf"

    if backend_type == "vllm":
        if model_name_or_path is None:
            raise ValueError("model_name_or_path is required for vLLM backend")
        return VLLMInferenceBackend(model_name_or_path, **kwargs)

    elif backend_type == "hf":
        if model is None or tokenizer is None:
            raise ValueError("model and tokenizer are required for HF backend")
        return HFInferenceBackend(model, tokenizer, device)

    else:
        raise ValueError(f"Unknown backend_type: {backend_type}")


def check_inference_backends() -> dict[str, bool]:
    """Check which inference backends are available.

    Returns:
        Dictionary mapping backend names to availability booleans
    """
    result = {"hf": True, "vllm": False}

    try:
        import vllm  # noqa: F401

        result["vllm"] = True
    except ImportError:
        pass

    return result



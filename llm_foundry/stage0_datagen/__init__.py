"""Stage 0: Data Generation Module.

This module provides data generation capabilities for LLM training, including:
- SFT (Supervised Fine-Tuning) instruction-following data generation
- GRPO (Group Relative Policy Optimization) math reasoning data generation
- DeepSeek API client with rate limiting and token bucket

The module uses async/await for efficient concurrent API calls and implements
resumable generation to handle interruptions gracefully.

Example:
    >>> from llm_foundry.stage0_datagen import DataGenConfig, run_datagen
    >>> config = DataGenConfig.from_yaml("configs/datagen.yaml")
    >>> stats = run_datagen(config)
    >>> print(f"Generated {stats['sft']} SFT samples, {stats['grpo']} GRPO samples")

Classes:
    DataGenConfig: Configuration for data generation
    SFTGenerator: Generator for SFT instruction-following data
    GRPOGenerator: Generator for GRPO math reasoning data
    DeepSeekClient: Async API client with rate limiting

Functions:
    run_datagen: Main entry point for data generation pipeline
"""

from .datagen import DataGenConfig, run_datagen
from .sft_gen import SFTGenerator
from .grpo_gen import GRPOGenerator
from .client import DeepSeekClient

__all__ = ["DataGenConfig", "run_datagen", "SFTGenerator", "GRPOGenerator", "DeepSeekClient"]

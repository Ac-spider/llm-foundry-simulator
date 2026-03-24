"""Common utilities shared across all stages.

This module provides core components used throughout the LLM Foundry Simulator:

- model: Transformer language model with configurable architecture
- optimizer: AdamW optimizer with cosine LR scheduler and ZeRO-1 sharding
- data: Data loading utilities for training
- config: YAML configuration management with hash validation
- env_check: Hardware environment detection
- hashing: SHA256 hashing for reproducibility

Example:
    >>> from llm_foundry.common import ModelConfig, create_model
    >>> from llm_foundry.common import load_config, check_environment
    >>>
    >>> # Check environment
    >>> env = check_environment()
    >>> env.print_report()
    >>>
    >>> # Load config and create model
    >>> config = load_config("configs/train.yaml")
    >>> model_config = ModelConfig(d_model=768, num_layers=12)
    >>> model = create_model(model_config)
"""

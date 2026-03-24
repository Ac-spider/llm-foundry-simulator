"""
Configuration management with YAML loading and hash validation.

Provides utilities for loading, validating, and caching configurations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from .hashing import compute_config_hash, verify_config_hash

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> SimpleNamespace:
    """
    Load YAML configuration file into a SimpleNamespace.

    SimpleNamespace allows attribute-style access (config.lr instead of config['lr']).

    Args:
        config_path: Path to YAML config file

    Returns:
        SimpleNamespace with config values as attributes

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        data = {}

    # Convert nested dicts to SimpleNamespace recursively
    def _convert(obj):
        if isinstance(obj, dict):
            return SimpleNamespace(**{k: _convert(v) for k, v in obj.items()})
        elif isinstance(obj, list):
            return [_convert(item) for item in obj]
        else:
            return obj

    config = _convert(data)
    logger.info(f"Loaded config from {config_path}")
    return config


def load_config_with_hash(config_path: str | Path) -> tuple[SimpleNamespace, str]:
    """
    Load config and compute its hash for caching/reproducibility.

    Args:
        config_path: Path to YAML config file

    Returns:
        Tuple of (config, hash_string)
    """
    config = load_config(config_path)
    config_dict = namespace_to_dict(config)
    config_hash = compute_config_hash(config_dict)
    return config, config_hash


def namespace_to_dict(ns: SimpleNamespace | Any) -> dict[str, Any] | Any:
    """
    Convert SimpleNamespace back to dictionary recursively.

    Args:
        ns: SimpleNamespace or any other value

    Returns:
        Dictionary representation or original value
    """
    if isinstance(ns, SimpleNamespace):
        return {k: namespace_to_dict(v) for k, v in ns.__dict__.items()}
    elif isinstance(ns, list):
        return [namespace_to_dict(item) for item in ns]
    else:
        return ns


def save_config(config: SimpleNamespace | dict, output_path: str | Path) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration as SimpleNamespace or dict
        output_path: Path to save YAML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(config, SimpleNamespace):
        config = namespace_to_dict(config)

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=True)

    logger.info(f"Saved config to {output_path}")


def validate_config(config: SimpleNamespace | dict, required_keys: list[str]) -> bool:
    """
    Validate that config contains all required keys.

    Args:
        config: Configuration as SimpleNamespace or dict
        required_keys: List of required key names (supports nested keys with dots)

    Returns:
        True if all required keys are present

    Raises:
        ValueError: If any required key is missing
    """
    if isinstance(config, SimpleNamespace):
        config = namespace_to_dict(config)

    for key in required_keys:
        keys = key.split(".")
        current = config
        for k in keys:
            if not isinstance(current, dict) or k not in current:
                raise ValueError(f"Missing required config key: {key}")
            current = current[k]

    return True


def merge_configs(base: SimpleNamespace | dict, override: SimpleNamespace | dict) -> SimpleNamespace:
    """
    Merge two configurations, with override taking precedence.

    Args:
        base: Base configuration
        override: Override configuration

    Returns:
        Merged configuration as SimpleNamespace
    """
    if isinstance(base, SimpleNamespace):
        base = namespace_to_dict(base)
    if isinstance(override, SimpleNamespace):
        override = namespace_to_dict(override)

    def _merge(base_dict: dict, override_dict: dict) -> dict:
        result = base_dict.copy()
        for key, value in override_dict.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = _merge(result[key], value)
            else:
                result[key] = value
        return result

    merged = _merge(base, override)
    return SimpleNamespace(**merged)


@dataclass
class TrainConfig:
    """Training configuration dataclass with validation."""

    # Model config
    vocab_size: int = 50257
    context_length: int = 1024
    d_model: int = 768
    num_layers: int = 12
    num_heads: int = 12
    d_ff: int = 3072

    # Training config
    batch_size: int = 32
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.01
    max_iters: int = 100000
    warmup_iters: int = 2000
    eval_interval: int = 1000
    eval_iters: int = 100

    # System config
    device: str = "cuda"
    compile: bool = True
    use_flash_attn: bool = True

    # Paths
    data_path: str = ""
    output_dir: str = "./outputs"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model": {
                "vocab_size": self.vocab_size,
                "context_length": self.context_length,
                "d_model": self.d_model,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "d_ff": self.d_ff,
            },
            "training": {
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "min_lr": self.min_lr,
                "weight_decay": self.weight_decay,
                "max_iters": self.max_iters,
                "warmup_iters": self.warmup_iters,
                "eval_interval": self.eval_interval,
                "eval_iters": self.eval_iters,
            },
            "system": {
                "device": self.device,
                "compile": self.compile,
                "use_flash_attn": self.use_flash_attn,
            },
            "paths": {
                "data_path": self.data_path,
                "output_dir": self.output_dir,
            },
        }

    @classmethod
    def from_namespace(cls, ns: SimpleNamespace) -> "TrainConfig":
        """Create from SimpleNamespace."""
        config = cls()

        if hasattr(ns, "model"):
            m = ns.model
            config.vocab_size = getattr(m, "vocab_size", config.vocab_size)
            config.context_length = getattr(m, "context_length", config.context_length)
            config.d_model = getattr(m, "d_model", config.d_model)
            config.num_layers = getattr(m, "num_layers", config.num_layers)
            config.num_heads = getattr(m, "num_heads", config.num_heads)
            config.d_ff = getattr(m, "d_ff", config.d_ff)

        if hasattr(ns, "training"):
            t = ns.training
            config.batch_size = getattr(t, "batch_size", config.batch_size)
            config.learning_rate = getattr(t, "learning_rate", config.learning_rate)
            config.min_lr = getattr(t, "min_lr", config.min_lr)
            config.weight_decay = getattr(t, "weight_decay", config.weight_decay)
            config.max_iters = getattr(t, "max_iters", config.max_iters)
            config.warmup_iters = getattr(t, "warmup_iters", config.warmup_iters)
            config.eval_interval = getattr(t, "eval_interval", config.eval_interval)
            config.eval_iters = getattr(t, "eval_iters", config.eval_iters)

        if hasattr(ns, "system"):
            s = ns.system
            config.device = getattr(s, "device", config.device)
            config.compile = getattr(s, "compile", config.compile)
            config.use_flash_attn = getattr(s, "use_flash_attn", config.use_flash_attn)

        if hasattr(ns, "paths"):
            p = ns.paths
            config.data_path = getattr(p, "data_path", config.data_path)
            config.output_dir = getattr(p, "output_dir", config.output_dir)

        return config

"""Tests for config module."""

import tempfile
from pathlib import Path

import pytest
from types import SimpleNamespace

from llm_foundry.common.config import (
    TrainConfig,
    load_config,
    merge_configs,
    namespace_to_dict,
    save_config,
    validate_config,
)


def test_load_config():
    """Test config loading."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("lr: 0.001\nbatch_size: 32\n")
        temp_path = f.name

    config = load_config(temp_path)

    assert isinstance(config, SimpleNamespace)
    assert config.lr == 0.001
    assert config.batch_size == 32

    # Clean up
    Path(temp_path).unlink()


def test_load_config_nested():
    """Test nested config loading."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("model:\n  d_model: 768\n  num_layers: 12\n")
        temp_path = f.name

    config = load_config(temp_path)

    assert isinstance(config.model, SimpleNamespace)
    assert config.model.d_model == 768
    assert config.model.num_layers == 12

    # Clean up
    Path(temp_path).unlink()


def test_namespace_to_dict():
    """Test namespace to dict conversion."""
    ns = SimpleNamespace(a=1, b=SimpleNamespace(c=2))
    d = namespace_to_dict(ns)

    assert d == {"a": 1, "b": {"c": 2}}


def test_save_and_load_config():
    """Test config saving and loading roundtrip."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SimpleNamespace(lr=0.001, batch_size=32)
        output_path = Path(tmpdir) / "config.yaml"

        save_config(config, output_path)
        assert output_path.exists()

        loaded = load_config(output_path)
        assert loaded.lr == 0.001
        assert loaded.batch_size == 32


def test_validate_config():
    """Test config validation."""
    config = {"lr": 0.001, "batch_size": 32}

    # Should pass with existing keys
    assert validate_config(config, ["lr", "batch_size"]) is True

    # Should fail with missing key
    with pytest.raises(ValueError):
        validate_config(config, ["lr", "missing_key"])


def test_merge_configs():
    """Test config merging."""
    base = SimpleNamespace(lr=0.001, batch_size=32)
    override = SimpleNamespace(lr=0.002)

    merged = merge_configs(base, override)

    assert merged.lr == 0.002  # Overridden
    assert merged.batch_size == 32  # Preserved


def test_train_config_defaults():
    """Test TrainConfig default values."""
    config = TrainConfig()

    assert config.vocab_size == 50257
    assert config.d_model == 768
    assert config.num_layers == 12


def test_train_config_to_dict():
    """Test TrainConfig to dict conversion."""
    config = TrainConfig()
    d = config.to_dict()

    assert "model" in d
    assert "training" in d
    assert d["model"]["d_model"] == 768


def test_train_config_from_namespace():
    """Test TrainConfig from namespace."""
    ns = SimpleNamespace(
        model=SimpleNamespace(d_model=512, num_layers=6),
        training=SimpleNamespace(batch_size=64),
    )

    config = TrainConfig.from_namespace(ns)

    assert config.d_model == 512
    assert config.num_layers == 6
    assert config.batch_size == 64
    # Defaults preserved
    assert config.vocab_size == 50257

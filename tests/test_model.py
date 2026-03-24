"""Tests for model module."""

import pytest
import torch

from llm_foundry.common.model import BasicsTransformerLM, ModelConfig, create_model


def test_model_config():
    """Test ModelConfig dataclass."""
    config = ModelConfig(
        vocab_size=1000,
        context_length=128,
        d_model=256,
        num_layers=4,
        num_heads=4,
        d_ff=512,
    )

    assert config.vocab_size == 1000
    assert config.d_model == 256

    config_dict = config.to_dict()
    assert config_dict["vocab_size"] == 1000

    restored = ModelConfig.from_dict(config_dict)
    assert restored.vocab_size == 1000


def test_create_model():
    """Test model creation."""
    config = ModelConfig(
        vocab_size=100,
        context_length=32,
        d_model=64,
        num_layers=2,
        num_heads=4,
        d_ff=128,
    )

    model = create_model(config, use_flash_attn=False)
    assert isinstance(model, BasicsTransformerLM)
    assert model.vocab_size == 100


def test_model_forward():
    """Test model forward pass."""
    config = ModelConfig(
        vocab_size=100,
        context_length=32,
        d_model=64,
        num_layers=2,
        num_heads=4,
        d_ff=128,
    )

    model = create_model(config, use_flash_attn=False)
    batch_size, seq_len = 2, 16

    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    logits = model(x)

    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    assert not torch.isnan(logits).any()


def test_model_generate():
    """Test model text generation."""
    config = ModelConfig(
        vocab_size=100,
        context_length=32,
        d_model=64,
        num_layers=2,
        num_heads=4,
        d_ff=128,
    )

    model = create_model(config, use_flash_attn=False)
    model.eval()

    prompt = torch.randint(0, config.vocab_size, (1, 5))
    output = model.generate(prompt, max_new_tokens=5, temperature=1.0)

    assert output.shape[0] == 1  # Batch size
    assert output.shape[1] <= 5  # At most max_new_tokens


def test_model_get_num_params():
    """Test parameter counting."""
    config = ModelConfig(
        vocab_size=100,
        context_length=32,
        d_model=64,
        num_layers=2,
        num_heads=4,
        d_ff=128,
    )

    model = create_model(config, use_flash_attn=False)
    n_params = model.get_num_params(non_embedding=True)
    assert n_params > 0

    n_params_all = model.get_num_params(non_embedding=False)
    assert n_params_all > n_params  # Including embeddings

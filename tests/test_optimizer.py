"""Tests for optimizer module."""

import pytest
import torch

from llm_foundry.common.optimizer import (
    AdamW,
    ShardedOptimizer,
    create_optimizer,
    get_cosine_lr,
)


def test_get_cosine_lr_warmup():
    """Test linear warmup phase."""
    lr = get_cosine_lr(
        it=0,
        max_learning_rate=1e-3,
        min_learning_rate=1e-4,
        warmup_iters=100,
        cosine_cycle_iters=1000,
    )
    assert lr == 0.0  # Start at 0

    lr = get_cosine_lr(
        it=50,
        max_learning_rate=1e-3,
        min_learning_rate=1e-4,
        warmup_iters=100,
        cosine_cycle_iters=1000,
    )
    assert abs(lr - 0.5e-3) < 1e-6  # Halfway through warmup


def test_get_cosine_lr_peak():
    """Test peak learning rate at end of warmup."""
    lr = get_cosine_lr(
        it=100,
        max_learning_rate=1e-3,
        min_learning_rate=1e-4,
        warmup_iters=100,
        cosine_cycle_iters=1000,
    )
    assert abs(lr - 1e-3) < 1e-6


def test_get_cosine_lr_decay():
    """Test cosine decay phase."""
    lr = get_cosine_lr(
        it=550,  # Middle of cosine decay
        max_learning_rate=1e-3,
        min_learning_rate=1e-4,
        warmup_iters=100,
        cosine_cycle_iters=1000,
    )
    # Should be between min and max
    assert 1e-4 < lr < 1e-3


def test_get_cosine_lr_minimum():
    """Test minimum learning rate after decay."""
    lr = get_cosine_lr(
        it=1001,
        max_learning_rate=1e-3,
        min_learning_rate=1e-4,
        warmup_iters=100,
        cosine_cycle_iters=1000,
    )
    assert abs(lr - 1e-4) < 1e-6


def test_adamw_step():
    """Test AdamW optimizer step."""
    param = torch.nn.Parameter(torch.randn(10, 10))
    param.grad = torch.randn(10, 10)

    optimizer = AdamW([param], lr=1e-3, weight_decay=0.01)
    optimizer.step()

    # Parameter should have been updated
    assert optimizer.state[param]["step"] == 1
    assert "exp_avg" in optimizer.state[param]
    assert "exp_avg_sq" in optimizer.state[param]


def test_adamw_weight_decay():
    """Test that weight decay is applied."""
    param = torch.nn.Parameter(torch.ones(10, 10))
    param.grad = torch.zeros(10, 10)

    optimizer = AdamW([param], lr=1e-3, weight_decay=0.1)
    initial_value = param.data.clone()

    optimizer.step()

    # Weight decay should have reduced the parameter
    assert not torch.allclose(param.data, initial_value)


def test_create_optimizer_basic():
    """Test basic optimizer creation."""
    param = torch.nn.Parameter(torch.randn(10))
    optimizer = create_optimizer([param], lr=1e-3, sharded=False)

    assert isinstance(optimizer, AdamW)


def test_sharded_optimizer_init():
    """Test ShardedOptimizer initialization."""
    params = [torch.nn.Parameter(torch.randn(10)) for _ in range(4)]

    optimizer = ShardedOptimizer(params, AdamW, lr=1e-3)
    assert optimizer.world_size == 1  # No distributed
    assert optimizer.rank == 0


def test_sharded_optimizer_step():
    """Test ShardedOptimizer step."""
    param = torch.nn.Parameter(torch.randn(10, 10))
    param.grad = torch.randn(10, 10)

    optimizer = ShardedOptimizer([param], AdamW, lr=1e-3)
    optimizer.step()

    # Should complete without error
    assert True

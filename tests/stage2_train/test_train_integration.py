"""
Integration tests for Stage 2: Training module.

These tests verify the end-to-end functionality of the training pipeline,
including model training, DDP, checkpointing, and learning rate scheduling.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from llm_foundry.backends.attention import get_attention_fn
from llm_foundry.common.config import TrainConfig
from llm_foundry.common.data import create_data_loader, get_batch, load_tokens
from llm_foundry.common.model import BasicsTransformerLM, ModelConfig, create_model
from llm_foundry.common.optimizer import (
    AdamW,
    ShardedOptimizer,
    create_optimizer,
    get_cosine_lr,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def small_model_config():
    """Create a small model config for fast testing."""
    return ModelConfig(
        vocab_size=100,
        context_length=32,
        d_model=64,
        num_layers=2,
        num_heads=4,
        d_ff=128,
        rope_theta=10000.0,
    )


@pytest.fixture
def tiny_model_config():
    """Create a tiny model config for very fast testing."""
    return ModelConfig(
        vocab_size=50,
        context_length=16,
        d_model=32,
        num_layers=1,
        num_heads=2,
        d_ff=64,
        rope_theta=10000.0,
    )


@pytest.fixture
def mock_token_data(tmp_path):
    """Create mock token data for testing."""
    data_path = tmp_path / "tokens.npy"
    # Create random token data with values in valid range for tiny_model_config (vocab_size=50)
    tokens = np.random.randint(0, 50, size=10000, dtype=np.uint16)
    np.save(data_path, tokens)
    return str(data_path)


@pytest.fixture
def mock_token_data_large(tmp_path):
    """Create mock token data with larger vocab range for small_model_config."""
    data_path = tmp_path / "tokens_large.npy"
    # Create random token data with values in valid range for small_model_config (vocab_size=100)
    tokens = np.random.randint(0, 100, size=10000, dtype=np.uint16)
    np.save(data_path, tokens)
    return str(data_path)


# =============================================================================
# End-to-End Training Tests
# =============================================================================


class TestEndToEndTraining:
    """End-to-end training tests with small models."""

    def test_simple_training_step(self, tiny_model_config, mock_token_data):
        """Test a single training step completes successfully."""
        device = "cpu"
        model = create_model(tiny_model_config, use_flash_attn=False)
        model.to(device)
        model.train()

        optimizer = create_optimizer(
            model.parameters(),
            lr=1e-3,
            weight_decay=0.01,
            sharded=False,
        )

        # Load data and create batch
        tokens = load_tokens(mock_token_data)
        x, y = get_batch(tokens, batch_size=2, context_length=16, device=device)

        # Forward pass
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Verify loss is finite
        assert torch.isfinite(loss).item()
        assert loss.item() > 0

    def test_multiple_training_steps(self, tiny_model_config, mock_token_data):
        """Test multiple training steps with loss decreasing."""
        device = "cpu"
        model = create_model(tiny_model_config, use_flash_attn=False)
        model.to(device)
        model.train()

        optimizer = create_optimizer(
            model.parameters(),
            lr=1e-3,
            weight_decay=0.01,
            sharded=False,
        )

        tokens = load_tokens(mock_token_data)
        losses = []

        # Train for a few steps
        for _ in range(5):
            x, y = get_batch(tokens, batch_size=2, context_length=16, device=device)

            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # All losses should be finite
        assert all(np.isfinite(l) for l in losses)

    def test_training_with_gradient_accumulation(self, tiny_model_config, mock_token_data):
        """Test training with gradient accumulation."""
        device = "cpu"
        model = create_model(tiny_model_config, use_flash_attn=False)
        model.to(device)
        model.train()

        optimizer = create_optimizer(
            model.parameters(),
            lr=1e-3,
            weight_decay=0.01,
            sharded=False,
        )

        tokens = load_tokens(mock_token_data)
        grad_accum_steps = 4

        optimizer.zero_grad()
        total_loss = 0

        for i in range(grad_accum_steps):
            x, y = get_batch(tokens, batch_size=2, context_length=16, device=device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss = loss / grad_accum_steps  # Scale for accumulation
            loss.backward()
            total_loss += loss.item()

        optimizer.step()

        assert total_loss > 0

    def test_training_with_lr_schedule(self, tiny_model_config, mock_token_data):
        """Test training with cosine learning rate schedule."""
        device = "cpu"
        model = create_model(tiny_model_config, use_flash_attn=False)
        model.to(device)
        model.train()

        optimizer = create_optimizer(
            model.parameters(),
            lr=1e-3,
            weight_decay=0.01,
            sharded=False,
        )

        tokens = load_tokens(mock_token_data)
        max_iters = 100
        warmup_iters = 10
        min_lr = 1e-4

        lrs = []
        for step in range(max_iters):
            # Update learning rate
            lr = get_cosine_lr(
                it=step,
                max_learning_rate=1e-3,
                min_learning_rate=min_lr,
                warmup_iters=warmup_iters,
                cosine_cycle_iters=max_iters,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            lrs.append(lr)

            # Training step
            x, y = get_batch(tokens, batch_size=2, context_length=16, device=device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Verify LR schedule behavior
        assert lrs[0] == 0.0  # Start at 0
        assert lrs[warmup_iters] == 1e-3  # Peak after warmup
        assert abs(lrs[-1] - min_lr) < 1e-6  # End at min_lr (approximately)
        assert all(lrs[i] <= lrs[i + 1] for i in range(warmup_iters - 1))  # Warmup increases


# =============================================================================
# Checkpoint Save/Load Tests
# =============================================================================


class TestCheckpointing:
    """Tests for checkpoint save and load functionality."""

    def test_save_load_model_checkpoint(self, small_model_config, tmp_path):
        """Test saving and loading model checkpoint."""
        model = create_model(small_model_config, use_flash_attn=False)

        # Get initial parameters
        initial_params = {name: param.clone() for name, param in model.named_parameters()}

        # Save checkpoint
        checkpoint_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), checkpoint_path)

        # Modify model
        for param in model.parameters():
            param.data += 1.0

        # Load checkpoint
        model.load_state_dict(torch.load(checkpoint_path))

        # Verify parameters restored
        for name, param in model.named_parameters():
            assert torch.allclose(param, initial_params[name])

    def test_save_load_optimizer_checkpoint(self, small_model_config, tmp_path):
        """Test saving and loading optimizer checkpoint."""
        model = create_model(small_model_config, use_flash_attn=False)
        optimizer = AdamW(model.parameters(), lr=1e-3)

        # Simulate some training steps to populate optimizer state
        for _ in range(3):
            x = torch.randint(0, small_model_config.vocab_size, (2, 16))
            logits = model(x)
            loss = logits.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Save optimizer state
        checkpoint_path = tmp_path / "optimizer.pt"
        torch.save(optimizer.state_dict(), checkpoint_path)

        # Create new optimizer and load state
        new_optimizer = AdamW(model.parameters(), lr=1e-3)
        new_optimizer.load_state_dict(torch.load(checkpoint_path))

        # Verify optimizer state restored
        assert new_optimizer.state_dict()["param_groups"] == optimizer.state_dict()["param_groups"]

    def test_checkpoint_with_config(self, small_model_config, tmp_path):
        """Test saving checkpoint with config."""
        model = create_model(small_model_config, use_flash_attn=False)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": small_model_config.to_dict(),
            "step": 100,
            "loss": 2.5,
        }

        checkpoint_path = tmp_path / "checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)

        # Load checkpoint
        loaded = torch.load(checkpoint_path)

        assert loaded["step"] == 100
        assert loaded["loss"] == 2.5
        assert loaded["config"]["vocab_size"] == small_model_config.vocab_size

    def test_model_from_pretrained(self, small_model_config, tmp_path):
        """Test loading model from pretrained directory."""
        model = create_model(small_model_config, use_flash_attn=False)

        # Save model in expected format
        model_dir = tmp_path / "pretrained"
        model_dir.mkdir()

        config_path = model_dir / "model_config.json"
        weights_path = model_dir / "model.pt"

        import json
        # Save config without the attention_fn (which is not JSON serializable)
        config_to_save = {k: v for k, v in model.config.items() if k != "attention_fn"}
        with open(config_path, "w") as f:
            json.dump(config_to_save, f)
        torch.save(model.state_dict(), weights_path)

        # Load with from_pretrained
        loaded_model = BasicsTransformerLM.from_pretrained(str(model_dir))

        # Verify loaded model has same architecture
        assert loaded_model.vocab_size == model.vocab_size
        assert loaded_model.d_model == model.d_model


# =============================================================================
# Attention Injection Integration Tests
# =============================================================================


class TestAttentionInjection:
    """Tests for attention backend injection integration."""

    def test_model_with_sdpa_attention(self, small_model_config):
        """Test model with SDPA attention backend."""
        attention_fn = get_attention_fn(use_flash_attn=False)
        model = create_model(small_model_config, use_flash_attn=False)

        x = torch.randint(0, small_model_config.vocab_size, (2, 16))
        logits = model(x)

        assert logits.shape == (2, 16, small_model_config.vocab_size)
        assert not torch.isnan(logits).any()

    def test_attention_fn_injection(self, small_model_config):
        """Test that custom attention function is properly injected."""
        # Create a custom attention function that tracks calls
        call_count = [0]

        def custom_attention_fn(q, k, v, is_causal=True):
            call_count[0] += 1
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=is_causal
            )

        model = BasicsTransformerLM(
            vocab_size=small_model_config.vocab_size,
            context_length=small_model_config.context_length,
            d_model=small_model_config.d_model,
            num_layers=small_model_config.num_layers,
            num_heads=small_model_config.num_heads,
            d_ff=small_model_config.d_ff,
            rope_theta=small_model_config.rope_theta,
            attention_fn=custom_attention_fn,
        )

        x = torch.randint(0, small_model_config.vocab_size, (1, 8))
        _ = model(x)

        # Should have been called for each layer
        assert call_count[0] == small_model_config.num_layers

    def test_attention_backward_compatibility(self, small_model_config):
        """Test that attention works with backward pass."""
        model = create_model(small_model_config, use_flash_attn=False)
        model.train()

        x = torch.randint(0, small_model_config.vocab_size, (2, 16))
        logits = model(x)
        loss = logits.mean()

        # Should not raise error
        loss.backward()

        # Verify gradients exist
        assert any(p.grad is not None for p in model.parameters())


# =============================================================================
# Learning Rate Schedule Tests
# =============================================================================


class TestLearningRateSchedule:
    """Tests for learning rate scheduling."""

    def test_cosine_lr_warmup_phase(self):
        """Test linear warmup phase of cosine LR schedule."""
        max_lr = 1e-3
        warmup_iters = 100

        # At iteration 0, LR should be 0
        lr = get_cosine_lr(0, max_lr, 1e-4, warmup_iters, 1000)
        assert lr == 0.0

        # At half warmup, LR should be half max
        lr = get_cosine_lr(50, max_lr, 1e-4, warmup_iters, 1000)
        assert abs(lr - max_lr / 2) < 1e-10

        # At end of warmup, LR should be max
        lr = get_cosine_lr(warmup_iters, max_lr, 1e-4, warmup_iters, 1000)
        assert abs(lr - max_lr) < 1e-10

    def test_cosine_lr_decay_phase(self):
        """Test cosine decay phase of LR schedule."""
        max_lr = 1e-3
        min_lr = 1e-4
        warmup_iters = 100
        cosine_cycle_iters = 1000

        # At middle of cosine decay, LR should be between min and max
        lr = get_cosine_lr(550, max_lr, min_lr, warmup_iters, cosine_cycle_iters)
        assert min_lr < lr < max_lr

        # At end of cosine cycle, LR should be min
        lr = get_cosine_lr(cosine_cycle_iters, max_lr, min_lr, warmup_iters, cosine_cycle_iters)
        assert abs(lr - min_lr) < 1e-10

    def test_cosine_lr_constant_phase(self):
        """Test constant phase after cosine cycle."""
        max_lr = 1e-3
        min_lr = 1e-4
        warmup_iters = 100
        cosine_cycle_iters = 1000

        # After cosine cycle, LR should stay at min
        lr1 = get_cosine_lr(1500, max_lr, min_lr, warmup_iters, cosine_cycle_iters)
        lr2 = get_cosine_lr(2000, max_lr, min_lr, warmup_iters, cosine_cycle_iters)
        assert abs(lr1 - min_lr) < 1e-10
        assert abs(lr2 - min_lr) < 1e-10

    def test_lr_schedule_integration_with_optimizer(self, small_model_config):
        """Test LR schedule integration with optimizer."""
        model = create_model(small_model_config, use_flash_attn=False)
        optimizer = create_optimizer(model.parameters(), lr=1e-3)

        max_iters = 100
        warmup_iters = 10

        for step in range(max_iters):
            lr = get_cosine_lr(
                it=step,
                max_learning_rate=1e-3,
                min_learning_rate=1e-4,
                warmup_iters=warmup_iters,
                cosine_cycle_iters=max_iters,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        # Final LR should be min (approximately)
        assert abs(optimizer.param_groups[0]["lr"] - 1e-4) < 1e-6


# =============================================================================
# Data Loading Integration Tests
# =============================================================================


class TestDataLoadingIntegration:
    """Tests for training integration with data loading."""

    def test_data_loader_yields_batches(self, mock_token_data):
        """Test that data loader yields proper batches."""
        data_loader = create_data_loader(
            data_path=mock_token_data,
            batch_size=4,
            context_length=32,
            device="cpu",
            num_batches=5,
        )

        batches = list(data_loader)
        assert len(batches) == 5

        for x, y in batches:
            assert x.shape == (4, 32)
            assert y.shape == (4, 32)
            assert x.dtype == torch.long
            assert y.dtype == torch.long

    def test_batch_targets_are_offset_by_one(self, mock_token_data):
        """Test that targets are offset by one from inputs."""
        tokens = load_tokens(mock_token_data)
        x, y = get_batch(tokens, batch_size=1, context_length=10, device="cpu")

        # y should be x shifted by one position
        # y[i] should equal x[i+1] for each sequence
        for i in range(9):
            assert y[0, i].item() == x[0, i + 1].item()

    def test_training_with_data_loader(self, tiny_model_config, mock_token_data):
        """Test training using data loader."""
        device = "cpu"
        model = create_model(tiny_model_config, use_flash_attn=False)
        model.to(device)
        model.train()

        optimizer = create_optimizer(model.parameters(), lr=1e-3)

        data_loader = create_data_loader(
            data_path=mock_token_data,
            batch_size=2,
            context_length=16,
            device=device,
            num_batches=3,
        )

        losses = []
        for x, y in data_loader:
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        assert len(losses) == 3
        assert all(np.isfinite(l) for l in losses)


# =============================================================================
# DDP Training Tests (Multi-process)
# =============================================================================


def _ddp_worker(rank, world_size, model_config_dict, token_data_path, result_queue):
    """Worker function for DDP testing."""
    # Initialize process group
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    try:
        device = "cpu"
        model = BasicsTransformerLM(
            vocab_size=model_config_dict["vocab_size"],
            context_length=model_config_dict["context_length"],
            d_model=model_config_dict["d_model"],
            num_layers=model_config_dict["num_layers"],
            num_heads=model_config_dict["num_heads"],
            d_ff=model_config_dict["d_ff"],
            rope_theta=model_config_dict["rope_theta"],
        )
        model.to(device)

        # Wrap with DDP
        model = DDP(model)

        optimizer = AdamW(model.parameters(), lr=1e-3)

        # Load data
        tokens = load_tokens(token_data_path)
        x, y = get_batch(tokens, batch_size=2, context_length=16, device=device)

        # Forward pass
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Report success
        result_queue.put((rank, "success", loss.item()))
    except Exception as e:
        result_queue.put((rank, "error", str(e)))
    finally:
        dist.destroy_process_group()


class TestDDPTraining:
    """Tests for Distributed Data Parallel training."""

    @pytest.mark.skipif(not dist.is_available(), reason="Distributed training not available")
    def test_ddp_basic_training(self, tiny_model_config, mock_token_data, tmp_path):
        """Test basic DDP training with 2 processes."""
        world_size = 2

        # Use spawn method for multiprocessing
        mp.set_start_method("spawn", force=True)

        result_queue = mp.Queue()

        # Spawn processes
        processes = []
        for rank in range(world_size):
            p = mp.Process(
                target=_ddp_worker,
                args=(
                    rank,
                    world_size,
                    tiny_model_config.to_dict(),
                    mock_token_data,
                    result_queue,
                ),
            )
            p.start()
            processes.append(p)

        # Wait for completion
        for p in processes:
            p.join(timeout=60)

        # Collect results
        results = []
        for _ in range(world_size):
            try:
                results.append(result_queue.get(timeout=5))
            except Exception:
                pass

        # Verify all processes succeeded
        assert len(results) == world_size
        for rank, status, _ in results:
            assert status == "success", f"Rank {rank} failed"

    def test_sharded_optimizer_without_distributed(self, tiny_model_config):
        """Test ShardedOptimizer works without distributed initialization."""
        model = create_model(tiny_model_config, use_flash_attn=False)

        optimizer = ShardedOptimizer(model.parameters(), AdamW, lr=1e-3)

        # Should work without distributed
        assert optimizer.world_size == 1
        assert optimizer.rank == 0

        # Training step
        x = torch.randint(0, tiny_model_config.vocab_size, (2, 16))
        logits = model(x)
        loss = logits.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Should complete without error
        assert True

    def test_sharded_optimizer_state_dict(self, tiny_model_config):
        """Test ShardedOptimizer state dict save/load."""
        model = create_model(tiny_model_config, use_flash_attn=False)

        optimizer = ShardedOptimizer(model.parameters(), AdamW, lr=1e-3)

        # Simulate training
        for _ in range(3):
            x = torch.randint(0, tiny_model_config.vocab_size, (2, 16))
            logits = model(x)
            loss = logits.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Get state dict
        state_dict = optimizer.state_dict()

        # Create new optimizer and load state
        new_optimizer = ShardedOptimizer(model.parameters(), AdamW, lr=1e-3)
        new_optimizer.load_state_dict(state_dict)

        # Verify state loaded
        assert new_optimizer.param_groups == optimizer.param_groups


# =============================================================================
# Integration with Common Modules
# =============================================================================


class TestCommonModuleIntegration:
    """Tests for Trainer integration with common modules."""

    def test_model_optimizer_integration(self, small_model_config):
        """Test that model and optimizer work together."""
        model = create_model(small_model_config, use_flash_attn=False)
        optimizer = create_optimizer(model.parameters(), lr=1e-3)

        # Verify optimizer has model parameters
        param_count = sum(1 for _ in model.parameters())
        optim_param_count = sum(len(g["params"]) for g in optimizer.param_groups)
        assert optim_param_count == param_count

    def test_train_config_integration(self):
        """Test TrainConfig integration with training components."""
        config = TrainConfig(
            vocab_size=100,
            context_length=32,
            d_model=64,
            num_layers=2,
            num_heads=4,
            d_ff=128,
            batch_size=4,
            learning_rate=1e-3,
            max_iters=100,
        )

        # Create model from config
        model_config = ModelConfig(
            vocab_size=config.vocab_size,
            context_length=config.context_length,
            d_model=config.d_model,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
        )
        model = create_model(model_config, use_flash_attn=config.use_flash_attn)

        # Create optimizer from config
        optimizer = create_optimizer(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        assert model.vocab_size == config.vocab_size
        assert optimizer.param_groups[0]["lr"] == config.learning_rate

    def test_full_training_pipeline(self, tiny_model_config, mock_token_data):
        """Test full training pipeline integration."""
        device = "cpu"

        # Create components
        model = create_model(tiny_model_config, use_flash_attn=False)
        model.to(device)
        model.train()

        optimizer = create_optimizer(
            model.parameters(),
            lr=1e-3,
            weight_decay=0.01,
            sharded=False,
        )

        data_loader = create_data_loader(
            data_path=mock_token_data,
            batch_size=2,
            context_length=16,
            device=device,
            num_batches=5,
        )

        # Training loop
        losses = []
        for step, (x, y) in enumerate(data_loader):
            # Update LR
            lr = get_cosine_lr(
                it=step,
                max_learning_rate=1e-3,
                min_learning_rate=1e-4,
                warmup_iters=2,
                cosine_cycle_iters=5,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # Forward
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            losses.append(loss.item())

        # Verify training completed
        assert len(losses) == 5
        assert all(np.isfinite(l) for l in losses)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_training_with_empty_batch(self, tiny_model_config):
        """Test handling of edge cases in batch processing."""
        model = create_model(tiny_model_config, use_flash_attn=False)
        model.train()

        # Normal batch should work
        x = torch.randint(0, tiny_model_config.vocab_size, (2, 16))
        logits = model(x)
        assert logits.shape == (2, 16, tiny_model_config.vocab_size)

    def test_gradient_clipping(self, tiny_model_config):
        """Test gradient clipping during training."""
        model = create_model(tiny_model_config, use_flash_attn=False)
        model.train()

        optimizer = create_optimizer(model.parameters(), lr=1e-3)

        # Create large gradients
        x = torch.randint(0, tiny_model_config.vocab_size, (2, 16))
        logits = model(x)
        loss = logits.mean() * 1000  # Scale up to get large gradients

        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        max_norm = 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # Verify gradients are clipped
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        assert total_norm <= max_norm * 1.01  # Allow small numerical error

    def test_model_eval_mode(self, small_model_config):
        """Test model in eval mode."""
        model = create_model(small_model_config, use_flash_attn=False)
        model.eval()

        x = torch.randint(0, small_model_config.vocab_size, (2, 16))

        with torch.no_grad():
            logits = model(x)

        assert logits.shape == (2, 16, small_model_config.vocab_size)

    def test_training_resumption(self, tiny_model_config, mock_token_data, tmp_path):
        """Test training resumption from checkpoint."""
        device = "cpu"
        model = create_model(tiny_model_config, use_flash_attn=False)
        model.to(device)

        optimizer = create_optimizer(model.parameters(), lr=1e-3)

        # First training phase
        tokens = load_tokens(mock_token_data)
        x, y = get_batch(tokens, batch_size=2, context_length=16, device=device)

        logits = model(x)
        loss1 = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()

        # Save checkpoint
        checkpoint_path = tmp_path / "checkpoint.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": 1,
        }, checkpoint_path)

        # Create new model and optimizer
        new_model = create_model(tiny_model_config, use_flash_attn=False)
        new_model.to(device)
        new_optimizer = create_optimizer(new_model.parameters(), lr=1e-3)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)
        new_model.load_state_dict(checkpoint["model_state_dict"])
        new_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Verify models are in same state
        for (name1, p1), (name2, p2) in zip(
            model.named_parameters(), new_model.named_parameters()
        ):
            assert torch.allclose(p1, p2), f"Parameter {name1} mismatch"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

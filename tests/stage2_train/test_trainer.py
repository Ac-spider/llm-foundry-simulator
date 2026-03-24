"""
Tests for Stage 2: Trainer module.

These tests verify the training loop, DDP support, gradient accumulation,
checkpoint saving/loading, and learning rate scheduling.
"""

import json
import os
import tempfile

import numpy as np
import pytest
import torch

from llm_foundry.stage2_train.trainer import Trainer, DDPIndividualParameters, DDPBucketed


class TestTrainer:
    """Tests for Trainer class."""

    def test_trainer_init(self):
        """Test Trainer initialization."""
        cfg = {
            "model": {
                "vocab_size": 100,
                "context_length": 32,
                "d_model": 64,
                "n_layers": 2,
                "n_heads": 2,
                "d_ff": 128,
                "rope_theta": 10000.0,
                "use_flash_attn": False,
            },
            "training": {
                "data_path": "/tmp/dummy.npy",
                "batch_size": 2,
                "max_steps": 5,
                "lr": 3e-4,
            },
            "output": {
                "base_dir": "/tmp/test_output",
            },
        }

        trainer = Trainer(cfg)
        assert trainer.cfg == cfg
        assert trainer.run_dir.startswith("/tmp/test_output/")
        assert trainer.ckpt_dir == os.path.join(trainer.run_dir, "checkpoints")
        assert trainer.metrics_path == os.path.join(trainer.run_dir, "metrics.jsonl")

    def test_trainer_run_dir_hash(self):
        """Test that run_dir contains consistent hash."""
        cfg1 = {
            "model": {"vocab_size": 100, "context_length": 32, "d_model": 64,
                     "n_layers": 2, "n_heads": 2, "d_ff": 128, "rope_theta": 10000.0,
                     "use_flash_attn": False},
            "training": {"data_path": "/tmp/dummy.npy", "batch_size": 2,
                        "max_steps": 5, "lr": 3e-4},
            "output": {"base_dir": "/tmp/test"},
        }
        cfg2 = {
            "model": {"vocab_size": 100, "context_length": 32, "d_model": 64,
                     "n_layers": 2, "n_heads": 2, "d_ff": 128, "rope_theta": 10000.0,
                     "use_flash_attn": False},
            "training": {"data_path": "/tmp/dummy.npy", "batch_size": 2,
                        "max_steps": 5, "lr": 3e-4},
            "output": {"base_dir": "/tmp/test"},
        }
        cfg3 = {
            "model": {"vocab_size": 200, "context_length": 32, "d_model": 64,
                     "n_layers": 2, "n_heads": 2, "d_ff": 128, "rope_theta": 10000.0,
                     "use_flash_attn": False},
            "training": {"data_path": "/tmp/dummy.npy", "batch_size": 2,
                        "max_steps": 5, "lr": 3e-4},
            "output": {"base_dir": "/tmp/test"},
        }

        trainer1 = Trainer(cfg1)
        trainer2 = Trainer(cfg2)
        trainer3 = Trainer(cfg3)

        # Same config should produce same hash
        assert trainer1.run_dir == trainer2.run_dir
        # Different config should produce different hash
        assert trainer1.run_dir != trainer3.run_dir

    def test_trainer_single_gpu_training(self):
        """Test single-GPU training with minimal config (CPU runnable)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create random token data
            data_path = os.path.join(tmpdir, "tokens.npy")
            tokens = np.random.randint(0, 100, size=(10000,), dtype=np.uint16)
            np.save(data_path, tokens)

            cfg = {
                "model": {
                    "vocab_size": 100,
                    "context_length": 32,
                    "d_model": 64,
                    "n_layers": 2,
                    "n_heads": 2,
                    "d_ff": 128,
                    "rope_theta": 10000.0,
                    "use_flash_attn": False,
                },
                "training": {
                    "data_path": data_path,
                    "batch_size": 2,
                    "gradient_accumulation_steps": 1,
                    "max_steps": 5,
                    "lr": 3e-4,
                    "min_lr": 3e-5,
                    "warmup_steps": 1,
                    "weight_decay": 0.1,
                    "grad_clip": 1.0,
                    "save_interval": 5,
                    "log_interval": 1,
                    "device": "cpu",
                },
                "output": {
                    "base_dir": tmpdir,
                },
            }

            trainer = Trainer(cfg)
            trainer.train()

            # Verify run_dir exists
            assert os.path.isdir(trainer.run_dir)

            # Verify metrics.jsonl exists and has valid JSON lines
            assert os.path.isfile(trainer.metrics_path)
            with open(trainer.metrics_path) as f:
                lines = [l.strip() for l in f if l.strip()]
            assert len(lines) > 0
            for line in lines:
                record = json.loads(line)
                assert "step" in record
                assert "loss" in record
                assert "lr" in record
                assert "epoch" in record

            # Verify checkpoint exists
            ckpt_dir = trainer.ckpt_dir
            assert os.path.isdir(ckpt_dir)
            ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
            assert len(ckpt_files) > 0

    def test_trainer_gradient_accumulation(self):
        """Test training with gradient accumulation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "tokens.npy")
            tokens = np.random.randint(0, 100, size=(10000,), dtype=np.uint16)
            np.save(data_path, tokens)

            cfg = {
                "model": {
                    "vocab_size": 100,
                    "context_length": 32,
                    "d_model": 64,
                    "n_layers": 2,
                    "n_heads": 2,
                    "d_ff": 128,
                    "rope_theta": 10000.0,
                    "use_flash_attn": False,
                },
                "training": {
                    "data_path": data_path,
                    "batch_size": 2,
                    "gradient_accumulation_steps": 4,  # Test grad accum
                    "max_steps": 3,
                    "lr": 3e-4,
                    "min_lr": 3e-5,
                    "warmup_steps": 1,
                    "weight_decay": 0.1,
                    "grad_clip": 1.0,
                    "save_interval": 3,
                    "log_interval": 1,
                    "device": "cpu",
                },
                "output": {
                    "base_dir": tmpdir,
                },
            }

            trainer = Trainer(cfg)
            trainer.train()

            # Verify training completed
            assert os.path.isfile(trainer.metrics_path)

    def test_trainer_checkpoint_loadable(self):
        """Test that saved checkpoints can be loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "tokens.npy")
            tokens = np.random.randint(0, 100, size=(10000,), dtype=np.uint16)
            np.save(data_path, tokens)

            cfg = {
                "model": {
                    "vocab_size": 100,
                    "context_length": 32,
                    "d_model": 64,
                    "n_layers": 2,
                    "n_heads": 2,
                    "d_ff": 128,
                    "rope_theta": 10000.0,
                    "use_flash_attn": False,
                },
                "training": {
                    "data_path": data_path,
                    "batch_size": 2,
                    "gradient_accumulation_steps": 1,
                    "max_steps": 5,
                    "lr": 3e-4,
                    "min_lr": 3e-5,
                    "warmup_steps": 1,
                    "weight_decay": 0.1,
                    "grad_clip": 1.0,
                    "save_interval": 5,
                    "log_interval": 1,
                    "device": "cpu",
                },
                "output": {
                    "base_dir": tmpdir,
                },
            }

            trainer = Trainer(cfg)
            trainer.train()

            # Load checkpoint
            ckpt_dir = trainer.ckpt_dir
            ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
            assert len(ckpt_files) > 0

            ckpt_path = os.path.join(ckpt_dir, ckpt_files[0])
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

            # Verify checkpoint structure
            assert "config" in checkpoint
            assert "state_dict" in checkpoint

            # Verify config matches
            assert checkpoint["config"]["vocab_size"] == 100
            assert checkpoint["config"]["d_model"] == 64

            # Verify state_dict is loadable
            from llm_foundry.common.model import create_model, ModelConfig
            model_config = ModelConfig(
                vocab_size=100,
                context_length=32,
                d_model=64,
                num_layers=2,
                num_heads=2,
                d_ff=128,
                rope_theta=10000.0,
            )
            model = create_model(model_config, use_flash_attn=False)
            model.load_state_dict(checkpoint["state_dict"])

            # Verify model has parameters
            assert sum(p.numel() for p in model.parameters()) > 0


class TestTrainerLRScheduling:
    """Tests for learning rate scheduling in Trainer."""

    def test_lr_warmup_and_decay(self):
        """Test that LR warms up then decays."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "tokens.npy")
            tokens = np.random.randint(0, 100, size=(10000,), dtype=np.uint16)
            np.save(data_path, tokens)

            cfg = {
                "model": {
                    "vocab_size": 100,
                    "context_length": 32,
                    "d_model": 64,
                    "n_layers": 2,
                    "n_heads": 2,
                    "d_ff": 128,
                    "rope_theta": 10000.0,
                    "use_flash_attn": False,
                },
                "training": {
                    "data_path": data_path,
                    "batch_size": 2,
                    "max_steps": 10,
                    "lr": 1e-3,  # max LR
                    "min_lr": 1e-4,  # min LR
                    "warmup_steps": 3,
                    "save_interval": 10,
                    "log_interval": 1,
                    "device": "cpu",
                },
                "output": {
                    "base_dir": tmpdir,
                },
            }

            trainer = Trainer(cfg)
            trainer.train()

            # Parse metrics
            with open(trainer.metrics_path) as f:
                records = [json.loads(l) for l in f if l.strip()]

            # Check warmup: LR should increase from step 1 to 3
            lr_step1 = records[0]["lr"]
            lr_step3 = records[2]["lr"]
            assert lr_step3 > lr_step1, "LR should increase during warmup"

            # Check decay: LR should decrease after warmup
            lr_step10 = records[-1]["lr"]
            assert lr_step10 < lr_step3, "LR should decrease after warmup"
            assert lr_step10 >= 1e-4, "LR should not go below min_lr"


class TestTrainerMetrics:
    """Tests for metrics logging."""

    def test_metrics_format(self):
        """Test that metrics are logged in correct format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "tokens.npy")
            tokens = np.random.randint(0, 100, size=(10000,), dtype=np.uint16)
            np.save(data_path, tokens)

            cfg = {
                "model": {
                    "vocab_size": 100,
                    "context_length": 32,
                    "d_model": 64,
                    "n_layers": 2,
                    "n_heads": 2,
                    "d_ff": 128,
                    "rope_theta": 10000.0,
                    "use_flash_attn": False,
                },
                "training": {
                    "data_path": data_path,
                    "batch_size": 2,
                    "max_steps": 3,
                    "lr": 3e-4,
                    "save_interval": 3,
                    "log_interval": 1,
                    "device": "cpu",
                },
                "output": {
                    "base_dir": tmpdir,
                },
            }

            trainer = Trainer(cfg)
            trainer.train()

            with open(trainer.metrics_path) as f:
                lines = f.readlines()

            for line in lines:
                record = json.loads(line)
                # Check all required fields
                assert isinstance(record["step"], int)
                assert isinstance(record["loss"], float)
                assert isinstance(record["lr"], float)
                assert isinstance(record["epoch"], float)

                # Check value ranges
                assert record["step"] > 0
                assert record["loss"] > 0
                assert record["lr"] > 0
                assert record["epoch"] >= 0


class TestDDPWrappers:
    """Tests for DDP wrapper classes."""

    def test_ddp_individual_parameters_init(self):
        """Test DDPIndividualParameters initialization."""
        from llm_foundry.common.model import create_model, ModelConfig

        config = ModelConfig(
            vocab_size=100,
            context_length=32,
            d_model=64,
            num_layers=2,
            num_heads=2,
            d_ff=128,
            rope_theta=10000.0,
        )
        model = create_model(config, use_flash_attn=False)

        # In non-distributed environment, should work but not sync
        ddp_model = DDPIndividualParameters(model)
        assert ddp_model.is_initialized is False  # No distributed env
        assert ddp_model.world_size == 1

    def test_ddp_bucketed_init(self):
        """Test DDPBucketed initialization."""
        from llm_foundry.common.model import create_model, ModelConfig

        config = ModelConfig(
            vocab_size=100,
            context_length=32,
            d_model=64,
            num_layers=2,
            num_heads=2,
            d_ff=128,
            rope_theta=10000.0,
        )
        model = create_model(config, use_flash_attn=False)

        # In non-distributed environment, should work but not sync
        ddp_model = DDPBucketed(model, bucket_size_mb=1.0)
        assert ddp_model.is_initialized is False  # No distributed env
        assert ddp_model.world_size == 1

    def test_ddp_forward_pass(self):
        """Test DDP wrapper forward pass."""
        from llm_foundry.common.model import create_model, ModelConfig

        config = ModelConfig(
            vocab_size=100,
            context_length=32,
            d_model=64,
            num_layers=2,
            num_heads=2,
            d_ff=128,
            rope_theta=10000.0,
        )
        model = create_model(config, use_flash_attn=False)
        ddp_model = DDPIndividualParameters(model)

        # Test forward pass
        x = torch.randint(0, 100, (2, 32))
        logits = ddp_model(x)
        assert logits.shape == (2, 32, 100)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

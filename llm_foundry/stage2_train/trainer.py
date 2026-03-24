"""
Stage 2: Trainer module for LLM training with DDP and ZeRO-1 support.

Adapted from CS336 Assignments 1, 2, and 4.
- Training loop: reference_resource/Assignment1-basics/Transformer/train.py
- DDP implementation: reference_resource/Assignment2-system/cs336_systems/ddp_training.py
"""

from __future__ import annotations

import dataclasses
import json
import hashlib
import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from llm_foundry.common.data import get_batch, load_tokens
from llm_foundry.common.model import ModelConfig, BasicsTransformerLM
from llm_foundry.common.optimizer import AdamW, ShardedOptimizer, get_cosine_lr


class Trainer:
    """LLM Trainer supporting single-GPU and DDP+ZeRO-1 multi-GPU training.

    Args:
        cfg: Complete configuration dictionary with 'model', 'training', 'output' keys.

    Attributes:
        run_dir: Output directory for this training run, format: {base_dir}/{run_hash}
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model_cfg_dict = cfg["model"]
        self.train_cfg = cfg["training"]
        self.out_cfg = cfg["output"]

        # Generate run_hash using SHA256 of sorted config JSON
        cfg_json = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
        run_hash = hashlib.sha256(cfg_json.encode("utf-8")).hexdigest()[:8]
        self.run_dir = os.path.join(self.out_cfg["base_dir"], run_hash)
        self.ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        self.metrics_path = os.path.join(self.run_dir, "metrics.jsonl")

    def train(self) -> None:
        """Main training loop.

        1. Initialize device, model, optimizer, data
        2. Detect single/multi-GPU, enable DDP + ZeRO-1 if needed
        3. Training step: forward -> loss -> backward -> clip -> step -> log -> save
        4. Save final checkpoint at end
        """
        # --- Device initialization ---
        device = self.train_cfg.get("device", "cpu")
        if device == "auto":
            device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )

        # --- DDP detection ---
        # Check if running under torchrun (RANK env var set automatically)
        is_ddp = dist.is_available() and "RANK" in os.environ

        if is_ddp:
            # Auto fallback: prefer NCCL, fallback to gloo
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            try:
                dist.init_process_group(backend=backend)
            except RuntimeError:
                print(f"[WARN] DDP backend '{backend}' init failed, trying 'gloo'")
                dist.init_process_group(backend="gloo")

            rank = int(os.environ["RANK"])
            local_rank = int(os.environ["LOCAL_RANK"])
            world_size = dist.get_world_size()
            device = f"cuda:{local_rank}"
            torch.cuda.set_device(device)
            is_master = rank == 0
        else:
            rank = 0
            world_size = 1
            is_master = True

        # --- Output directory setup (master only) ---
        if is_master:
            os.makedirs(self.run_dir, exist_ok=True)
            os.makedirs(self.ckpt_dir, exist_ok=True)
            # Clear old metrics.jsonl to avoid accumulation on re-runs
            if os.path.exists(self.metrics_path):
                os.remove(self.metrics_path)

        # --- Data loading ---
        data_path = self.train_cfg["data_path"]
        tokens = load_tokens(data_path)  # np.ndarray, memory-mapped

        # --- Model initialization ---
        model_config = ModelConfig(
            vocab_size=self.model_cfg_dict["vocab_size"],
            context_length=self.model_cfg_dict["context_length"],
            d_model=self.model_cfg_dict["d_model"],
            num_layers=self.model_cfg_dict["n_layers"],
            num_heads=self.model_cfg_dict["n_heads"],
            d_ff=self.model_cfg_dict["d_ff"],
            rope_theta=self.model_cfg_dict.get("rope_theta", 10000.0),
        )

        # Create model with attention backend
        from llm_foundry.common.model import create_model
        use_flash_attn = self.model_cfg_dict.get("use_flash_attn", False)
        model = create_model(model_config, use_flash_attn=use_flash_attn)
        model = model.to(device)

        # --- DDP wrapper (multi-GPU only) ---
        if is_ddp and world_size > 1:
            model = DDPIndividualParameters(model)

        # --- Optimizer initialization ---
        # Differential weight decay: 2D+ params (matrices) decay, 1D params (norm/bias) don't
        param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
        params_to_decay = [p for _, p in param_dict.items() if p.dim() >= 2]
        params_to_not_decay = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": params_to_decay, "weight_decay": self.train_cfg.get("weight_decay", 0.1)},
            {"params": params_to_not_decay, "weight_decay": 0.0},
        ]

        if is_ddp and world_size > 1:
            # ZeRO-1: use ShardedOptimizer to wrap AdamW, sharding optimizer states
            optimizer = ShardedOptimizer(
                optim_groups,
                AdamW,
                lr=self.train_cfg["lr"],
                betas=(0.9, 0.999),
                eps=1e-8,
            )
        else:
            # Single-GPU: use AdamW directly
            optimizer = AdamW(
                optim_groups,
                lr=self.train_cfg["lr"],
                betas=(0.9, 0.999),
                eps=1e-8,
            )

        # --- Training hyperparameters ---
        max_steps = self.train_cfg["max_steps"]
        batch_size = self.train_cfg["batch_size"]
        context_length = model_config.context_length
        grad_accum_steps = self.train_cfg.get("gradient_accumulation_steps", 1)
        # DDP multi-GPU: disable gradient accumulation
        # DDPIndividualParameters triggers AllReduce on each backward(),
        # so grad accumulation would average each micro-step separately
        if is_ddp and world_size > 1 and grad_accum_steps > 1:
            print("[WARN] DDP mode: gradient_accumulation_steps forced to 1")
            grad_accum_steps = 1
        lr_max = self.train_cfg["lr"]
        lr_min = self.train_cfg.get("min_lr", lr_max * 0.1)
        warmup_steps = self.train_cfg.get("warmup_steps", max_steps // 20)
        grad_clip = self.train_cfg.get("grad_clip", 1.0)
        save_interval = self.train_cfg.get("save_interval", 500)
        log_interval = self.train_cfg.get("log_interval", 10)

        model.train()

        # --- Main training loop ---
        for step in range(1, max_steps + 1):
            # Cosine annealing LR schedule with linear warmup
            lr = get_cosine_lr(
                step - 1,  # 0-indexed for scheduler
                max_learning_rate=lr_max,
                min_learning_rate=lr_min,
                warmup_iters=warmup_steps,
                cosine_cycle_iters=max_steps,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            total_loss = 0.0
            for micro_step in range(grad_accum_steps):
                x, y = get_batch(
                    tokens,
                    batch_size=batch_size,
                    context_length=context_length,
                    device=device,
                )
                logits = model(x)
                # Cross-entropy loss, divided by grad_accum_steps for averaging
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                ) / grad_accum_steps
                loss.backward()
                total_loss += loss.item()

            # DDP multi-GPU: wait for all async AllReduce to complete
            if is_ddp and world_size > 1 and hasattr(model, "finish_gradient_synchronization"):
                model.finish_gradient_synchronization()

            # Gradient clipping: prevent gradient explosion
            if grad_clip > 0:
                raw_model = model.module if hasattr(model, "module") else model
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # DDP multi-GPU (DDPBucketed): reset bucket ready counts
            if is_ddp and world_size > 1 and hasattr(model, "reset_buckets"):
                model.reset_buckets()

            # Calculate epoch (token-based estimate)
            total_tokens_seen = step * batch_size * context_length * grad_accum_steps
            epoch = total_tokens_seen / max(len(tokens), 1)

            # --- Logging (every log_interval steps, master only) ---
            if is_master and step % log_interval == 0:
                actual_loss = total_loss
                self._log_metrics(step, actual_loss, lr, epoch)
                print(f"[train] step={step}, loss={actual_loss:.4f}, lr={lr:.6f}, epoch={epoch:.2f}")

            # --- Checkpoint saving (every save_interval steps, master only) ---
            if is_master and step % save_interval == 0:
                raw_model = model.module if hasattr(model, "module") else model
                self._save_checkpoint(raw_model, model_config, step)

        # --- Training end: save final checkpoint (if last step didn't trigger save) ---
        if is_master and max_steps % save_interval != 0:
            raw_model = model.module if hasattr(model, "module") else model
            self._save_checkpoint(raw_model, model_config, max_steps)

        # --- DDP cleanup ---
        if is_ddp:
            dist.destroy_process_group()

    def _save_checkpoint(self, model: BasicsTransformerLM, model_config: ModelConfig, step: int) -> None:
        """Save Transformer checkpoint (compatible with from_pretrained() format).

        Args:
            model: Transformer instance (non-DDP wrapped)
            model_config: ModelConfig dataclass with model configuration
            step: Current training step for filename

        File format: {"config": dict, "state_dict": OrderedDict}
        """
        ckpt_path = os.path.join(self.ckpt_dir, f"step_{step:06d}.pt")

        torch.save(
            {
                "config": dataclasses.asdict(model_config),
                "state_dict": model.state_dict(),
            },
            ckpt_path,
        )
        print(f"[train] Saved checkpoint: {ckpt_path}")

    def _log_metrics(self, step: int, loss: float, lr: float, epoch: float) -> None:
        """Append a metrics line to metrics.jsonl.

        Args:
            step: Current training step (1-indexed)
            loss: Current step loss value
            lr: Current step learning rate
            epoch: Current epoch number (token-based estimate)

        Output format (one JSON object per line):
            {"step": 1, "loss": 4.32, "lr": 0.0003, "epoch": 0.0}
        """
        record = {
            "step": step,
            "loss": round(float(loss), 6),
            "lr": float(lr),
            "epoch": round(float(epoch), 4),
        }
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


class DDPIndividualParameters(nn.Module):
    """DDP wrapper with per-parameter async AllReduce.

    Each parameter triggers async AllReduce immediately after gradient computation,
    enabling communication-computation overlap.

    Adapted from CS336 Assignment 2.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.is_initialized = dist.is_available() and dist.is_initialized()
        self.communication_handles = []
        self.world_size = dist.get_world_size() if self.is_initialized else 1

        if self.is_initialized:
            # Broadcast from rank 0 to ensure all ranks start with same weights
            for param in self.module.parameters():
                dist.broadcast(param.data, src=0)

                if param.requires_grad:
                    param.register_post_accumulate_grad_hook(self._make_hook(param))

    def _make_hook(self, param):
        """Factory: create gradient sync hook for each parameter."""
        def hook(param):
            if not self.is_initialized:
                return

            # Divide by world_size first, then SUM AllReduce = MEAN
            param.grad.data.div_(self.world_size)

            # Async AllReduce (non-blocking, allows overlap with computation)
            handle = dist.all_reduce(
                param.grad.data, op=dist.ReduceOp.SUM, async_op=True
            )
            self.communication_handles.append(handle)

        return hook

    def forward(self, *args, **kwargs):
        """Forward pass: delegate to wrapped module."""
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        """Wait for all async AllReduce operations to complete.

        Must be called before optimizer.step() to ensure gradients are synchronized.
        """
        for handle in self.communication_handles:
            handle.wait()
        self.communication_handles.clear()


class DDPBucketed(nn.Module):
    """DDP wrapper with bucketed AllReduce for reduced kernel launch overhead.

    Groups parameters into buckets and performs one AllReduce per bucket.

    Adapted from CS336 Assignment 2.
    """

    def __init__(self, model: nn.Module, bucket_size_mb: float = 25.0):
        super().__init__()
        self.model = model
        self.bucket_size_bytes = bucket_size_mb * 1024 * 1024
        self.is_initialized = dist.is_available() and dist.is_initialized()
        self.world_size = dist.get_world_size() if self.is_initialized else 1

        self.handles = []
        self.buckets = []
        self.ready_buckets = []
        self.total_buckets = []

        if self.is_initialized:
            for p in self.model.parameters():
                dist.broadcast(p.data, src=0)
            self._build_buckets()

    def _build_buckets(self):
        """Greedy bucketing: assign parameters to buckets by size."""
        current_bucket = []
        current_size = 0

        params = [p for p in self.model.parameters() if p.requires_grad]

        for param in params:
            param_size = param.numel() * param.element_size()

            if current_size + param_size > self.bucket_size_bytes and len(current_bucket) > 0:
                self.buckets.append(current_bucket)
                current_bucket = []
                current_size = 0

            current_bucket.append(param)
            current_size += param_size

        if len(current_bucket) > 0:
            self.buckets.append(current_bucket)

        self.ready_buckets = [0] * len(self.buckets)
        self.total_buckets = [len(b) for b in self.buckets]

        for bucket_id, bucket in enumerate(self.buckets):
            for param in bucket:
                param.register_post_accumulate_grad_hook(
                    self._make_hook(param, bucket_id)
                )

    def _make_hook(self, param, bucket_id):
        """Factory: create bucket-aware gradient hook."""
        def hook(param):
            self.ready_buckets[bucket_id] += 1

            if self.ready_buckets[bucket_id] == self.total_buckets[bucket_id]:
                grads = [p.grad for p in self.buckets[bucket_id]]
                flat_grad = torch._utils._flatten_dense_tensors(grads)
                flat_grad.div_(self.world_size)

                handle = dist.all_reduce(
                    flat_grad, op=dist.ReduceOp.SUM, async_op=True
                )
                self.handles.append((handle, bucket_id, flat_grad))

        return hook

    def forward(self, *args, **kwargs):
        """Forward pass: delegate to wrapped module."""
        return self.model(*args, **kwargs)

    def finish_gradient_synchronization(self):
        """Wait for all bucket AllReduce to complete and unflatten gradients."""
        for handle, bucket_id, flat_grad in self.handles:
            handle.wait()

            grads = [p.grad for p in self.buckets[bucket_id]]
            unflat_grads = torch._utils._unflatten_dense_tensors(flat_grad, grads)

            for orig_grad, new_grad in zip(grads, unflat_grads):
                orig_grad.copy_(new_grad)

        self.handles.clear()

    def reset_buckets(self):
        """Reset bucket ready counts for next iteration."""
        for i in range(len(self.ready_buckets)):
            self.ready_buckets[i] = 0

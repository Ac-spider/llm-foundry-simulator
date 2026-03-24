"""
Optimizer utilities: AdamW, cosine LR scheduler, and ZeRO-1 ShardedOptimizer.

Adapted from CS336 Assignments 1, 2, and 4.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Type

import torch
import torch.distributed as dist


def get_cosine_lr(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    Cosine learning rate scheduler with linear warmup.

    Three-phase schedule:
    1. Linear warmup (0 <= it < warmup_iters): lr increases from 0 to max_lr
    2. Cosine decay (warmup_iters <= it <= cosine_cycle_iters): lr decays from max_lr to min_lr
    3. Constant (it > cosine_cycle_iters): lr stays at min_lr

    Args:
        it: Current iteration
        max_learning_rate: Peak learning rate after warmup
        min_learning_rate: Minimum learning rate at end of cosine cycle
        warmup_iters: Number of warmup iterations
        cosine_cycle_iters: Total iterations for cosine cycle (including warmup)

    Returns:
        Learning rate for current iteration
    """
    # Phase 1: Linear warmup
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters

    # Phase 3: Constant minimum
    if it > cosine_cycle_iters:
        return min_learning_rate

    # Phase 2: Cosine decay
    decay_ratio = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)


class AdamW(torch.optim.Optimizer):
    """
    AdamW optimizer with weight decay decoupling.

    Implements the AdamW algorithm from "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2019).
    Weight decay is applied directly to parameters, not to gradients.

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 1e-3)
        betas: Coefficients for running averages (default: (0.9, 0.999))
        eps: Term added for numerical stability (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 0.01)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                state["step"] += 1
                step = state["step"]

                # Decoupled weight decay
                if group["weight_decay"] != 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])

                # Bias correction
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute step size with bias correction
                step_size = group["lr"] / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])

                # Update parameters
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class ShardedOptimizer(torch.optim.Optimizer):
    """
    ZeRO Stage 1 sharded optimizer.

    Distributes optimizer states (momentum, variance) across ranks to reduce memory.
    Each rank only maintains optimizer states for 1/world_size of parameters.

    Adapted from CS336 Assignment 2 with bug fix for world_size=1.

    Args:
        params: Parameters to optimize
        optimizer_cls: Inner optimizer class (e.g., AdamW)
        **kwargs: Arguments passed to inner optimizer
    """

    def __init__(self, params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs):
        # Global counter for round-robin assignment
        self.global_index = 0
        self.param_to_rank = {}

        # Handle distributed initialization
        if dist.is_available() and dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

        defaults = kwargs.copy()
        super().__init__(params, defaults)

        # Build inner optimizer with only this rank's parameters
        inner_params = []
        for group in self.param_groups:
            sharded_params = [p for p in group["params"] if self.param_to_rank.get(p, 0) == self.rank]
            if len(sharded_params) > 0:
                inner_param = {**group, "params": sharded_params}
                inner_params.append(inner_param)

        self.inner_optimizer = optimizer_cls(inner_params)

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        """Add a parameter group with round-robin assignment."""
        if self.world_size > 1:
            for p in param_group["params"]:
                self.param_to_rank[p] = self.global_index % self.world_size
                self.global_index += 1
        else:
            # For world_size=1, assign all params to rank 0
            for p in param_group["params"]:
                self.param_to_rank[p] = 0

        super().add_param_group(param_group)

        # Update inner optimizer if already created
        if hasattr(self, "inner_optimizer") and self.inner_optimizer is not None:
            sharded_params = [p for p in param_group["params"] if self.param_to_rank.get(p, 0) == self.rank]
            if len(sharded_params) > 0:
                inner_param = {**param_group, "params": sharded_params}
                self.inner_optimizer.add_param_group(inner_param)

    def step(self, closure: Callable[[], float] | None = None, **kwargs) -> float | None:
        """Perform optimization step with broadcast synchronization."""
        loss = None

        # Phase 1: Update parameters owned by this rank
        if len(self.inner_optimizer.param_groups) > 0:
            loss = self.inner_optimizer.step(closure)

        # Phase 2: Broadcast updated parameters from owner to all ranks
        if self.world_size > 1:
            for group in self.param_groups:
                for p in group["params"]:
                    owner_rank = self.param_to_rank.get(p, 0)
                    dist.broadcast(p.data, owner_rank)

        return loss

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Zero gradients of all parameters."""
        # Zero all gradients (not just sharded ones)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.zero_()

    def state_dict(self) -> dict:
        """Return state dict with inner optimizer state."""
        return {
            "state": self.inner_optimizer.state_dict(),
            "param_groups": self.param_groups,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load state dict."""
        self.inner_optimizer.load_state_dict(state_dict["state"])


def create_optimizer(
    model_parameters,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    sharded: bool = False,
    **kwargs,
) -> torch.optim.Optimizer:
    """
    Create an optimizer with optional ZeRO-1 sharding.

    Args:
        model_parameters: Model parameters to optimize
        lr: Learning rate
        weight_decay: Weight decay coefficient
        sharded: Whether to use ZeRO-1 sharded optimizer
        **kwargs: Additional optimizer arguments

    Returns:
        Optimizer instance
    """
    if sharded:
        return ShardedOptimizer(
            model_parameters,
            AdamW,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs,
        )
    else:
        return AdamW(
            model_parameters,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs,
        )

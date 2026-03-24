"""GRPO (Group Relative Policy Optimization) Trainer."""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from ..common.model import BasicsTransformerLM
    from ..backends.inference import InferenceBackend


class GRPOTrainer:
    """Group Relative Policy Optimization Trainer.

    GRPO uses group-based advantage estimation instead of a value network.
    For each prompt, sample G responses, compute rewards, normalize within group.

    Key features:
    - Rollout: Generate responses using inference backend
    - Reward computation: External reward function
    - Group normalization: Advantage = (r - mean) / (std + eps)
    - Three loss types: "no_baseline", "reinforce_with_baseline", "grpo_clip"

    Attributes:
        model: Policy model being trained
        optimizer: Optimizer
        inference_backend: Backend for generation (rollout)
        reward_fn: Function to compute reward(response, ground_truth) -> dict
        group_size: Number of responses per prompt (G)
        beta: GRPO temperature/clipping parameter
        loss_type: One of "no_baseline", "reinforce_with_baseline", "grpo_clip"
    """

    def __init__(
        self,
        model: BasicsTransformerLM,
        optimizer: torch.optim.Optimizer,
        inference_backend: InferenceBackend,
        reward_fn: Callable[[str, str], dict[str, float]],
        group_size: int = 8,
        beta: float = 0.1,
        device: str = "cuda",
        config: dict | None = None,
    ):
        """Initialize GRPO Trainer.

        Args:
            model: Policy model to train
            optimizer: Optimizer for model parameters
            inference_backend: Backend for text generation
            reward_fn: Reward function (response, gt) -> dict with 'reward' key
            group_size: Number of responses per prompt
            beta: Clipping/temperature parameter
            device: Device for training
            config: Additional configuration dict
        """
        self.model = model
        self.optimizer = optimizer
        self.inference_backend = inference_backend
        self.reward_fn = reward_fn
        self.group_size = group_size
        self.beta = beta
        self.device = device
        self.config = config or {}
        self.global_step = 0

        # Extract config parameters with defaults
        self.loss_type = self.config.get("loss_type", "reinforce_with_baseline")
        self.advantage_eps = self.config.get("advantage_eps", 1e-6)
        self.normalize_by_std = self.config.get("normalize_by_std", True)
        self.cliprange = self.config.get("cliprange", 0.2)
        self.grad_clip_norm = self.config.get("grad_clip_norm", 1.0)

        self.model.to(self.device)

    def rollout(
        self,
        prompts: list[str],
        ground_truths: list[str],
        max_new_tokens: int = 256,
        temperature: float = 1.0,
    ) -> dict:
        """Generate responses and compute rewards.

        Args:
            prompts: List of prompt strings (length N)
            ground_truths: List of ground truth answers (length N)
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature

        Returns:
            Dict with:
                - "responses": list of response strings (length N * G)
                - "rewards": tensor of rewards (shape: N * G)
                - "advantages": tensor of normalized advantages (shape: N * G)
                - "prompts": repeated prompts (each prompt repeated G times)
                - "ground_truths": repeated ground truths
        """
        from ..backends.inference import GenerationConfig

        n_prompts = len(prompts)
        config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
        )

        # Repeat each prompt group_size times, then batch generate all at once
        repeated_prompts = [p for p in prompts for _ in range(self.group_size)]
        repeated_gts = [gt for gt in ground_truths for _ in range(self.group_size)]

        all_responses = self.inference_backend.generate_batch(repeated_prompts, config)
        all_prompts = repeated_prompts
        all_ground_truths = repeated_gts

        # Compute rewards
        rewards_list = []
        for response, gt in zip(all_responses, all_ground_truths):
            reward_dict = self.reward_fn(response, gt)
            rewards_list.append(reward_dict.get("reward", 0.0))

        rewards = torch.tensor(rewards_list, dtype=torch.float32, device=self.device)

        # Compute group-normalized advantages
        advantages = self.compute_group_advantages(
            rewards, self.group_size, eps=self.advantage_eps
        )

        return {
            "responses": all_responses,
            "rewards": rewards,
            "advantages": advantages,
            "prompts": all_prompts,
            "ground_truths": all_ground_truths,
        }

    def compute_group_advantages(
        self,
        rewards: torch.Tensor,
        group_size: int,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Compute group-normalized advantages.

        Args:
            rewards: Flat tensor of shape (N * G,)
            group_size: G (number of responses per prompt)
            eps: Small constant for numerical stability

        Returns:
            Advantages tensor of shape (N * G,)
        """
        # Reshape to (N, G)
        n_groups = rewards.shape[0] // group_size
        reshaped = rewards.view(n_groups, group_size)

        # Compute mean per group: (N, 1)
        group_means = reshaped.mean(dim=1, keepdim=True)

        if self.normalize_by_std:
            # Compute std per group (biased estimator for stability)
            group_stds = reshaped.std(dim=1, unbiased=False, keepdim=True)
            # Normalize: (r - mean) / (std + eps)
            advantages_reshaped = (reshaped - group_means) / (group_stds + eps)
        else:
            # Only subtract mean
            advantages_reshaped = reshaped - group_means

        # Flatten back to (N * G,)
        return advantages_reshaped.view(-1)

    def compute_policy_gradient_loss(
        self,
        log_probs: torch.Tensor,
        advantages: torch.Tensor,
        loss_type: str = "reinforce_with_baseline",
        old_log_probs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Compute policy gradient loss.

        Args:
            log_probs: Current policy log probs (batch, seq_len)
            advantages: Advantage values (batch,)
            loss_type: One of "no_baseline", "reinforce_with_baseline", "grpo_clip"
            old_log_probs: Old policy log probs for PPO clipping (grpo_clip only)

        Returns:
            loss: Scalar loss tensor
            metrics: Dict with loss breakdown
        """
        batch_size, seq_len = log_probs.shape

        # Expand advantages to match log_probs shape: (batch,) -> (batch, 1) -> (batch, seq_len)
        adv_expanded = advantages.unsqueeze(-1)

        if loss_type == "no_baseline":
            # REINFORCE: -advantage * log_prob
            # Note: advantages should be raw rewards for this case
            per_token_loss = -adv_expanded * log_probs
            metrics = {"loss_type": "no_baseline"}

        elif loss_type == "reinforce_with_baseline":
            # REINFORCE with baseline (group-normalized advantage)
            per_token_loss = -adv_expanded * log_probs
            metrics = {"loss_type": "reinforce_with_baseline"}

        elif loss_type == "grpo_clip":
            # PPO Clip objective
            if old_log_probs is None:
                raise ValueError("old_log_probs required for grpo_clip loss")

            # Compute ratio: pi_theta / pi_theta_old = exp(log_new - log_old)
            ratio = torch.exp(log_probs - old_log_probs)

            # Clipped objective
            unclipped = ratio * adv_expanded
            clipped = torch.clamp(ratio, 1.0 - self.cliprange, 1.0 + self.cliprange) * adv_expanded

            # Take minimum (conservative update) and negate for minimization
            per_token_loss = -torch.min(unclipped, clipped)

            # Compute clip fraction for monitoring
            is_clipped = (clipped < unclipped).float()
            clip_fraction = is_clipped.mean().item()
            metrics = {"loss_type": "grpo_clip", "clip_fraction": clip_fraction}

        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        # Average over batch and sequence
        loss = per_token_loss.mean()

        return loss, metrics

    def train_step(
        self,
        prompts: list[str],
        ground_truths: list[str],
        gradient_accumulation_steps: int = 1,
        epochs_per_rollout: int = 1,
    ) -> dict[str, float]:
        """Single GRPO training step.

        1. Rollout: Generate responses and compute advantages
        2. For each epoch:
           - Forward pass to get log probs
           - Compute policy gradient loss
           - Backward and update

        Args:
            prompts: List of prompt strings
            ground_truths: List of ground truth answers
            gradient_accumulation_steps: Number of gradient accumulation steps
            epochs_per_rollout: Number of training epochs per rollout

        Returns:
            Metrics dict with loss, reward_mean, etc.
        """
        import random

        from ..backends.inference import GenerationConfig

        # === Phase 1: Rollout ===
        self.model.eval()

        rollout_data = self.rollout(
            prompts=prompts,
            ground_truths=ground_truths,
            max_new_tokens=self.config.get("max_new_tokens", 256),
            temperature=self.config.get("temperature", 1.0),
        )

        responses = rollout_data["responses"]
        rewards = rollout_data["rewards"]
        advantages = rollout_data["advantages"]
        flat_prompts = rollout_data["prompts"]
        flat_ground_truths = rollout_data["ground_truths"]

        # Compute old_log_probs if using grpo_clip
        old_log_probs = None
        if self.loss_type == "grpo_clip":
            old_log_probs = self._compute_old_log_probs(
                flat_prompts, responses, gradient_accumulation_steps
            )

        # === Phase 2: Training ===
        self.model.train()

        total_loss = 0.0
        n_updates = 0

        indices = list(range(len(flat_prompts)))
        train_batch_size = self.config.get("train_batch_size", len(flat_prompts))

        for epoch in range(epochs_per_rollout):
            random.shuffle(indices)

            for i in range(0, len(indices), train_batch_size):
                batch_indices = indices[i:i + train_batch_size]

                self.optimizer.zero_grad()
                epoch_loss = 0.0

                # Gradient accumulation loop
                micro_batch_size = max(1, len(batch_indices) // gradient_accumulation_steps)

                for j in range(0, len(batch_indices), micro_batch_size):
                    micro_indices = batch_indices[j:j + micro_batch_size]

                    mb_prompts = [flat_prompts[idx] for idx in micro_indices]
                    mb_responses = [responses[idx] for idx in micro_indices]
                    mb_advantages = advantages[micro_indices]

                    # Validate shape consistency before indexing old_log_probs
                    if old_log_probs is not None:
                        assert old_log_probs.shape[0] == advantages.shape[0]

                    # Forward pass to get log probs
                    log_probs = self._get_response_log_probs(mb_prompts, mb_responses)

                    # Get old_log_probs for this microbatch if using grpo_clip
                    mb_old_log_probs = None
                    if self.loss_type == "grpo_clip" and old_log_probs is not None:
                        mb_old_log_probs = old_log_probs[micro_indices]

                    # Compute loss
                    loss, loss_meta = self.compute_policy_gradient_loss(
                        log_probs=log_probs,
                        advantages=mb_advantages,
                        loss_type=self.loss_type,
                        old_log_probs=mb_old_log_probs,
                    )

                    # Scale for gradient accumulation
                    scaled_loss = loss / gradient_accumulation_steps
                    scaled_loss.backward()

                    epoch_loss += loss.item()

                # Gradient clipping and update
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.grad_clip_norm
                )

                self.optimizer.step()
                self.global_step += 1
                n_updates += 1
                total_loss += epoch_loss

        metrics = {
            "loss": total_loss / max(1, n_updates),
            "reward_mean": rewards.mean().item(),
            "reward_max": rewards.max().item(),
            "reward_min": rewards.min().item(),
            "global_step": self.global_step,
        }

        return metrics

    def _compute_old_log_probs(
        self,
        prompts: list[str],
        responses: list[str],
        gradient_accumulation_steps: int,
    ) -> torch.Tensor:
        """Compute old policy log probs for PPO clipping.

        Args:
            prompts: List of prompts
            responses: List of responses
            gradient_accumulation_steps: Number of accumulation steps

        Returns:
            Tensor of old log probs (N * G, seq_len)
        """
        old_log_probs_list = []

        micro_batch_size = max(1, len(prompts) // gradient_accumulation_steps)

        with torch.inference_mode():
            for i in range(0, len(prompts), micro_batch_size):
                mb_prompts = prompts[i:i + micro_batch_size]
                mb_responses = responses[i:i + micro_batch_size]

                log_probs = self._get_response_log_probs(mb_prompts, mb_responses)
                old_log_probs_list.append(log_probs.cpu())

        return torch.cat(old_log_probs_list, dim=0).to(self.device)

    def _get_response_log_probs(
        self,
        prompts: list[str],
        responses: list[str],
    ) -> torch.Tensor:
        """Get log probabilities for response tokens given prompts.

        Args:
            prompts: List of prompt strings
            responses: List of response strings

        Returns:
            Log probs tensor (batch, seq_len)
        """
        # Get tokenizer from inference backend if available, else assume model has it
        tokenizer = getattr(self.inference_backend, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError(
                "Tokenizer not available. Please use HFInferenceBackend or provide tokenizer."
            )

        # Tokenize prompts and responses together
        batch_input_ids = []
        batch_labels = []

        for prompt, response in zip(prompts, responses):
            # Tokenize prompt with special tokens
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
            # Tokenize response without special tokens
            response_ids = tokenizer.encode(response, add_special_tokens=False)

            # Concatenate
            concat_ids = prompt_ids + response_ids

            # Shift: input_ids predicts labels
            input_ids = concat_ids[:-1]
            labels = concat_ids[1:]

            batch_input_ids.append(input_ids)
            batch_labels.append(labels)

        # Pad to max length
        max_len = max(len(ids) for ids in batch_input_ids)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        padded_input_ids = []
        padded_labels = []

        for input_ids, labels in zip(batch_input_ids, batch_labels):
            pad_len = max_len - len(input_ids)
            padded_input_ids.append(input_ids + [pad_id] * pad_len)
            padded_labels.append(labels + [pad_id] * pad_len)

        input_ids_tensor = torch.tensor(padded_input_ids, dtype=torch.long, device=self.device)
        labels_tensor = torch.tensor(padded_labels, dtype=torch.long, device=self.device)

        # Forward pass
        logits = self.model(input_ids_tensor)

        # Compute log probabilities using log-sum-exp trick
        max_logits = torch.max(logits, dim=-1, keepdim=True)[0]
        exp_logits = torch.exp(logits - max_logits)
        sum_exp = torch.sum(exp_logits, dim=-1, keepdim=True)
        log_probs_all = logits - max_logits - torch.log(sum_exp)

        # Gather log probs for actual tokens
        log_probs = torch.gather(
            log_probs_all,
            dim=-1,
            index=labels_tensor.unsqueeze(-1),
        ).squeeze(-1)

        # Mask padding positions
        mask = (labels_tensor != pad_id).float()
        log_probs = log_probs * mask

        return log_probs

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        import os

        os.makedirs(path, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "config": self.config,
            "group_size": self.group_size,
            "beta": self.beta,
        }

        torch.save(checkpoint, os.path.join(path, "checkpoint.pt"))

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint.

        Args:
            path: Path to checkpoint
        """
        import os

        checkpoint_path = os.path.join(path, "checkpoint.pt")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)

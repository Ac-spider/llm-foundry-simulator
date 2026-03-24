"""Alignment utilities for SFT, DPO, and GRPO training."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING, Callable, Literal

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
else:
    PreTrainedTokenizerBase = object


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    """Tokenize prompt-output pairs and create response mask.

    Only output tokens are marked with response_mask=1 for loss computation.
    Shift alignment: input_ids = concat[:-1], labels = concat[1:].

    Args:
        prompt_strs: List of prompt strings
        output_strs: List of output/response strings
        tokenizer: HuggingFace tokenizer

    Returns:
        Dict with input_ids, labels, response_mask tensors
    """
    batch_input_ids = []
    batch_labels = []
    batch_response_mask = []

    for p_str, o_str in zip(prompt_strs, output_strs):
        p_ids = tokenizer.encode(p_str, add_special_tokens=True)
        o_ids = tokenizer.encode(o_str, add_special_tokens=False)

        concat_ids = p_ids + o_ids
        mask = [0] * (len(p_ids) - 1) + [1] * len(o_ids)

        input_ids = concat_ids[:-1]
        labels = concat_ids[1:]

        batch_input_ids.append(input_ids)
        batch_labels.append(labels)
        batch_response_mask.append(mask)

    max_len = max(len(ids) for ids in batch_input_ids)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    padded_input_ids = []
    padded_labels = []
    padded_mask = []

    for i_ids, l_ids, m in zip(batch_input_ids, batch_labels, batch_response_mask):
        pad_len = max_len - len(i_ids)
        padded_input_ids.append(i_ids + [pad_id] * pad_len)
        padded_labels.append(l_ids + [pad_id] * pad_len)
        padded_mask.append(m + [0] * pad_len)

    return {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "labels": torch.tensor(padded_labels, dtype=torch.long),
        "response_mask": torch.tensor(padded_mask, dtype=torch.long)
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute entropy H(p) = -sum p(x) log p(x) for logits.

    Uses log-sum-exp trick for numerical stability.

    Args:
        logits: Unnormalized logits, shape (..., vocab_size)

    Returns:
        Entropy values, shape (...)
    """
    max_logits = torch.max(logits, dim=-1, keepdim=True)[0]
    exp_logits = torch.exp(logits - max_logits)
    sum_exp = torch.sum(exp_logits, dim=-1, keepdim=True)
    log_probs = logits - max_logits - torch.log(sum_exp)
    probs = torch.exp(log_probs)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """Get log probabilities for response tokens.

    Args:
        model: Causal language model
        input_ids: Input token ids, shape (batch, seq_len)
        labels: Target token ids, shape (batch, seq_len)
        return_token_entropy: Whether to return per-token entropy

    Returns:
        Dict with log_probs and optionally token_entropy
    """
    logits = model(input_ids).logits

    max_logits = torch.max(logits, dim=-1, keepdim=True)[0]
    exp_logits = torch.exp(logits - max_logits)
    sum_exp = torch.sum(exp_logits, dim=-1, keepdim=True)
    log_probs_all = logits - max_logits - torch.log(sum_exp)

    log_probs = torch.gather(
        log_probs_all,
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)

    result = {"log_probs": log_probs}
    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)

    return result


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """Sum masked elements and divide by constant.

    Args:
        tensor: Input tensor
        mask: 0/1 mask
        normalize_constant: Division constant
        dim: Dimension to sum over (None for global)

    Returns:
        Normalized tensor
    """
    masked_tensor = tensor * mask.to(tensor.dtype)

    if dim is None:
        summed = torch.sum(masked_tensor)
    else:
        summed = torch.sum(masked_tensor, dim=dim)

    return summed / normalize_constant


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    """Compute mean of masked elements.

    Args:
        tensor: Input tensor
        mask: 0/1 mask
        dim: Dimension to compute mean over

    Returns:
        Mean tensor
    """
    mask = mask.to(tensor.dtype)
    masked_tensor = tensor * mask

    if dim is None:
        sum_tensor = torch.sum(masked_tensor)
        count_tensor = torch.sum(mask)
    else:
        sum_tensor = torch.sum(masked_tensor, dim=dim)
        count_tensor = torch.sum(mask, dim=dim)

    count_tensor = torch.clamp(count_tensor, min=1e-8)
    return sum_tensor / count_tensor


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """SFT training step for a microbatch.

    Args:
        policy_log_probs: Log probabilities, shape (batch, seq_len)
        response_mask: Response mask, shape (batch, seq_len)
        gradient_accumulation_steps: Number of gradient accumulation steps
        normalize_constant: Normalization constant

    Returns:
        Tuple of (scaled_loss, metadata)
    """
    sum_log_probs = masked_normalize(
        tensor=policy_log_probs,
        mask=response_mask,
        normalize_constant=normalize_constant,
        dim=None
    )

    loss = -sum_log_probs
    scaled_loss = loss / gradient_accumulation_steps
    scaled_loss.backward()

    metadata = {
        "sft_loss": loss.detach(),
        "scaled_sft_loss": scaled_loss.detach()
    }

    return scaled_loss, metadata


def log_generations(
    prompts: list[str],
    responses: list[str],
    ground_truths: list[str],
    reward_fn: Callable[[str, str], dict[str, float]],
    token_entropies: list[float] | None = None,
    step: int = 0,
) -> dict:
    """Log generation metrics for evaluation.

    Args:
        prompts: Input prompts
        responses: Generated responses
        ground_truths: Ground truth answers
        reward_fn: Reward function
        token_entropies: Per-response token entropies
        step: Training step

    Returns:
        Dict of metrics
    """
    total_rewards = []
    format_rewards = []
    answer_rewards = []
    response_lengths = []
    correct_lengths = []
    incorrect_lengths = []

    if token_entropies is None:
        token_entropies = [0.0] * len(prompts)

    for i, (p, r, gt, entropy) in enumerate(zip(prompts, responses, ground_truths, token_entropies)):
        reward_info = reward_fn(r, gt)
        t_reward = reward_info.get("reward", 0.0)
        f_reward = reward_info.get("format_reward", 0.0)
        a_reward = reward_info.get("answer_reward", 0.0)

        total_rewards.append(t_reward)
        format_rewards.append(f_reward)
        answer_rewards.append(a_reward)

        length = len(r)
        response_lengths.append(length)
        if a_reward > 0:
            correct_lengths.append(length)
        else:
            incorrect_lengths.append(length)

        if i < 2:
            print(f"--- Step {step} | Sample {i + 1} ---")
            print(f"Prompt: {p[:100]}...")
            print(f"Response: {r}")
            print(f"Ground Truth: {gt}")
            print(f"Rewards: Total={t_reward}, Format={f_reward}, Answer={a_reward}")
            print(f"Avg Entropy: {entropy:.4f}")
            print("-" * 30)

    def safe_mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    return {
        "eval/reward_total": safe_mean(total_rewards),
        "eval/reward_format": safe_mean(format_rewards),
        "eval/reward_answer": safe_mean(answer_rewards),
        "eval/avg_token_entropy": safe_mean(token_entropies),
        "eval/avg_response_len": safe_mean(response_lengths),
        "eval/avg_correct_len": safe_mean(correct_lengths),
        "eval/avg_incorrect_len": safe_mean(incorrect_lengths),
    }


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compute group-normalized advantages for GRPO.

    Args:
        reward_fn: Reward function
        rollout_responses: List of rollout responses
        repeated_ground_truths: Repeated ground truths
        group_size: Group size G
        advantage_eps: Epsilon for numerical stability
        normalize_by_std: Whether to normalize by std

    Returns:
        Tuple of (advantages, raw_rewards, metadata)
    """
    raw_rewards_list = []

    for resp, gt in zip(rollout_responses, repeated_ground_truths):
        reward_dict = reward_fn(resp, gt)
        raw_rewards_list.append(reward_dict.get('reward', 0))

    raw_rewards = torch.tensor(raw_rewards_list, dtype=torch.float32)
    reshaped_rewards = raw_rewards.view(-1, group_size)
    group_means = reshaped_rewards.mean(dim=-1, keepdim=True)

    if normalize_by_std:
        group_stds = reshaped_rewards.std(dim=-1, unbiased=False, keepdim=True)
        advantage_reshaped = (reshaped_rewards - group_means) / (group_stds + advantage_eps)
    else:
        advantage_reshaped = reshaped_rewards - group_means

    advantages = advantage_reshaped.view(-1)

    metadata = {
        "reward_mean": raw_rewards.mean().item(),
        "reward_max": raw_rewards.max().item(),
        "reward_min": raw_rewards.min().item(),
    }

    return advantages, raw_rewards, metadata


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute REINFORCE policy gradient loss.

    Args:
        raw_rewards_or_advantages: Rewards or advantages, shape (batch,)
        policy_log_probs: Log probabilities, shape (batch, seq_len)

    Returns:
        Per-token loss, shape (batch, seq_len)
    """
    adv_expanded = raw_rewards_or_advantages.unsqueeze(-1)
    return -adv_expanded * policy_log_probs


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute PPO clip loss for GRPO.

    Args:
        advantages: Normalized advantages, shape (batch,)
        policy_log_probs: Current policy log probs, shape (batch, seq_len)
        old_log_probs: Old policy log probs, shape (batch, seq_len)
        cliprange: Clip range epsilon

    Returns:
        Tuple of (per_token_loss, metadata)
    """
    adv_expanded = advantages.unsqueeze(-1)
    ratio = torch.exp(policy_log_probs - old_log_probs)

    unclipped_term = ratio * adv_expanded
    clipped_term = torch.clamp(ratio, 1 - cliprange, 1 + cliprange) * adv_expanded

    per_token_loss = -torch.min(unclipped_term, clipped_term)

    is_clipped = (clipped_term < unclipped_term).to(torch.float32)
    clip_fraction = is_clipped.mean()

    return per_token_loss, {"clip_fraction": clip_fraction.detach()}


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Dispatcher for policy gradient loss computation.

    Args:
        policy_log_probs: Policy log probabilities
        loss_type: Loss type
        raw_rewards: Raw rewards (for no_baseline)
        advantages: Advantages (for reinforce_with_baseline, grpo_clip)
        old_log_probs: Old policy log probs (for grpo_clip)
        cliprange: Clip range (for grpo_clip)

    Returns:
        Tuple of (loss, metadata)
    """
    metadata = {}

    if loss_type == "no_baseline":
        assert raw_rewards is not None
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
    elif loss_type == "grpo_clip":
        assert advantages is not None
        assert old_log_probs is not None
        assert cliprange is not None
        loss, clip_meta = compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
        metadata.update(clip_meta)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    return loss, metadata


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """GRPO training step for a microbatch.

    Args:
        policy_log_probs: Policy log probabilities
        response_mask: Response mask
        gradient_accumulation_steps: Number of gradient accumulation steps
        loss_type: Loss type
        raw_rewards: Raw rewards
        advantages: Normalized advantages
        old_log_probs: Old policy log probs
        cliprange: Clip range

    Returns:
        Tuple of (scaled_loss, metadata)
    """
    per_token_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    per_example_loss = masked_mean(per_token_loss, response_mask, dim=1)
    loss = torch.mean(per_example_loss)
    scaled_loss = loss / gradient_accumulation_steps
    scaled_loss.backward()

    metadata['loss'] = loss.detach()
    metadata['scaled_loss'] = scaled_loss.detach()

    return scaled_loss, metadata

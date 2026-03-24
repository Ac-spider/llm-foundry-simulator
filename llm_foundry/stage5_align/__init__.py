"""Stage 5: Alignment Module (SFT, DPO, GRPO).

This module provides training capabilities for aligning language models with human
preferences using state-of-the-art RLHF (Reinforcement Learning from Human Feedback)
techniques.

Training Methods:
    SFT (Supervised Fine-Tuning):
        - Standard instruction-following training
        - Loss computed only on response tokens
        - Supports both HuggingFace and custom tokenizers

    DPO (Direct Preference Optimization):
        - Direct optimization from preference pairs (chosen/rejected)
        - No explicit reward model needed
        - Implicit reward through policy/reference ratio
        - Loss: -log σ(β * (log π_θ(y_w)/π_ref(y_w) - log π_θ(y_l)/π_ref(y_l)))

    GRPO (Group Relative Policy Optimization):
        - Group-based advantage estimation (no value network)
        - Sample G responses per prompt, normalize rewards within group
        - Three loss variants: no_baseline, reinforce_with_baseline, grpo_clip
        - Supports PPO-style clipping for stable training

Core Classes:
    SFTTrainer: Supervised fine-tuning trainer
    SFTDataset: Dataset for SFT with prompt-response pairs
    DPOTrainer: Direct Preference Optimization trainer
    DPODataset: Dataset for DPO with preference pairs
    GRPOTrainer: Group Relative Policy Optimization trainer

Utility Functions:
    tokenize_prompt_and_output: Tokenize prompt-output pairs with response masking
    get_response_log_probs: Get per-token log probabilities for responses
    sft_microbatch_train_step: SFT training step for microbatches
    compute_group_normalized_rewards: Compute group-normalized advantages for GRPO
    grpo_microbatch_train_step: GRPO training step for microbatches

Example:
    >>> from llm_foundry.stage5_align import SFTTrainer, DPOTrainer, GRPOTrainer
    >>> # SFT training
    >>> sft_trainer = SFTTrainer(model, tokenizer, optimizer)
    >>> sft_metrics = sft_trainer.train_epoch(train_dataloader)
    >>> # DPO training
    >>> dpo_trainer = DPOTrainer(model, ref_model, tokenizer, optimizer, beta=0.1)
    >>> dpo_metrics = dpo_trainer.train_epoch(dpo_dataloader)
    >>> # GRPO training
    >>> grpo_trainer = GRPOTrainer(model, optimizer, inference_backend, reward_fn)
    >>> grpo_metrics = grpo_trainer.train_step(prompts, ground_truths)

References:
    - SFT: Standard supervised fine-tuning (Ouyang et al., 2022)
    - DPO: Rafailov et al., "Direct Preference Optimization", NeurIPS 2023
    - GRPO: DeepSeekMath paper (group-based RL without value model)
"""

from .sft import SFTTrainer, SFTDataset, collate_fn
from .dpo import DPOTrainer, DPODataset
from .grpo import GRPOTrainer
from .utils import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    sft_microbatch_train_step,
    compute_group_normalized_rewards,
    grpo_microbatch_train_step,
)

__all__ = [
    # Trainers
    "SFTTrainer",
    "DPOTrainer",
    "GRPOTrainer",
    # Datasets
    "SFTDataset",
    "DPODataset",
    # Collate function
    "collate_fn",
    # Utils
    "tokenize_prompt_and_output",
    "get_response_log_probs",
    "sft_microbatch_train_step",
    "compute_group_normalized_rewards",
    "grpo_microbatch_train_step",
]

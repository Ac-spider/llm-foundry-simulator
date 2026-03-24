"""DPO (Direct Preference Optimization) Trainer.

This module implements Direct Preference Optimization (DPO), which directly optimizes
language models from preference data without explicit reward model training.

Reference:
    Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly
    a Reward Model", NeurIPS 2023.

DPO Loss Formula:
    L = -log σ(β * (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))

Where:
    - y_w: chosen (preferred) response
    - y_l: rejected (less preferred) response
    - π_θ: policy model being trained
    - π_ref: reference model (frozen, typically the SFT model)
    - β: temperature parameter controlling deviation from reference
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from ..common.model import BasicsTransformerLM

logger = logging.getLogger(__name__)


def get_sequence_log_prob(
    model: BasicsTransformerLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute log P(sequence) for a given model.

    This function computes the log probability of a sequence by:
    1. Forward pass to get logits
    2. Shift logits and labels for next-token prediction
    3. Compute log_softmax over vocabulary
    4. Gather log probs for actual tokens
    5. Sum over sequence length

    Args:
        model: Language model
        input_ids: Token IDs, shape (batch_size, seq_len)
        attention_mask: Optional attention mask, shape (batch_size, seq_len)

    Returns:
        Log probabilities, shape (batch_size,)
    """
    # Forward pass: get logits for each position
    logits = model(input_ids)  # (batch_size, seq_len, vocab_size)

    # Shift for next-token prediction: position i predicts token i+1
    # logits[:, :-1, :] corresponds to input positions 0..seq_len-2
    # input_ids[:, 1:] corresponds to target tokens 1..seq_len-1
    shift_logits = logits[:, :-1, :]  # (batch_size, seq_len-1, vocab_size)
    shift_labels = input_ids[:, 1:]  # (batch_size, seq_len-1)

    # Compute log softmax over vocabulary for numerical stability
    log_probs_all = F.log_softmax(shift_logits, dim=-1)  # (batch_size, seq_len-1, vocab_size)

    # Gather log probs for the actual tokens
    # unsqueeze(-1) adds a dimension for gather: (batch_size, seq_len-1, 1)
    token_log_probs = torch.gather(
        log_probs_all,
        dim=-1,
        index=shift_labels.unsqueeze(-1),
    ).squeeze(-1)  # (batch_size, seq_len-1)

    # Apply attention mask if provided (for padded sequences)
    if attention_mask is not None:
        # Shift mask to align with targets
        shift_mask = attention_mask[:, 1:].to(token_log_probs.dtype)
        token_log_probs = token_log_probs * shift_mask
        # Sum and divide by number of valid tokens for mean log prob
        seq_log_probs = token_log_probs.sum(dim=-1) / shift_mask.sum(dim=-1).clamp(min=1.0)
    else:
        # Mean log prob per sequence
        seq_log_probs = token_log_probs.mean(dim=-1)

    return seq_log_probs


def get_response_log_probs(
    model: BasicsTransformerLM,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Get per-token log probabilities for responses.

    Similar to get_sequence_log_prob but returns per-token log probs
    and optionally computes entropy.

    Args:
        model: Language model
        input_ids: Token IDs, shape (batch_size, seq_len)
        response_mask: Mask indicating response tokens (1=response, 0=prompt/pad)

    Returns:
        Dictionary with:
            - log_probs: Per-token log probabilities, shape (batch_size, seq_len)
    """
    logits = model(input_ids)  # (batch_size, seq_len, vocab_size)

    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :]  # (batch_size, seq_len-1, vocab_size)
    shift_labels = input_ids[:, 1:]  # (batch_size, seq_len-1)

    # Log-sum-exp trick for numerical stability
    max_logits = torch.max(shift_logits, dim=-1, keepdim=True)[0]
    exp_logits = torch.exp(shift_logits - max_logits)
    sum_exp = torch.sum(exp_logits, dim=-1, keepdim=True)
    log_probs_all = shift_logits - max_logits - torch.log(sum_exp)

    # Gather log probs for actual tokens
    log_probs = torch.gather(
        log_probs_all,
        dim=-1,
        index=shift_labels.unsqueeze(-1),
    ).squeeze(-1)  # (batch_size, seq_len-1)

    # Pad to match original sequence length (first position has no prediction)
    batch_size = input_ids.size(0)
    zero_pad = torch.zeros(batch_size, 1, device=log_probs.device, dtype=log_probs.dtype)
    log_probs = torch.cat([zero_pad, log_probs], dim=1)  # (batch_size, seq_len)

    # Zero out non-response positions if mask provided
    if response_mask is not None:
        log_probs = log_probs * response_mask.to(log_probs.dtype)

    return {"log_probs": log_probs}


class DPODataset(Dataset):
    """Dataset for DPO training with preference pairs.

    Data format: {"prompt": str, "chosen": str, "rejected": str}
    Each sample contains a prompt and two responses (chosen=preferred, rejected=less preferred).

    Attributes:
        data: List of preference samples
        tokenizer: Tokenizer for encoding text
        max_length: Maximum sequence length
        prompt_template: Template for formatting prompts
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 2048,
        prompt_template: str | None = None,
    ):
        """Initialize DPO dataset.

        Args:
            data_path: Path to JSONL file with preference data
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
            prompt_template: Optional template for formatting prompts
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Default Alpaca-style template
        self.prompt_template = prompt_template or (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{prompt}\n\n### Response:\n{response}"
        )

        # Load data
        data_file = Path(data_path)
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        self.data = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))

        logger.info(f"Loaded {len(self.data)} preference pairs from {data_path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """Get a single preference pair.

        Returns a dictionary with:
            - prompt: Original prompt string
            - chosen: Chosen response string
            - rejected: Rejected response string
            - prompt_input_ids: Tokenized prompt
            - chosen_input_ids: Tokenized prompt + chosen
            - rejected_input_ids: Tokenized prompt + rejected
        """
        item = self.data[idx]
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]

        # Format full texts with template
        chosen_text = self.prompt_template.format(prompt=prompt, response=chosen)
        rejected_text = self.prompt_template.format(prompt=prompt, response=rejected)

        # Add EOS token
        if self.tokenizer.eos_token:
            chosen_text += self.tokenizer.eos_token
            rejected_text += self.tokenizer.eos_token

        # Tokenize prompt only (for reference)
        prompt_input_ids = self.tokenizer.encode(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
        )

        # Tokenize prompt + chosen
        chosen_input_ids = self.tokenizer.encode(
            chosen_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )

        # Tokenize prompt + rejected
        rejected_input_ids = self.tokenizer.encode(
            rejected_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )

        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "prompt_input_ids": prompt_input_ids,
            "chosen_input_ids": chosen_input_ids,
            "rejected_input_ids": rejected_input_ids,
        }


def dpo_collate_fn(batch: list[dict], pad_token_id: int = 0) -> dict[str, torch.Tensor]:
    """Collate function for DPO dataloader.

    Pads sequences to max length in batch and creates attention masks.

    Args:
        batch: List of samples from dataset
        pad_token_id: Token ID for padding

    Returns:
        Dictionary with padded tensors:
            - chosen_input_ids: (batch_size, max_seq_len)
            - chosen_attention_mask: (batch_size, max_seq_len)
            - rejected_input_ids: (batch_size, max_seq_len)
            - rejected_attention_mask: (batch_size, max_seq_len)
    """
    # Find max lengths
    max_chosen_len = max(len(item["chosen_input_ids"]) for item in batch)
    max_rejected_len = max(len(item["rejected_input_ids"]) for item in batch)

    batch_chosen_ids = []
    batch_chosen_mask = []
    batch_rejected_ids = []
    batch_rejected_mask = []

    for item in batch:
        # Pad chosen
        chosen_ids = item["chosen_input_ids"]
        chosen_pad_len = max_chosen_len - len(chosen_ids)
        batch_chosen_ids.append(chosen_ids + [pad_token_id] * chosen_pad_len)
        batch_chosen_mask.append([1] * len(chosen_ids) + [0] * chosen_pad_len)

        # Pad rejected
        rejected_ids = item["rejected_input_ids"]
        rejected_pad_len = max_rejected_len - len(rejected_ids)
        batch_rejected_ids.append(rejected_ids + [pad_token_id] * rejected_pad_len)
        batch_rejected_mask.append([1] * len(rejected_ids) + [0] * rejected_pad_len)

    return {
        "chosen_input_ids": torch.tensor(batch_chosen_ids, dtype=torch.long),
        "chosen_attention_mask": torch.tensor(batch_chosen_mask, dtype=torch.long),
        "rejected_input_ids": torch.tensor(batch_rejected_ids, dtype=torch.long),
        "rejected_attention_mask": torch.tensor(batch_rejected_mask, dtype=torch.long),
    }


class DPOTrainer:
    """Direct Preference Optimization Trainer.

    DPO directly optimizes language models from preference data without training
    an explicit reward model. The key insight is that the reward can be expressed
    implicitly through the ratio of policy to reference model likelihoods.

    Attributes:
        model: Policy model being trained
        ref_model: Reference model (frozen, typically the SFT model)
        tokenizer: Tokenizer
        optimizer: Optimizer
        beta: DPO temperature parameter
        device: Device for training
        config: Training configuration
        global_step: Current training step

    Example:
        >>> trainer = DPOTrainer(
        ...     model=policy_model,
        ...     ref_model=reference_model,
        ...     tokenizer=tokenizer,
        ...     optimizer=optimizer,
        ...     beta=0.1,
        ... )
        >>> metrics = trainer.train_epoch(dataloader)
    """

    def __init__(
        self,
        model: BasicsTransformerLM,
        ref_model: BasicsTransformerLM,
        tokenizer: PreTrainedTokenizerBase,
        optimizer: torch.optim.Optimizer,
        beta: float = 0.1,
        device: str = "cuda",
        config: dict | None = None,
    ):
        """Initialize DPO trainer.

        Args:
            model: Policy model being trained
            ref_model: Reference model (frozen, typically the SFT checkpoint)
            tokenizer: Tokenizer
            optimizer: Optimizer for policy model
            beta: DPO temperature parameter (default 0.1)
            device: Device for training
            config: Additional training configuration
        """
        self.model = model.to(device)
        self.ref_model = ref_model.to(device)
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.beta = beta
        self.device = device
        self.config = config or {}
        self.global_step = 0

        # Freeze reference model - no gradients needed
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()

        logger.info(f"Initialized DPOTrainer with beta={beta}, device={device}")

    def _compute_sequence_log_probs(
        self,
        model: BasicsTransformerLM,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute log P(sequence) for a given model.

        Args:
            model: Language model
            input_ids: Token IDs, shape (batch_size, seq_len)
            attention_mask: Optional attention mask

        Returns:
            Log probabilities, shape (batch_size,)
        """
        return get_sequence_log_prob(model, input_ids, attention_mask)

    def compute_dpo_loss(
        self,
        chosen_input_ids: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        chosen_attention_mask: torch.Tensor | None = None,
        rejected_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute DPO loss for a batch of preference pairs.

        DPO Loss:
            L = -log σ(β * (log_ratio_chosen - log_ratio_rejected))

        Where:
            log_ratio_chosen = log π_θ(y_w|x) - log π_ref(y_w|x)
            log_ratio_rejected = log π_θ(y_l|x) - log π_ref(y_l|x)

        Args:
            chosen_input_ids: Tokenized prompt + chosen, shape (batch_size, seq_len)
            rejected_input_ids: Tokenized prompt + rejected, shape (batch_size, seq_len)
            chosen_attention_mask: Attention mask for chosen
            rejected_attention_mask: Attention mask for rejected

        Returns:
            loss: Scalar loss tensor
            metrics: Dictionary with training metrics:
                - chosen_logprob: Mean log prob of chosen responses (policy)
                - rejected_logprob: Mean log prob of rejected responses (policy)
                - chosen_ref_logprob: Mean log prob of chosen (reference)
                - rejected_ref_logprob: Mean log prob of rejected (reference)
                - logits_diff: Difference in implicit rewards
                - accuracy: Percentage of samples where chosen > rejected
        """
        batch_size = chosen_input_ids.size(0)

        # Policy model log probs (with gradients)
        log_pi_theta_chosen = self._compute_sequence_log_probs(
            self.model, chosen_input_ids, chosen_attention_mask
        )
        log_pi_theta_rejected = self._compute_sequence_log_probs(
            self.model, rejected_input_ids, rejected_attention_mask
        )

        # Reference model log probs (no gradients)
        with torch.no_grad():
            log_pi_ref_chosen = self._compute_sequence_log_probs(
                self.ref_model, chosen_input_ids, chosen_attention_mask
            )
            log_pi_ref_rejected = self._compute_sequence_log_probs(
                self.ref_model, rejected_input_ids, rejected_attention_mask
            )

        # Compute implicit reward differences (log ratios)
        # r(x,y) ∝ log π_θ(y|x) - log π_ref(y|x)
        pi_logratios = log_pi_theta_chosen - log_pi_theta_rejected
        ref_logratios = log_pi_ref_chosen - log_pi_ref_rejected
        logits_diff = pi_logratios - ref_logratios  # Implicit reward difference

        # DPO loss: -log σ(β * diff)
        # This encourages the policy to prefer chosen over rejected
        loss = -F.logsigmoid(self.beta * logits_diff).mean()

        # Compute metrics
        with torch.no_grad():
            # Accuracy: percentage where chosen > rejected
            accuracy = (logits_diff > 0).float().mean().item()

            metrics = {
                "loss": loss.item(),
                "chosen_logprob": log_pi_theta_chosen.mean().item(),
                "rejected_logprob": log_pi_theta_rejected.mean().item(),
                "chosen_ref_logprob": log_pi_ref_chosen.mean().item(),
                "rejected_ref_logprob": log_pi_ref_rejected.mean().item(),
                "logits_diff": logits_diff.mean().item(),
                "accuracy": accuracy,
            }

        return loss, metrics

    def train_step(
        self,
        batch: dict[str, torch.Tensor],
        gradient_accumulation_steps: int = 1,
    ) -> dict[str, float]:
        """Single training step.

        Args:
            batch: Dictionary with chosen/rejected input_ids and attention_masks
            gradient_accumulation_steps: Number of steps to accumulate gradients

        Returns:
            Dictionary with training metrics
        """
        # Move batch to device
        chosen_input_ids = batch["chosen_input_ids"].to(self.device)
        rejected_input_ids = batch["rejected_input_ids"].to(self.device)
        chosen_attention_mask = batch.get("chosen_attention_mask", None)
        if chosen_attention_mask is not None:
            chosen_attention_mask = chosen_attention_mask.to(self.device)
        rejected_attention_mask = batch.get("rejected_attention_mask", None)
        if rejected_attention_mask is not None:
            rejected_attention_mask = rejected_attention_mask.to(self.device)

        # Compute loss
        loss, metrics = self.compute_dpo_loss(
            chosen_input_ids,
            rejected_input_ids,
            chosen_attention_mask,
            rejected_attention_mask,
        )

        # Scale for gradient accumulation
        scaled_loss = loss / gradient_accumulation_steps
        scaled_loss.backward()

        # Add scaled loss to metrics
        metrics["scaled_loss"] = scaled_loss.item()

        self.global_step += 1

        return metrics

    def train_epoch(
        self,
        dataloader: DataLoader,
        gradient_accumulation_steps: int = 1,
        max_steps: int | None = None,
        log_interval: int = 10,
    ) -> dict[str, float]:
        """Train for one epoch.

        Args:
            dataloader: DataLoader with preference pairs
            gradient_accumulation_steps: Number of steps to accumulate gradients
            max_steps: Maximum number of steps (None for full epoch)
            log_interval: Log metrics every N steps

        Returns:
            Dictionary with average metrics for the epoch
        """
        self.model.train()

        total_metrics = {
            "loss": 0.0,
            "accuracy": 0.0,
            "chosen_logprob": 0.0,
            "rejected_logprob": 0.0,
            "logits_diff": 0.0,
        }
        num_batches = 0

        pbar = tqdm(dataloader, desc="DPO Training")

        for step, batch in enumerate(pbar):
            if max_steps is not None and step >= max_steps:
                break

            # Determine if we should update weights
            is_accumulation_step = (step + 1) % gradient_accumulation_steps != 0
            is_last_step = step == len(dataloader) - 1

            if not is_accumulation_step or is_last_step:
                # Last step in accumulation, do optimizer step
                metrics = self.train_step(batch, gradient_accumulation_steps)
                self.optimizer.step()
                self.optimizer.zero_grad()
            else:
                # Accumulation step, don't zero gradients
                with torch.no_grad():
                    metrics = self.train_step(batch, gradient_accumulation_steps)

            # Accumulate metrics
            for key in total_metrics:
                if key in metrics:
                    total_metrics[key] += metrics[key]
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{metrics['loss']:.4f}",
                "acc": f"{metrics['accuracy']:.2%}",
            })

            # Log periodically
            if (step + 1) % log_interval == 0:
                logger.info(
                    f"Step {self.global_step}: loss={metrics['loss']:.4f}, "
                    f"accuracy={metrics['accuracy']:.2%}"
                )

        # Average metrics
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        logger.info(f"Epoch complete: {avg_metrics}")

        return avg_metrics

    def train(
        self,
        dataloader: DataLoader,
        num_epochs: int = 1,
        gradient_accumulation_steps: int = 1,
        max_steps: int | None = None,
        save_interval: int | None = None,
        checkpoint_dir: str | None = None,
    ) -> list[dict[str, float]]:
        """Main training loop.

        Args:
            dataloader: DataLoader with preference pairs
            num_epochs: Number of epochs to train
            gradient_accumulation_steps: Number of steps to accumulate gradients
            max_steps: Maximum total steps across all epochs
            save_interval: Save checkpoint every N steps
            checkpoint_dir: Directory to save checkpoints

        Returns:
            List of metrics dictionaries, one per epoch
        """
        all_metrics = []
        total_steps = 0

        for epoch in range(num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")

            # Calculate remaining steps
            remaining_steps = None
            if max_steps is not None:
                remaining_steps = max_steps - total_steps
                if remaining_steps <= 0:
                    break

            # Train one epoch
            epoch_metrics = self.train_epoch(
                dataloader,
                gradient_accumulation_steps=gradient_accumulation_steps,
                max_steps=remaining_steps,
            )
            epoch_metrics["epoch"] = epoch + 1
            epoch_metrics["global_step"] = self.global_step
            all_metrics.append(epoch_metrics)

            total_steps = self.global_step

            # Save checkpoint if requested
            if checkpoint_dir and save_interval and total_steps % save_interval == 0:
                checkpoint_path = Path(checkpoint_dir) / f"checkpoint-{total_steps}"
                self.save_checkpoint(str(checkpoint_path))

        return all_metrics

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint.

        Args:
            path: Directory path to save checkpoint
        """
        checkpoint_dir = Path(path)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model state
        model_path = checkpoint_dir / "model.pt"
        torch.save(self.model.state_dict(), model_path)

        # Save config if available
        if hasattr(self.model, "config"):
            config_path = checkpoint_dir / "model_config.json"
            import json

            with open(config_path, "w") as f:
                json.dump(self.model.config, f, indent=2)

        # Save optimizer state
        optimizer_path = checkpoint_dir / "optimizer.pt"
        torch.save(self.optimizer.state_dict(), optimizer_path)

        # Save training state
        trainer_state = {
            "global_step": self.global_step,
            "beta": self.beta,
            "config": self.config,
        }
        trainer_path = checkpoint_dir / "trainer_state.pt"
        torch.save(trainer_state, trainer_path)

        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint.

        Args:
            path: Directory path to checkpoint
        """
        checkpoint_dir = Path(path)

        # Load model state
        model_path = checkpoint_dir / "model.pt"
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        # Load optimizer state
        optimizer_path = checkpoint_dir / "optimizer.pt"
        self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))

        # Load trainer state
        trainer_path = checkpoint_dir / "trainer_state.pt"
        trainer_state = torch.load(trainer_path, map_location=self.device)
        self.global_step = trainer_state["global_step"]
        self.beta = trainer_state.get("beta", self.beta)
        self.config.update(trainer_state.get("config", {}))

        logger.info(f"Loaded checkpoint from {path}, global_step={self.global_step}")

    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        """Evaluate the model on a validation set.

        Args:
            dataloader: DataLoader with preference pairs

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()

        total_metrics = {
            "loss": 0.0,
            "accuracy": 0.0,
            "chosen_logprob": 0.0,
            "rejected_logprob": 0.0,
            "logits_diff": 0.0,
        }
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                chosen_input_ids = batch["chosen_input_ids"].to(self.device)
                rejected_input_ids = batch["rejected_input_ids"].to(self.device)
                chosen_attention_mask = batch.get("chosen_attention_mask")
                if chosen_attention_mask is not None:
                    chosen_attention_mask = chosen_attention_mask.to(self.device)
                rejected_attention_mask = batch.get("rejected_attention_mask")
                if rejected_attention_mask is not None:
                    rejected_attention_mask = rejected_attention_mask.to(self.device)

                _, metrics = self.compute_dpo_loss(
                    chosen_input_ids,
                    rejected_input_ids,
                    chosen_attention_mask,
                    rejected_attention_mask,
                )

                for key in total_metrics:
                    if key in metrics:
                        total_metrics[key] += metrics[key]
                num_batches += 1

        # Average metrics
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        logger.info(f"Evaluation complete: {avg_metrics}")

        return avg_metrics

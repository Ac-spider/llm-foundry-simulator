"""SFT (Supervised Fine-Tuning) Trainer."""
from __future__ import annotations

import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..common.model import BasicsTransformerLM
    from ..backends.inference import InferenceBackend


class SFTDataset(Dataset):
    """Dataset for SFT training with prompt-response pairs.

    Loads JSONL data with {"prompt": str, "response": str} format.
    Each sample is tokenized and processed to compute loss only on response tokens.
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 2048, pad_token_id: int = 0):
        """Initialize SFT dataset.

        Args:
            data_path: Path to JSONL file with {"prompt": str, "response": str} format
            tokenizer: Tokenizer (BPETokenizer or HuggingFace tokenizer)
            max_length: Maximum sequence length
            pad_token_id: ID to use for padding
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = pad_token_id

        # Load data
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line.strip()))

    def _encode_text(self, text: str) -> list[int]:
        """Encode text using tokenizer (handles both HF and BPE tokenizer)."""
        # Check if tokenizer has encode method with add_special_tokens param
        import inspect
        sig = inspect.signature(self.tokenizer.encode)
        if 'add_special_tokens' in sig.parameters:
            # HuggingFace tokenizer
            return self.tokenizer.encode(text, add_special_tokens=False)
        else:
            # BPETokenizer - simple encode
            return self.tokenizer.encode(text)

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.data)

    def __getitem__(self, idx) -> dict:
        """Get a single sample.

        Returns dict with:
            - input_ids: tensor of token IDs (prompt + response, shifted)
            - labels: tensor of target token IDs
            - response_mask: tensor (1 for response tokens, 0 for prompt/padding)
        """
        sample = self.data[idx]
        prompt = sample["prompt"]
        response = sample["response"]

        # Tokenize prompt and response separately
        prompt_ids = self._encode_text(prompt)
        response_ids = self._encode_text(response)

        # Concatenate: prompt + response
        concat_ids = prompt_ids + response_ids

        # Truncate if exceeds max_length, ensuring at least 1 response token survives
        if len(prompt_ids) >= self.max_length:
            prompt_ids = prompt_ids[:self.max_length - 1]
        concat_ids = (prompt_ids + response_ids)[:self.max_length]

        # Shift for next-token prediction: input[i] predicts label[i]
        # input_ids = concat[:-1], labels = concat[1:]
        input_ids = concat_ids[:-1]
        labels = concat_ids[1:]

        # response_mask: 0 for prompt positions (after shift), 1 for response
        # After shift: prompt contributes (len(prompt_ids) - 1) positions
        # Response contributes len(response_ids) positions
        prompt_len_after_shift = len(prompt_ids) - 1
        response_len = len(response_ids)
        response_mask = [0] * prompt_len_after_shift + [1] * response_len

        # Truncate mask to match labels length
        response_mask = response_mask[:len(labels)]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "response_mask": torch.tensor(response_mask, dtype=torch.long),
        }


def collate_fn(batch: list[dict], pad_token_id: int = 0) -> dict[str, torch.Tensor]:
    """Collate function for batching SFT samples.

    Pads sequences to max length in batch.
    """
    max_len = max(len(sample["input_ids"]) for sample in batch)

    padded_input_ids = []
    padded_labels = []
    padded_masks = []

    for sample in batch:
        pad_len = max_len - len(sample["input_ids"])

        # Pad with pad_token_id (for input_ids and labels)
        padded_input_ids.append(
            sample["input_ids"].tolist() + [pad_token_id] * pad_len
        )
        padded_labels.append(
            sample["labels"].tolist() + [pad_token_id] * pad_len
        )
        # Pad with 0 for response_mask (don't compute loss on padding)
        padded_masks.append(
            sample["response_mask"].tolist() + [0] * pad_len
        )

    return {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "labels": torch.tensor(padded_labels, dtype=torch.long),
        "response_mask": torch.tensor(padded_masks, dtype=torch.long),
    }


class SFTTrainer:
    """Supervised Fine-Tuning Trainer.

    Attributes:
        model: The model to train
        tokenizer: Tokenizer instance
        optimizer: Optimizer instance
        device: Training device
        config: Training configuration dict
    """

    def __init__(
        self,
        model: BasicsTransformerLM,
        tokenizer,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda",
        config: dict | None = None,
    ):
        """Initialize SFT trainer.

        Args:
            model: The transformer model to train
            tokenizer: Tokenizer for encoding/decoding
            optimizer: PyTorch optimizer
            device: Device to run training on
            config: Optional training configuration dict
        """
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.device = device
        self.config = config or {}
        self.global_step = 0
        self.pad_token_id = config.get("pad_token_id", 0) if config else 0

        self.model.to(self.device)

    def _compute_loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute SFT loss (negative log likelihood on response tokens only).

        Only compute loss on tokens where response_mask == 1.

        Args:
            input_ids: Input token IDs, shape (batch_size, seq_len)
            labels: Target token IDs, shape (batch_size, seq_len)
            response_mask: Binary mask (1 for response, 0 for prompt/padding)

        Returns:
            Loss tensor (scalar)
        """
        # Forward pass
        logits = self.model(input_ids)  # (batch_size, seq_len, vocab_size)

        # Compute cross-entropy loss per token
        # Flatten for F.cross_entropy
        logits_flat = logits.view(-1, logits.size(-1))  # (batch_size * seq_len, vocab_size)
        labels_flat = labels.view(-1)  # (batch_size * seq_len,)

        # Compute loss
        loss = F.cross_entropy(
            logits_flat,
            labels_flat,
            reduction='none',
            ignore_index=self.pad_token_id
        )

        # Reshape to match mask
        loss = loss.view(labels.shape)  # (batch_size, seq_len)

        # Apply response mask (only compute loss on response tokens)
        loss = loss * response_mask.float()

        # Normalize by number of response tokens
        num_response_tokens = response_mask.sum()
        if num_response_tokens > 0:
            loss = loss.sum() / num_response_tokens
        else:
            loss = loss.sum() * 0  # No response tokens, return zero loss

        return loss

    def train_step(
        self,
        batch: dict[str, torch.Tensor],
        gradient_accumulation_steps: int = 1,
    ) -> dict[str, float]:
        """Single training step with gradient accumulation.

        Args:
            batch: Dict with input_ids, labels, response_mask tensors
            gradient_accumulation_steps: Number of steps to accumulate gradients

        Returns:
            Dict with "loss" key
        """
        # Move batch to device
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        response_mask = batch["response_mask"].to(self.device)

        # Compute loss
        loss = self._compute_loss(input_ids, labels, response_mask)

        # Scale loss for gradient accumulation
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps

        # Backward pass
        loss.backward()

        return {
            "loss": loss.item() * gradient_accumulation_steps,
        }

    def train_epoch(
        self,
        dataloader: DataLoader,
        gradient_accumulation_steps: int = 1,
        eval_fn: callable | None = None,
    ) -> dict[str, float]:
        """Train for one epoch.

        Args:
            dataloader: Training data loader
            gradient_accumulation_steps: Number of steps to accumulate gradients
            eval_fn: Optional evaluation function to call periodically

        Returns:
            Dict with training metrics (loss, etc.)
        """
        self.model.train()

        total_loss = 0.0
        num_batches = 0
        num_steps = 0

        for batch_idx, batch in enumerate(dataloader):
            # Training step
            step_metrics = self.train_step(batch, gradient_accumulation_steps)
            total_loss += step_metrics["loss"]
            num_batches += 1

            # Gradient update
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.get("max_grad_norm", 1.0)
                )

                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()

                self.global_step += 1
                num_steps += 1

        # Handle any remaining gradients
        if num_batches % gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.get("max_grad_norm", 1.0)
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.global_step += 1
            num_steps += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        return {
            "loss": avg_loss,
            "num_batches": num_batches,
            "num_steps": num_steps,
        }

    def evaluate(
        self,
        eval_dataloader: DataLoader,
        inference_backend: InferenceBackend | None = None,
    ) -> dict[str, float]:
        """Evaluate on validation set.

        If inference_backend is provided, use it for generation-based eval.
        Otherwise, just compute validation loss.

        Args:
            eval_dataloader: Evaluation data loader
            inference_backend: Optional inference backend for generation

        Returns:
            Dict with eval metrics
        """
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in eval_dataloader:
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                response_mask = batch["response_mask"].to(self.device)

                # Compute loss
                loss = self._compute_loss(input_ids, labels, response_mask)
                total_loss += loss.item()
                num_batches += 1

        metrics = {
            "eval_loss": total_loss / num_batches if num_batches > 0 else 0.0,
        }

        # TODO: Add generation-based evaluation if inference_backend is provided
        if inference_backend is not None:
            # Generation-based evaluation would go here
            pass

        return metrics

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "config": self.config,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint.

        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)

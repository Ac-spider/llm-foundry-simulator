"""
Data loading utilities for training.

Adapted from CS336 Assignment 4.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from llm_foundry.stage1_tokenize.binarize import load_binarized_file

if TYPE_CHECKING:
    import numpy.typing as npt

logger = logging.getLogger(__name__)


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a random training batch from memory-mapped dataset.

    Uses the standard "input-target offset by one" strategy for language modeling:
    - Input x is positions [i, i+context_length)
    - Target y is positions [i+1, i+1+context_length)
    So y[t] is the next token after x[t].

    Args:
        dataset: Memory-mapped token ID array, shape (total_tokens,)
        batch_size: Number of sequences in batch
        context_length: Sequence length (context window)
        device: Target device ("cpu", "cuda:0", etc.)

    Returns:
        Tuple of (x, y) tensors, each shape (batch_size, context_length)
    """
    # Random starting indices
    starting_idxs = torch.randint(len(dataset) - context_length, (batch_size,))

    # Build input sequences
    x = torch.stack([
        torch.from_numpy((dataset[i : i + context_length]).astype(np.int64))
        for i in starting_idxs
    ])

    # Build target sequences (offset by 1)
    y = torch.stack([
        torch.from_numpy((dataset[i + 1 : i + 1 + context_length]).astype(np.int64))
        for i in starting_idxs
    ])

    if "cuda" in device:
        # Use pinned memory + async transfer for better throughput
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)

    return x, y


def load_tokens(data_path: str | Path, dtype: np.dtype = np.uint16) -> npt.NDArray:
    """
    Load token data as memory-mapped numpy array.

    Args:
        data_path: Path to binarized file containing token IDs
        dtype: Data type of tokens (default: uint16 for up to 65535 vocab)

    Returns:
        Memory-mapped numpy array of token IDs
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Use load_binarized_file to handle custom binary format
    tokens = load_binarized_file(data_path, mmap_mode="r")
    logger.info(f"Loaded {len(tokens):,} tokens from {data_path}")
    return tokens


def create_data_loader(
    data_path: str | Path,
    batch_size: int,
    context_length: int,
    device: str,
    num_batches: int | None = None,
):
    """
    Create a simple data loader that yields batches indefinitely.

    Args:
        data_path: Path to token data file
        batch_size: Batch size
        context_length: Sequence length
        device: Device to place tensors on
        num_batches: If specified, stop after this many batches

    Yields:
        Tuples of (x, y) tensors
    """
    tokens = load_tokens(data_path)
    batch_count = 0

    while num_batches is None or batch_count < num_batches:
        x, y = get_batch(tokens, batch_size, context_length, device)
        yield x, y
        batch_count += 1

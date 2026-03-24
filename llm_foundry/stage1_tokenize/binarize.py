"""
Binarization module for tokenizing text and saving as binary format.

Converts text files to token ID sequences stored as binary files for efficient
memory-mapped access during training.
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from typing import Protocol

    class TokenizerProtocol(Protocol):
        """Protocol for tokenizer with encode method."""

        def encode(self, text: str) -> list[int]:
            """Encode text to token IDs."""
            ...


logger = logging.getLogger(__name__)

# Magic number for binarized file format validation
# Used to identify valid binarized token files
MAGIC_NUMBER = b"LLMFBIN"  # LLM Foundry Binary
VERSION = 1


@dataclass
class BinarizeConfig:
    """Configuration for binarization process.

    Attributes:
        input_path: Path to input text file
        output_path: Path to output binary file
        dtype: NumPy dtype for token storage (default: uint16 for vocab < 65536)
        append_eos: Whether to append EOS token after each document
        eos_token_id: Token ID to use as EOS (default: 50256 for GPT-2)
        chunk_size: Number of lines to process before writing to disk
    """

    input_path: str = ""
    output_path: str = ""
    dtype: str = "uint16"
    append_eos: bool = True
    eos_token_id: int = 50256
    chunk_size: int = 10000

    def __post_init__(self):
        """Validate configuration."""
        if not self.input_path:
            raise ValueError("input_path is required")
        if not self.output_path:
            raise ValueError("output_path is required")
        if self.dtype not in ("uint16", "uint32"):
            raise ValueError(f"Unsupported dtype: {self.dtype}. Use 'uint16' or 'uint32'")


def binarize_file(
    tokenizer: TokenizerProtocol,
    config: BinarizeConfig,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, int]:
    """Binarize a text file into token IDs and save as binary format.

    Processes text file line by line, tokenizes each line using the provided
    tokenizer, optionally appends EOS token, and writes token IDs to a binary
    file in uint16 format. Uses memory-efficient chunked processing.

    Args:
        tokenizer: Tokenizer with encode(text: str) -> list[int] method
        config: BinarizeConfig with input/output paths and options
        progress_callback: Optional callback(current_lines, total_lines) for progress

    Returns:
        Dictionary with statistics:
            - total_lines: Number of lines processed
            - total_tokens: Total number of tokens written
            - avg_tokens_per_line: Average tokens per line

    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If tokenizer fails to encode

    Example:
        >>> from llm_foundry.stage1_tokenize.tokenizer import BPETokenizer
        >>> tokenizer = BPETokenizer.from_files("vocab.pkl", "merges.pkl")
        >>> config = BinarizeConfig(
        ...     input_path="corpus.txt",
        ...     output_path="corpus.bin",
        ...     append_eos=True,
        ... )
        >>> stats = binarize_file(tokenizer, config)
        >>> print(f"Processed {stats['total_lines']} lines, {stats['total_tokens']} tokens")
    """
    input_path = Path(config.input_path)
    output_path = Path(config.output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Count total lines for progress reporting
    logger.info(f"Counting lines in {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    logger.info(f"Found {total_lines:,} lines to process")

    # Determine numpy dtype
    np_dtype = np.dtype(config.dtype)

    # Statistics
    total_tokens = 0
    processed_lines = 0
    chunk_tokens: list[int] = []

    # Write binary file with header
    logger.info(f"Binarizing {input_path} -> {output_path}")

    with (
        open(input_path, "r", encoding="utf-8") as f_in,
        open(output_path, "wb") as f_out,
    ):
        # Write header: magic number (7 bytes) + version (1 byte) + dtype (1 byte)
        # dtype: 0 = uint16, 1 = uint32
        dtype_code = 0 if config.dtype == "uint16" else 1
        header = MAGIC_NUMBER + struct.pack("<BB", VERSION, dtype_code)
        f_out.write(header)

        # Process lines
        for line_num, line in enumerate(f_in, 1):
            line = line.rstrip("\n\r")

            # Skip empty lines
            if not line.strip():
                continue

            # Tokenize
            try:
                token_ids = tokenizer.encode(line)
            except Exception as e:
                logger.warning(f"Failed to encode line {line_num}: {e}")
                continue

            # Append EOS if configured
            if config.append_eos:
                token_ids = token_ids + [config.eos_token_id]

            # Add to chunk
            chunk_tokens.extend(token_ids)
            processed_lines += 1

            # Write chunk when it reaches chunk_size
            if len(chunk_tokens) >= config.chunk_size:
                arr = np.array(chunk_tokens, dtype=np_dtype)
                f_out.write(arr.tobytes())
                total_tokens += len(chunk_tokens)
                chunk_tokens = []

            # Progress callback
            if progress_callback and line_num % 1000 == 0:
                progress_callback(line_num, total_lines)

        # Write remaining tokens
        if chunk_tokens:
            arr = np.array(chunk_tokens, dtype=np_dtype)
            f_out.write(arr.tobytes())
            total_tokens += len(chunk_tokens)

    # Calculate statistics
    avg_tokens = total_tokens / processed_lines if processed_lines > 0 else 0

    stats = {
        "total_lines": processed_lines,
        "total_tokens": total_tokens,
        "avg_tokens_per_line": round(avg_tokens, 2),
    }

    logger.info(
        f"Binarization complete: {stats['total_lines']:,} lines, "
        f"{stats['total_tokens']:,} tokens (avg {stats['avg_tokens_per_line']:.1f}/line)"
    )

    return stats


def load_binarized_file(
    bin_path: str | Path,
    mmap_mode: str = "r",
) -> np.ndarray:
    """Load a binarized token file as numpy array.

    Supports memory-mapped access for efficient random access to large
    datasets without loading everything into RAM.

    Args:
        bin_path: Path to binarized file
        mmap_mode: Memory mapping mode ('r' = read-only, 'r+' = read-write)

    Returns:
        NumPy array of token IDs (memory-mapped if mmap_mode is set)

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file has invalid format

    Example:
        >>> tokens = load_binarized_file("corpus.bin", mmap_mode="r")
        >>> print(f"Dataset contains {len(tokens):,} tokens")
        >>> # Access random slice without loading entire file
        >>> sample = tokens[1000000:1000100]
    """
    bin_path = Path(bin_path)

    if not bin_path.exists():
        raise FileNotFoundError(f"Binarized file not found: {bin_path}")

    # Read header to determine dtype
    with open(bin_path, "rb") as f:
        magic = f.read(len(MAGIC_NUMBER))
        if magic != MAGIC_NUMBER:
            raise ValueError(f"Invalid binarized file format: {bin_path}")

        version, dtype_code = struct.unpack("<BB", f.read(2))
        if version != VERSION:
            raise ValueError(f"Unsupported file version: {version}")

        dtype = np.uint16 if dtype_code == 0 else np.uint32
        header_size = len(MAGIC_NUMBER) + 2

    # Calculate number of tokens
    file_size = bin_path.stat().st_size
    data_size = file_size - header_size
    num_tokens = data_size // dtype().itemsize

    logger.debug(f"Loading {num_tokens:,} tokens from {bin_path} (dtype={dtype.__name__})")

    # Memory-map the file
    if mmap_mode:
        # Use numpy.memmap for memory-mapped access
        tokens = np.memmap(bin_path, dtype=dtype, mode=mmap_mode, offset=header_size)
    else:
        # Load entire file into memory
        with open(bin_path, "rb") as f:
            f.seek(header_size)
            tokens = np.fromfile(f, dtype=dtype)

    return tokens


def binarize_file_simple(
    tokenizer: TokenizerProtocol,
    input_path: str | Path,
    output_path: str | Path,
    dtype: np.dtype = np.uint16,
    append_eos: bool = True,
    eos_token_id: int = 50256,
) -> int:
    """Simple version of binarize_file without progress tracking.

    Loads entire file into memory and writes at once. Suitable for
    smaller files where memory usage is not a concern.

    Args:
        tokenizer: Tokenizer with encode method
        input_path: Path to input text file
        output_path: Path to output binary file
        dtype: NumPy dtype for tokens (uint16 or uint32)
        append_eos: Whether to append EOS token after each line
        eos_token_id: Token ID for EOS

    Returns:
        Total number of tokens written
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read and tokenize all lines
    all_token_ids: list[int] = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if not line.strip():
                continue

            token_ids = tokenizer.encode(line)
            if append_eos:
                token_ids = token_ids + [eos_token_id]
            all_token_ids.extend(token_ids)

    # Write to binary file
    arr = np.array(all_token_ids, dtype=dtype)

    with open(output_path, "wb") as f:
        # Write simple header
        f.write(MAGIC_NUMBER)
        dtype_code = 0 if dtype == np.uint16 else 1
        f.write(struct.pack("<BB", VERSION, dtype_code))
        f.write(arr.tobytes())

    return len(all_token_ids)

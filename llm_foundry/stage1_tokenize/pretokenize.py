"""
Pre-tokenization module for Stage 1 Tokenization.

This module provides pre-tokenization functionality that splits raw text into
"words" before BPE tokenization is applied. This follows the GPT-2 approach
where text is first segmented using a regex pattern before byte-pair encoding.
"""

import os
import regex as re
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator, Union


# ─────────────────────────────────────────
# GPT-2 Pre-tokenization Regex
# ─────────────────────────────────────────
# This regex pattern is used to split text into "words" before BPE processing.
# It handles:
#   - English contractions: 's, 'd, 'm, 't, 'll, 've, 're
#   - Words with optional leading space: ?\p{L}+
#   - Numbers with optional leading space: ?\p{N}+
#   - Punctuation/symbols with optional leading space: ?[^\s\p{L}\p{N}]+
#   - Trailing whitespace: \s+(?!\S)
#   - Other whitespace sequences: \s+
#
# Purpose: Prevents cross-word boundary merges during BPE training,
# ensuring contractions, numbers, and punctuation remain distinct tokens.
# ─────────────────────────────────────────
PRETOKENIZE_REGEX = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def pretokenize(text: str) -> list[str]:
    """
    Pre-tokenize raw text using GPT-2 regex pattern.

    Purpose: Split text into "words" before BPE tokenization. This ensures
    that contractions, numbers, and punctuation are treated as separate units
    and won't be merged across word boundaries during BPE training.

    Key concept: Pre-tokenization creates boundaries that BPE cannot cross,
    preserving semantic units like contractions ('ll, 're) and keeping
    numbers/punctuation distinct from words.

    Args:
        text (str): Raw input text to pre-tokenize.

    Returns:
        list[str]: List of pre-token strings ("words") extracted from text.
            Each element is a substring matching the GPT-2 regex pattern.
            Example: "Hello world!" -> ["Hello", " world", "!"]

    Example:
        >>> pretokenize("Hello world!")
        ['Hello', ' world', '!']
        >>> pretokenize("It's a test.")
        ["It", "'s", " a", " test", "."]
    """
    return PRETOKENIZE_REGEX.findall(text)


def pretokenize_file(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    special_tokens: list[str] | None = None,
) -> int:
    """
    Pre-tokenize an entire file and write results to output.

    Purpose: Process large text files by reading them, splitting on special
    tokens (if provided), pre-tokenizing each segment, and writing the
    pre-tokenized output. This prepares text for BPE training.

    Key concept: File-level pre-tokenization with special token handling
    ensures document boundaries are respected (prevents cross-document merges).

    Args:
        input_path (Union[str, Path]): Path to input text file (UTF-8 encoded).
        output_path (Union[str, Path]): Path to output file for pre-tokenized text.
        special_tokens (list[str] | None): Optional list of special tokens to use
            as document separators. Text is split on these tokens before
            pre-tokenization. Defaults to None (no splitting).

    Returns:
        int: Total number of pre-tokens extracted from the file.

    Raises:
        FileNotFoundError: If input_path does not exist.
        PermissionError: If unable to read input or write output.

    Example:
        >>> pretokenize_file("input.txt", "output.txt", ["<|endoftext|>"])
        15000  # Total pre-tokens extracted
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    total_tokens = 0

    with open(input_path, "r", encoding="utf-8", errors="ignore") as f_in:
        text = f_in.read()

    # Split on special tokens if provided (prevents cross-document merges)
    if special_tokens:
        escaped_specials = [re.escape(st) for st in special_tokens]
        split_pat = "|".join(escaped_specials)
        parts = re.split(split_pat, text)
    else:
        parts = [text]

    # Pre-tokenize each part and write to output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f_out:
        for part in parts:
            tokens = pretokenize(part)
            total_tokens += len(tokens)
            # Write each pre-token on its own line for clarity
            for token in tokens:
                f_out.write(token + "\n")

    return total_tokens


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Find byte boundaries for chunking a file, aligned to special tokens.

    Purpose: Divide a large file into chunks suitable for parallel processing,
    ensuring boundaries align with special tokens (e.g., <|endoftext|>) to
    prevent splitting documents across chunks.

    Key concept: File chunking with boundary alignment - each chunk starts
    and ends at document boundaries, enabling safe parallel processing.

    Args:
        file (BinaryIO): Binary file object opened for reading.
        desired_num_chunks (int): Desired number of chunks (typically CPU count).
        split_special_token (bytes): Special token bytes to align boundaries to.
            Example: b"<|endoftext|>"

    Returns:
        list[int]: Sorted list of byte offsets marking chunk boundaries.
            Adjacent pairs define [start, end) ranges for each chunk.
            May return fewer chunks if boundaries overlap.

    Raises:
        AssertionError: If split_special_token is not bytes type.

    Example:
        >>> with open("corpus.txt", "rb") as f:
        ...     boundaries = find_chunk_boundaries(f, 4, b"<|endoftext|>")
        >>> boundaries
        [0, 1048576, 2097152, 3145728, 4000000]  # Byte offsets
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial uniform boundary guesses
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size  # Last boundary is always EOF

    mini_chunk_size = 4096  # Read 4KB at a time

    # Adjust each intermediate boundary to align with special token
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)

        while True:
            mini_chunk = file.read(mini_chunk_size)

            # EOF reached - boundary goes to end of file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Look for special token in this mini-chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break

            initial_position += mini_chunk_size

    # Remove duplicates and sort
    return sorted(set(chunk_boundaries))


@dataclass
class PretokenizeConfig:
    """
    Configuration for pre-tokenization operations.

    Purpose: Encapsulate all parameters needed for pre-tokenization,
    enabling consistent configuration across different stages of the
    tokenization pipeline.

    Attributes:
        special_tokens (list[str]): Special tokens to use as document
            boundaries. These are removed from text before pre-tokenization
            and prevent cross-document BPE merges. Defaults to ["<|endoftext|>"].
        num_workers (int): Number of parallel workers for file processing.
            Defaults to -1 (use all CPU cores).
        output_format (str): Format for output files. "lines" writes each
            pre-token on its own line. "json" writes as JSON array.
            Defaults to "lines".

    Example:
        >>> config = PretokenizeConfig(
        ...     special_tokens=["<|endoftext|>", "<|padding|>"],
        ...     num_workers=4,
        ...     output_format="lines",
        ... )
        >>> config.get_effective_num_workers()
        4
    """

    special_tokens: list[str] = None
    num_workers: int = -1  # -1 means use all CPU cores
    output_format: str = "lines"  # "lines" or "json"

    def __post_init__(self):
        """Set default values after initialization."""
        if self.special_tokens is None:
            self.special_tokens = ["<|endoftext|>"]

    def get_effective_num_workers(self) -> int:
        """
        Get the effective number of workers.

        Returns:
            int: Number of workers to use. If num_workers is -1,
                returns the number of CPU cores.
        """
        if self.num_workers == -1:
            return os.cpu_count() or 1
        return max(1, self.num_workers)

    def get_split_token_bytes(self) -> bytes:
        """
        Get the primary special token as bytes for file chunking.

        Returns:
            bytes: The first special token encoded as UTF-8 bytes.
                Used by find_chunk_boundaries for alignment.
        """
        if self.special_tokens:
            return self.special_tokens[0].encode("utf-8")
        return b"<|endoftext|>"

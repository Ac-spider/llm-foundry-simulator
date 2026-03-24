"""
Data Pipeline module for Stage 4: Data quality filtering.

This module provides DataPipeline class for filtering and deduplicating
text data using Gopher quality rules, length filtering, and SHA256 deduplication.
Adapted from CS336 Assignment4.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from typing import Protocol

    class TextSourceProtocol(Protocol):
        """Protocol for text source with read method."""

        def read(self) -> str:
            """Read and return text content."""
            ...


logger = logging.getLogger(__name__)


@dataclass
class DataPipelineConfig:
    """Configuration for DataPipeline.

    Attributes:
        min_length: Minimum document length in characters
        max_length: Maximum document length in characters
        enable_gopher_filter: Whether to apply Gopher quality filter
        enable_deduplication: Whether to apply SHA256 deduplication
        dedup_key: Key function for deduplication (default: hash full text)
    """

    min_length: int = 100
    max_length: int = 100000
    enable_gopher_filter: bool = True
    enable_deduplication: bool = True
    dedup_key: Callable[[str], str] | None = None

    def __post_init__(self):
        """Validate configuration."""
        if self.min_length < 0:
            raise ValueError("min_length must be non-negative")
        if self.max_length <= self.min_length:
            raise ValueError("max_length must be greater than min_length")


def gopher_quality_filter(text: str) -> bool:
    """Apply Gopher quality filter rules.

    Implements DeepMind Gopher paper heuristic rules for text quality filtering:

    Rule 1: Word count between 50 and 100,000
        - Too short (< 50): navigation, ads, error fragments
        - Too long (> 100k): concatenated garbage or data dumps

    Rule 2: Mean word length between 3 and 10 characters
        - Too short: single-letter abbreviations or symbols
        - Too long: unspaced URLs or random strings

    Rule 3: Ellipsis line ratio <= 30%
        - High ellipsis density indicates truncated summaries or low-quality lists

    Rule 4: Alpha word ratio >= 80%
        - Low alpha ratio indicates excessive numbers, symbols, or code

    Args:
        text: Input text to filter

    Returns:
        True if text passes all quality checks, False to discard

    Reference:
        "Scaling Language Models: Methods, Analysis & Insights from Training Gopher"
        https://arxiv.org/abs/2112.11446
    """
    words = text.split()
    num_words = len(words)

    # Rule 1: Word count between 50 and 100,000
    if num_words < 50 or num_words > 100000:
        return False

    # Rule 2: Mean word length between 3 and 10 characters
    total_chars = sum(len(w) for w in words)
    mean_word_length = total_chars / num_words if num_words > 0 else 0
    if mean_word_length < 3 or mean_word_length > 10:
        return False

    # Rule 3: Ellipsis line ratio <= 30%
    lines = text.splitlines()
    if len(lines) > 0:
        ellipsis_lines = sum(1 for line in lines if line.strip().endswith("..."))
        if ellipsis_lines / len(lines) > 0.3:
            return False

    # Rule 4: Alpha word ratio >= 80%
    alpha_words = sum(1 for word in words if any(c.isalpha() for c in word))
    if num_words > 0 and (alpha_words / num_words) < 0.8:
        return False

    return True


def length_filter(text: str, min_length: int = 100, max_length: int = 100000) -> bool:
    """Filter text by character length.

    Args:
        text: Input text to filter
        min_length: Minimum character length (inclusive)
        max_length: Maximum character length (inclusive)

    Returns:
        True if text length is within bounds, False otherwise
    """
    text_len = len(text)
    return min_length <= text_len <= max_length


class DataPipeline:
    """Data quality pipeline for filtering and deduplicating text data.

    This class provides a configurable pipeline for:
    1. Length-based filtering (min/max character count)
    2. Gopher quality filtering (heuristic rules)
    3. SHA256-based deduplication

    The pipeline processes documents one at a time and maintains a set of
    seen hashes for deduplication.

    Example:
        >>> config = DataPipelineConfig(
        ...     min_length=100,
        ...     max_length=50000,
        ...     enable_gopher_filter=True,
        ...     enable_deduplication=True,
        ... )
        >>> pipeline = DataPipeline(config)
        >>>
        >>> for doc in documents:
        ...     if pipeline.process(doc):
        ...         # Document passed all filters
        ...         print(f"Kept: {doc[:50]}...")
        ...
        >>> stats = pipeline.get_stats()
        >>> print(f"Kept {stats['kept']}/{stats['total']} documents")
    """

    def __init__(self, config: DataPipelineConfig | None = None):
        """Initialize the pipeline with configuration.

        Args:
            config: Pipeline configuration. Uses defaults if not provided.
        """
        self.config = config or DataPipelineConfig()
        self.seen_hashes: set[str] = set()
        self.stats = {
            "total": 0,
            "kept": 0,
            "filtered_length": 0,
            "filtered_gopher": 0,
            "filtered_duplicate": 0,
        }

    def process(self, text: str) -> bool:
        """Process a single document through the pipeline.

        Applies filters in order:
        1. Length filter
        2. Gopher quality filter (if enabled)
        3. Deduplication (if enabled)

        Args:
            text: Document text to process

        Returns:
            True if document passes all filters, False otherwise
        """
        self.stats["total"] += 1

        # Filter 1: Length check
        if not length_filter(text, self.config.min_length, self.config.max_length):
            self.stats["filtered_length"] += 1
            return False

        # Filter 2: Gopher quality filter
        if self.config.enable_gopher_filter:
            if not gopher_quality_filter(text):
                self.stats["filtered_gopher"] += 1
                return False

        # Filter 3: Deduplication
        if self.config.enable_deduplication:
            # Use custom dedup key if provided, otherwise hash full text
            if self.config.dedup_key:
                key = self.config.dedup_key(text)
            else:
                key = hashlib.sha256(text.encode("utf-8")).hexdigest()

            if key in self.seen_hashes:
                self.stats["filtered_duplicate"] += 1
                return False
            self.seen_hashes.add(key)

        self.stats["kept"] += 1
        return True

    def process_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        doc_separator: str = "\n\n",
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[str, int]:
        """Process a file through the pipeline.

        Reads documents from input file, applies all filters, and writes
        kept documents to output file.

        Args:
            input_path: Path to input text file
            output_path: Path to output text file
            doc_separator: Separator between documents (default: double newline)
            progress_callback: Optional callback(current, total) for progress

        Returns:
            Dictionary with processing statistics
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Stream input file to avoid loading entire corpus into memory
        logger.info(f"Processing {input_path} (streaming)...")
        total_docs = 0
        first_kept = True
        pending = ""

        with open(input_path, "r", encoding="utf-8") as fin, \
             open(output_path, "w", encoding="utf-8") as fout:
            for chunk in iter(lambda: fin.read(65536), ""):
                pending += chunk
                parts = pending.split(doc_separator)
                # Last part may be incomplete — hold it for next iteration
                pending = parts[-1]
                for doc_raw in parts[:-1]:
                    doc = doc_raw.strip()
                    if not doc:
                        continue
                    total_docs += 1
                    if self.process(doc):
                        if not first_kept:
                            fout.write(doc_separator)
                        fout.write(doc)
                        first_kept = False
                    if progress_callback and total_docs % 100 == 0:
                        progress_callback(total_docs, total_docs)

            # Handle remaining content after EOF
            if pending.strip():
                doc = pending.strip()
                total_docs += 1
                if self.process(doc):
                    if not first_kept:
                        fout.write(doc_separator)
                    fout.write(doc)

        return self.get_stats()

    def get_stats(self) -> dict[str, int]:
        """Get current pipeline statistics.

        Returns:
            Dictionary with keys:
                - total: Total documents processed
                - kept: Documents that passed all filters
                - filtered_length: Documents filtered by length
                - filtered_gopher: Documents filtered by Gopher rules
                - filtered_duplicate: Documents filtered as duplicates
        """
        return self.stats.copy()

    def reset(self) -> None:
        """Reset the pipeline state.

        Clears deduplication history and statistics.
        """
        self.seen_hashes.clear()
        self.stats = {
            "total": 0,
            "kept": 0,
            "filtered_length": 0,
            "filtered_gopher": 0,
            "filtered_duplicate": 0,
        }

    def __len__(self) -> int:
        """Return number of unique documents seen (for deduplication)."""
        return len(self.seen_hashes)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"DataPipeline("
            f"min_length={self.config.min_length}, "
            f"max_length={self.config.max_length}, "
            f"gopher={self.config.enable_gopher_filter}, "
            f"dedup={self.config.enable_deduplication}, "
            f"stats={self.stats}"
            f")"
        )

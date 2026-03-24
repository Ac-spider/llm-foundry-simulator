"""Stage 4: Data Quality Pipeline Module.

This module provides data quality filtering and deduplication capabilities for LLM
training data preparation. It implements Gopher quality rules and SHA256-based
deduplication following DeepMind's data processing pipeline.

Key Features:
- Gopher quality filtering: Heuristic rules for text quality assessment
- Length-based filtering: Configurable min/max character limits
- SHA256 deduplication: Exact duplicate detection and removal
- File-level processing: Process entire datasets with progress tracking
- Statistics tracking: Detailed filtering metrics and reporting

Core Classes:
    DataPipeline: Main pipeline class for filtering and deduplication
    DataPipelineConfig: Configuration for pipeline behavior

Filter Functions:
    gopher_quality_filter: Apply Gopher paper quality heuristics
        - Word count check (50-100,000 words)
        - Mean word length check (3-10 characters)
        - Ellipsis line ratio check (< 30%)
        - Alpha word ratio check (> 80%)
    length_filter: Simple character length-based filtering

Example:
    >>> from llm_foundry.stage4_data import DataPipeline, DataPipelineConfig
    >>> config = DataPipelineConfig(
    ...     min_length=100,
    ...     max_length=50000,
    ...     enable_gopher_filter=True,
    ...     enable_deduplication=True,
    ... )
    >>> pipeline = DataPipeline(config)
    >>> stats = pipeline.process_file("input.txt", "output.txt")
    >>> print(f"Kept {stats['kept']}/{stats['total']} documents")

Reference:
    "Scaling Language Models: Methods, Analysis & Insights from Training Gopher"
    https://arxiv.org/abs/2112.11446
"""

from .pipeline import (
    DataPipeline,
    DataPipelineConfig,
    gopher_quality_filter,
    length_filter,
)

__all__ = [
    "DataPipeline",
    "DataPipelineConfig",
    "gopher_quality_filter",
    "length_filter",
]

"""
Unified hashing utilities for reproducibility.

All modules use SHA256 for consistent hashing of configs, data, and model states.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def compute_config_hash(config: dict[str, Any]) -> str:
    """
    Compute SHA256 hash of a configuration dictionary.

    The config is serialized to a canonical JSON string (sorted keys, no whitespace)
    before hashing to ensure consistent results regardless of dict ordering.

    Args:
        config: Configuration dictionary with serializable values

    Returns:
        Hexadecimal hash string (64 characters)

    Example:
        >>> config = {"lr": 0.001, "batch_size": 32}
        >>> compute_config_hash(config)
        'a3f5c8...'
    """
    # Serialize to canonical JSON (sorted keys, no extra whitespace)
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_file_hash(file_path: str | Path) -> str:
    """
    Compute SHA256 hash of a file.

    Reads the file in chunks to handle large files efficiently.

    Args:
        file_path: Path to the file to hash

    Returns:
        Hexadecimal hash string (64 characters)

    Raises:
        FileNotFoundError: If the file does not exist
        IOError: If the file cannot be read
    """
    file_path = Path(file_path)
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def compute_string_hash(text: str) -> str:
    """
    Compute SHA256 hash of a string.

    Args:
        text: String to hash

    Returns:
        Hexadecimal hash string (64 characters)
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_bytes_hash(data: bytes) -> str:
    """
    Compute SHA256 hash of raw bytes.

    Args:
        data: Bytes to hash

    Returns:
        Hexadecimal hash string (64 characters)
    """
    return hashlib.sha256(data).hexdigest()


def verify_config_hash(config: dict[str, Any], expected_hash: str) -> bool:
    """
    Verify that a config matches an expected hash.

    Args:
        config: Configuration dictionary
        expected_hash: Expected SHA256 hash

    Returns:
        True if the computed hash matches expected_hash
    """
    return compute_config_hash(config) == expected_hash


def verify_file_hash(file_path: str | Path, expected_hash: str) -> bool:
    """
    Verify that a file matches an expected hash.

    Args:
        file_path: Path to the file
        expected_hash: Expected SHA256 hash

    Returns:
        True if the computed hash matches expected_hash
    """
    return compute_file_hash(file_path) == expected_hash

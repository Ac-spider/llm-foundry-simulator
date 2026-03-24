"""Tests for hashing utilities."""

import json
import tempfile
from pathlib import Path

from llm_foundry.common.hashing import (
    compute_bytes_hash,
    compute_config_hash,
    compute_file_hash,
    compute_string_hash,
    verify_config_hash,
    verify_file_hash,
)


def test_compute_config_hash():
    """Test config hash computation."""
    config = {"lr": 0.001, "batch_size": 32}
    hash1 = compute_config_hash(config)

    # Same config should give same hash
    hash2 = compute_config_hash(config)
    assert hash1 == hash2

    # Different config should give different hash
    config2 = {"lr": 0.002, "batch_size": 32}
    hash3 = compute_config_hash(config2)
    assert hash1 != hash3

    # Order shouldn't matter (sorted keys)
    config3 = {"batch_size": 32, "lr": 0.001}
    hash4 = compute_config_hash(config3)
    assert hash1 == hash4


def test_compute_config_hash_length():
    """Test that hash has correct length (SHA256 = 64 hex chars)."""
    config = {"test": "value"}
    hash_str = compute_config_hash(config)
    assert len(hash_str) == 64


def test_compute_string_hash():
    """Test string hash computation."""
    hash1 = compute_string_hash("hello")
    hash2 = compute_string_hash("hello")
    hash3 = compute_string_hash("world")

    assert hash1 == hash2
    assert hash1 != hash3
    assert len(hash1) == 64


def test_compute_bytes_hash():
    """Test bytes hash computation."""
    hash1 = compute_bytes_hash(b"hello")
    hash2 = compute_bytes_hash(b"hello")
    hash3 = compute_bytes_hash(b"world")

    assert hash1 == hash2
    assert hash1 != hash3


def test_compute_file_hash():
    """Test file hash computation."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("test content")
        temp_path = f.name

    hash1 = compute_file_hash(temp_path)
    hash2 = compute_file_hash(temp_path)

    assert hash1 == hash2
    assert len(hash1) == 64

    # Clean up
    Path(temp_path).unlink()


def test_verify_config_hash():
    """Test config hash verification."""
    config = {"lr": 0.001}
    hash_str = compute_config_hash(config)

    assert verify_config_hash(config, hash_str) is True
    assert verify_config_hash(config, "wrong_hash") is False


def test_verify_file_hash():
    """Test file hash verification."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("test content")
        temp_path = f.name

    hash_str = compute_file_hash(temp_path)

    assert verify_file_hash(temp_path, hash_str) is True
    assert verify_file_hash(temp_path, "wrong_hash") is False

    # Clean up
    Path(temp_path).unlink()

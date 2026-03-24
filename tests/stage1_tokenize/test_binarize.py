"""
Tests for binarize module.

Tests the binarization process: converting text to token IDs and saving
as binary format with memory-mapped access.
"""

from __future__ import annotations

import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

from llm_foundry.stage1_tokenize.binarize import (
    MAGIC_NUMBER,
    VERSION,
    BinarizeConfig,
    binarize_file,
    binarize_file_simple,
    load_binarized_file,
)


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self, vocab: dict[str, int] | None = None):
        """Initialize with optional vocabulary mapping."""
        self.vocab = vocab or {}
        self.call_count = 0

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs (simple word-based encoding)."""
        self.call_count += 1
        if self.vocab:
            # Use vocabulary if provided
            tokens = []
            for word in text.split():
                tokens.append(self.vocab.get(word, self.vocab.get("<unk>", 0)))
            return tokens
        else:
            # Simple char-based encoding for testing
            return [ord(c) for c in text if c.isalnum()]


class TestBinarizeConfig:
    """Tests for BinarizeConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BinarizeConfig(
            input_path="input.txt",
            output_path="output.bin",
        )
        assert config.input_path == "input.txt"
        assert config.output_path == "output.bin"
        assert config.dtype == "uint16"
        assert config.append_eos is True
        assert config.eos_token_id == 50256
        assert config.chunk_size == 10000

    def test_validation_missing_input_path(self):
        """Test validation fails when input_path is empty."""
        with pytest.raises(ValueError, match="input_path is required"):
            BinarizeConfig(output_path="output.bin")

    def test_validation_missing_output_path(self):
        """Test validation fails when output_path is empty."""
        with pytest.raises(ValueError, match="output_path is required"):
            BinarizeConfig(input_path="input.txt")

    def test_validation_invalid_dtype(self):
        """Test validation fails for unsupported dtype."""
        with pytest.raises(ValueError, match="Unsupported dtype"):
            BinarizeConfig(
                input_path="input.txt",
                output_path="output.bin",
                dtype="float32",
            )

    def test_valid_dtypes(self):
        """Test that uint16 and uint32 are accepted."""
        config16 = BinarizeConfig(
            input_path="input.txt",
            output_path="output.bin",
            dtype="uint16",
        )
        assert config16.dtype == "uint16"

        config32 = BinarizeConfig(
            input_path="input.txt",
            output_path="output.bin",
            dtype="uint32",
        )
        assert config32.dtype == "uint32"


class TestBinarizeFile:
    """Tests for binarize_file function."""

    def test_basic_binarization(self, tmp_path: Path):
        """Test basic binarization of a text file."""
        # Create test input file
        input_file = tmp_path / "input.txt"
        input_file.write_text("hello world\ntest line\n")

        output_file = tmp_path / "output.bin"

        tokenizer = MockTokenizer()
        config = BinarizeConfig(
            input_path=str(input_file),
            output_path=str(output_file),
            append_eos=False,
        )

        stats = binarize_file(tokenizer, config)

        assert stats["total_lines"] == 2
        assert stats["total_tokens"] > 0
        assert output_file.exists()

    def test_binarization_with_eos(self, tmp_path: Path):
        """Test binarization with EOS token appended."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("hello\nworld\n")

        output_file = tmp_path / "output.bin"

        tokenizer = MockTokenizer()
        config = BinarizeConfig(
            input_path=str(input_file),
            output_path=str(output_file),
            append_eos=True,
            eos_token_id=999,
        )

        stats = binarize_file(tokenizer, config)

        # Load and verify EOS tokens are present
        tokens = load_binarized_file(output_file, mmap_mode=None)
        # EOS should be at the end of each line
        assert 999 in tokens

    def test_empty_lines_skipped(self, tmp_path: Path):
        """Test that empty lines are skipped during binarization."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("hello\n\n\nworld\n")

        output_file = tmp_path / "output.bin"

        tokenizer = MockTokenizer()
        config = BinarizeConfig(
            input_path=str(input_file),
            output_path=str(output_file),
            append_eos=False,
        )

        stats = binarize_file(tokenizer, config)

        # Should only process 2 non-empty lines
        assert stats["total_lines"] == 2

    def test_file_not_found(self, tmp_path: Path):
        """Test that FileNotFoundError is raised for missing input."""
        config = BinarizeConfig(
            input_path=str(tmp_path / "nonexistent.txt"),
            output_path=str(tmp_path / "output.bin"),
        )

        tokenizer = MockTokenizer()

        with pytest.raises(FileNotFoundError):
            binarize_file(tokenizer, config)

    def test_progress_callback(self, tmp_path: Path):
        """Test progress callback is called during binarization."""
        input_file = tmp_path / "input.txt"
        # Create file with many lines to trigger progress updates
        lines = [f"line {i}\n" for i in range(2500)]
        input_file.write_text("".join(lines))

        output_file = tmp_path / "output.bin"

        progress_calls = []

        def callback(current: int, total: int):
            progress_calls.append((current, total))

        tokenizer = MockTokenizer()
        config = BinarizeConfig(
            input_path=str(input_file),
            output_path=str(output_file),
            append_eos=False,
        )

        binarize_file(tokenizer, config, progress_callback=callback)

        # Should have received progress updates
        assert len(progress_calls) > 0

    def test_output_directory_created(self, tmp_path: Path):
        """Test that output directory is created if it doesn't exist."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("hello world\n")

        output_file = tmp_path / "nested" / "dir" / "output.bin"

        tokenizer = MockTokenizer()
        config = BinarizeConfig(
            input_path=str(input_file),
            output_path=str(output_file),
            append_eos=False,
        )

        binarize_file(tokenizer, config)

        assert output_file.exists()


class TestLoadBinarizedFile:
    """Tests for load_binarized_file function."""

    def test_load_existing_file(self, tmp_path: Path):
        """Test loading a valid binarized file."""
        # Create a binarized file first
        input_file = tmp_path / "input.txt"
        input_file.write_text("hello world\n")

        output_file = tmp_path / "output.bin"

        tokenizer = MockTokenizer()
        config = BinarizeConfig(
            input_path=str(input_file),
            output_path=str(output_file),
            append_eos=False,
        )

        binarize_file(tokenizer, config)

        # Load the file
        tokens = load_binarized_file(output_file)

        assert isinstance(tokens, np.ndarray)
        assert len(tokens) > 0

    def test_mmap_mode(self, tmp_path: Path):
        """Test memory-mapped loading."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("hello world\ntest line\nanother line\n")

        output_file = tmp_path / "output.bin"

        tokenizer = MockTokenizer()
        config = BinarizeConfig(
            input_path=str(input_file),
            output_path=str(output_file),
            append_eos=False,
        )

        binarize_file(tokenizer, config)

        # Load with memory mapping
        tokens = load_binarized_file(output_file, mmap_mode="r")

        # Should be a memmap object
        assert isinstance(tokens, np.ndarray)

        # Should be able to slice without loading everything
        slice_data = tokens[0:5]
        assert len(slice_data) <= 5

    def test_file_not_found(self, tmp_path: Path):
        """Test FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_binarized_file(tmp_path / "nonexistent.bin")

    def test_invalid_format(self, tmp_path: Path):
        """Test ValueError for invalid file format."""
        invalid_file = tmp_path / "invalid.bin"
        invalid_file.write_bytes(b"INVALID_DATA")

        with pytest.raises(ValueError, match="Invalid binarized file format"):
            load_binarized_file(invalid_file)

    def test_unsupported_version(self, tmp_path: Path):
        """Test ValueError for unsupported file version."""
        invalid_file = tmp_path / "invalid.bin"
        # Write magic number but with wrong version
        invalid_file.write_bytes(MAGIC_NUMBER + struct.pack("<BB", 255, 0))

        with pytest.raises(ValueError, match="Unsupported file version"):
            load_binarized_file(invalid_file)

    def test_dtype_uint32(self, tmp_path: Path):
        """Test loading uint32 format file."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("hello world\n")

        output_file = tmp_path / "output.bin"

        tokenizer = MockTokenizer()
        config = BinarizeConfig(
            input_path=str(input_file),
            output_path=str(output_file),
            dtype="uint32",
            append_eos=False,
        )

        binarize_file(tokenizer, config)

        tokens = load_binarized_file(output_file)

        # Should be uint32
        assert tokens.dtype == np.uint32


class TestBinarizeFileSimple:
    """Tests for binarize_file_simple function."""

    def test_simple_binarization(self, tmp_path: Path):
        """Test simple binarization without progress tracking."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("hello world\ntest line\n")

        output_file = tmp_path / "output.bin"

        tokenizer = MockTokenizer()
        token_count = binarize_file_simple(
            tokenizer,
            str(input_file),
            str(output_file),
            append_eos=False,
        )

        assert token_count > 0
        assert output_file.exists()

        # Verify file can be loaded
        tokens = load_binarized_file(output_file)
        assert len(tokens) == token_count

    def test_simple_with_eos(self, tmp_path: Path):
        """Test simple binarization with EOS."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("hello\nworld\n")

        output_file = tmp_path / "output.bin"

        tokenizer = MockTokenizer()
        token_count = binarize_file_simple(
            tokenizer,
            str(input_file),
            str(output_file),
            append_eos=True,
            eos_token_id=100,
        )

        tokens = load_binarized_file(output_file)
        assert 100 in tokens


class TestFileFormat:
    """Tests for binarized file format."""

    def test_header_format(self, tmp_path: Path):
        """Test that file header is correctly written."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("test\n")

        output_file = tmp_path / "output.bin"

        tokenizer = MockTokenizer()
        config = BinarizeConfig(
            input_path=str(input_file),
            output_path=str(output_file),
            append_eos=False,
        )

        binarize_file(tokenizer, config)

        # Read and verify header
        with open(output_file, "rb") as f:
            magic = f.read(len(MAGIC_NUMBER))
            assert magic == MAGIC_NUMBER

            version, dtype_code = struct.unpack("<BB", f.read(2))
            assert version == VERSION
            assert dtype_code == 0  # uint16

    def test_data_integrity(self, tmp_path: Path):
        """Test that token data is correctly written and can be recovered."""
        # Use a tokenizer with known vocabulary
        vocab = {"hello": 1, "world": 2, "test": 3}
        tokenizer = MockTokenizer(vocab)

        input_file = tmp_path / "input.txt"
        input_file.write_text("hello world\ntest hello\n")

        output_file = tmp_path / "output.bin"

        config = BinarizeConfig(
            input_path=str(input_file),
            output_path=str(output_file),
            append_eos=False,
        )

        binarize_file(tokenizer, config)

        # Load and verify tokens
        tokens = load_binarized_file(output_file, mmap_mode=None)

        # Should have [1, 2, 3, 1] for "hello world" and "test hello"
        expected = [1, 2, 3, 1]
        assert list(tokens) == expected


class TestIntegration:
    """Integration tests for binarize workflow."""

    def test_full_workflow(self, tmp_path: Path):
        """Test complete binarize -> load workflow."""
        # Create input file
        input_file = tmp_path / "corpus.txt"
        lines = [
            "The quick brown fox",
            "jumps over the lazy dog",
            "Pack my box with five dozen liquor jugs",
        ]
        input_file.write_text("\n".join(lines))

        output_file = tmp_path / "corpus.bin"

        # Binarize
        tokenizer = MockTokenizer()
        config = BinarizeConfig(
            input_path=str(input_file),
            output_path=str(output_file),
            append_eos=True,
            eos_token_id=0,
        )

        stats = binarize_file(tokenizer, config)

        # Verify stats
        assert stats["total_lines"] == 3
        assert stats["total_tokens"] > 0
        assert stats["avg_tokens_per_line"] > 0

        # Load and verify
        tokens = load_binarized_file(output_file, mmap_mode="r")
        assert len(tokens) == stats["total_tokens"]

        # EOS tokens should be present (one per line)
        eos_count = np.sum(tokens == 0)
        assert eos_count == 3

    def test_large_file_chunking(self, tmp_path: Path):
        """Test that large files are properly chunked during writing."""
        input_file = tmp_path / "input.txt"
        # Create many lines
        lines = [f"line number {i} with some content\n" for i in range(100)]
        input_file.write_text("".join(lines))

        output_file = tmp_path / "output.bin"

        tokenizer = MockTokenizer()
        config = BinarizeConfig(
            input_path=str(input_file),
            output_path=str(output_file),
            chunk_size=10,  # Small chunk size to force multiple writes
            append_eos=False,
        )

        stats = binarize_file(tokenizer, config)

        assert stats["total_lines"] == 100

        # Verify all tokens are present
        tokens = load_binarized_file(output_file, mmap_mode=None)
        assert len(tokens) == stats["total_tokens"]

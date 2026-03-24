"""Tests for BPETokenizer in stage1_tokenize module."""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import pytest

from llm_foundry.stage1_tokenize.tokenizer import BPETokenizer, TokenizerConfig


class TestBPETokenizer:
    """Test suite for BPETokenizer class."""

    def test_tokenizer_initialization(self):
        """Test basic tokenizer initialization."""
        vocab = {i: bytes([i]) for i in range(256)}
        vocab[256] = b"<|endoftext|>"
        vocab[257] = b"th"
        vocab[258] = b"e"
        merges = [(b"t", b"h"), (b"th", b"e")]

        tokenizer = BPETokenizer(vocab, merges, special_tokens=["<|endoftext|>"])

        assert len(tokenizer) == 259
        assert tokenizer.special_tokens == ["<|endoftext|>"]
        assert len(tokenizer.merges) == 2

    def test_encode_decode_simple(self):
        """Test basic encoding and decoding."""
        # Create a minimal vocab
        vocab = {i: bytes([i]) for i in range(256)}
        vocab[256] = b"<|endoftext|>"

        # Add some merge tokens
        vocab[257] = b"th"
        vocab[258] = b"he"
        vocab[259] = b"the"

        merges = [
            (b"t", b"h"),
            (b"h", b"e"),
            (b"th", b"e"),
        ]

        tokenizer = BPETokenizer(vocab, merges, special_tokens=["<|endoftext|>"])

        # Test encoding
        text = "the"
        ids = tokenizer.encode(text)
        # Should encode to single token for "the" if merge is applied
        assert isinstance(ids, list)
        assert len(ids) > 0

        # Test decoding
        decoded = tokenizer.decode(ids)
        assert decoded == text

    def test_special_tokens_handling(self):
        """Test special token encoding/decoding."""
        vocab = {i: bytes([i]) for i in range(256)}
        vocab[256] = b"<|endoftext|>"
        vocab[257] = b"<|padding|>"

        merges = []
        special_tokens = ["<|endoftext|>", "<|padding|>"]

        tokenizer = BPETokenizer(vocab, merges, special_tokens=special_tokens)

        # Test encoding with special tokens
        text = "hello<|endoftext|>world"
        ids = tokenizer.encode(text)

        # Should contain the special token ID
        assert 256 in ids  # <|endoftext|> ID

        # Test decoding
        decoded = tokenizer.decode(ids)
        assert "<|endoftext|>" in decoded
        assert "hello" in decoded
        assert "world" in decoded

    def test_save_and_load(self):
        """Test saving and loading tokenizer."""
        vocab = {i: bytes([i]) for i in range(256)}
        vocab[256] = b"<|endoftext|>"
        vocab[257] = b"th"

        merges = [(b"t", b"h")]
        special_tokens = ["<|endoftext|>"]

        tokenizer = BPETokenizer(vocab, merges, special_tokens=special_tokens)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save tokenizer
            tokenizer.save(tmpdir)

            # Check files exist
            assert (Path(tmpdir) / "vocab.pkl").exists()
            assert (Path(tmpdir) / "merges.pkl").exists()
            assert (Path(tmpdir) / "special_tokens.txt").exists()

            # Load tokenizer
            loaded = BPETokenizer.load(tmpdir)

            assert len(loaded) == len(tokenizer)
            assert loaded.special_tokens == tokenizer.special_tokens
            assert loaded.merges == tokenizer.merges
            assert loaded.vocab == tokenizer.vocab

    def test_empty_text(self):
        """Test encoding empty text."""
        vocab = {i: bytes([i]) for i in range(256)}
        merges = []

        tokenizer = BPETokenizer(vocab, merges)

        ids = tokenizer.encode("")
        assert ids == []

        decoded = tokenizer.decode([])
        assert decoded == ""

    def test_unicode_handling(self):
        """Test handling of unicode characters."""
        vocab = {i: bytes([i]) for i in range(256)}
        merges = []

        tokenizer = BPETokenizer(vocab, merges)

        # Test with unicode text
        text = "Hello, 世界! 🌍"
        ids = tokenizer.encode(text)
        assert len(ids) > 0

        decoded = tokenizer.decode(ids)
        assert decoded == text

    def test_repr(self):
        """Test string representation."""
        vocab = {i: bytes([i]) for i in range(256)}
        merges = []
        special_tokens = ["<|endoftext|>"]

        tokenizer = BPETokenizer(vocab, merges, special_tokens=special_tokens)

        repr_str = repr(tokenizer)
        assert "BPETokenizer" in repr_str
        assert "vocab_size=257" in repr_str
        assert "num_merges=0" in repr_str


class TestTokenizerConfig:
    """Test suite for TokenizerConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TokenizerConfig()

        assert config.vocab_size == 10000
        assert config.special_tokens == ["<|endoftext|>"]

    def test_custom_config(self):
        """Test custom configuration values."""
        config = TokenizerConfig(
            vocab_size=5000,
            special_tokens=["<|start|>", "<|end|>"],
        )

        assert config.vocab_size == 5000
        assert config.special_tokens == ["<|start|>", "<|end|>"]


class TestBPETokenizerTraining:
    """Test suite for BPE training functionality."""

    def test_train_basic(self):
        """Test basic training on a small text file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            # Write simple training text with more variety
            for i in range(50):
                f.write(f"the cat sat on the mat number {i} with some extra words\n")
                f.write(f"the dog ran through the park quickly on day {i}\n")
            input_path = f.name

        try:
            tokenizer = BPETokenizer.train(
                input_path=input_path,
                vocab_size=300,
                special_tokens=["<|endoftext|>"],
            )

            # Check tokenizer was created
            assert isinstance(tokenizer, BPETokenizer)
            # Vocab should have at least 256 bytes + special tokens
            assert len(tokenizer) >= 257

            # Test encoding/decoding
            text = "the cat"
            ids = tokenizer.encode(text)
            decoded = tokenizer.decode(ids)
            assert decoded == text

        finally:
            Path(input_path).unlink(missing_ok=True)

    def test_train_vocab_size_validation(self):
        """Test that training validates vocab_size."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test text\n")
            input_path = f.name

        try:
            with pytest.raises(ValueError, match="vocab_size must be at least"):
                BPETokenizer.train(
                    input_path=input_path,
                    vocab_size=100,  # Too small
                    special_tokens=["<|endoftext|>"],
                )
        finally:
            Path(input_path).unlink(missing_ok=True)

    def test_train_with_special_tokens(self):
        """Test training with custom special tokens."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello world<|endoftext|>goodbye world\n")
            input_path = f.name

        try:
            tokenizer = BPETokenizer.train(
                input_path=input_path,
                vocab_size=300,
                special_tokens=["<|endoftext|>", "<|padding|>"],
            )

            # Check special tokens are in vocab
            assert b"<|endoftext|>" in tokenizer.inverse_vocab
            assert b"<|padding|>" in tokenizer.inverse_vocab

        finally:
            Path(input_path).unlink(missing_ok=True)


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_train_save_load_encode_decode(self):
        """Test full workflow: train, save, load, encode, decode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create training file
            train_file = Path(tmpdir) / "train.txt"
            train_file.write_text(
                "the quick brown fox jumps over the lazy dog\n" * 100,
                encoding="utf-8",
            )

            # Train tokenizer
            tokenizer = BPETokenizer.train(
                input_path=str(train_file),
                vocab_size=400,
                special_tokens=["<|endoftext|>"],
            )

            # Save tokenizer
            save_dir = Path(tmpdir) / "tokenizer"
            tokenizer.save(save_dir)

            # Load tokenizer
            loaded = BPETokenizer.load(save_dir)

            # Test encoding/decoding with loaded tokenizer
            test_texts = [
                "the quick",
                "brown fox",
                "lazy dog",
                "<|endoftext|>",
            ]

            for text in test_texts:
                ids = loaded.encode(text)
                decoded = loaded.decode(ids)
                assert decoded == text, f"Failed for text: {text}"

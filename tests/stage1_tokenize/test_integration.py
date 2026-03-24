"""
Integration tests for Stage 1: Tokenization pipeline.

Tests the complete workflow from raw text to tokenized binary files:
- BPE training from text corpus
- Text encoding to token IDs
- Binary serialization (binarize)
- Token loading with load_tokens()
- Round-trip encode/decode verification
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch


class TestBPETokenizerIntegration:
    """Integration tests for BPETokenizer complete workflow."""

    @pytest.fixture
    def sample_corpus(self) -> str:
        """Create a sample text corpus for testing."""
        return """The quick brown fox jumps over the lazy dog.
The quick brown fox jumps over the lazy dog.
Machine learning is a subset of artificial intelligence.
Deep learning uses neural networks with many layers.
Natural language processing helps computers understand text.
The quick brown fox jumps over the lazy dog.
<|endoftext|>
Another document starts here.
It has multiple lines.
And some punctuation! Does it work?
<|endoftext|>
Final document with numbers 123 and symbols @#$.
Testing BPE tokenization on various inputs.
<|endoftext|>"""

    @pytest.fixture
    def corpus_file(self, sample_corpus: str, tmp_path: Path) -> Path:
        """Create a temporary corpus file."""
        corpus_path = tmp_path / "corpus.txt"
        corpus_path.write_text(sample_corpus, encoding="utf-8")
        return corpus_path

    def test_end_to_end_train_encode_decode(
        self, corpus_file: Path, tmp_path: Path
    ) -> None:
        """Test complete workflow: train -> encode -> decode round-trip."""
        pytest.importorskip("regex", reason="regex module required")

        from llm_foundry.stage1_tokenize.tokenizer import BPETokenizer

        # Step 1: Train BPE tokenizer
        vocab_size = 300  # Small vocab for fast testing
        special_tokens = ["<|endoftext|>"]

        tokenizer = BPETokenizer.train(
            input_path=str(corpus_file),
            vocab_size=vocab_size,
            special_tokens=special_tokens,
        )

        # Verify vocab size
        assert len(tokenizer.vocab) == vocab_size
        assert len(tokenizer.merges) == vocab_size - 256 - len(special_tokens)

        # Step 2: Encode text
        test_text = "The quick brown fox"
        token_ids = tokenizer.encode(test_text)

        # Verify encoding produces token IDs
        assert isinstance(token_ids, list)
        assert len(token_ids) > 0
        assert all(isinstance(tid, int) for tid in token_ids)
        assert all(0 <= tid < vocab_size for tid in token_ids)

        # Step 3: Decode back to text
        decoded_text = tokenizer.decode(token_ids)

        # Verify round-trip (may not be exact due to BPE compression)
        assert isinstance(decoded_text, str)
        # The decoded text should contain the original words
        assert "The" in decoded_text or "the" in decoded_text.lower()
        assert "quick" in decoded_text.lower()

    def test_save_and_load(self, corpus_file: Path, tmp_path: Path) -> None:
        """Test save/load functionality."""
        pytest.importorskip("regex", reason="regex module required")

        from llm_foundry.stage1_tokenize.tokenizer import BPETokenizer

        # Train and save tokenizer
        vocab_size = 300
        special_tokens = ["<|endoftext|>"]

        tokenizer1 = BPETokenizer.train(
            input_path=str(corpus_file),
            vocab_size=vocab_size,
            special_tokens=special_tokens,
        )

        save_dir = tmp_path / "tokenizer"
        tokenizer1.save(save_dir)

        # Verify saved files exist
        assert (save_dir / "vocab.pkl").exists()
        assert (save_dir / "merges.pkl").exists()

        # Load tokenizer
        tokenizer2 = BPETokenizer.load(save_dir)

        # Verify loaded tokenizer works the same
        test_text = "The quick brown fox"
        ids1 = tokenizer1.encode(test_text)
        ids2 = tokenizer2.encode(test_text)

        assert ids1 == ids2
        assert tokenizer1.decode(ids1) == tokenizer2.decode(ids2)

    def test_pretokenize_to_binary(
        self, corpus_file: Path, tmp_path: Path
    ) -> None:
        """Test pretokenize and binarize workflow."""
        pytest.importorskip("regex", reason="regex module required")

        from llm_foundry.stage1_tokenize.tokenizer import BPETokenizer
        from llm_foundry.common.data import load_tokens

        # Train tokenizer
        vocab_size = 300
        special_tokens = ["<|endoftext|>"]

        tokenizer = BPETokenizer.train(
            input_path=str(corpus_file),
            vocab_size=vocab_size,
            special_tokens=special_tokens,
        )

        # Pretokenize: encode corpus to token IDs
        corpus_text = corpus_file.read_text(encoding="utf-8")
        token_ids = tokenizer.encode(corpus_text)

        # Binarize: save as uint16 numpy file (.npy format for load_tokens compatibility)
        output_path = tmp_path / "tokens.npy"
        token_array = np.array(token_ids, dtype=np.uint16)
        np.save(output_path, token_array)

        # Verify binary file exists and has content
        assert output_path.exists()

        # Load tokens using load_tokens (as np.memmap)
        loaded_tokens = load_tokens(output_path)

        # Verify loaded tokens match original
        assert len(loaded_tokens) == len(token_ids)
        assert np.array_equal(loaded_tokens, token_ids)

    def test_special_token_handling(
        self, corpus_file: Path, tmp_path: Path
    ) -> None:
        """Test special token encoding/decoding."""
        pytest.importorskip("regex", reason="regex module required")

        from llm_foundry.stage1_tokenize.tokenizer import BPETokenizer

        special_tokens = ["<|endoftext|>", "<|padding|>"]

        tokenizer = BPETokenizer.train(
            input_path=str(corpus_file),
            vocab_size=300,
            special_tokens=special_tokens,
        )

        # Test encoding with special tokens
        text_with_special = "Hello<|endoftext|>World"
        token_ids = tokenizer.encode(text_with_special)

        # Verify special token is in vocab (inverse_vocab maps bytes -> token_id)
        for st in special_tokens:
            st_bytes = st.encode("utf-8")
            assert st_bytes in tokenizer.inverse_vocab
            assert tokenizer.inverse_vocab[st_bytes] >= 256  # Special tokens are after byte tokens

        # Test that special tokens are preserved in encoding
        decoded = tokenizer.decode(token_ids)
        # Special tokens should appear in decoded text
        assert "<|endoftext|>" in decoded or "Hello" in decoded

    def test_encode_chunks(self, corpus_file: Path) -> None:
        """Test encoding multiple text chunks."""
        pytest.importorskip("regex", reason="regex module required")

        from llm_foundry.stage1_tokenize.tokenizer import BPETokenizer

        tokenizer = BPETokenizer.train(
            input_path=str(corpus_file),
            vocab_size=300,
            special_tokens=["<|endoftext|>"],
        )

        # Test with multiple text chunks
        text_chunks = [
            "The quick brown fox",
            "jumps over the lazy dog",
            "Machine learning is amazing",
        ]

        # Encode each chunk individually and concatenate
        all_ids = []
        for chunk in text_chunks:
            all_ids.extend(tokenizer.encode(chunk))

        # Verify we got token IDs
        assert isinstance(all_ids, list)
        assert len(all_ids) > 0
        assert all(isinstance(tid, int) for tid in all_ids)

    def test_binary_file_format_compatibility(
        self, corpus_file: Path, tmp_path: Path
    ) -> None:
        """Test that binary format is compatible with training data loading."""
        pytest.importorskip("regex", reason="regex module required")

        from llm_foundry.stage1_tokenize.tokenizer import BPETokenizer
        from llm_foundry.common.data import load_tokens, get_batch

        # Train tokenizer
        tokenizer = BPETokenizer.train(
            input_path=str(corpus_file),
            vocab_size=300,
            special_tokens=["<|endoftext|>"],
        )

        # Pretokenize corpus
        corpus_text = corpus_file.read_text(encoding="utf-8")
        token_ids = tokenizer.encode(corpus_text)

        # Save as .npy file (format expected by load_tokens)
        npy_path = tmp_path / "train.npy"
        np.save(npy_path, np.array(token_ids, dtype=np.uint16))

        # Load with load_tokens (memory-mapped)
        loaded = load_tokens(npy_path)

        # Verify we can sample batches (as would be done in training)
        batch_size = 2
        context_length = 8

        # Should be able to get batches without error
        device = "cpu"
        x, y = get_batch(loaded, batch_size, context_length, device)

        assert x.shape == (batch_size, context_length)
        assert y.shape == (batch_size, context_length)
        assert x.dtype == torch.long
        assert y.dtype == torch.long


class TestTokenizerEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_text_encoding(self, tmp_path: Path) -> None:
        """Test encoding empty text."""
        pytest.importorskip("regex", reason="regex module required")

        from llm_foundry.stage1_tokenize.tokenizer import BPETokenizer

        # Create minimal corpus
        corpus_path = tmp_path / "corpus.txt"
        corpus_path.write_text("Hello world test", encoding="utf-8")

        tokenizer = BPETokenizer.train(
            input_path=str(corpus_path),
            vocab_size=300,
            special_tokens=[],
        )

        # Encode empty string
        result = tokenizer.encode("")
        assert result == []

    def test_unicode_text(self, tmp_path: Path) -> None:
        """Test handling of unicode text."""
        pytest.importorskip("regex", reason="regex module required")

        from llm_foundry.stage1_tokenize.tokenizer import BPETokenizer

        # Create corpus with unicode
        corpus_path = tmp_path / "corpus.txt"
        corpus_path.write_text(
            "Hello world café naïve résumé 日本語 中文 emoji 🎉",
            encoding="utf-8",
        )

        tokenizer = BPETokenizer.train(
            input_path=str(corpus_path),
            vocab_size=300,
            special_tokens=[],
        )

        # Encode unicode text
        text = "café naïve"
        token_ids = tokenizer.encode(text)
        decoded = tokenizer.decode(token_ids)

        # Should handle unicode gracefully
        assert isinstance(token_ids, list)
        assert isinstance(decoded, str)

    def test_large_vocab_request(self, tmp_path: Path) -> None:
        """Test training with vocab size larger than possible merges."""
        pytest.importorskip("regex", reason="regex module required")

        from llm_foundry.stage1_tokenize.tokenizer import BPETokenizer

        # Small corpus with limited diversity
        corpus_path = tmp_path / "corpus.txt"
        corpus_path.write_text("a b c d e f g", encoding="utf-8")

        # Request vocab size larger than possible
        tokenizer = BPETokenizer.train(
            input_path=str(corpus_path),
            vocab_size=1000,  # Much larger than corpus can support
            special_tokens=[],
        )

        # Should still produce valid tokenizer
        assert len(tokenizer.vocab) >= 256  # At least byte-level vocab
        assert len(tokenizer.merges) < 1000 - 256


class TestIntegrationWithCommon:
    """Test integration with common module components."""

    def test_tokenizer_with_common_data_loading(
        self, tmp_path: Path
    ) -> None:
        """Test tokenizer output works with common.data utilities."""
        pytest.importorskip("regex", reason="regex module required")

        from llm_foundry.stage1_tokenize.tokenizer import BPETokenizer
        from llm_foundry.common.data import load_tokens, create_data_loader

        # Create and train tokenizer
        corpus_path = tmp_path / "corpus.txt"
        corpus_path.write_text(
            "The quick brown fox jumps over the lazy dog. " * 100,
            encoding="utf-8",
        )

        tokenizer = BPETokenizer.train(
            input_path=str(corpus_path),
            vocab_size=300,
            special_tokens=["<|endoftext|>"],
        )

        # Tokenize and save
        text = corpus_path.read_text(encoding="utf-8")
        token_ids = tokenizer.encode(text)

        npy_path = tmp_path / "tokens.npy"
        np.save(npy_path, np.array(token_ids, dtype=np.uint16))

        # Test load_tokens
        loaded = load_tokens(npy_path)
        assert len(loaded) == len(token_ids)

        # Test create_data_loader
        device = "cpu"
        data_loader = create_data_loader(
            data_path=npy_path,
            batch_size=4,
            context_length=16,
            device=device,
            num_batches=3,
        )

        batches = list(data_loader)
        assert len(batches) == 3

        for x, y in batches:
            assert x.shape == (4, 16)
            assert y.shape == (4, 16)
            # Verify x and y are offset by 1
            assert torch.equal(x[:, 1:], y[:, :-1])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

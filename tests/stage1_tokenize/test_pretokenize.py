"""
Tests for the pretokenize module.

This test suite verifies the pre-tokenization functionality including:
- pretokenize_regex pattern matching
- pretokenize() function
- pretokenize_file() function
- find_chunk_boundaries() function
- PretokenizeConfig class
"""

import io
import tempfile
from pathlib import Path

import pytest
import regex as re

from llm_foundry.stage1_tokenize.pretokenize import (
    PRETOKENIZE_REGEX,
    PretokenizeConfig,
    find_chunk_boundaries,
    pretokenize,
    pretokenize_file,
)


class TestPretokenizeRegex:
    """Tests for the PRETOKENIZE_REGEX constant."""

    def test_regex_compiles(self):
        """Test that the regex compiles successfully."""
        assert PRETOKENIZE_REGEX is not None
        assert isinstance(PRETOKENIZE_REGEX, type(re.compile("")))

    def test_regex_matches_contractions(self):
        """Test regex matches English contractions."""
        text = "It's don't won't can't I'll you've we're"
        matches = PRETOKENIZE_REGEX.findall(text)
        # Should match: It, 's, don, 't, won, 't, can, 't, I, 'll, you, 've, we, 're
        assert "'s" in matches
        assert "'t" in matches
        assert "'ll" in matches
        assert "'ve" in matches
        assert "'re" in matches

    def test_regex_matches_words(self):
        """Test regex matches words with optional leading space."""
        text = "Hello world"
        matches = PRETOKENIZE_REGEX.findall(text)
        assert "Hello" in matches
        assert " world" in matches

    def test_regex_matches_numbers(self):
        """Test regex matches numbers."""
        text = "Price is 123 dollars"
        matches = PRETOKENIZE_REGEX.findall(text)
        assert " 123" in matches

    def test_regex_matches_punctuation(self):
        """Test regex matches punctuation."""
        text = "Hello, world!"
        matches = PRETOKENIZE_REGEX.findall(text)
        assert "," in matches or " ," in matches
        assert "!" in matches or " !" in matches

    def test_regex_matches_whitespace(self):
        """Test regex handles whitespace correctly."""
        text = "Line1\n\nLine2"
        matches = PRETOKENIZE_REGEX.findall(text)
        # Should capture whitespace between lines
        assert len(matches) > 0


class TestPretokenizeFunction:
    """Tests for the pretokenize() function."""

    def test_empty_string(self):
        """Test pretokenize handles empty string."""
        result = pretokenize("")
        assert result == []

    def test_simple_text(self):
        """Test pretokenize with simple text."""
        result = pretokenize("Hello world!")
        assert "Hello" in result
        assert " world" in result
        assert "!" in result

    def test_contractions(self):
        """Test pretokenize with contractions."""
        result = pretokenize("It's a test.")
        assert "It" in result
        assert "'s" in result
        assert " a" in result
        assert " test" in result
        assert "." in result

    def test_numbers(self):
        """Test pretokenize with numbers."""
        result = pretokenize("123 456.789")
        assert "123" in result
        assert " 456" in result
        assert "." in result
        assert "789" in result

    def test_unicode(self):
        """Test pretokenize with unicode text."""
        result = pretokenize("Hello 世界")
        assert "Hello" in result
        assert " 世界" in result


class TestPretokenizeFile:
    """Tests for the pretokenize_file() function."""

    def test_basic_file_processing(self, tmp_path):
        """Test basic file pre-tokenization."""
        input_file = tmp_path / "input.txt"
        output_file = tmp_path / "output.txt"

        input_file.write_text("Hello world! Test.", encoding="utf-8")

        count = pretokenize_file(input_file, output_file)

        assert count > 0
        assert output_file.exists()

        output_content = output_file.read_text(encoding="utf-8")
        assert "Hello" in output_content
        assert " world" in output_content

    def test_file_with_special_tokens(self, tmp_path):
        """Test pre-tokenization with special token splitting."""
        input_file = tmp_path / "input.txt"
        output_file = tmp_path / "output.txt"

        input_file.write_text(
            "Hello<|endoftext|>World<|endoftext|>Test",
            encoding="utf-8"
        )

        count = pretokenize_file(
            input_file,
            output_file,
            special_tokens=["<|endoftext|>"]
        )

        assert count > 0
        output_content = output_file.read_text(encoding="utf-8")
        # Should have processed "Hello", "World", "Test" separately
        assert "Hello" in output_content
        assert "World" in output_content
        assert "Test" in output_content

    def test_empty_input_file(self, tmp_path):
        """Test pre-tokenization of empty file."""
        input_file = tmp_path / "input.txt"
        output_file = tmp_path / "output.txt"

        input_file.write_text("", encoding="utf-8")

        count = pretokenize_file(input_file, output_file)

        assert count == 0
        assert output_file.exists()

    def test_nonexistent_input_file(self, tmp_path):
        """Test error handling for non-existent input file."""
        input_file = tmp_path / "nonexistent.txt"
        output_file = tmp_path / "output.txt"

        with pytest.raises(FileNotFoundError):
            pretokenize_file(input_file, output_file)

    def test_creates_output_directory(self, tmp_path):
        """Test that output directory is created if needed."""
        input_file = tmp_path / "input.txt"
        output_file = tmp_path / "subdir" / "output.txt"

        input_file.write_text("Hello world!", encoding="utf-8")

        pretokenize_file(input_file, output_file)

        assert output_file.exists()


class TestFindChunkBoundaries:
    """Tests for the find_chunk_boundaries() function."""

    def test_basic_chunking(self):
        """Test basic file chunking."""
        content = b"Hello<|endoftext|>World<|endoftext|>Test"
        file = io.BytesIO(content)

        boundaries = find_chunk_boundaries(file, 2, b"<|endoftext|>")

        assert boundaries[0] == 0
        assert boundaries[-1] == len(content)
        assert len(boundaries) >= 2

    def test_chunk_alignment(self):
        """Test that chunks align to special tokens."""
        # Create content with clear special token boundaries
        content = b"A<|endoftext|>B<|endoftext|>C<|endoftext|>D"
        file = io.BytesIO(content)

        boundaries = find_chunk_boundaries(file, 2, b"<|endoftext|>")

        # Boundaries should align with special token positions
        assert 0 in boundaries
        assert len(content) in boundaries

    def test_single_chunk(self):
        """Test with single chunk request."""
        content = b"Hello<|endoftext|>World"
        file = io.BytesIO(content)

        boundaries = find_chunk_boundaries(file, 1, b"<|endoftext|>")

        assert boundaries == [0, len(content)]

    def test_more_chunks_than_content(self):
        """Test requesting more chunks than possible."""
        content = b"Hello"
        file = io.BytesIO(content)

        boundaries = find_chunk_boundaries(file, 10, b"<|endoftext|>")

        # Should return at least start and end
        assert len(boundaries) >= 2
        assert boundaries[0] == 0
        assert boundaries[-1] == len(content)

    def test_invalid_token_type(self):
        """Test that non-bytes token raises assertion error."""
        content = b"Hello"
        file = io.BytesIO(content)

        with pytest.raises(AssertionError):
            find_chunk_boundaries(file, 2, "<|endoftext|>")  # string, not bytes


class TestPretokenizeConfig:
    """Tests for the PretokenizeConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PretokenizeConfig()

        assert config.special_tokens == ["<|endoftext|>"]
        assert config.num_workers == -1
        assert config.output_format == "lines"

    def test_custom_values(self):
        """Test custom configuration values."""
        config = PretokenizeConfig(
            special_tokens=["<|padding|>", "<|endoftext|>"],
            num_workers=4,
            output_format="json",
        )

        assert config.special_tokens == ["<|padding|>", "<|endoftext|>"]
        assert config.num_workers == 4
        assert config.output_format == "json"

    def test_get_effective_num_workers_default(self):
        """Test effective workers with default (-1)."""
        config = PretokenizeConfig(num_workers=-1)

        effective = config.get_effective_num_workers()

        assert effective >= 1
        assert effective == (os.cpu_count() or 1)

    def test_get_effective_num_workers_custom(self):
        """Test effective workers with custom value."""
        config = PretokenizeConfig(num_workers=4)

        assert config.get_effective_num_workers() == 4

    def test_get_effective_num_workers_zero(self):
        """Test effective workers with zero (should become 1)."""
        config = PretokenizeConfig(num_workers=0)

        assert config.get_effective_num_workers() == 1

    def test_get_split_token_bytes_default(self):
        """Test getting split token bytes with default config."""
        config = PretokenizeConfig()

        token_bytes = config.get_split_token_bytes()

        assert token_bytes == b"<|endoftext|>"

    def test_get_split_token_bytes_custom(self):
        """Test getting split token bytes with custom tokens."""
        config = PretokenizeConfig(special_tokens=["<|custom|>"])

        token_bytes = config.get_split_token_bytes()

        assert token_bytes == b"<|custom|>"

    def test_get_split_token_bytes_empty(self):
        """Test getting split token bytes with empty list."""
        config = PretokenizeConfig(special_tokens=[])

        token_bytes = config.get_split_token_bytes()

        assert token_bytes == b"<|endoftext|>"


import os  # noqa: E402

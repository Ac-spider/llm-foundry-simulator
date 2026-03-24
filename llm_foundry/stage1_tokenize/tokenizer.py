"""BPE Tokenizer implementation for LLM Foundry Simulator.

This module provides a unified BPETokenizer class that combines
the BPE training algorithm from bpe.py with the Tokenizer class
for encoding/decoding.
"""

from __future__ import annotations

import os
import pickle
import regex as re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import BinaryIO

import multiprocessing as mp


# GPT-2 pre-tokenization regex pattern
# Matches: contractions, words, numbers, punctuation, and whitespace
GPT2_PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """Find byte boundaries for parallel processing, aligned to special tokens.

    Args:
        file: Binary file object
        desired_num_chunks: Number of chunks to create
        split_special_token: Token to align boundaries to

    Returns:
        List of byte offsets for chunk boundaries
    """
    assert isinstance(split_special_token, bytes)

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)

        while True:
            mini_chunk = file.read(mini_chunk_size)

            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break

            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def _process_chunk(args) -> Counter:
    """Process a file chunk and return word counts.

    Args:
        args: Tuple of (input_path, start, end, special_tokens)

    Returns:
        Counter mapping byte tuples to frequencies
    """
    input_path, start, end, special_tokens = args

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)

    chunk_text = chunk_bytes.decode("utf-8", errors="ignore")

    # Split on special tokens to avoid cross-document merging
    if special_tokens:
        escaped_specials = [re.escape(st) for st in special_tokens]
        split_pat = "|".join(escaped_specials)
        parts = re.split(split_pat, chunk_text)
    else:
        parts = [chunk_text]

    word_counts = Counter()
    for part in parts:
        for match in re.finditer(GPT2_PAT, part):
            token_bytes = match.group().encode("utf-8")
            token_tuple = tuple(bytes([b]) for b in token_bytes)
            word_counts[token_tuple] += 1

    return word_counts


@dataclass
class TokenizerConfig:
    """Configuration for BPETokenizer.

    Attributes:
        vocab_size: Target vocabulary size
        special_tokens: List of special tokens to include
    """

    vocab_size: int = 10000
    special_tokens: list[str] = field(default_factory=lambda: ["<|endoftext|>"])

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> SimpleNamespace:
        """Load tokenizer configuration from YAML file.

        Returns a SimpleNamespace with the full config structure matching
        the tokenize.yaml schema: data, training, output, encoding, logging.

        Args:
            config_path: Path to YAML config file

        Returns:
            SimpleNamespace with full config structure
        """
        import yaml
        from types import SimpleNamespace

        config_path = Path(config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if data is None:
            data = {}

        # Convert nested dict to SimpleNamespace recursively
        def _convert(obj):
            if isinstance(obj, dict):
                return SimpleNamespace(**{k: _convert(v) for k, v in obj.items()})
            elif isinstance(obj, list):
                return [_convert(item) for item in obj]
            return obj

        config = _convert(data)

        # Ensure required sections exist with defaults
        if not hasattr(config, "data"):
            config.data = SimpleNamespace()
        if not hasattr(config.data, "train_file"):
            config.data.train_file = None
        if not hasattr(config.data, "val_file"):
            config.data.val_file = None
        if not hasattr(config.data, "test_file"):
            config.data.test_file = None

        if not hasattr(config, "training"):
            config.training = SimpleNamespace()
        if not hasattr(config.training, "vocab_size"):
            config.training.vocab_size = 10000
        if not hasattr(config.training, "special_tokens"):
            config.training.special_tokens = ["<|endoftext|>"]

        if not hasattr(config, "output"):
            config.output = SimpleNamespace()
        if not hasattr(config.output, "output_dir"):
            config.output.output_dir = "results/tokenizer"
        if not hasattr(config.output, "name"):
            config.output.name = "bpe_tokenizer"

        # Add skip_training flag (for CLI usage)
        config.skip_training = False

        return config


class BPETokenizer:
    """Byte Pair Encoding Tokenizer with training and inference capabilities.

    This class combines BPE training (from train_bpe) with encoding/decoding
    functionality. It can be trained from scratch on text data or loaded
    from saved vocab/merges files.

    Attributes:
        vocab: Dictionary mapping token IDs to byte sequences
        merges: List of merge rules as (bytes, bytes) tuples
        special_tokens: List of special tokens
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """Initialize tokenizer with vocab and merges.

        Args:
            vocab: Mapping from token IDs to byte sequences
            merges: Ordered list of merge rules
            special_tokens: Optional list of special tokens
        """
        self.vocab = vocab.copy()
        self.merges = merges
        self.special_tokens = special_tokens or []

        # Add any missing special tokens to vocab
        if self.special_tokens:
            existing_bytes = set(self.vocab.values())
            next_id = max(self.vocab.keys()) + 1 if self.vocab else 0
            for st in self.special_tokens:
                st_bytes = st.encode("utf-8")
                if st_bytes not in existing_bytes:
                    self.vocab[next_id] = st_bytes
                    existing_bytes.add(st_bytes)
                    next_id += 1

        # Build inverse vocab for encoding
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

        # Build merge rank lookup
        self.merges_rank = {pair: i for i, pair in enumerate(self.merges)}

        # Compile special token pattern for encoding
        if self.special_tokens:
            escaped = [re.escape(st) for st in self.special_tokens]
            self.special_pat = re.compile("(" + "|".join(escaped) + ")")
        else:
            self.special_pat = None

    @classmethod
    def train(
        cls,
        input_path: str | Path,
        vocab_size: int,
        special_tokens: list[str] | None = None,
    ) -> BPETokenizer:
        """Train a new BPE tokenizer from text data.

        This method implements the BPE training algorithm with incremental
        updates for efficiency. It uses multiprocessing for parallel word
        frequency counting.

        Args:
            input_path: Path to training text file
            vocab_size: Target vocabulary size (must be >= 256 + len(special_tokens))
            special_tokens: List of special tokens to include

        Returns:
            Trained BPETokenizer instance

        Raises:
            ValueError: If vocab_size is too small
        """
        special_tokens = special_tokens or ["<|endoftext|>"]
        min_vocab_size = 256 + len(special_tokens)
        if vocab_size < min_vocab_size:
            raise ValueError(
                f"vocab_size must be at least {min_vocab_size} "
                f"(256 bytes + {len(special_tokens)} special tokens)"
            )

        # Initialize vocab with byte tokens
        vocab = {i: bytes([i]) for i in range(256)}
        next_id = 256

        # Add special tokens
        for st in special_tokens:
            vocab[next_id] = st.encode("utf-8")
            next_id += 1

        # Parallel word frequency counting
        num_processes = mp.cpu_count()
        split_token = special_tokens[0].encode("utf-8") if special_tokens else b"<|endoftext|>"

        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_processes, split_token)

        chunk_args = [
            (input_path, boundaries[i], boundaries[i + 1], special_tokens)
            for i in range(len(boundaries) - 1)
        ]

        word_counts = Counter()
        with mp.Pool(num_processes) as pool:
            for chunk_counts in pool.imap_unordered(_process_chunk, chunk_args):
                word_counts.update(chunk_counts)

        merges = []
        current_vocab_size = len(vocab)

        # Initialize pair counting structures
        pair_counts = defaultdict(int)
        pair_to_words = defaultdict(set)

        for word, count in word_counts.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_counts[pair] += count
                pair_to_words[pair].add(word)

        # Main BPE merge loop
        while current_vocab_size < vocab_size:
            if not pair_counts:
                break

            # Find most frequent pair (tie-break by lexicographic order)
            best_pair = max(pair_counts.keys(), key=lambda p: (pair_counts[p], p))
            merges.append(best_pair)

            new_token_bytes = best_pair[0] + best_pair[1]
            vocab[next_id] = new_token_bytes
            next_id += 1
            current_vocab_size += 1

            words_to_process = list(pair_to_words[best_pair])

            del pair_counts[best_pair]
            del pair_to_words[best_pair]

            for word in words_to_process:
                count = word_counts[word]
                if count == 0:
                    continue

                # Remove old pair contributions
                for i in range(len(word) - 1):
                    p = (word[i], word[i + 1])
                    if p == best_pair:
                        continue

                    pair_counts[p] -= count
                    if pair_counts[p] <= 0:
                        del pair_counts[p]

                    if word in pair_to_words.get(p, set()):
                        pair_to_words[p].remove(word)
                        if not pair_to_words[p]:
                            del pair_to_words[p]

                # Create merged word
                new_word = []
                i = 0
                while i < len(word):
                    if (
                        i < len(word) - 1
                        and word[i] == best_pair[0]
                        and word[i + 1] == best_pair[1]
                    ):
                        new_word.append(new_token_bytes)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_word = tuple(new_word)

                # Add new pair contributions
                for i in range(len(new_word) - 1):
                    p = (new_word[i], new_word[i + 1])
                    pair_counts[p] += count
                    pair_to_words[p].add(new_word)

                del word_counts[word]
                word_counts[new_word] += count

        return cls(vocab, merges, special_tokens)

    def _encode_chunk(self, text: str) -> list[int]:
        """Encode a text chunk without special tokens.

        Args:
            text: Text to encode

        Returns:
            List of token IDs
        """
        ids = []

        for match in re.finditer(GPT2_PAT, text):
            token_bytes = match.group().encode("utf-8")
            b_list = [bytes([b]) for b in token_bytes]

            # Apply BPE merges greedily
            while len(b_list) >= 2:
                best_pair = None
                min_rank = float("inf")

                for i in range(len(b_list) - 1):
                    pair = (b_list[i], b_list[i + 1])
                    rank = self.merges_rank.get(pair, float("inf"))
                    if rank < min_rank:
                        best_pair = pair
                        min_rank = rank

                if not best_pair:
                    break

                # Apply the merge
                i = 0
                new_b_list = []
                while i < len(b_list):
                    if (
                        i < len(b_list) - 1
                        and b_list[i] == best_pair[0]
                        and b_list[i + 1] == best_pair[1]
                    ):
                        new_b_list.append(b_list[i] + b_list[i + 1])
                        i += 2
                    else:
                        new_b_list.append(b_list[i])
                        i += 1
                b_list = new_b_list

            # Convert bytes to token IDs
            for b in b_list:
                ids.append(self.inverse_vocab[b])

        return ids

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Text to encode

        Returns:
            List of token IDs
        """
        if not self.special_tokens:
            return self._encode_chunk(text)

        ids = []
        parts = self.special_pat.split(text)
        for part in parts:
            if not part:
                continue
            if part in self.special_tokens:
                part_bytes = part.encode("utf-8")
                ids.append(self.inverse_vocab[part_bytes])
            else:
                ids.extend(self._encode_chunk(part))

        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to text.

        Args:
            ids: List of token IDs

        Returns:
            Decoded text string

        Raises:
            ValueError: If an unknown token ID is encountered
        """
        b_list = []
        for token_id in ids:
            if token_id in self.vocab:
                b_list.append(self.vocab[token_id])
            else:
                raise ValueError(f"Token ID {token_id} not found in vocabulary")

        b_text = b"".join(b_list)
        return b_text.decode("utf-8", errors="replace")

    def save(self, save_directory: str | Path) -> None:
        """Save tokenizer vocab and merges to directory.

        Args:
            save_directory: Directory to save tokenizer files
        """
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save vocab
        vocab_path = save_path / "vocab.pkl"
        with open(vocab_path, "wb") as f:
            pickle.dump(self.vocab, f)

        # Save merges
        merges_path = save_path / "merges.pkl"
        with open(merges_path, "wb") as f:
            pickle.dump(self.merges, f)

        # Save special tokens as text file for readability
        special_tokens_path = save_path / "special_tokens.txt"
        with open(special_tokens_path, "w", encoding="utf-8") as f:
            for token in self.special_tokens:
                f.write(token + "\n")

    @classmethod
    def load(cls, load_directory: str | Path) -> BPETokenizer:
        """Load tokenizer from directory.

        Args:
            load_directory: Directory containing saved tokenizer files

        Returns:
            Loaded BPETokenizer instance

        Raises:
            FileNotFoundError: If required files are missing
        """
        load_path = Path(load_directory)

        vocab_path = load_path / "vocab.pkl"
        merges_path = load_path / "merges.pkl"
        special_tokens_path = load_path / "special_tokens.txt"

        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_path, "rb") as f:
            merges = pickle.load(f)

        special_tokens = []
        if special_tokens_path.exists():
            with open(special_tokens_path, "r", encoding="utf-8") as f:
                special_tokens = [line.strip() for line in f if line.strip()]

        return cls(vocab, merges, special_tokens)

    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"BPETokenizer("
            f"vocab_size={len(self.vocab)}, "
            f"num_merges={len(self.merges)}, "
            f"special_tokens={self.special_tokens}"
            f")"
        )

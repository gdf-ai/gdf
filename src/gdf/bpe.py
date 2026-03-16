"""BPE (Byte Pair Encoding) tokenizer that learns from data.

Instead of treating every byte as a token (256 vocab), BPE learns common
pairs from the training data and merges them into single tokens. This means
the model can work with word-level and subword-level concepts.

Example:
    Raw bytes:  t h e _ c a t _ s a t
    After BPE:  the _ cat _ sat          (fewer tokens, more meaning per token)
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


# Start with all 256 byte values as base tokens
BASE_VOCAB_SIZE = 256


class BPETokenizer:
    """Byte Pair Encoding tokenizer that learns merges from data."""

    def __init__(self, merges: list[tuple[int, int]] | None = None, vocab_size: int = BASE_VOCAB_SIZE):
        self.merges: list[tuple[int, int]] = merges or []
        # Token ID for each merged pair = BASE_VOCAB_SIZE + index in merges list
        self.vocab_size = vocab_size

    @classmethod
    def train(cls, texts: list[str], target_vocab_size: int = 1024,
              on_progress: callable = None) -> BPETokenizer:
        """Learn BPE merges from a list of texts.

        Args:
            texts: Training texts to learn vocabulary from.
            target_vocab_size: How many tokens total (256 base + N merges).
            on_progress: Optional callback(step, total, pair) for progress.

        Returns:
            Trained BPETokenizer.
        """
        num_merges = target_vocab_size - BASE_VOCAB_SIZE
        if num_merges <= 0:
            return cls([], BASE_VOCAB_SIZE)

        # Convert all text to byte sequences
        sequences: list[list[int]] = []
        for text in texts:
            tokens = list(text.encode("utf-8"))
            if tokens:
                sequences.append(tokens)

        merges: list[tuple[int, int]] = []

        for step in range(num_merges):
            # Count all adjacent pairs
            pair_counts: Counter[tuple[int, int]] = Counter()
            for seq in sequences:
                for i in range(len(seq) - 1):
                    pair_counts[(seq[i], seq[i + 1])] += 1

            if not pair_counts:
                break

            # Find most frequent pair
            best_pair = pair_counts.most_common(1)[0][0]
            new_token_id = BASE_VOCAB_SIZE + len(merges)
            merges.append(best_pair)

            if on_progress:
                on_progress(step, num_merges, best_pair)

            # Replace all occurrences of best_pair with new_token_id
            for i, seq in enumerate(sequences):
                sequences[i] = _apply_merge(seq, best_pair, new_token_id)

        return cls(merges, BASE_VOCAB_SIZE + len(merges))

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs using learned merges."""
        tokens = list(text.encode("utf-8"))
        for i, (a, b) in enumerate(self.merges):
            new_id = BASE_VOCAB_SIZE + i
            tokens = _apply_merge(tokens, (a, b), new_id)
        return tokens

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs back to text."""
        # Build reverse mapping: each merged token → its two components
        # Then recursively expand back to bytes
        bytes_out = []
        for token in tokens:
            bytes_out.extend(self._expand_token(token))
        return bytes(bytes_out).decode("utf-8", errors="replace")

    def _expand_token(self, token: int) -> list[int]:
        """Expand a token back to its byte components."""
        if token < BASE_VOCAB_SIZE:
            return [token]
        merge_idx = token - BASE_VOCAB_SIZE
        if merge_idx >= len(self.merges):
            return [token % 256]  # fallback
        a, b = self.merges[merge_idx]
        return self._expand_token(a) + self._expand_token(b)

    def save(self, path: str | Path) -> None:
        """Save tokenizer to a JSON file."""
        data = {
            "merges": self.merges,
            "vocab_size": self.vocab_size,
        }
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        """Load tokenizer from a JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        merges = [tuple(m) for m in data["merges"]]
        return cls(merges=merges, vocab_size=data["vocab_size"])

    def get_vocab_tokens(self) -> dict[int, str]:
        """Return a sample of what each token represents (for debugging)."""
        result = {}
        for i in range(min(BASE_VOCAB_SIZE, self.vocab_size)):
            try:
                result[i] = bytes([i]).decode("utf-8", errors="replace")
            except Exception:
                result[i] = f"<byte {i}>"
        for i, (a, b) in enumerate(self.merges):
            token_id = BASE_VOCAB_SIZE + i
            expanded = bytes(self._expand_token(token_id))
            try:
                result[token_id] = expanded.decode("utf-8", errors="replace")
            except Exception:
                result[token_id] = f"<merge {a}+{b}>"
        return result


def _apply_merge(tokens: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace all occurrences of pair in tokens with new_id."""
    if len(tokens) < 2:
        return tokens
    result = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            result.append(new_id)
            i += 2
        else:
            result.append(tokens[i])
            i += 1
    return result

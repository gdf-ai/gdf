"""Byte-level tokenizer: every byte (0-255) is a token."""

VOCAB_SIZE = 256


def encode(text: str) -> list[int]:
    """Encode a string to a list of byte values."""
    return list(text.encode("utf-8"))


def decode(tokens: list[int]) -> str:
    """Decode a list of byte values back to a string."""
    return bytes(tokens).decode("utf-8", errors="replace")

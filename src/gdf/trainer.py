"""OnlineTrainer: Adam optimizer with replay buffer and feedback handling."""

from __future__ import annotations

import re
import random
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F

from .model import TinyTransformer
from .tokenizer import encode
from .device import detect_device


@dataclass
class TrainerConfig:
    lr: float = 1e-3
    correction_lr_mult: float = 3.0
    replay_buffer_size: int = 100
    replay_samples: int = 3
    max_seq_len: int = 512
    use_amp: bool = True          # automatic mixed precision (fp16/bf16)
    grad_clip: float = 1.0        # gradient clipping max norm


class OnlineTrainer:
    """Trains a TinyTransformer online, one input at a time."""

    def __init__(self, model: TinyTransformer, config: TrainerConfig | None = None,
                 device: torch.device | None = None):
        self.device = device or detect_device()
        self.model = model.to(self.device)
        self.config = config or TrainerConfig()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        self.replay_buffer: deque[list[int]] = deque(maxlen=self.config.replay_buffer_size)
        self.step_count = 0

        # Mixed precision — only on CUDA
        self._use_amp = self.config.use_amp and self.device.type == "cuda"
        self._scaler = torch.amp.GradScaler("cuda") if self._use_amp else None
        # Determine autocast dtype
        if self._use_amp:
            self._amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            self._amp_dtype = torch.float32

    def _amp_context(self):
        """Return the appropriate autocast context."""
        if self._use_amp:
            return torch.amp.autocast("cuda", dtype=self._amp_dtype)
        return nullcontext()

    def _train_on_tokens(self, tokens: list[int], lr_mult: float = 1.0) -> float:
        """Train on a single token sequence. Returns loss value."""
        if len(tokens) < 2:
            return 0.0

        tokens = tokens[: self.config.max_seq_len]
        x = torch.tensor([tokens[:-1]], dtype=torch.long, device=self.device)
        y = torch.tensor([tokens[1:]], dtype=torch.long, device=self.device)

        # Adjust LR if needed
        if lr_mult != 1.0:
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.config.lr * lr_mult

        self.model.train()

        with self._amp_context():
            logits = self.model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        self.optimizer.zero_grad()

        if self._scaler is not None:
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()

        # Restore LR
        if lr_mult != 1.0:
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.config.lr

        self.step_count += 1
        return loss.item()

    def _replay(self) -> list[float]:
        """Replay random samples from the buffer. Returns list of losses."""
        if not self.replay_buffer:
            return []
        n = min(self.config.replay_samples, len(self.replay_buffer))
        samples = random.sample(list(self.replay_buffer), n)
        return [self._train_on_tokens(s) for s in samples]

    def train_step(self, text: str, feedback: str = "good", correction: str | None = None) -> dict:
        """Process one user interaction.

        Args:
            text: The input text.
            feedback: One of "good", "bad", "correction".
            correction: The corrected text (required if feedback=="correction").

        Returns:
            Dict with loss info.
        """
        tokens = encode(text)
        result: dict = {"feedback": feedback, "step": self.step_count}

        if feedback == "good":
            loss = self._train_on_tokens(tokens)
            self.replay_buffer.append(tokens)
            replay_losses = self._replay()
            result["loss"] = loss
            result["replay_losses"] = replay_losses

        elif feedback == "bad":
            # Skip training on bad input
            result["loss"] = None
            result["replay_losses"] = []

        elif feedback == "correction":
            if correction is None:
                raise ValueError("correction text required when feedback='correction'")
            corr_tokens = encode(correction)
            loss = self._train_on_tokens(corr_tokens, lr_mult=self.config.correction_lr_mult)
            self.replay_buffer.append(corr_tokens)
            replay_losses = self._replay()
            result["loss"] = loss
            result["replay_losses"] = replay_losses

        else:
            raise ValueError(f"Unknown feedback type: {feedback}")

        return result

    def train_bulk(
        self,
        text: str,
        epochs: int = 3,
        chunk_size: int = 256,
        overlap: int = 64,
        on_step: Callable[[int, int, float], None] | None = None,
    ) -> dict:
        """Train on a large text block, chunked with overlap, for multiple epochs.

        Args:
            text: Full text to train on.
            epochs: Number of passes over the data.
            chunk_size: Max tokens per chunk.
            overlap: Token overlap between chunks for context continuity.
            on_step: Optional callback(step, total_steps, loss) for progress.

        Returns:
            Dict with training stats.
        """
        # Split into sentences, then pack into chunks
        chunks = self._chunk_text(text, chunk_size, overlap)
        if not chunks:
            return {"epochs": 0, "steps": 0, "final_loss": 0.0}

        total_steps = len(chunks) * epochs
        step_i = 0
        losses = []

        for epoch in range(epochs):
            random.shuffle(chunks)
            epoch_losses = []
            for tokens in chunks:
                loss = self._train_on_tokens(tokens)
                epoch_losses.append(loss)
                self.replay_buffer.append(tokens)
                # Replay less frequently during bulk (every 5 chunks)
                if step_i % 5 == 0:
                    self._replay()
                step_i += 1
                if on_step:
                    on_step(step_i, total_steps, loss)
            losses.extend(epoch_losses)

        return {
            "epochs": epochs,
            "chunks": len(chunks),
            "steps": step_i,
            "avg_loss": sum(losses) / len(losses) if losses else 0.0,
            "final_loss": losses[-1] if losses else 0.0,
            "first_loss": losses[0] if losses else 0.0,
        }

    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> list[list[int]]:
        """Split text into overlapping token chunks, splitting at sentence boundaries."""
        # Split into sentences first
        sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Encode all sentences
        encoded_sentences = [encode(s) for s in sentences]

        # Pack sentences into chunks up to chunk_size
        chunks: list[list[int]] = []
        current: list[int] = []

        for tokens in encoded_sentences:
            if not tokens:
                continue
            # If adding this sentence would exceed chunk_size, flush
            if current and len(current) + len(tokens) + 1 > chunk_size:
                chunks.append(current)
                # Keep overlap from end of previous chunk
                if overlap > 0 and len(current) > overlap:
                    current = current[-overlap:] + [ord(" ")] + tokens
                else:
                    current = tokens
            else:
                if current:
                    current += [ord(" ")] + tokens
                else:
                    current = tokens

        if current:
            chunks.append(current)

        return chunks

    def get_state(self) -> dict:
        """Get trainer state for serialization."""
        return {
            "optimizer_state": self.optimizer.state_dict(),
            "replay_buffer": list(self.replay_buffer),
            "step_count": self.step_count,
            "config": {
                "lr": self.config.lr,
                "correction_lr_mult": self.config.correction_lr_mult,
                "replay_buffer_size": self.config.replay_buffer_size,
                "replay_samples": self.config.replay_samples,
                "max_seq_len": self.config.max_seq_len,
            },
        }

    def load_state(self, state: dict) -> None:
        """Load trainer state from dict."""
        self.optimizer.load_state_dict(state["optimizer_state"])
        buf = state.get("replay_buffer", [])
        self.replay_buffer = deque(buf, maxlen=self.config.replay_buffer_size)
        self.step_count = state.get("step_count", 0)
        if "config" in state:
            cfg = state["config"]
            self.config = TrainerConfig(**cfg)
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.config.lr

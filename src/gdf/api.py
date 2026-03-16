"""GDFModel: high-level wrapper for create, load, train, generate, merge, save."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from .model import TinyTransformer, ModelConfig
from .trainer import OnlineTrainer, TrainerConfig
from .serialization import save_model, load_model, get_model_info, _compute_hash
from .tokenizer import encode, decode
from .merging import merge_models
from .device import detect_device


class GDFModel:
    """High-level API for gdf models."""

    def __init__(
        self,
        model: TinyTransformer,
        trainer: OnlineTrainer,
        base_model_hash: str | None = None,
    ):
        self.model = model
        self.trainer = trainer
        self.device = trainer.device
        self.base_model_hash = base_model_hash

    @classmethod
    def create(cls, config: ModelConfig | None = None) -> GDFModel:
        """Create a fresh model."""
        config = config or ModelConfig()
        model = TinyTransformer(config)
        device = detect_device()
        trainer = OnlineTrainer(model, device=device)
        base_hash = _compute_hash(model.state_dict())
        return cls(model, trainer, base_model_hash=base_hash)

    @classmethod
    def load(cls, path: str | Path) -> GDFModel:
        """Load a model from a .pt file."""
        model, trainer, meta = load_model(path, load_trainer=True)
        device = detect_device()
        if trainer is None:
            trainer = OnlineTrainer(model, device=device)
        else:
            # Move to best device
            trainer.device = device
            trainer.model = model.to(device)
        return cls(model, trainer, base_model_hash=meta.get("base_model_hash"))

    def save(self, path: str | Path) -> None:
        """Save model to a .pt file."""
        save_model(path, self.model, self.trainer, self.base_model_hash)

    def train(self, text: str, feedback: str = "good", correction: str | None = None) -> dict:
        """Train on a single text input with feedback."""
        return self.trainer.train_step(text, feedback=feedback, correction=correction)

    def train_file(
        self,
        file_path: str | Path,
        epochs: int = 3,
        chunk_size: int = 256,
        on_step=None,
    ) -> dict:
        """Train on a text file."""
        text = Path(file_path).read_text(encoding="utf-8", errors="replace")
        return self.trainer.train_bulk(
            text, epochs=epochs, chunk_size=chunk_size, on_step=on_step,
        )

    def generate(
        self,
        prompt: str = "",
        max_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> str:
        """Generate text autoregressively."""
        self.model.eval()
        tokens = encode(prompt) if prompt else []

        if not tokens:
            tokens = [ord(" ")]

        with torch.no_grad():
            for _ in range(max_tokens):
                ctx = tokens[-self.model.config.max_seq_len :]
                x = torch.tensor([ctx], dtype=torch.long, device=self.device)
                logits = self.model(x)
                logits = logits[0, -1, :]  # last position

                # Temperature
                if temperature > 0:
                    logits = logits / temperature
                else:
                    # Greedy
                    idx = logits.argmax().item()
                    tokens.append(idx)
                    continue

                # Top-k filtering
                if top_k > 0:
                    topk_vals, topk_idx = torch.topk(logits, min(top_k, logits.size(-1)))
                    mask = torch.full_like(logits, float("-inf"))
                    mask.scatter_(0, topk_idx, topk_vals)
                    logits = mask

                probs = F.softmax(logits, dim=-1)
                idx = torch.multinomial(probs, 1).item()
                tokens.append(idx)

        return decode(tokens)

    @staticmethod
    def merge(
        paths: list[str | Path],
        strategy: str = "ties",
        **kwargs,
    ) -> GDFModel:
        """Merge multiple saved models into one."""
        models = []
        base = None
        base_hash = None

        for p in paths:
            m, _, meta = load_model(p, load_trainer=False)
            models.append(m)
            if base_hash is None:
                base_hash = meta.get("base_model_hash")

        # Try to find a shared base — use the first model's base hash
        # In practice, the base would be loaded separately
        # For now, if all share same base_model_hash, use first model as approx base
        # This is a simplification; real usage would store the base model separately

        if strategy in ("ties", "task_arithmetic") and len(models) >= 2:
            # Check if all share same base
            all_hashes = set()
            for p in paths:
                info = get_model_info(p)
                all_hashes.add(info.get("base_model_hash"))

            if len(all_hashes) == 1 and None not in all_hashes:
                # All share same base — but we don't have it stored
                # Fall back to fedavg for now, or use ties without base (which falls back)
                pass

        merged, new_hash = merge_models(models, base=base, strategy=strategy, **kwargs)
        trainer = OnlineTrainer(merged)
        return GDFModel(merged, trainer, base_model_hash=new_hash)

    @staticmethod
    def info(path: str | Path) -> dict:
        """Get info about a saved model."""
        return get_model_info(path)

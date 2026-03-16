"""Save/load model+trainer state to .pt files with base_model_hash tracking."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from .model import TinyTransformer, ModelConfig
from .trainer import OnlineTrainer, TrainerConfig


def _compute_hash(state_dict: dict) -> str:
    """Compute a deterministic hash of model weights."""
    h = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        h.update(key.encode())
        h.update(state_dict[key].cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def save_model(
    path: str | Path,
    model: TinyTransformer,
    trainer: OnlineTrainer | None = None,
    base_model_hash: str | None = None,
) -> None:
    """Save model (and optionally trainer state) to a .pt file."""
    data: dict = {
        "config": model.config.to_dict(),
        "weights": model.state_dict(),
        "model_hash": _compute_hash(model.state_dict()),
        "base_model_hash": base_model_hash,
    }
    if trainer is not None:
        data["trainer_state"] = trainer.get_state()
    torch.save(data, str(path))


def load_model(
    path: str | Path,
    load_trainer: bool = True,
) -> tuple[TinyTransformer, OnlineTrainer | None, dict]:
    """Load model (and optionally trainer) from a .pt file.

    Returns:
        (model, trainer_or_None, metadata_dict)
        metadata_dict has keys: model_hash, base_model_hash
    """
    data = torch.load(str(path), map_location="cpu", weights_only=False)

    config = ModelConfig.from_dict(data["config"])
    model = TinyTransformer(config)
    model.load_state_dict(data["weights"])

    trainer = None
    if load_trainer and "trainer_state" in data:
        trainer = OnlineTrainer(model)
        trainer.load_state(data["trainer_state"])

    meta = {
        "model_hash": data.get("model_hash"),
        "base_model_hash": data.get("base_model_hash"),
    }
    return model, trainer, meta


def get_model_info(path: str | Path) -> dict:
    """Get metadata about a saved model without fully loading it."""
    data = torch.load(str(path), map_location="cpu", weights_only=False)
    config = ModelConfig.from_dict(data["config"])
    model = TinyTransformer(config)
    model.load_state_dict(data["weights"])
    return {
        "config": data["config"],
        "model_hash": data.get("model_hash"),
        "base_model_hash": data.get("base_model_hash"),
        "has_trainer_state": "trainer_state" in data,
        "parameters": model.count_parameters(),
        "step_count": data.get("trainer_state", {}).get("step_count", 0),
        "replay_buffer_size": len(data.get("trainer_state", {}).get("replay_buffer", [])),
    }

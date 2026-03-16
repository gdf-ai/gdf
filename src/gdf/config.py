"""Persistent config for gdf — default model, settings."""

from __future__ import annotations

import json
from pathlib import Path

CONFIG_DIR = Path.home() / ".gdf"
CONFIG_FILE = CONFIG_DIR / "config.json"


def _ensure_dir() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    _ensure_dir()
    if CONFIG_FILE.exists():
        return json.loads(CONFIG_FILE.read_text())
    return {}


def save_config(cfg: dict) -> None:
    _ensure_dir()
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))


def get_default_model() -> str | None:
    cfg = load_config()
    path = cfg.get("default_model")
    if path and Path(path).exists():
        return path
    return None


def set_default_model(path: str) -> None:
    p = Path(path).resolve()
    cfg = load_config()
    cfg["default_model"] = str(p)
    save_config(cfg)


def resolve_model(path: str | None) -> str:
    """Resolve a model path: use given path, or fall back to default."""
    if path:
        return path
    default = get_default_model()
    if default:
        return default
    raise SystemExit(
        "No model specified and no default set.\n"
        "  Create one:  gdf init\n"
        "  Or set one:  gdf default mymodel.pt"
    )

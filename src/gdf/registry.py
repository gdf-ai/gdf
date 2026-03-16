"""Model registry — fetch predefined models from GitHub.

The registry is a simple JSON file listing available models, their
hub URLs, and public tokens. Cached locally to work offline.
"""

from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path


REGISTRY_URL = "https://raw.githubusercontent.com/gdf-ai/gdf/main/models.json"
LOCAL_REGISTRY = Path(__file__).resolve().parents[2] / "models.json"
CACHE_PATH = Path.home() / ".gdf" / "registry.json"
CACHE_TTL = 3600  # 1 hour


@dataclass
class ModelEntry:
    """A model available in the network."""
    name: str           # "general-7b"
    description: str    # "General-purpose 7B language model"
    hub_url: str        # "http://hub1.gdf.network:7677"
    token: str          # public token
    size: str           # "14GB"
    status: str         # "active" | "paused"

    @classmethod
    def from_dict(cls, d: dict) -> ModelEntry:
        return cls(
            name=d["name"],
            description=d.get("description", ""),
            hub_url=d["hub_url"],
            token=d["token"],
            size=d.get("size", "?"),
            status=d.get("status", "active"),
        )


def _load_local() -> list[ModelEntry] | None:
    """Try loading models from the local models.json bundled with the repo."""
    if LOCAL_REGISTRY.exists():
        try:
            data = json.loads(LOCAL_REGISTRY.read_text(encoding="utf-8"))
            return [ModelEntry.from_dict(e) for e in data]
        except Exception:
            pass
    return None


def fetch_registry() -> list[ModelEntry]:
    """Fetch available models. Uses cache / local file / remote, in that order."""
    # 1. Check cache first
    if CACHE_PATH.exists():
        try:
            cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            age = time.time() - cache.get("fetched_at", 0)
            if age < CACHE_TTL:
                return [ModelEntry.from_dict(e) for e in cache.get("models", [])]
        except Exception:
            pass

    # 2. Try local models.json (instant, always available in dev)
    local = _load_local()
    if local:
        return local

    # 3. Fetch from GitHub (remote registry)
    try:
        req = urllib.request.Request(REGISTRY_URL, headers={
            "User-Agent": "gdf/0.1",
        })
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status != 200:
                raise ValueError(f"HTTP {resp.status}")
            data = json.loads(resp.read().decode("utf-8"))

        # Cache it
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        cache = {"fetched_at": time.time(), "models": data}
        CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")

        return [ModelEntry.from_dict(e) for e in data]

    except Exception:
        # Fall back to cache (even if expired)
        if CACHE_PATH.exists():
            try:
                cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
                return [ModelEntry.from_dict(e) for e in cache.get("models", [])]
            except Exception:
                pass
        return []


def get_model(name: str) -> ModelEntry | None:
    """Look up a model by name."""
    for entry in fetch_registry():
        if entry.name == name:
            return entry
    return None

"""Unified model registry and routing.

Every model in the network — whether a local domain model or a remote hub
model — is a ModelInfo. The ModelRegistry manages both local models
(~/.gdf/models.json) and remote models (fetched from GitHub with caching).

The Router picks the best model for a query using keyword matching.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable


REGISTRY_URL = "https://raw.githubusercontent.com/gdf-ai/gdf/main/models.json"
LOCAL_REGISTRY = Path(__file__).resolve().parents[2] / "models.json"
CACHE_PATH = Path.home() / ".gdf" / "registry_cache.json"
LOCAL_MODELS_PATH = Path.home() / ".gdf" / "models.json"
CACHE_TTL = 3600  # 1 hour


@dataclass
class ModelInfo:
    """A model in the network — local or remote."""
    name: str
    description: str = ""
    # Location
    model_path: str | None = None       # local .pt file
    hub_url: str | None = None          # remote hub
    token: str | None = None
    # Domain metadata (for routing)
    domain: str | None = None
    keywords: list[str] = field(default_factory=list)
    # Stats
    quality_score: float = 0.0
    training_steps: int = 0
    training_sources: int = 0
    contributors: int = 0
    generation: int = 0
    size: str = "?"
    status: str = "active"
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ModelInfo:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Registry ────────────────────────────────────────────────────────────────

class ModelRegistry:
    """Unified registry: local models + remote models from GitHub."""

    def __init__(self, local_path: str | Path | None = None):
        self.local_path = Path(local_path) if local_path else LOCAL_MODELS_PATH
        self.models: dict[str, ModelInfo] = {}
        self._load()

    def _load(self) -> None:
        # Migrate from old specialists.json if it exists
        old_path = self.local_path.parent / "specialists.json"
        if old_path.exists() and not self.local_path.exists():
            self._migrate_specialists(old_path)

        if self.local_path.exists():
            try:
                data = json.loads(self.local_path.read_text(encoding="utf-8"))
                for name, info in data.items():
                    self.models[name] = ModelInfo.from_dict(info)
            except Exception:
                pass

    def _migrate_specialists(self, old_path: Path) -> None:
        """Convert old specialists.json to new models.json format."""
        try:
            data = json.loads(old_path.read_text(encoding="utf-8"))
            for name, info in data.items():
                self.models[name] = ModelInfo.from_dict(info)
            self._save()
            old_path.unlink()
        except Exception:
            pass

    def _save(self) -> None:
        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        data = {name: info.to_dict() for name, info in self.models.items()}
        self.local_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    def register(self, info: ModelInfo) -> None:
        """Register or update a model."""
        self.models[info.name] = info
        self._save()

    def unregister(self, name: str) -> bool:
        if name in self.models:
            del self.models[name]
            self._save()
            return True
        return False

    def get(self, name: str) -> ModelInfo | None:
        """Look up a model by name (local first, then remote)."""
        if name in self.models:
            return self.models[name]
        # Check remote
        for m in self.list_remote():
            if m.name == name:
                return m
        return None

    def list_all(self) -> list[ModelInfo]:
        """All models: local + remote (deduplicated by name)."""
        seen = set()
        result = []
        for m in self.models.values():
            seen.add(m.name)
            result.append(m)
        for m in self.list_remote():
            if m.name not in seen:
                seen.add(m.name)
                result.append(m)
        return result

    def list_local(self) -> list[ModelInfo]:
        return list(self.models.values())

    def list_remote(self) -> list[ModelInfo]:
        """Fetch remote models from GitHub registry (cached)."""
        return _fetch_remote_models()

    def list_routable(self) -> list[ModelInfo]:
        """Models that have keywords and a model_path (can be loaded for routing)."""
        return [m for m in self.models.values() if m.keywords and m.model_path]

    def domains(self) -> list[str]:
        return sorted(set(m.domain for m in self.models.values() if m.domain))


# ── Remote registry fetch (with caching) ────────────────────────────────────

def _apply_env_overrides(models: list[ModelInfo]) -> list[ModelInfo]:
    """Override hub_url/token from environment variables.

    Env vars checked (per model, falling back to global):
      GDF_{NAME}_HUB_URL, GDF_{NAME}_HUB_TOKEN  (model-specific)
      GDF_HUB_URL, GDF_HUB_TOKEN                (global fallback)
    """
    hub_url = os.environ.get("GDF_HUB_URL")
    hub_token = os.environ.get("GDF_HUB_TOKEN")

    for m in models:
        prefix = f"GDF_{m.name.upper().replace('-', '_')}"
        m.hub_url = os.environ.get(f"{prefix}_HUB_URL", hub_url or m.hub_url)
        m.token = os.environ.get(f"{prefix}_HUB_TOKEN", hub_token or m.token)

    return models


def _fetch_remote_models() -> list[ModelInfo]:
    """Fetch models from GitHub. Uses cache / local file / remote.

    Environment variables GDF_HUB_URL and GDF_HUB_TOKEN override the
    hub_url and token from models.json, so production credentials never
    need to be committed.
    """
    # 1. Check cache
    if CACHE_PATH.exists():
        try:
            cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            age = time.time() - cache.get("fetched_at", 0)
            if age < CACHE_TTL:
                return _apply_env_overrides(
                    [ModelInfo.from_dict(e) for e in cache.get("models", [])]
                )
        except Exception:
            pass

    # 2. Try local models.json (bundled with repo)
    if LOCAL_REGISTRY.exists():
        try:
            data = json.loads(LOCAL_REGISTRY.read_text(encoding="utf-8"))
            return _apply_env_overrides(
                [ModelInfo.from_dict(e) for e in data]
            )
        except Exception:
            pass

    # 3. Fetch from GitHub
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

        return _apply_env_overrides(
            [ModelInfo.from_dict(e) for e in data]
        )

    except Exception:
        # Fall back to expired cache
        if CACHE_PATH.exists():
            try:
                cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
                return _apply_env_overrides(
                    [ModelInfo.from_dict(e) for e in cache.get("models", [])]
                )
            except Exception:
                pass
        return []


# ── Convenience functions (backward compat) ──────────────────────────────────

def fetch_registry() -> list[ModelInfo]:
    """Fetch available models (local + remote)."""
    return ModelRegistry().list_all()


def get_model(name: str) -> ModelInfo | None:
    """Look up a model by name."""
    return ModelRegistry().get(name)


# ── Router ──────────────────────────────────────────────────────────────────

class Router:
    """Routes queries to the best model(s) using keyword matching."""

    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    def route(self, query: str, top_k: int = 3) -> list[tuple[ModelInfo, float]]:
        """Find the best model(s) for a query."""
        query_lower = query.lower()
        query_words = set(query_lower.split())

        scored: list[tuple[ModelInfo, float]] = []

        for model in self.registry.list_routable():
            score = self._score(query_lower, query_words, model)
            if score > 0:
                scored.append((model, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def _score(self, query: str, query_words: set[str], model: ModelInfo) -> float:
        """Score how relevant a model is for a query."""
        score = 0.0

        # Keyword matches (strongest signal)
        for kw in model.keywords:
            kw_lower = kw.lower()
            if kw_lower in query:
                score += 3.0
            elif kw_lower in query_words:
                score += 2.0
            else:
                kw_words = set(kw_lower.split())
                overlap = len(query_words & kw_words)
                if overlap:
                    score += overlap * 1.0

        # Domain name match
        if model.domain and model.domain.lower() in query:
            score += 2.0

        # Description word overlap
        desc_words = set(model.description.lower().split())
        desc_overlap = len(query_words & desc_words)
        score += desc_overlap * 0.3

        # Quality bonus
        if model.quality_score > 0:
            quality_bonus = 1.0 / (1.0 + model.quality_score / 100.0)
            score *= (1.0 + quality_bonus * 0.2)

        # Recency bonus
        age_days = (time.time() - model.updated) / 86400
        if age_days < 7:
            score *= 1.1

        return score

    def route_or_general(self, query: str, general_name: str = "general",
                          threshold: float = 1.0) -> list[tuple[ModelInfo, float]]:
        """Route to models, falling back to general model if no good match."""
        results = self.route(query, top_k=3)
        good_results = [(m, score) for m, score in results if score >= threshold]

        if good_results:
            return good_results

        general = self.registry.get(general_name)
        if general:
            return [(general, 0.1)]

        return results[:1] if results else []


# ── Multi-model query ──────────────────────────────────────────────────────

def query_models(
    query: str,
    registry: ModelRegistry,
    top_k: int = 2,
    max_tokens: int = 200,
    temperature: float = 0.7,
) -> list[dict]:
    """Query the best model(s) and return their responses."""
    from .api import GDFModel

    router = Router(registry)
    matches = router.route(query, top_k=top_k)

    results = []
    for model_info, score in matches:
        try:
            bm = GDFModel.load(model_info.model_path)
            response = bm.generate(
                prompt=query,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            results.append({
                "model": model_info.name,
                "domain": model_info.domain,
                "score": score,
                "response": response,
                "quality": model_info.quality_score,
            })
        except Exception as e:
            results.append({
                "model": model_info.name,
                "domain": model_info.domain,
                "score": score,
                "response": f"[Error: {e}]",
                "quality": model_info.quality_score,
            })

    return results


# ── Suggested domains ──────────────────────────────────────────────────────

SUGGESTED_DOMAINS = {
    "code-python": {
        "description": "Python programming, libraries, best practices",
        "keywords": ["python", "pip", "django", "flask", "pandas", "numpy",
                     "pytorch", "def", "class", "import"],
        "data_sources": "GitHub Python repos, Python docs, Stack Overflow Python",
    },
    "code-rust": {
        "description": "Rust programming, ownership, lifetimes, crates",
        "keywords": ["rust", "cargo", "lifetime", "borrow", "trait", "impl",
                     "unsafe", "async", "tokio"],
        "data_sources": "Rust docs, crates.io, Rust GitHub repos",
    },
    "medical": {
        "description": "Medical knowledge, clinical guidelines, anatomy",
        "keywords": ["medical", "clinical", "diagnosis", "treatment", "patient",
                     "symptom", "disease", "drug", "therapy", "surgery"],
        "data_sources": "PubMed abstracts, medical textbooks, clinical guidelines",
    },
    "legal": {
        "description": "Legal concepts, case law, regulations",
        "keywords": ["legal", "law", "court", "statute", "regulation", "contract",
                     "liability", "plaintiff", "defendant", "jurisdiction"],
        "data_sources": "Legal textbooks, case law databases, regulatory docs",
    },
    "science-physics": {
        "description": "Physics concepts, equations, experiments",
        "keywords": ["physics", "quantum", "relativity", "force", "energy",
                     "momentum", "wave", "particle", "field", "thermodynamics"],
        "data_sources": "Physics textbooks, arXiv physics papers",
    },
    "science-chemistry": {
        "description": "Chemistry, reactions, molecular structures",
        "keywords": ["chemistry", "molecule", "reaction", "element", "compound",
                     "organic", "inorganic", "catalyst", "bond", "acid"],
        "data_sources": "Chemistry textbooks, PubChem, reaction databases",
    },
    "finance": {
        "description": "Financial concepts, markets, accounting, investing",
        "keywords": ["finance", "stock", "bond", "investment", "portfolio",
                     "accounting", "revenue", "profit", "market", "trading"],
        "data_sources": "Financial textbooks, SEC filings, market analysis",
    },
    "history": {
        "description": "World history, historical events, civilizations",
        "keywords": ["history", "war", "civilization", "empire", "revolution",
                     "century", "ancient", "medieval", "colonial", "dynasty"],
        "data_sources": "Wikipedia history, history textbooks, primary sources",
    },
    "cooking": {
        "description": "Recipes, cooking techniques, food science",
        "keywords": ["recipe", "cook", "bake", "ingredient", "sauce", "temperature",
                     "cuisine", "flavor", "kitchen", "meal"],
        "data_sources": "Recipe databases, cooking blogs, food science texts",
    },
    "electronics": {
        "description": "Electronics, circuits, microcontrollers, signals",
        "keywords": ["circuit", "resistor", "capacitor", "arduino", "voltage",
                     "current", "pcb", "microcontroller", "signal", "amplifier"],
        "data_sources": "Electronics textbooks, datasheets, hobbyist forums",
    },
}

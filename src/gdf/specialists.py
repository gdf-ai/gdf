"""Specialist model registry and routing.

Instead of one giant general model, gdf maintains a network of
specialist models, each deeply trained on a specific domain. When a
query comes in, the router picks the best specialist(s) to answer.

Why this beats a general LLM:
- A 1B model trained on 10GB of medical papers beats a 70B model
  trained on 10TB of "everything" — for medical questions.
- Each specialist can be trained by people who actually know the domain.
- Specialists update independently — no retraining the whole network.
- 100 GPUs per specialty = deep expertise. 10k GPUs total = 100 specialties.

Architecture:
    Specialist = model + metadata (domain, keywords, description, quality score)
    Registry   = collection of specialists, stored locally or on a hub
    Router     = given a query, picks the best specialist(s)
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

from .tokenizer import encode


@dataclass
class SpecialistInfo:
    """Metadata about a specialist model."""
    name: str                              # e.g., "medical-cardiology"
    domain: str                            # e.g., "medicine"
    description: str                       # what this specialist knows
    keywords: list[str]                    # topic keywords for routing
    model_path: str                        # path to the .pt file
    quality_score: float = 0.0             # perplexity on domain eval set (lower = better)
    training_steps: int = 0
    training_sources: int = 0              # number of documents trained on
    contributors: int = 0                  # number of people who contributed
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    generation: int = 0                    # merge generation
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> SpecialistInfo:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Registry ────────────────────────────────────────────────────────────────

class SpecialistRegistry:
    """Local registry of specialist models."""

    def __init__(self, registry_path: str | Path = "~/.gdf/specialists.json"):
        self.path = Path(registry_path).expanduser()
        self.specialists: dict[str, SpecialistInfo] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            for name, info in data.items():
                self.specialists[name] = SpecialistInfo.from_dict(info)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {name: info.to_dict() for name, info in self.specialists.items()}
        self.path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    def register(self, info: SpecialistInfo) -> None:
        """Register or update a specialist."""
        self.specialists[info.name] = info
        self._save()

    def unregister(self, name: str) -> bool:
        if name in self.specialists:
            del self.specialists[name]
            self._save()
            return True
        return False

    def get(self, name: str) -> SpecialistInfo | None:
        return self.specialists.get(name)

    def list_all(self) -> list[SpecialistInfo]:
        return list(self.specialists.values())

    def list_domain(self, domain: str) -> list[SpecialistInfo]:
        return [s for s in self.specialists.values() if s.domain == domain]

    def domains(self) -> list[str]:
        return sorted(set(s.domain for s in self.specialists.values()))


# ── Router ──────────────────────────────────────────────────────────────────

class Router:
    """Routes queries to the best specialist model(s).

    Uses keyword matching + domain scoring. Simple but effective.
    A more advanced version could use a small classifier model.
    """

    def __init__(self, registry: SpecialistRegistry):
        self.registry = registry

    def route(self, query: str, top_k: int = 3) -> list[tuple[SpecialistInfo, float]]:
        """Find the best specialist(s) for a query.

        Returns list of (specialist, relevance_score) sorted by relevance.
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())

        scored: list[tuple[SpecialistInfo, float]] = []

        for specialist in self.registry.list_all():
            score = self._score(query_lower, query_words, specialist)
            if score > 0:
                scored.append((specialist, score))

        # Sort by relevance score (descending)
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def _score(self, query: str, query_words: set[str], spec: SpecialistInfo) -> float:
        """Score how relevant a specialist is for a query."""
        score = 0.0

        # Keyword matches (strongest signal)
        for kw in spec.keywords:
            kw_lower = kw.lower()
            if kw_lower in query:
                # Exact substring match — strong
                score += 3.0
            elif kw_lower in query_words:
                # Word match
                score += 2.0
            else:
                # Partial word overlap
                kw_words = set(kw_lower.split())
                overlap = len(query_words & kw_words)
                if overlap:
                    score += overlap * 1.0

        # Domain name match
        if spec.domain.lower() in query:
            score += 2.0

        # Description word overlap
        desc_words = set(spec.description.lower().split())
        desc_overlap = len(query_words & desc_words)
        score += desc_overlap * 0.3

        # Quality bonus — better-trained specialists get a small boost
        if spec.quality_score > 0:
            # Lower perplexity = better. Normalize to 0-1 range.
            quality_bonus = 1.0 / (1.0 + spec.quality_score / 100.0)
            score *= (1.0 + quality_bonus * 0.2)

        # Recency bonus
        age_days = (time.time() - spec.updated) / 86400
        if age_days < 7:
            score *= 1.1  # recently updated = slightly preferred

        return score

    def route_or_general(self, query: str, general_name: str = "general",
                          threshold: float = 1.0) -> list[tuple[SpecialistInfo, float]]:
        """Route to specialists, falling back to general model if no good match.

        Args:
            query: The user query.
            general_name: Name of the general/fallback specialist.
            threshold: Minimum score to be considered a match.
        """
        results = self.route(query, top_k=3)

        # Filter by threshold
        good_results = [(s, score) for s, score in results if score >= threshold]

        if good_results:
            return good_results

        # Fall back to general model
        general = self.registry.get(general_name)
        if general:
            return [(general, 0.1)]

        # No specialists at all — return whatever we have
        return results[:1] if results else []


# ── Multi-expert query ──────────────────────────────────────────────────────

def query_specialists(
    query: str,
    registry: SpecialistRegistry,
    top_k: int = 2,
    max_tokens: int = 200,
    temperature: float = 0.7,
) -> list[dict]:
    """Query the best specialist(s) and return their responses.

    Args:
        query: User's question/prompt.
        registry: Specialist registry to search.
        top_k: Number of specialists to query.
        max_tokens: Max tokens per response.
        temperature: Generation temperature.

    Returns:
        List of dicts with specialist name, score, and generated response.
    """
    from .api import GDFModel

    router = Router(registry)
    matches = router.route(query, top_k=top_k)

    results = []
    for specialist, score in matches:
        try:
            bm = GDFModel.load(specialist.model_path)
            response = bm.generate(
                prompt=query,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            results.append({
                "specialist": specialist.name,
                "domain": specialist.domain,
                "score": score,
                "response": response,
                "quality": specialist.quality_score,
            })
        except Exception as e:
            results.append({
                "specialist": specialist.name,
                "domain": specialist.domain,
                "score": score,
                "response": f"[Error: {e}]",
                "quality": specialist.quality_score,
            })

    return results


# ── Suggested specialist domains ────────────────────────────────────────────

# These are starting points — the community defines what specialists exist.
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

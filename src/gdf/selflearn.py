"""Autonomous self-learning loop: fetch → train → evaluate → repeat."""

from __future__ import annotations

import json
import math
import random
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from .tokenizer import encode, decode
from .model import TinyTransformer


@dataclass
class SelfLearnConfig:
    """Configuration for the autonomous learning loop."""
    eval_fraction: float = 0.15          # fraction of text held out for evaluation
    min_text_length: int = 500           # skip articles shorter than this
    max_text_length: int = 50_000        # truncate articles longer than this
    save_every: int = 5                  # save checkpoint every N cycles
    pause_seconds: float = 2.0           # pause between fetches (be polite)
    max_cycles: int = 0                  # 0 = run forever
    improvement_threshold: float = 0.0   # save best only if perplexity drops by at least this
    chunk_size: int = 256
    epochs_per_article: int = 3
    log_file: str | None = None          # optional JSON-lines log


@dataclass
class CycleResult:
    """Result of one fetch→train→eval cycle."""
    cycle: int
    source: str
    title: str
    text_length: int
    train_loss_start: float
    train_loss_end: float
    perplexity_before: float
    perplexity_after: float
    improved: bool
    elapsed_seconds: float


# ── Wikipedia fetching ──────────────────────────────────────────────────────

def fetch_wikipedia_random(lang: str = "en") -> tuple[str, str]:
    """Fetch a random Wikipedia article. Returns (title, plain_text)."""
    # Step 1: get a random article title
    api = f"https://{lang}.wikipedia.org/api/rest_v1/page/random/summary"
    req = urllib.request.Request(api, headers={
        "User-Agent": "gdf/0.1 (self-learner; educational project)",
        "Accept": "application/json",
    })
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    title = data.get("title", "Unknown")

    # Step 2: get the full article text via the TextExtracts API
    params = (
        f"https://{lang}.wikipedia.org/w/api.php?"
        f"action=query&titles={urllib.request.quote(title)}"
        f"&prop=extracts&explaintext=1&exsectionformat=plain&format=json"
    )
    req2 = urllib.request.Request(params, headers={
        "User-Agent": "gdf/0.1 (self-learner; educational project)",
    })
    with urllib.request.urlopen(req2, timeout=15) as resp2:
        result = json.loads(resp2.read().decode("utf-8"))

    pages = result.get("query", {}).get("pages", {})
    for page in pages.values():
        text = page.get("extract", "")
        if text:
            return title, text

    # Fallback to summary extract
    return title, data.get("extract", "")


def fetch_wikipedia_topic(topic: str, lang: str = "en") -> list[tuple[str, str]]:
    """Search Wikipedia for a topic and return up to 5 (title, text) pairs."""
    search_url = (
        f"https://{lang}.wikipedia.org/w/api.php?"
        f"action=query&list=search&srsearch={urllib.request.quote(topic)}"
        f"&srlimit=5&format=json"
    )
    req = urllib.request.Request(search_url, headers={
        "User-Agent": "gdf/0.1 (self-learner; educational project)",
    })
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    results = []
    for item in data.get("query", {}).get("search", []):
        title = item["title"]
        # Fetch full text
        params = (
            f"https://{lang}.wikipedia.org/w/api.php?"
            f"action=query&titles={urllib.request.quote(title)}"
            f"&prop=extracts&explaintext=1&exsectionformat=plain&format=json"
        )
        req2 = urllib.request.Request(params, headers={
            "User-Agent": "gdf/0.1 (self-learner; educational project)",
        })
        try:
            with urllib.request.urlopen(req2, timeout=15) as resp2:
                result = json.loads(resp2.read().decode("utf-8"))
            pages = result.get("query", {}).get("pages", {})
            for page in pages.values():
                text = page.get("extract", "")
                if text and len(text) > 200:
                    results.append((title, text))
        except Exception:
            continue

    return results


# ── Evaluation ──────────────────────────────────────────────────────────────

def compute_perplexity(model: TinyTransformer, text: str, chunk_size: int = 256,
                       device: torch.device | None = None) -> float:
    """Compute perplexity of the model on a text string.

    Lower perplexity = model predicts the text better = it has learned.
    """
    tokens = encode(text)
    if len(tokens) < 2:
        return float("inf")

    if device is None:
        # Infer device from model parameters
        device = next(model.parameters()).device

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        # Process in chunks
        for start in range(0, len(tokens) - 1, chunk_size):
            end = min(start + chunk_size, len(tokens))
            chunk = tokens[start:end]
            if len(chunk) < 2:
                break

            x = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
            y = torch.tensor([chunk[1:]], dtype=torch.long, device=device)

            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            n = len(chunk) - 1
            total_loss += loss.item() * n
            total_tokens += n

    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


# ── Autonomous loop ─────────────────────────────────────────────────────────

class SelfLearner:
    """Autonomous learning loop manager."""

    def __init__(self, bit_model, config: SelfLearnConfig | None = None):
        self.bm = bit_model
        self.config = config or SelfLearnConfig()
        self.cycle_count = 0
        self.best_perplexity = float("inf")
        self.history: list[CycleResult] = []
        self._stop = False

    def stop(self):
        """Signal the loop to stop after the current cycle."""
        self._stop = True

    def _split_train_eval(self, text: str) -> tuple[str, str]:
        """Split text into train and eval portions."""
        sentences = [s.strip() for s in text.replace("\n", ". ").split(". ") if s.strip()]
        if len(sentences) < 4:
            # Too short to split meaningfully
            return text, text

        n_eval = max(2, int(len(sentences) * self.config.eval_fraction))
        # Take eval sentences from the middle (less likely to be headers/footers)
        mid = len(sentences) // 2
        eval_start = mid - n_eval // 2
        eval_end = eval_start + n_eval

        eval_sentences = sentences[eval_start:eval_end]
        train_sentences = sentences[:eval_start] + sentences[eval_end:]

        return ". ".join(train_sentences), ". ".join(eval_sentences)

    def run_cycle(
        self,
        topic: str | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> CycleResult | None:
        """Run one fetch→train→evaluate cycle.

        Args:
            topic: If given, search for this topic. Otherwise fetch random.
            on_status: Callback for status messages.
        """
        def status(msg: str):
            if on_status:
                on_status(msg)

        cfg = self.config
        self.cycle_count += 1

        # 1. Fetch
        status(f"Cycle {self.cycle_count}: Fetching...")
        try:
            if topic:
                articles = fetch_wikipedia_topic(topic)
                if not articles:
                    status(f"  No articles found for '{topic}', trying random...")
                    title, text = fetch_wikipedia_random()
                else:
                    title, text = articles[0]
            else:
                title, text = fetch_wikipedia_random()
        except Exception as e:
            status(f"  Fetch failed: {e}")
            return None

        # Truncate if too long
        if len(text) > cfg.max_text_length:
            text = text[:cfg.max_text_length]

        if len(text) < cfg.min_text_length:
            status(f"  Skipping '{title}' — too short ({len(text)} chars)")
            return None

        status(f"  Article: {title} ({len(text):,} chars)")

        # 2. Split into train/eval
        train_text, eval_text = self._split_train_eval(text)

        # 3. Measure perplexity BEFORE training
        ppl_before = compute_perplexity(self.bm.model, eval_text, cfg.chunk_size)
        status(f"  Perplexity before: {ppl_before:.2f}")

        # 4. Train
        t0 = time.time()
        result = self.bm.trainer.train_bulk(
            train_text,
            epochs=cfg.epochs_per_article,
            chunk_size=cfg.chunk_size,
        )
        elapsed = time.time() - t0

        # 5. Measure perplexity AFTER training
        ppl_after = compute_perplexity(self.bm.model, eval_text, cfg.chunk_size)
        improved = ppl_after < ppl_before - cfg.improvement_threshold
        status(f"  Perplexity after:  {ppl_after:.2f} ({'improved' if improved else 'no change'})")
        status(f"  Loss: {result['first_loss']:.4f} → {result['final_loss']:.4f} ({elapsed:.1f}s)")

        # Track best
        if ppl_after < self.best_perplexity:
            self.best_perplexity = ppl_after

        cycle_result = CycleResult(
            cycle=self.cycle_count,
            source="wikipedia",
            title=title,
            text_length=len(text),
            train_loss_start=result["first_loss"],
            train_loss_end=result["final_loss"],
            perplexity_before=ppl_before,
            perplexity_after=ppl_after,
            improved=improved,
            elapsed_seconds=elapsed,
        )
        self.history.append(cycle_result)

        # Log to file if configured
        if cfg.log_file:
            self._append_log(cycle_result)

        return cycle_result

    def run(
        self,
        save_path: str | None = None,
        topics: list[str] | None = None,
        on_status: Callable[[str], None] | None = None,
        on_cycle: Callable[[CycleResult], None] | None = None,
    ) -> list[CycleResult]:
        """Run the autonomous learning loop.

        Args:
            save_path: Where to save the model (auto-saves periodically).
            topics: Optional list of topics to cycle through. If None, uses random.
            on_status: Callback for log messages.
            on_cycle: Callback after each cycle with the result.

        Returns:
            List of all cycle results.
        """
        def status(msg: str):
            if on_status:
                on_status(msg)

        cfg = self.config
        topic_idx = 0
        self._stop = False

        status("Starting autonomous learning loop...")
        status(f"  Eval fraction: {cfg.eval_fraction}")
        status(f"  Epochs per article: {cfg.epochs_per_article}")
        status(f"  Save every: {cfg.save_every} cycles")
        if cfg.max_cycles:
            status(f"  Max cycles: {cfg.max_cycles}")
        status("")

        while not self._stop:
            if cfg.max_cycles and self.cycle_count >= cfg.max_cycles:
                status("Reached max cycles.")
                break

            # Pick topic
            topic = None
            if topics:
                topic = topics[topic_idx % len(topics)]
                topic_idx += 1

            # Run one cycle
            result = self.run_cycle(topic=topic, on_status=on_status)

            if result and on_cycle:
                on_cycle(result)

            # Auto-save
            if save_path and self.cycle_count % cfg.save_every == 0:
                self.bm.save(save_path)
                status(f"  Checkpoint saved ({self.cycle_count} cycles)")

            # Be polite to Wikipedia
            if not self._stop:
                time.sleep(cfg.pause_seconds)

        # Final save
        if save_path:
            self.bm.save(save_path)
            status(f"Final save ({self.cycle_count} cycles, best ppl: {self.best_perplexity:.2f})")

        return self.history

    def summary(self) -> dict:
        """Return a summary of all learning so far."""
        if not self.history:
            return {"cycles": 0}

        ppls = [r.perplexity_after for r in self.history]
        improved_count = sum(1 for r in self.history if r.improved)

        return {
            "cycles": len(self.history),
            "total_text": sum(r.text_length for r in self.history),
            "avg_perplexity": sum(ppls) / len(ppls),
            "best_perplexity": min(ppls),
            "worst_perplexity": max(ppls),
            "latest_perplexity": ppls[-1],
            "improved_cycles": improved_count,
            "improvement_rate": improved_count / len(self.history),
            "total_time": sum(r.elapsed_seconds for r in self.history),
        }

    def _append_log(self, result: CycleResult) -> None:
        """Append a cycle result to the log file."""
        entry = {
            "cycle": result.cycle,
            "source": result.source,
            "title": result.title,
            "text_length": result.text_length,
            "train_loss_start": result.train_loss_start,
            "train_loss_end": result.train_loss_end,
            "perplexity_before": result.perplexity_before,
            "perplexity_after": result.perplexity_after,
            "improved": result.improved,
            "elapsed": result.elapsed_seconds,
            "timestamp": time.time(),
        }
        with open(self.config.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

"""File tree crawler — scan folders for text files, crawl web URLs, auto-learn."""

from __future__ import annotations

import os
import time
import urllib.parse
from collections import deque
from pathlib import Path
from typing import Callable, Iterator

from .fetcher import fetch_url_raw, extract_links, fetch_url
from .selflearn import (
    SelfLearnConfig, CycleResult, compute_perplexity,
    fetch_wikipedia_random, fetch_wikipedia_topic,
)

# Built-in diverse seed pool — used when the hub doesn't provide reseed URLs
DEFAULT_RESEED_URLS = [
    # Science & Tech
    "https://www.quantamagazine.org/",
    "https://phys.org/",
    "https://www.scientificamerican.com/",
    "https://www.nature.com/subjects",
    "https://science.nasa.gov/solar-system/",
    "https://home.cern/science",
    "https://arxiv.org/list/cs.AI/recent",
    "https://developer.mozilla.org/en-US/docs/Web",
    "https://www.space.com/science",
    # Education
    "https://ocw.mit.edu/courses/mathematics/",
    "https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/",
    "https://plato.stanford.edu/contents.html",
    "https://www.khanacademy.org/science",
    "https://brilliant.org/wiki/",
    # Literature & Humanities
    "https://www.gutenberg.org/browse/scores/top",
    "https://www.poetryfoundation.org/poems",
    "https://publicdomainreview.org/collections/",
    "https://www.bl.uk/discovering-literature",
    # Reference & Government
    "https://www.loc.gov/collections/",
    "https://www.cia.gov/the-world-factbook/",
    "https://www.law.cornell.edu/wex",
    "https://www.bls.gov/ooh/",
    # Health
    "https://medlineplus.gov/healthtopics.html",
    "https://www.who.int/health-topics",
    "https://www.ncbi.nlm.nih.gov/books/",
    # News
    "https://www.bbc.com/news/science_and_environment",
    "https://www.reuters.com/science/",
    "https://apnews.com/hub/science",
    "https://www.npr.org/sections/science/",
    # Math & CS
    "https://mathworld.wolfram.com/",
    "https://cp-algorithms.com/",
    "https://docs.python.org/3/tutorial/",
    "https://doc.rust-lang.org/book/",
    # Geography & Environment
    "https://earthobservatory.nasa.gov/",
    "https://www.nationalgeographic.com/science/",
    # Arts
    "https://www.metmuseum.org/art/collection",
    "https://artsandculture.google.com/",
    # Wikipedia portals (diverse entry points, not just Special:Random)
    "https://en.wikipedia.org/wiki/Portal:Science",
    "https://en.wikipedia.org/wiki/Portal:Technology",
    "https://en.wikipedia.org/wiki/Portal:History",
    "https://en.wikipedia.org/wiki/Portal:Arts",
    "https://en.wikipedia.org/wiki/Portal:Philosophy",
]

# Extensions we can learn from
TEXT_EXTENSIONS = {
    ".txt", ".md", ".rst", ".csv", ".json", ".xml",
    ".py", ".js", ".ts", ".html", ".css", ".java", ".c", ".cpp", ".h",
    ".go", ".rs", ".rb", ".php", ".sh", ".bat", ".ps1",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".log", ".sql", ".r", ".m", ".swift", ".kt",
}

# Folders to always skip
SKIP_DIRS = {
    "__pycache__", "node_modules", ".git", ".svn", ".hg",
    "venv", ".venv", "env", ".env", ".tox",
    "dist", "build", ".eggs", "*.egg-info",
    ".idea", ".vscode", ".vs",
    "target", "bin", "obj",
}

MAX_FILE_SIZE = 1_000_000  # 1MB max per file


def discover_files(root: str | Path, extensions: set[str] | None = None) -> list[Path]:
    """Walk a directory tree and return all trainable text files, sorted by size."""
    root = Path(root)
    exts = extensions or TEXT_EXTENSIONS
    files: list[Path] = []

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune skipped dirs
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]

        for fname in filenames:
            p = Path(dirpath) / fname
            if p.suffix.lower() in exts and p.stat().st_size <= MAX_FILE_SIZE:
                files.append(p)

    files.sort(key=lambda p: p.stat().st_size)
    return files


def crawl_and_train(
    model,
    root: str | Path,
    epochs: int = 3,
    extensions: set[str] | None = None,
    on_file: Callable[[int, int, Path], None] | None = None,
    on_step: Callable[[int, int, float], None] | None = None,
    check_stop: Callable[[], bool] | None = None,
) -> dict:
    """Crawl a folder tree and train on every text file found.

    Args:
        model: GDFModel instance.
        root: Root directory to crawl.
        epochs: Epochs per file.
        extensions: File extensions to include (default: TEXT_EXTENSIONS).
        on_file: Callback(file_index, total_files, file_path) called before each file.
        on_step: Callback(step, total_steps, loss) for training progress.
        check_stop: Callback that returns True if we should stop.

    Returns:
        Dict with crawl stats.
    """
    files = discover_files(root, extensions)
    total_files = len(files)
    files_trained = 0
    total_steps = 0
    total_bytes = 0
    stopped = False

    for i, fp in enumerate(files):
        if check_stop and check_stop():
            stopped = True
            break

        if on_file:
            on_file(i, total_files, fp)

        try:
            result = model.train_file(fp, epochs=epochs, on_step=on_step)
            files_trained += 1
            total_steps += result.get("steps", 0)
            total_bytes += fp.stat().st_size
        except Exception:
            # Skip files that fail to read/train
            continue

    return {
        "root": str(root),
        "files_found": total_files,
        "files_trained": files_trained,
        "total_steps": total_steps,
        "total_bytes": total_bytes,
        "stopped_early": stopped,
    }


# ── Web crawling ────────────────────────────────────────────────────────────

def crawl_url(
    url: str,
    depth: int = 1,
    max_pages: int = 50,
    pause: float = 1.0,
) -> list[tuple[str, str]]:
    """BFS crawl starting from *url*, following same-domain links up to *depth*.

    Args:
        url: Starting URL.
        depth: How many link-hops to follow (1 = starting page only).
        max_pages: Stop after fetching this many pages.
        pause: Seconds to wait between requests.

    Returns:
        List of (url, extracted_text) pairs.
    """
    visited: set[str] = set()
    results: list[tuple[str, str]] = []
    # queue items: (url, current_depth)
    queue: deque[tuple[str, int]] = deque([(url, 0)])

    while queue and len(results) < max_pages:
        current_url, current_depth = queue.popleft()
        if current_url in visited:
            continue
        visited.add(current_url)

        try:
            raw_html, text = fetch_url_raw(current_url)
        except Exception:
            continue

        if text and len(text.strip()) > 50:
            results.append((current_url, text))

        # Follow links if we haven't reached max depth
        if current_depth < depth - 1:
            links = extract_links(raw_html, current_url)
            for link in links:
                if link not in visited:
                    queue.append((link, current_depth + 1))

        if pause and len(results) < max_pages and queue:
            time.sleep(pause)

    return results


def autonomous_crawl(
    seed_url: str,
    pause: float = 1.0,
    max_visited: int = 500,
    min_text_length: int = 200,
    max_text_length: int = 50_000,
    check_stop: Callable[[], bool] | None = None,
    reseed_urls: list[str] | None = None,
) -> Iterator[tuple[str, str]]:
    """Yield (url, text) pairs indefinitely from a BFS web crawl.

    Each peer crawls autonomously from a seed URL. When the queue empties,
    reseeds from a diverse pool of URLs. When visited exceeds max_visited,
    clears state and reseeds to bound memory.

    Args:
        seed_url: Starting URL for the crawl.
        pause: Seconds to wait between fetches.
        max_visited: Clear visited set and reseed after this many pages.
        min_text_length: Skip pages with less text than this.
        max_text_length: Truncate page text to this length.
        check_stop: Callback that returns True to stop crawling.
        reseed_urls: Pool of diverse URLs to pick from when reseeding.
            Falls back to Wikipedia random if empty.
    """
    import random as _random

    seeds_pool = list(reseed_urls) if reseed_urls else list(DEFAULT_RESEED_URLS)
    _random.shuffle(seeds_pool)
    visited: set[str] = set()
    # Prime the queue with the seed + several diverse sources so we don't
    # get trapped on a single domain (e.g. Wikipedia linking to Wikipedia)
    initial_seeds = [seed_url] + _random.sample(
        seeds_pool, min(5, len(seeds_pool))
    )
    queue: deque[str] = deque(initial_seeds)
    pages_since_inject = 0
    inject_every = 3  # inject a fresh diverse seed every N pages

    def _pick_reseed() -> str:
        """Pick a random reseed URL from the pool, or fall back to Wikipedia."""
        if seeds_pool:
            return _random.choice(seeds_pool)
        return "https://en.wikipedia.org/wiki/Special:Random"

    while True:
        if check_stop and check_stop():
            return

        # Auto-reseed when queue is empty
        if not queue:
            queue.append(_pick_reseed())

        # Memory bound: clear and reseed when visited is too large
        if len(visited) > max_visited:
            visited.clear()
            queue.clear()
            queue.extend(_random.sample(seeds_pool, min(5, len(seeds_pool))))
            continue

        url = queue.popleft()
        if url in visited:
            continue
        visited.add(url)

        # Fetch
        try:
            raw_html, text = fetch_url_raw(url)
        except Exception:
            continue

        # Extract and enqueue same-domain links (follow depth on current site)
        try:
            links = extract_links(raw_html, url, same_domain_only=True)
            for link in links[:10]:  # limit per-page links to avoid domain flooding
                if link not in visited:
                    queue.append(link)
        except Exception:
            pass

        # Periodically inject a diverse seed so we don't stay on one domain
        pages_since_inject += 1
        if pages_since_inject >= inject_every:
            queue.appendleft(_pick_reseed())
            pages_since_inject = 0

        # Yield if enough text
        text = text.strip()
        if len(text) >= min_text_length:
            if len(text) > max_text_length:
                text = text[:max_text_length]
            yield (url, text)

        # Be polite
        if pause:
            time.sleep(pause)


def crawl_sources_file(path: str | Path) -> list[str]:
    """Read a text file containing one source (URL or path) per line.

    Lines starting with ``#`` and blank lines are skipped.
    """
    sources: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            sources.append(line)
    return sources


# ── Auto mode (replaces SelfLearner.run) ────────────────────────────────────

def crawl_auto(
    model,
    config: SelfLearnConfig | None = None,
    topics: list[str] | None = None,
    on_status: Callable[[str], None] | None = None,
    on_cycle: Callable[[CycleResult], None] | None = None,
    check_stop: Callable[[], bool] | None = None,
    save_path: str | None = None,
) -> dict:
    """Autonomous fetch-train-evaluate loop (replaces SelfLearner.run).

    Fetches Wikipedia articles (random or topic-based), trains on them,
    measures perplexity improvement, and saves checkpoints.

    Args:
        model: GDFModel instance (has .model, .trainer, .save()).
        config: Learning configuration.
        topics: Optional topic list to cycle through.
        on_status: Status message callback.
        on_cycle: Called after each cycle with a CycleResult.
        check_stop: Returns True to stop the loop.
        save_path: Where to auto-save checkpoints.

    Returns:
        Summary dict with stats.
    """
    cfg = config or SelfLearnConfig()

    def status(msg: str):
        if on_status:
            on_status(msg)

    topic_idx = 0
    cycle_count = 0
    best_perplexity = float("inf")
    history: list[CycleResult] = []

    status("Starting autonomous learning loop...")
    status(f"  Eval fraction: {cfg.eval_fraction}")
    status(f"  Epochs per article: {cfg.epochs_per_article}")
    status(f"  Save every: {cfg.save_every} cycles")
    if cfg.max_cycles:
        status(f"  Max cycles: {cfg.max_cycles}")
    status("")

    while True:
        if check_stop and check_stop():
            break
        if cfg.max_cycles and cycle_count >= cfg.max_cycles:
            status("Reached max cycles.")
            break

        cycle_count += 1

        # Pick topic
        topic = None
        if topics:
            topic = topics[topic_idx % len(topics)]
            topic_idx += 1

        # Fetch
        status(f"Cycle {cycle_count}: Fetching...")
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
            continue

        if len(text) > cfg.max_text_length:
            text = text[:cfg.max_text_length]
        if len(text) < cfg.min_text_length:
            status(f"  Skipping '{title}' — too short ({len(text)} chars)")
            continue

        status(f"  Article: {title} ({len(text):,} chars)")

        # Split train/eval
        sentences = [s.strip() for s in text.replace("\n", ". ").split(". ") if s.strip()]
        if len(sentences) < 4:
            train_text, eval_text = text, text
        else:
            n_eval = max(2, int(len(sentences) * cfg.eval_fraction))
            mid = len(sentences) // 2
            eval_start = mid - n_eval // 2
            eval_end = eval_start + n_eval
            eval_text = ". ".join(sentences[eval_start:eval_end])
            train_text = ". ".join(sentences[:eval_start] + sentences[eval_end:])

        # Measure perplexity before
        ppl_before = compute_perplexity(model.model, eval_text, cfg.chunk_size)
        status(f"  Perplexity before: {ppl_before:.2f}")

        # Train
        t0 = time.time()
        result = model.trainer.train_bulk(
            train_text,
            epochs=cfg.epochs_per_article,
            chunk_size=cfg.chunk_size,
        )
        elapsed = time.time() - t0

        # Measure perplexity after
        ppl_after = compute_perplexity(model.model, eval_text, cfg.chunk_size)
        improved = ppl_after < ppl_before - cfg.improvement_threshold
        status(f"  Perplexity after:  {ppl_after:.2f} ({'improved' if improved else 'no change'})")
        status(f"  Loss: {result['first_loss']:.4f} → {result['final_loss']:.4f} ({elapsed:.1f}s)")

        if ppl_after < best_perplexity:
            best_perplexity = ppl_after

        cr = CycleResult(
            cycle=cycle_count,
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
        history.append(cr)

        if on_cycle:
            on_cycle(cr)

        # Auto-save
        if save_path and cycle_count % cfg.save_every == 0:
            model.save(save_path)
            status(f"  Checkpoint saved ({cycle_count} cycles)")

        # Be polite
        if not (check_stop and check_stop()):
            time.sleep(cfg.pause_seconds)

    # Final save
    if save_path:
        model.save(save_path)
        status(f"Final save ({cycle_count} cycles, best ppl: {best_perplexity:.2f})")

    # Build summary
    if not history:
        return {"cycles": 0}

    ppls = [r.perplexity_after for r in history]
    improved_count = sum(1 for r in history if r.improved)
    return {
        "cycles": len(history),
        "total_text": sum(r.text_length for r in history),
        "avg_perplexity": sum(ppls) / len(ppls),
        "best_perplexity": min(ppls),
        "worst_perplexity": max(ppls),
        "latest_perplexity": ppls[-1],
        "improved_cycles": improved_count,
        "improvement_rate": improved_count / len(history),
        "total_time": sum(r.elapsed_seconds for r in history),
    }

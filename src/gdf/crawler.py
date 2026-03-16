"""File tree crawler — scan folders for text files and train on them."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

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

"""Version checking and auto-update for gdf."""

from __future__ import annotations

import re
import sys
import subprocess
import time
import urllib.request

from .config import load_config, save_config


PYPROJECT_URL = "https://raw.githubusercontent.com/gdf-ai/gdf/main/pyproject.toml"
UPDATE_CHECK_TTL = 86400  # 24 hours


def _parse_version(v: str) -> tuple[int, ...]:
    """Parse '0.1.2' into (0, 1, 2)."""
    return tuple(int(x) for x in v.strip().split("."))


def check_for_update() -> tuple[bool, str | None]:
    """Check if a newer version is available on GitHub.

    Returns (update_available, remote_version) or (False, None).
    Only checks once per 24h (based on config timestamp).
    """
    try:
        cfg = load_config()
        last_check = cfg.get("last_update_check", 0)
        if time.time() - last_check < UPDATE_CHECK_TTL:
            return False, None

        # Fetch remote pyproject.toml
        req = urllib.request.Request(PYPROJECT_URL, headers={
            "User-Agent": "gdf/0.1",
        })
        with urllib.request.urlopen(req, timeout=3) as resp:
            content = resp.read().decode("utf-8")

        # Parse version
        match = re.search(r'version\s*=\s*"([^"]+)"', content)
        if not match:
            return False, None
        remote_version = match.group(1)

        # Save check timestamp
        cfg["last_update_check"] = time.time()
        save_config(cfg)

        # Compare versions
        from . import __version__
        try:
            if _parse_version(remote_version) > _parse_version(__version__):
                return True, remote_version
        except (ValueError, TypeError):
            pass

        return False, None

    except Exception:
        return False, None


def do_update() -> bool:
    """Run pip install --upgrade to update gdf."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade",
             "git+https://github.com/gdf-ai/gdf.git"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        return result.returncode == 0
    except Exception:
        return False


def maybe_auto_update() -> None:
    """Check for updates and auto-install if available. Called on CLI startup."""
    try:
        available, remote_version = check_for_update()
        if available and remote_version:
            from . import __version__
            print(f"  Updating gdf {__version__} -> {remote_version}...")
            if do_update():
                print(f"  Updated! Restart gdf to use {remote_version}.")
            else:
                print(f"  Update failed. Run manually: gdf update")
    except Exception:
        pass

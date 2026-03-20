"""CLI for gdf — volunteer GPU compute network."""

from __future__ import annotations

import sys
import time
import signal
import threading
from pathlib import Path

import click

from .api import GDFModel
from .device import device_info, format_device_info
from .crawler import autonomous_crawl
from .distributed import Hub, Peer
from .registry import (
    ModelInfo, ModelRegistry, fetch_registry, get_model,
)
from .updater import maybe_auto_update


# ── Helpers ──────────────────────────────────────────────────────────────────

def _pick(prompt: str, options: list[str]) -> int:
    """Show numbered choices, return index."""
    for i, opt in enumerate(options, 1):
        click.echo(f"  {i}) {opt}")
    while True:
        try:
            raw = click.prompt(prompt, prompt_suffix=" > ")
            n = int(raw)
            if 1 <= n <= len(options):
                return n - 1
        except (ValueError, EOFError, click.Abort):
            pass
        click.echo(f"  Enter 1-{len(options)}")


def _safe_echo(text: str) -> None:
    safe = text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    click.echo(safe)


# ── CLI Group ────────────────────────────────────────────────────────────────

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """gdf — Volunteer GPU Compute Network."""
    maybe_auto_update()
    if ctx.invoked_subcommand is not None:
        return

    # Bare `gdf` → show dashboard
    info = device_info()
    gpu_line = format_device_info(info)

    click.echo()
    click.echo("  gdf — Volunteer GPU Compute Network")
    click.echo("  " + "=" * 42)
    click.echo()
    click.echo(f"  {gpu_line}")
    click.echo()
    click.echo("  Commands:")
    click.echo("    gdf contribute [model]    Train a model with your GPU")
    click.echo("    gdf chat [model]          Chat with a model")
    click.echo("    gdf status                Show contribution stats")
    click.echo("    gdf model list            See available models")
    click.echo("    gdf hub                   Run a coordination server")
    click.echo("    gdf version / update      Version info")
    click.echo()


# ── contribute ──────────────────────────────────────────────────────────────

@cli.command()
@click.argument("model_name", required=False)
@click.option("--push-every", default=5, help="Push delta every N pages crawled")
@click.option("--epochs", "-e", default=3, help="Training epochs per page")
def contribute(model_name: str | None, push_every: int, epochs: int):
    """Train a model with your GPU.

    Crawls the web autonomously, trains locally, and pushes weight
    deltas back. Fully autonomous — hub assigns a seed URL at registration.

    \b
      gdf contribute              Pick a model interactively
      gdf contribute general      Contribute to a specific model
    """
    registry = ModelRegistry()
    remote = registry.list_remote()
    active = [e for e in remote if e.status == "active"]

    entry = None

    # 1. Explicit model name
    if model_name:
        entry = registry.get(model_name)
        if not entry:
            for e in active:
                if e.name == model_name:
                    entry = e
                    break
        if not entry:
            click.echo(f"  Model '{model_name}' not found.")
            if active:
                click.echo("  Available: " + ", ".join(e.name for e in active))
            return

    # 2. Pick from remote models
    if not entry:
        if not active:
            click.echo()
            click.echo("  No models available in the network yet.")
            click.echo()
            click.echo("  To run your own hub: gdf hub --model <path>")
            return

        click.echo()
        click.echo("  Available models:")
        click.echo()
        options = []
        for e in active:
            label = f"{e.name} — {e.description} ({e.size})"
            options.append(label)
        choice = _pick("Pick a model", options)
        entry = active[choice]

    if not entry.hub_url or not entry.token:
        click.echo(f"  Model '{entry.name}' has no hub configured.")
        return

    # Create peer and register
    peer = Peer(hub_url=entry.hub_url, token=entry.token)

    click.echo()
    click.echo(f"  Model:   {entry.name}")
    click.echo(f"  Hub:     {entry.hub_url}")
    click.echo(f"  Peer ID: {peer.peer_id}")
    click.echo()

    info = device_info()
    click.echo(f"  {format_device_info(info)}")
    click.echo()

    click.echo("  Registering with hub...")
    try:
        reg = peer.register()
        click.echo(f"  Registered! Hub generation: {reg.get('generation', '?')}")
    except Exception as e:
        click.echo(f"  Failed to connect: {e}")
        return

    # Pull model from hub
    click.echo("  Pulling model from hub...")
    try:
        _, pull_info = peer.pull_model()
        click.echo(f"  Model ready ({pull_info.get('params', '?'):,} params)")
    except Exception as e:
        click.echo(f"  Failed to pull model: {e}")
        return

    bm = GDFModel.load(peer.local_model_path)

    seed_url = reg.get("seed_url", "https://en.wikipedia.org/wiki/Special:Random")
    reseed_urls = reg.get("reseed_urls", [])
    click.echo()
    click.echo("  Contributing. Press Ctrl+C to stop.")
    click.echo(f"  Seed: {seed_url}")
    if reseed_urls:
        click.echo(f"  Reseed pool: {len(reseed_urls)} sources (from hub)")
    else:
        from .crawler import DEFAULT_RESEED_URLS
        click.echo(f"  Reseed pool: {len(DEFAULT_RESEED_URLS)} built-in sources")
    click.echo()

    # Ctrl+C handler
    stop = threading.Event()
    original_handler = signal.getsignal(signal.SIGINT)

    def _handle_stop(sig, frame):
        if stop.is_set():
            signal.signal(signal.SIGINT, original_handler)
            raise KeyboardInterrupt
        click.echo("\n  Stopping after current batch... (Ctrl+C again to force quit)")
        stop.set()

    signal.signal(signal.SIGINT, _handle_stop)

    pages_trained = 0
    total_chars = 0
    push_count = 0

    crawl_iter = autonomous_crawl(
        seed_url=seed_url, pause=1.0, check_stop=stop.is_set,
        reseed_urls=reseed_urls,
    )

    try:
        for url, text in crawl_iter:
            if stop.is_set():
                break

            # Train locally
            try:
                result = bm.trainer.train_bulk(text, epochs=epochs, chunk_size=256)
                bm.save(peer.local_model_path)
            except Exception as e:
                click.echo(f"  Train error: {e}")
                continue

            pages_trained += 1
            total_chars += len(text)

            # Display progress
            loss = result.get("final_loss", 0)
            click.echo(f"  [{pages_trained}] {url[:60]:60s} "
                       f"{len(text):>6,} chars  loss={loss:.4f}")

            # Push delta + pull latest every N pages
            if pages_trained % push_every == 0:
                click.echo("  >> Pushing delta...")
                try:
                    peer.push_model()
                    push_count += 1
                except Exception as e:
                    click.echo(f"  Push failed: {e}")

                click.echo("  << Pulling latest model...")
                try:
                    peer.pull_model()
                    bm = GDFModel.load(peer.local_model_path)
                except Exception as e:
                    click.echo(f"  Pull failed: {e}")

    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGINT, original_handler)

    # Final push
    if pages_trained > 0:
        click.echo()
        click.echo("  Final push...")
        try:
            peer.push_model()
            push_count += 1
        except Exception:
            pass

    click.echo()
    click.echo(f"  Session complete!")
    click.echo(f"  Pages trained: {pages_trained}")
    click.echo(f"  Total chars:   {total_chars:,}")
    click.echo(f"  Pushes:        {push_count}")
    click.echo("  Thanks for contributing!")


# ── chat (interactive) ───────────────────────────────────────────────────────

@cli.command()
@click.argument("model_name", required=False)
def chat(model_name: str | None):
    """Chat with a model.

    Talk to any model in the network, or pick a specific model.

    \b
      gdf chat                    Pick a model interactively
      gdf chat general            Chat with a specific model

    \b
    Commands:
      /models   List available models
      /switch   Pin to a specific model
      /quit     Exit
    """
    pinned_model = None
    pinned_name = None

    registry = ModelRegistry()

    if model_name:
        entry = registry.get(model_name)
        if entry and entry.hub_url and entry.token:
            click.echo(f"  Downloading {entry.name} from hub...")
            try:
                p = Peer(hub_url=entry.hub_url, token=entry.token)
                p.register()
                p.pull_model()
                pinned_model = GDFModel.load(p.local_model_path)
                pinned_name = entry.name
            except Exception as e:
                click.echo(f"  Failed to download: {e}")
                return
        elif entry and entry.model_path:
            try:
                pinned_model = GDFModel.load(entry.model_path)
                pinned_name = entry.name
            except Exception as e:
                click.echo(f"  Failed to load: {e}")
                return
        else:
            path = Path(model_name)
            if path.exists():
                pinned_model = GDFModel.load(str(path))
                pinned_name = path.stem
            else:
                click.echo(f"  Model '{model_name}' not found.")
                click.echo("  Try: gdf model list")
                return

    if not pinned_model:
        # Pick from available models
        remote = registry.list_remote()
        active = [e for e in remote if e.status == "active"]
        local = registry.list_local()

        all_options = []
        all_entries = []
        for m in local:
            if m.model_path:
                all_options.append(f"{m.name} (local)")
                all_entries.append(m)
        for e in active:
            all_options.append(f"{e.name} — {e.description} ({e.size})")
            all_entries.append(e)

        if not all_options:
            click.echo("  No models available. Run: gdf model list")
            return

        click.echo()
        choice = _pick("Pick a model", all_options)
        entry = all_entries[choice]

        if entry.hub_url and entry.token:
            click.echo(f"  Downloading {entry.name}...")
            try:
                p = Peer(hub_url=entry.hub_url, token=entry.token)
                p.register()
                p.pull_model()
                pinned_model = GDFModel.load(p.local_model_path)
                pinned_name = entry.name
            except Exception as e:
                click.echo(f"  Failed: {e}")
                return
        elif entry.model_path:
            try:
                pinned_model = GDFModel.load(entry.model_path)
                pinned_name = entry.name
            except Exception as e:
                click.echo(f"  Failed to load: {e}")
                return

    click.echo()
    click.echo(f"  Chatting with: {pinned_name}")
    click.echo("  Commands: /models /switch <name> /quit")
    click.echo()

    while True:
        try:
            line = click.prompt("", prompt_suffix="You: ")
        except (EOFError, click.Abort):
            click.echo()
            break

        line = line.strip()
        if not line:
            continue

        if line in ("/quit", "/q"):
            break

        elif line == "/models":
            click.echo()
            local = registry.list_local()
            if local:
                click.echo("  Local models:")
                for m in local:
                    click.echo(f"    {m.name}")
            remote = registry.list_remote()
            active = [e for e in remote if e.status == "active"]
            if active:
                click.echo("  Network models:")
                for e in active:
                    click.echo(f"    {e.name:25s} {e.description} ({e.size})")
            click.echo()
            continue

        elif line.startswith("/switch"):
            name = line[7:].strip()
            if not name:
                click.echo("  Usage: /switch <model-name>")
                continue
            entry = registry.get(name)
            if entry and entry.model_path:
                try:
                    pinned_model = GDFModel.load(entry.model_path)
                    pinned_name = entry.name
                    click.echo(f"  Switched to: {pinned_name}")
                except Exception as e:
                    click.echo(f"  Failed to load: {e}")
            elif entry and entry.hub_url and entry.token:
                click.echo(f"  Downloading {entry.name}...")
                try:
                    p = Peer(hub_url=entry.hub_url, token=entry.token)
                    p.register()
                    p.pull_model()
                    pinned_model = GDFModel.load(p.local_model_path)
                    pinned_name = entry.name
                    click.echo(f"  Switched to: {pinned_name}")
                except Exception as e:
                    click.echo(f"  Failed: {e}")
            else:
                click.echo(f"  Model '{name}' not found.")
            continue

        # Generate response
        response = pinned_model.generate(prompt=line, max_tokens=200, temperature=0.7)
        _safe_echo(f"  {response}")
        click.echo()


# ── status ───────────────────────────────────────────────────────────────────

@cli.command()
def status():
    """Show device info and network model statuses."""
    click.echo()
    info = device_info()
    click.echo(f"  {format_device_info(info)}")
    click.echo()

    registry = ModelRegistry()
    remote = registry.list_remote()
    active = [e for e in remote if e.status == "active"]

    if not active:
        click.echo("  No network models available.")
        click.echo()
        return

    click.echo(f"  Network models ({len(active)}):")
    click.echo()

    for e in active:
        click.echo(f"  {e.name:20s} {e.description}")
        if e.hub_url and e.token:
            try:
                p = Peer(hub_url=e.hub_url, token=e.token)
                st = p.hub_status()
                click.echo(f"    Peers: {st.get('active_peers', '?')}  "
                           f"Gen: {st.get('generation', '?')}  "
                           f"Contributions: {st.get('total_contributions', '?')}")
            except Exception:
                click.echo("    Hub unreachable")
        click.echo()


# ── hub (coordination server) ───────────────────────────────────────────────

@cli.command()
@click.option("--model", "-m", required=True, help="Model file to serve")
@click.option("--port", "-p", default=7677, help="Port to listen on")
@click.option("--host", default="0.0.0.0", help="Host to bind")
@click.option("--merge-every", default=3, help="Merge after N contributions")
@click.option("--strategy", "-s", default="fedavg",
              type=click.Choice(["fedavg", "ties"]))
@click.option("--token", default=None, envvar="GDF_HUB_TOKEN",
              help="Auth token (env: GDF_HUB_TOKEN, auto-generated if not set)")
@click.option("--parent", default=None, help="Parent hub URL (for hierarchy)")
@click.option("--seeds", default=None, help="Text file with seed URLs (one per line)")
def hub(model: str, port: int, host: str, merge_every: int,
        strategy: str, token: str | None, parent: str | None,
        seeds: str | None):
    """Run a distributed training hub.

    Deploy this on a VPS/cloud server so peers can connect
    and contribute GPU to train the model.

    \b
    Examples:
      gdf hub --model general.pt
      gdf hub --model general.pt --seeds urls.txt
      gdf hub --model general.pt --token mysecret
    """
    if not Path(model).exists():
        click.echo(f"  Model file not found: {model}")
        click.echo("  Create one first, or provide a valid path.")
        return

    # Auto-discover seeds.txt next to the model or in cwd
    if seeds is None:
        for candidate in [
            Path(model).parent / "seeds.txt",
            Path("seeds.txt"),
        ]:
            if candidate.exists():
                seeds = str(candidate)
                break

    h = Hub(
        model_path=model,
        merge_strategy=strategy,
        merge_every=merge_every,
        host=host,
        port=port,
        token=token,
        parent_hub=parent,
        seeds_file=seeds,
    )

    # Detect real IP addresses
    import socket
    local_ip = "127.0.0.1"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        pass

    click.echo()
    click.echo("  gdf hub")
    click.echo("  " + "=" * 30)
    click.echo(f"  Model:    {Path(model).name}")
    click.echo(f"  Strategy: {strategy}")
    click.echo(f"  Merge every {merge_every} contributions")
    click.echo(f"  Seeds: {len(h._seed_urls)}")
    click.echo(f"  Listening: {host}:{port}")
    if parent:
        click.echo(f"  Parent hub: {parent}")
    click.echo()
    click.echo(f"  TOKEN: {h.token}")
    click.echo()
    click.echo("  Share this with peers:")
    click.echo(f"    gdf contribute <model-name>")
    click.echo()
    click.echo(f"  Hub URL: http://{local_ip}:{port}")
    click.echo("  If running on a VPS, replace the IP with your server's public IP.")
    click.echo("  Press Ctrl+C to stop.")
    click.echo()

    h.start(on_log=lambda msg: _safe_echo(f"  {msg}"))


# ── model commands ──────────────────────────────────────────────────────────

@cli.group()
def model():
    """Manage models."""
    pass


@model.command("list")
def model_list():
    """List all models (local + network)."""
    registry = ModelRegistry()
    local = registry.list_local()
    remote = registry.list_remote()

    if not local and not remote:
        click.echo()
        click.echo("  No models available yet.")
        click.echo("  Once hubs are deployed, models will appear here.")
        click.echo()
        return

    click.echo()

    if local:
        click.echo(f"  Local models ({len(local)}):")
        for m in local:
            click.echo(f"    {m.name:25s} {m.description or ''}")
        click.echo()

    active = [e for e in remote if e.status == "active"]
    if active:
        click.echo(f"  Network models ({len(active)}):")
        for e in active:
            click.echo(f"    {e.name:25s} {e.description} ({e.size})")
        click.echo()


# ── version + update ─────────────────────────────────────────────────────────

@cli.command()
def version():
    """Show gdf version."""
    from . import __version__
    click.echo(f"  gdf {__version__}")


@cli.command()
def update():
    """Check for updates and install if available."""
    from . import __version__
    from .updater import check_for_update, do_update

    click.echo(f"  Current version: {__version__}")
    click.echo("  Checking for updates...")

    from .config import load_config, save_config
    cfg = load_config()
    cfg["last_update_check"] = 0
    save_config(cfg)

    available, remote_version = check_for_update()
    if available and remote_version:
        click.echo(f"  Update available: {__version__} -> {remote_version}")
        click.echo("  Installing...")
        if do_update():
            click.echo(f"  Updated to {remote_version}! Restart gdf to use.")
        else:
            click.echo("  Update failed. Try: pip install --upgrade git+https://github.com/gdf-ai/gdf.git")
    else:
        click.echo("  Already up to date.")


if __name__ == "__main__":
    cli()

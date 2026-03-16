"""CLI for gdf — intuitive, guided interface."""

from __future__ import annotations

import sys
import time
import signal
import threading
from pathlib import Path

import click

from .api import GDFModel
from .config import (
    get_default_model, set_default_model, resolve_model, load_config,
)
from .fetcher import fetch_url, is_url
from .crawler import discover_files, crawl_and_train
from .selflearn import SelfLearner, SelfLearnConfig
from .growth import grow_model, grow_wider, grow_deeper, GROWTH_STAGES, suggest_next_stage
from .bpe import BPETokenizer
from .device import device_info, format_device_info
from .distributed import Hub, Peer
from .specialists import (
    SpecialistInfo, SpecialistRegistry, Router, query_specialists,
    SUGGESTED_DOMAINS,
)
from .registry import fetch_registry, get_model


# ── Helpers ──────────────────────────────────────────────────────────────────

def _status_line() -> str:
    """One-line status of current default model."""
    default = get_default_model()
    if not default:
        return "  No model yet. Run: gdf init"
    p = Path(default)
    from .serialization import get_model_info
    try:
        info = get_model_info(p)
        steps = info["step_count"]
        buf = info["replay_buffer_size"]
        return f"  Model: {p.name}  |  {steps} steps  |  {buf} memories"
    except Exception:
        return f"  Model: {p.name} (cannot read)"


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


def _progress_bar(step: int, total: int, loss: float) -> None:
    pct = step * 100 // total
    bar = "#" * (pct // 2) + "-" * (50 - pct // 2)
    click.echo(f"\r  [{bar}] {pct}% step {step}/{total} loss={loss:.4f}", nl=False)


def _safe_echo(text: str) -> None:
    safe = text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    click.echo(safe)


# ── CLI Group ────────────────────────────────────────────────────────────────

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """gdf — Learn from anything."""
    if ctx.invoked_subcommand is not None:
        return

    # Bare `gdf` → show dashboard
    click.echo()
    click.echo("  gdf — Distributed Federated LLM Training")
    click.echo("  " + "=" * 45)
    click.echo()
    click.echo(_status_line())
    click.echo()
    click.echo("  Quick start:")
    click.echo("    gdf init                 Create a new model")
    click.echo("    gdf learn <url-or-file>  Learn from a URL or file")
    click.echo("    gdf crawl [folder]       Learn from all files in a folder")
    click.echo("    gdf autolearn            Self-learn from the web autonomously")
    click.echo("    gdf chat                 Interactive training + generation")
    click.echo("    gdf generate [prompt]    Generate text")
    click.echo("    gdf status               Show model info")
    click.echo()
    click.echo("  Specialists:")
    click.echo("    gdf specialist create    Create a domain expert model")
    click.echo("    gdf specialist train     Train a specialist on data")
    click.echo("    gdf specialist ask       Ask a question (auto-routed)")
    click.echo("    gdf specialist list      List all specialists")
    click.echo()
    click.echo("  Network:")
    click.echo("    gdf contribute           Contribute your GPU to train a model")
    click.echo("    gdf hub                  Run a coordination server")
    click.echo("    gdf peer <url> --token X  Join a hub as a training peer")
    click.echo()
    click.echo("  Advanced:")
    click.echo("    gdf grow                 Grow model to next size")
    click.echo("    gdf train-tokenizer      Train BPE tokenizer on data")
    click.echo("    gdf default <path>       Set default model")
    click.echo("    gdf merge                Merge multiple models")
    click.echo("    gdf new <path>           Create model at specific path")
    click.echo()


# ── init ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--name", "-n", default=None, help="Model filename")
def init(name: str | None):
    """Create a new model and set it as default."""
    if not name:
        name = click.prompt("  Model name", default="brain.pt", prompt_suffix=" > ")
        if not name.endswith(".pt"):
            name += ".pt"

    bm = GDFModel.create()
    bm.save(name)
    set_default_model(name)
    params = bm.model.count_parameters()
    click.echo(f"  Created {name} ({params:,} parameters)")
    click.echo(f"  Set as default model.")
    click.echo()
    click.echo(f"  Next: gdf learn <url-or-file>")


# ── default ──────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("path", required=False)
def default(path: str | None):
    """Set or show the default model."""
    if path:
        if not Path(path).exists():
            click.echo(f"  File not found: {path}")
            return
        set_default_model(path)
        click.echo(f"  Default model set to: {Path(path).resolve()}")
    else:
        d = get_default_model()
        if d:
            click.echo(f"  Default model: {d}")
        else:
            click.echo("  No default model set.")
            click.echo("  Run: gdf init")


# ── learn ────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("source")
@click.option("--model", "-m", default=None, help="Model to train (default: current)")
@click.option("--epochs", "-e", default=5, help="Training epochs")
def learn(source: str, model: str | None, epochs: int):
    """Learn from a URL or file.

    Examples:
      gdf learn https://example.com/article
      gdf learn mybook.txt
      gdf learn notes.md --epochs 10
    """
    model_path = resolve_model(model)
    bm = GDFModel.load(model_path)

    # Auto-detect URL vs file
    if is_url(source):
        click.echo(f"  Fetching {source}...")
        try:
            text = fetch_url(source)
        except Exception as e:
            click.echo(f"  Failed to fetch: {e}")
            return
        char_count = len(text)
        word_count = len(text.split())
        click.echo(f"  Got {word_count:,} words ({char_count:,} chars)")

        if char_count < 10:
            click.echo("  Not enough text content found.")
            return

        # Preview
        preview = text[:200].replace("\n", " ")
        click.echo(f"  Preview: {preview}...")
        click.echo()

        result = bm.trainer.train_bulk(text, epochs=epochs, on_step=_progress_bar)
        click.echo()
    else:
        fp = Path(source)
        if not fp.exists():
            click.echo(f"  File not found: {source}")
            return
        fsize = fp.stat().st_size
        click.echo(f"  Training on {fp.name} ({fsize:,} bytes), {epochs} epochs...")
        result = bm.train_file(source, epochs=epochs, on_step=_progress_bar)
        click.echo()

    click.echo(f"  Done! {result['steps']} steps | Loss: {result['first_loss']:.4f} -> {result['final_loss']:.4f}")
    bm.save(model_path)
    click.echo(f"  Model saved.")


# ── crawl ────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("folder", required=False)
@click.option("--model", "-m", default=None, help="Model to train")
@click.option("--epochs", "-e", default=3, help="Epochs per file")
def crawl(folder: str | None, model: str | None, epochs: int):
    """Crawl a folder tree and learn from all text files.

    Press Ctrl+C at any time to stop — progress is saved.

    Examples:
      gdf crawl
      gdf crawl ~/Documents
      gdf crawl ./src --epochs 5
    """
    model_path = resolve_model(model)

    if not folder:
        click.echo("  Where should I look for files to learn from?")
        click.echo()
        options = [
            f"Current folder ({Path.cwd()})",
            "Documents folder",
            "Pick a folder",
        ]
        choice = _pick("Choose", options)
        if choice == 0:
            folder = str(Path.cwd())
        elif choice == 1:
            docs = Path.home() / "Documents"
            if not docs.exists():
                docs = Path.home()
            folder = str(docs)
        else:
            folder = click.prompt("  Folder path", prompt_suffix=" > ")

    root = Path(folder)
    if not root.is_dir():
        click.echo(f"  Not a directory: {folder}")
        return

    # Discover files first
    click.echo(f"  Scanning {root}...")
    files = discover_files(root)

    if not files:
        click.echo("  No text files found.")
        return

    total_size = sum(f.stat().st_size for f in files)
    click.echo(f"  Found {len(files)} files ({total_size:,} bytes)")
    click.echo()

    # Show preview of what we'll train on
    for f in files[:5]:
        rel = f.relative_to(root) if f.is_relative_to(root) else f.name
        click.echo(f"    {rel} ({f.stat().st_size:,}b)")
    if len(files) > 5:
        click.echo(f"    ... and {len(files) - 5} more")
    click.echo()

    if not click.confirm("  Start learning?", default=True):
        click.echo("  Cancelled.")
        return

    bm = GDFModel.load(model_path)

    # Ctrl+C handling — stop gracefully
    stop = threading.Event()
    original_handler = signal.getsignal(signal.SIGINT)

    def _handle_stop(sig, frame):
        if stop.is_set():
            # Second Ctrl+C — hard exit
            signal.signal(signal.SIGINT, original_handler)
            raise KeyboardInterrupt
        click.echo("\n  Stopping after current file... (Ctrl+C again to force quit)")
        stop.set()

    signal.signal(signal.SIGINT, _handle_stop)

    current_file_name = ""

    def on_file(i, total, fp):
        nonlocal current_file_name
        rel = fp.relative_to(root) if fp.is_relative_to(root) else fp.name
        current_file_name = str(rel)
        click.echo(f"\n  [{i + 1}/{total}] {rel}")

    def on_step(step, total, loss):
        pct = step * 100 // total
        click.echo(f"\r    {pct}% loss={loss:.4f}", nl=False)

    try:
        result = crawl_and_train(
            bm, root,
            epochs=epochs,
            on_file=on_file,
            on_step=on_step,
            check_stop=stop.is_set,
        )
    finally:
        signal.signal(signal.SIGINT, original_handler)

    click.echo()
    click.echo()
    if result["stopped_early"]:
        click.echo(f"  Stopped early.")
    click.echo(f"  Files: {result['files_trained']}/{result['files_found']}")
    click.echo(f"  Steps: {result['total_steps']} | Data: {result['total_bytes']:,} bytes")

    bm.save(model_path)
    click.echo(f"  Model saved.")


# ── chat (interactive) ───────────────────────────────────────────────────────

@cli.command()
@click.argument("model_name", required=False)
def chat(model_name: str | None):
    """Chat with a model.

    Talk to any model in the network. A router picks the best
    specialist for your question, or you can pick a specific model.

    \b
      gdf chat                    Auto-routes to best specialist
      gdf chat general-7b         Chat with a specific model

    \b
    Commands:
      /models   List available models/specialists
      /switch   Pin to a specific model
      /auto     Back to auto-routing
      /quit     Exit
    """
    # Determine mode: specific model or auto-routing
    pinned_model = None   # GDFModel instance when pinned
    pinned_name = None

    if model_name:
        # Try registry first, then local specialist, then local file
        entry = get_model(model_name)
        if entry:
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
        else:
            # Try as local file path
            path = Path(model_name)
            if path.exists():
                pinned_model = GDFModel.load(str(path))
                pinned_name = path.stem
            else:
                # Try as specialist name
                registry = SpecialistRegistry()
                spec = registry.get(model_name)
                if spec:
                    pinned_model = GDFModel.load(spec.model_path)
                    pinned_name = spec.name
                else:
                    click.echo(f"  Model '{model_name}' not found.")
                    click.echo("  Try: gdf chat (for auto-routing)")
                    return

    registry = SpecialistRegistry()
    router = Router(registry)
    auto_mode = pinned_model is None

    click.echo()
    if pinned_name:
        click.echo(f"  Chatting with: {pinned_name}")
    else:
        specialists = registry.list_all()
        if specialists:
            click.echo(f"  Auto-routing across {len(specialists)} specialist(s)")
        else:
            click.echo("  No specialists registered. Using default model.")
            try:
                model_path = resolve_model(None)
                pinned_model = GDFModel.load(model_path)
                pinned_name = Path(model_path).stem
                auto_mode = False
            except Exception:
                click.echo("  No model available. Run: gdf init")
                return
    click.echo("  Commands: /models /switch <name> /auto /quit")
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
            specialists = registry.list_all()
            if specialists:
                click.echo("  Local specialists:")
                for s in specialists:
                    click.echo(f"    {s.name:25s} [{s.domain}]")
            entries = fetch_registry()
            active = [e for e in entries if e.status == "active"]
            if active:
                click.echo("  Network models:")
                for e in active:
                    click.echo(f"    {e.name:25s} {e.description} ({e.size})")
            if not specialists and not active:
                click.echo("  No models available.")
            click.echo()
            continue

        elif line.startswith("/switch"):
            name = line[7:].strip()
            if not name:
                click.echo("  Usage: /switch <model-name>")
                continue
            # Try specialist
            spec = registry.get(name)
            if spec:
                try:
                    pinned_model = GDFModel.load(spec.model_path)
                    pinned_name = spec.name
                    auto_mode = False
                    click.echo(f"  Switched to: {pinned_name}")
                except Exception as e:
                    click.echo(f"  Failed to load: {e}")
            else:
                # Try registry
                entry = get_model(name)
                if entry:
                    click.echo(f"  Downloading {entry.name}...")
                    try:
                        p = Peer(hub_url=entry.hub_url, token=entry.token)
                        p.register()
                        p.pull_model()
                        pinned_model = GDFModel.load(p.local_model_path)
                        pinned_name = entry.name
                        auto_mode = False
                        click.echo(f"  Switched to: {pinned_name}")
                    except Exception as e:
                        click.echo(f"  Failed: {e}")
                else:
                    click.echo(f"  Model '{name}' not found.")
            continue

        elif line == "/auto":
            pinned_model = None
            pinned_name = None
            auto_mode = True
            click.echo("  Auto-routing enabled.")
            continue

        # Generate response
        if auto_mode:
            matches = router.route(line, top_k=1)
            if matches:
                spec, score = matches[0]
                try:
                    bm = GDFModel.load(spec.model_path)
                    response = bm.generate(prompt=line, max_tokens=200, temperature=0.7)
                    click.echo(f"  [{spec.name}, relevance: {score:.2f}]")
                    _safe_echo(f"  {response}")
                except Exception as e:
                    click.echo(f"  [{spec.name}] Error: {e}")
            else:
                click.echo("  No matching specialist found. Try /switch <model>")
        else:
            response = pinned_model.generate(prompt=line, max_tokens=200, temperature=0.7)
            _safe_echo(f"  {response}")
        click.echo()


# ── generate ─────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("prompt", required=False, default="")
@click.option("--model", "-m", default=None, help="Model to use")
@click.option("--length", "-n", default=200, help="Max tokens")
@click.option("--temperature", "-t", default=0.8, help="Temperature")
def generate(prompt: str, model: str | None, length: int, temperature: float):
    """Generate text from the model."""
    model_path = resolve_model(model)
    bm = GDFModel.load(model_path)
    text = bm.generate(prompt=prompt, max_tokens=length, temperature=temperature)
    _safe_echo(text)


# ── status ───────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--model", "-m", default=None, help="Model to check")
def status(model: str | None):
    """Show model info."""
    model_path = resolve_model(model)
    data = GDFModel.info(model_path)
    click.echo()
    click.echo(f"  Model: {Path(model_path).name}")
    click.echo(f"  Path:  {Path(model_path).resolve()}")
    click.echo(f"  Parameters:  {data['parameters']:,}")
    click.echo(f"  Steps:       {data['step_count']}")
    click.echo(f"  Memories:    {data['replay_buffer_size']}")
    click.echo(f"  Hash:        {data['model_hash']}")
    click.echo(f"  Base hash:   {data['base_model_hash']}")
    click.echo(f"  Architecture: d={data['config']['d_model']} heads={data['config']['n_heads']} "
               f"layers={data['config']['n_layers']} ff={data['config']['d_ff']}")
    click.echo()
    # Device info
    info = device_info()
    click.echo(f"  {format_device_info(info)}")
    click.echo()


# ── merge ────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("output", required=False)
@click.argument("inputs", nargs=-1)
@click.option("--strategy", "-s", default="fedavg",
              type=click.Choice(["fedavg", "ties", "task_arithmetic"]))
def merge(output: str | None, inputs: tuple[str, ...], strategy: str):
    """Merge multiple models into one.

    Examples:
      gdf merge combined.pt model1.pt model2.pt
      gdf merge   (guided mode)
    """
    if not output or not inputs:
        # Guided mode
        click.echo("  Merge models together.")
        click.echo()

        # Collect input models
        model_paths: list[str] = []
        click.echo("  Enter model paths to merge (empty line when done):")
        while True:
            try:
                p = click.prompt(f"  Model {len(model_paths) + 1}", default="", prompt_suffix=" > ")
            except (EOFError, click.Abort):
                break
            if not p:
                break
            if not Path(p).exists():
                click.echo(f"    Not found: {p}")
                continue
            model_paths.append(p)

        if len(model_paths) < 2:
            click.echo("  Need at least 2 models to merge.")
            return

        out_name = click.prompt("  Output filename", default="merged.pt", prompt_suffix=" > ")

        click.echo()
        strategies = ["fedavg (simple average)", "ties (smart merge)", "task_arithmetic (delta sum)"]
        idx = _pick("Strategy", strategies)
        strategy = ["fedavg", "ties", "task_arithmetic"][idx]

        output = out_name
        inputs = tuple(model_paths)

    click.echo(f"  Merging {len(inputs)} models ({strategy})...")
    merged = GDFModel.merge(list(inputs), strategy=strategy)
    merged.save(output)
    click.echo(f"  Saved to {output}")

    if click.confirm("  Set as default model?", default=True):
        set_default_model(output)
        click.echo("  Done!")


# ── new (advanced) ───────────────────────────────────────────────────────────

@cli.command()
@click.argument("path")
def new(path: str):
    """Create a new model at a specific path (advanced)."""
    bm = GDFModel.create()
    bm.save(path)
    params = bm.model.count_parameters()
    click.echo(f"  Created {path} ({params:,} parameters)")


# ── autolearn (autonomous) ──────────────────────────────────────────────────

@cli.command()
@click.option("--model", "-m", default=None, help="Model to train")
@click.option("--cycles", "-c", default=0, help="Max cycles (0 = run forever)")
@click.option("--epochs", "-e", default=3, help="Epochs per article")
@click.option("--pause", "-p", default=2.0, help="Seconds between fetches")
@click.option("--topics", "-t", multiple=True, help="Topics to learn about (repeatable)")
@click.option("--log", "-l", default=None, help="Log file (JSON lines)")
def autolearn(model: str | None, cycles: int, epochs: int, pause: float,
              topics: tuple[str, ...], log: str | None):
    """Autonomous self-learning: fetch from web, train, evaluate, repeat.

    The model fetches Wikipedia articles, trains on them, measures improvement
    via perplexity, and saves checkpoints automatically.

    Press Ctrl+C to stop — progress is saved.

    Examples:
      gdf autolearn
      gdf autolearn --topics "physics" --topics "history"
      gdf autolearn --cycles 20 --log learning.jsonl
    """
    model_path = resolve_model(model)
    bm = GDFModel.load(model_path)

    config = SelfLearnConfig(
        max_cycles=cycles,
        epochs_per_article=epochs,
        pause_seconds=pause,
        log_file=log,
    )
    learner = SelfLearner(bm, config)

    click.echo()
    click.echo("  Autonomous self-learning mode")
    click.echo("  " + "=" * 35)
    click.echo(f"  Model: {Path(model_path).name}")
    if topics:
        click.echo(f"  Topics: {', '.join(topics)}")
    else:
        click.echo("  Source: Random Wikipedia articles")
    click.echo(f"  Epochs per article: {epochs}")
    if cycles:
        click.echo(f"  Max cycles: {cycles}")
    else:
        click.echo("  Running until Ctrl+C")
    click.echo()

    # Ctrl+C handling
    original_handler = signal.getsignal(signal.SIGINT)

    def _handle_stop(sig, frame):
        click.echo("\n  Stopping after current cycle...")
        learner.stop()
        signal.signal(signal.SIGINT, original_handler)

    signal.signal(signal.SIGINT, _handle_stop)

    def on_status(msg: str):
        _safe_echo(msg)

    def on_cycle(result):
        arrow = "v" if result.improved else "="
        click.echo(f"  [{result.cycle}] {result.title[:40]:40s} "
                    f"ppl: {result.perplexity_before:.1f} {arrow} {result.perplexity_after:.1f} "
                    f"| loss: {result.train_loss_end:.4f}")

    try:
        topic_list = list(topics) if topics else None
        learner.run(
            save_path=model_path,
            topics=topic_list,
            on_status=on_status,
            on_cycle=on_cycle,
        )
    finally:
        signal.signal(signal.SIGINT, original_handler)

    # Print summary
    s = learner.summary()
    click.echo()
    click.echo("  Summary")
    click.echo("  " + "-" * 30)
    click.echo(f"  Cycles:         {s['cycles']}")
    click.echo(f"  Total text:     {s.get('total_text', 0):,} chars")
    click.echo(f"  Best perplexity: {s.get('best_perplexity', 0):.2f}")
    click.echo(f"  Improvement rate: {s.get('improvement_rate', 0):.0%}")
    click.echo(f"  Total time:     {s.get('total_time', 0):.1f}s")
    click.echo()
    click.echo(f"  Model saved to {model_path}")


# ── grow ────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--model", "-m", default=None, help="Model to grow")
@click.option("--stage", "-s", default=None,
              type=click.Choice(list(GROWTH_STAGES.keys())),
              help="Target growth stage")
@click.option("--output", "-o", default=None, help="Output path (default: overwrites)")
def grow(model: str | None, stage: str | None, output: str | None):
    """Grow the model to a larger architecture.

    Preserves everything the model has learned while adding more capacity.
    The grown model produces identical output initially, then can learn more.

    Growth stages: micro (128d/2L) → tiny (256d/4L) → small (384d/6L)
                   → medium (512d/8L) → large (768d/12L)

    Examples:
      gdf grow                    Grow to next stage
      gdf grow --stage small      Grow to specific stage
      gdf grow -o bigger.pt       Save to new file
    """
    model_path = resolve_model(model)
    bm = GDFModel.load(model_path)

    current = bm.model.config
    click.echo()
    click.echo(f"  Current: d={current.d_model} heads={current.n_heads} "
               f"layers={current.n_layers} ff={current.d_ff} "
               f"({bm.model.count_parameters():,} params)")

    if not stage:
        next_stage = suggest_next_stage(current)
        if not next_stage:
            click.echo("  Already at largest predefined stage.")
            click.echo("  Use --stage to pick a specific target.")
            return
        stage = next_stage

    target = GROWTH_STAGES[stage]
    click.echo(f"  Target ({stage}): d={target.d_model} heads={target.n_heads} "
               f"layers={target.n_layers} ff={target.d_ff}")

    # Check if growth is actually needed
    if (current.d_model >= target.d_model and
            current.n_layers >= target.n_layers and
            current.d_ff >= target.d_ff):
        click.echo("  Model is already at or above this stage.")
        return

    click.echo()
    if not click.confirm("  Grow model?", default=True):
        click.echo("  Cancelled.")
        return

    click.echo("  Growing...")
    # Need to set vocab_size to match current model
    target_cfg = ModelConfig(
        vocab_size=current.vocab_size,
        d_model=target.d_model,
        n_heads=target.n_heads,
        n_layers=target.n_layers,
        d_ff=target.d_ff,
        max_seq_len=current.max_seq_len,
        dropout=current.dropout,
    )
    grown_model = grow_model(bm.model, target_cfg)

    # Create new GDFModel with the grown model
    from .trainer import OnlineTrainer
    grown_bm = GDFModel(grown_model, OnlineTrainer(grown_model), bm.base_model_hash)

    out_path = output or model_path
    grown_bm.save(out_path)

    click.echo(f"  Done! {grown_model.count_parameters():,} parameters")
    click.echo(f"  Saved to {out_path}")
    click.echo()
    click.echo("  The grown model is ready — it behaves identically to before")
    click.echo("  but has room to learn more. Train it with more data now.")


# ── train-tokenizer ─────────────────────────────────────────────────────────

@cli.command("train-tokenizer")
@click.argument("sources", nargs=-1)
@click.option("--vocab-size", "-v", default=1024, help="Target vocabulary size")
@click.option("--output", "-o", default="tokenizer.json", help="Output file")
def train_tokenizer(sources: tuple[str, ...], vocab_size: int, output: str):
    """Train a BPE tokenizer from text files or URLs.

    Learns common byte patterns from data so the model can think in
    words/subwords instead of individual characters.

    Examples:
      gdf train-tokenizer book1.txt book2.txt
      gdf train-tokenizer corpus/ --vocab-size 2048
    """
    if not sources:
        click.echo("  Provide text files or a folder to train from.")
        click.echo("  Usage: gdf train-tokenizer <file1> <file2> ...")
        return

    # Collect texts
    texts: list[str] = []
    for src in sources:
        p = Path(src)
        if p.is_dir():
            from .crawler import discover_files
            files = discover_files(p)
            click.echo(f"  Found {len(files)} files in {src}")
            for f in files:
                try:
                    texts.append(f.read_text(encoding="utf-8", errors="replace"))
                except Exception:
                    continue
        elif p.is_file():
            texts.append(p.read_text(encoding="utf-8", errors="replace"))
        else:
            click.echo(f"  Skipping {src} (not found)")

    if not texts:
        click.echo("  No text found to train on.")
        return

    total_chars = sum(len(t) for t in texts)
    click.echo(f"  Training BPE on {total_chars:,} characters...")
    click.echo(f"  Target vocab: {vocab_size} tokens (256 base + {vocab_size - 256} merges)")
    click.echo()

    def on_progress(step, total, pair):
        if step % 50 == 0 or step == total - 1:
            pct = (step + 1) * 100 // total
            click.echo(f"\r  {pct}% ({step + 1}/{total} merges)", nl=False)

    tokenizer = BPETokenizer.train(texts, target_vocab_size=vocab_size, on_progress=on_progress)
    click.echo()
    click.echo()

    tokenizer.save(output)
    click.echo(f"  Saved tokenizer to {output}")
    click.echo(f"  Vocabulary: {tokenizer.vocab_size} tokens")

    # Show some learned tokens
    vocab = tokenizer.get_vocab_tokens()
    click.echo()
    click.echo("  Sample learned tokens:")
    sample_ids = list(range(256, min(256 + 20, tokenizer.vocab_size)))
    for tid in sample_ids:
        token_str = vocab.get(tid, "?")
        display = repr(token_str)
        click.echo(f"    [{tid}] {display}")

    # Demo encoding
    demo = "The quick brown fox jumps."
    encoded = tokenizer.encode(demo)
    click.echo()
    click.echo(f"  Demo: \"{demo}\"")
    click.echo(f"  Byte tokens:  {len(demo.encode('utf-8'))} tokens")
    click.echo(f"  BPE tokens:   {len(encoded)} tokens ({len(demo.encode('utf-8')) - len(encoded)} saved)")


# ── hub (distributed server) ───────────────────────────────────────────────

@cli.command()
@click.option("--model", "-m", default=None, help="Model to serve")
@click.option("--port", "-p", default=7677, help="Port to listen on")
@click.option("--host", default="0.0.0.0", help="Host to bind")
@click.option("--merge-every", default=3, help="Merge after N contributions")
@click.option("--strategy", "-s", default="fedavg",
              type=click.Choice(["fedavg", "ties"]))
@click.option("--token", default=None, help="Auth token (auto-generated if not set)")
@click.option("--parent", default=None, help="Parent hub URL (for hierarchy)")
def hub(model: str | None, port: int, host: str, merge_every: int,
        strategy: str, token: str | None, parent: str | None):
    """Run a distributed training hub.

    Deploy this on a VPS/cloud server (DigitalOcean, Hetzner, AWS, etc.)
    so peers anywhere in the world can connect.

    The hub auto-generates an auth token — share it with your peers.
    Only peers with the token can connect.

    \b
    Quick deploy on a VPS:
      1. Install: pip install gdf
      2. Init:    gdf init
      3. Run:     gdf hub --port 7677
      4. Share the token + your server IP with peers

    Examples:
      gdf hub                           Start with auto-generated token
      gdf hub --token mysecret          Use a specific token
      gdf hub --merge-every 10          Merge after 10 contributions
    """
    model_path = resolve_model(model)

    h = Hub(
        model_path=model_path,
        merge_strategy=strategy,
        merge_every=merge_every,
        host=host,
        port=port,
        token=token,
        parent_hub=parent,
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
    click.echo(f"  Model:    {Path(model_path).name}")
    click.echo(f"  Strategy: {strategy}")
    click.echo(f"  Merge every {merge_every} contributions")
    click.echo(f"  Listening: {host}:{port}")
    if parent:
        click.echo(f"  Parent hub: {parent}")
    click.echo()
    click.echo(f"  TOKEN: {h.token}")
    click.echo()
    click.echo("  Share this command with peers:")
    click.echo(f"    gdf peer http://{local_ip}:{port} --token {h.token}")
    click.echo()
    click.echo("  If running on a VPS, replace the IP with your server's public IP.")
    click.echo("  Press Ctrl+C to stop.")
    click.echo()

    h.start(on_log=lambda msg: _safe_echo(f"  {msg}"))


# ── peer (distributed worker) ──────────────────────────────────────────────

@cli.command()
@click.argument("hub_url")
@click.option("--token", "-t", required=True, help="Auth token from the hub")
@click.option("--cycles", "-c", default=5, help="Training cycles per round")
@click.option("--rounds", "-r", default=0, help="Rounds to run (0 = forever)")
@click.option("--pause", default=5.0, help="Seconds between rounds")
def peer(hub_url: str, token: str, cycles: int, rounds: int, pause: float):
    """Join a distributed training network as a peer.

    You need the hub URL and auth token from whoever is running the hub.

    Examples:
      gdf peer http://hub.example.com:7677 --token abc123
      gdf peer http://hub.example.com:7677 --token abc123 --cycles 10
    """
    p = Peer(hub_url=hub_url, token=token)

    click.echo()
    click.echo("  gdf peer")
    click.echo("  " + "=" * 30)
    click.echo(f"  Hub:     {hub_url}")
    click.echo(f"  Peer ID: {p.peer_id}")
    click.echo(f"  Cycles per round: {cycles}")
    click.echo()

    # Show device info
    info = device_info()
    click.echo(f"  {format_device_info(info)}")
    click.echo()

    # Register
    click.echo("  Registering with hub...")
    try:
        reg = p.register()
        click.echo(f"  Registered! Hub generation: {reg.get('generation', '?')}")
    except Exception as e:
        click.echo(f"  Failed to connect: {e}")
        return

    click.echo()
    click.echo("  Press Ctrl+C to stop.")
    click.echo()

    round_num = 0
    try:
        while True:
            round_num += 1
            if rounds and round_num > rounds:
                break

            click.echo(f"  === Round {round_num} ===")
            try:
                result = p.train_and_push(
                    cycles=cycles,
                    on_status=lambda msg: _safe_echo(f"  {msg}"),
                )
                rate = result.get("improvement_rate", 0)
                click.echo(f"  Round {round_num} done. "
                           f"Improvement rate: {rate:.0%}")
            except Exception as e:
                click.echo(f"  Round {round_num} failed: {e}")

            if rounds == 0 or round_num < rounds:
                click.echo(f"  Waiting {pause}s before next round...")
                time.sleep(pause)

    except KeyboardInterrupt:
        click.echo("\n  Stopped.")

    click.echo(f"  Completed {round_num} rounds.")


# ── contribute ──────────────────────────────────────────────────────────────

@cli.command()
@click.argument("model_name", required=False)
def contribute(model_name: str | None):
    """Contribute your GPU to train a model.

    Pick a model and your machine starts training automatically.
    Your GPU does the work, results get merged into the global model.

    \b
      gdf contribute              Pick from available models
      gdf contribute general-7b   Start directly
    """
    # Fetch available models
    entries = fetch_registry()
    active = [e for e in entries if e.status == "active"]

    if not active:
        click.echo()
        click.echo("  No models available in the network yet.")
        click.echo()
        click.echo("  The model registry is empty. Once hubs are deployed,")
        click.echo("  models will appear here automatically.")
        click.echo()
        click.echo("  To run your own hub: gdf hub")
        return

    if not model_name:
        # Show picker
        click.echo()
        click.echo("  Available models:")
        click.echo()
        options = []
        for e in active:
            label = f"{e.name} — {e.description} ({e.size})"
            options.append(label)
        choice = _pick("Pick a model", options)
        entry = active[choice]
    else:
        entry = None
        for e in active:
            if e.name == model_name:
                entry = e
                break
        if not entry:
            click.echo(f"  Model '{model_name}' not found in registry.")
            click.echo("  Available: " + ", ".join(e.name for e in active))
            return

    click.echo()
    click.echo(f"  Model:  {entry.name}")
    click.echo(f"  Hub:    {entry.hub_url}")
    click.echo(f"  Size:   {entry.size}")
    click.echo()

    # Show device info
    info = device_info()
    click.echo(f"  {format_device_info(info)}")
    click.echo()

    # Create peer and register
    p = Peer(hub_url=entry.hub_url, token=entry.token)
    click.echo(f"  Peer ID: {p.peer_id}")
    click.echo("  Registering with hub...")

    try:
        reg = p.register()
        click.echo(f"  Registered! Hub generation: {reg.get('generation', '?')}")
    except Exception as e:
        click.echo(f"  Failed to connect: {e}")
        if "localhost" in entry.hub_url or "127.0.0.1" in entry.hub_url:
            click.echo()
            click.echo("  Start the hub first:")
            click.echo("    ./scripts/run_hub.sh")
            click.echo("  Or:")
            click.echo("    gdf init --name hub_model.pt && gdf hub --model hub_model.pt --token local-dev")
        return

    click.echo()
    click.echo("  Contributing. Press Ctrl+C to stop.")
    click.echo()

    # Infinite train loop
    round_num = 0
    try:
        while True:
            round_num += 1
            click.echo(f"  === Round {round_num} ===")
            try:
                result = p.train_and_push(
                    cycles=5,
                    on_status=lambda msg: _safe_echo(f"  {msg}"),
                )
                rate = result.get("improvement_rate", 0)
                click.echo(f"  Round {round_num} done. Improvement rate: {rate:.0%}")
            except Exception as e:
                click.echo(f"  Round {round_num} failed: {e}")

            click.echo(f"  Waiting 5s before next round...")
            time.sleep(5)

    except KeyboardInterrupt:
        click.echo(f"\n  Stopped after {round_num} rounds. Thanks for contributing!")


# ── specialist commands ─────────────────────────────────────────────────────

@cli.group()
def specialist():
    """Manage specialist models — domain-specific experts.

    Instead of one giant general model, train focused specialists
    that deeply understand specific domains.
    """
    pass


@specialist.command("create")
@click.argument("name")
@click.option("--domain", "-d", required=True, help="Domain (e.g., medical, code-python)")
@click.option("--description", "desc", default="", help="What this specialist knows")
@click.option("--keywords", "-k", multiple=True, help="Routing keywords (repeatable)")
def specialist_create(name: str, domain: str, desc: str, keywords: tuple[str, ...]):
    """Create a new specialist model.

    Examples:
      gdf specialist create cardiology -d medical -k heart -k cardiac
      gdf specialist create python-web -d code-python -k flask -k django
    """
    # Use suggested domain info if available
    suggested = SUGGESTED_DOMAINS.get(domain, {})
    if not desc and suggested:
        desc = suggested.get("description", "")
    if not keywords and suggested:
        keywords = tuple(suggested.get("keywords", []))

    if not desc:
        desc = click.prompt("  Description", prompt_suffix=" > ")
    if not keywords:
        kw_str = click.prompt("  Keywords (comma-separated)", prompt_suffix=" > ")
        keywords = tuple(k.strip() for k in kw_str.split(",") if k.strip())

    # Create the model
    model_path = f"{name}.pt"
    bm = GDFModel.create()
    bm.save(model_path)

    # Register the specialist
    info = SpecialistInfo(
        name=name,
        domain=domain,
        description=desc,
        keywords=list(keywords),
        model_path=str(Path(model_path).resolve()),
    )
    registry = SpecialistRegistry()
    registry.register(info)

    click.echo()
    click.echo(f"  Created specialist: {name}")
    click.echo(f"  Domain: {domain}")
    click.echo(f"  Keywords: {', '.join(keywords)}")
    click.echo(f"  Model: {model_path}")
    click.echo()
    click.echo(f"  Train it:  gdf specialist train {name} <source>")
    if suggested:
        click.echo(f"  Suggested data: {suggested.get('data_sources', 'N/A')}")


@specialist.command("list")
def specialist_list():
    """List all registered specialists."""
    registry = SpecialistRegistry()
    specialists = registry.list_all()

    if not specialists:
        click.echo()
        click.echo("  No specialists registered yet.")
        click.echo()
        click.echo("  Create one:")
        click.echo("    gdf specialist create <name> -d <domain>")
        click.echo()
        click.echo("  Suggested domains:")
        for domain, info in list(SUGGESTED_DOMAINS.items())[:5]:
            click.echo(f"    {domain:20s} {info['description'][:50]}")
        click.echo(f"    ... and {len(SUGGESTED_DOMAINS) - 5} more")
        return

    click.echo()
    click.echo(f"  {len(specialists)} specialist(s) registered:")
    click.echo()

    # Group by domain
    domains: dict[str, list[SpecialistInfo]] = {}
    for s in specialists:
        domains.setdefault(s.domain, []).append(s)

    for domain, specs in sorted(domains.items()):
        click.echo(f"  [{domain}]")
        for s in specs:
            quality = f"ppl={s.quality_score:.1f}" if s.quality_score else "untested"
            click.echo(f"    {s.name:25s} {s.training_steps:>6} steps  "
                       f"{s.contributors:>3} contributors  {quality}")
    click.echo()


@specialist.command("train")
@click.argument("name")
@click.argument("source")
@click.option("--epochs", "-e", default=5, help="Training epochs")
def specialist_train(name: str, source: str, epochs: int):
    """Train a specialist on domain-specific data.

    Examples:
      gdf specialist train cardiology heart_papers.txt
      gdf specialist train python-web https://flask.palletsprojects.com/
    """
    registry = SpecialistRegistry()
    info = registry.get(name)
    if not info:
        click.echo(f"  Specialist '{name}' not found. Run: gdf specialist list")
        return

    bm = GDFModel.load(info.model_path)

    if is_url(source):
        click.echo(f"  Fetching {source}...")
        try:
            text = fetch_url(source)
        except Exception as e:
            click.echo(f"  Failed: {e}")
            return
        click.echo(f"  Got {len(text.split()):,} words")
        result = bm.trainer.train_bulk(text, epochs=epochs, on_step=_progress_bar)
        click.echo()
    else:
        fp = Path(source)
        if not fp.exists():
            click.echo(f"  File not found: {source}")
            return
        result = bm.train_file(source, epochs=epochs, on_step=_progress_bar)
        click.echo()

    bm.save(info.model_path)

    # Update registry
    info.training_steps += result.get("steps", 0)
    info.training_sources += 1
    info.updated = time.time()
    registry.register(info)

    click.echo(f"  Done! {result['steps']} steps | "
               f"Loss: {result['first_loss']:.4f} -> {result['final_loss']:.4f}")
    click.echo(f"  Specialist '{name}' updated ({info.training_steps} total steps)")


@specialist.command("autolearn")
@click.argument("name")
@click.option("--cycles", "-c", default=10, help="Learning cycles")
@click.option("--topics", "-t", multiple=True, help="Topic keywords (repeatable)")
def specialist_autolearn(name: str, cycles: int, topics: tuple[str, ...]):
    """Auto-learn from the web for a specialist's domain.

    If no topics given, uses the specialist's keywords.

    Examples:
      gdf specialist autolearn cardiology
      gdf specialist autolearn cardiology -t "heart disease" -t "cardiac surgery"
    """
    registry = SpecialistRegistry()
    info = registry.get(name)
    if not info:
        click.echo(f"  Specialist '{name}' not found.")
        return

    bm = GDFModel.load(info.model_path)

    # Use specialist keywords as topics if none provided
    topic_list = list(topics) if topics else info.keywords[:5]

    click.echo(f"  Auto-learning for specialist: {name}")
    click.echo(f"  Topics: {', '.join(topic_list)}")
    click.echo()

    config = SelfLearnConfig(max_cycles=cycles, epochs_per_article=3)
    learner = SelfLearner(bm, config)

    original_handler = signal.getsignal(signal.SIGINT)
    def _handle_stop(sig, frame):
        click.echo("\n  Stopping...")
        learner.stop()
        signal.signal(signal.SIGINT, original_handler)
    signal.signal(signal.SIGINT, _handle_stop)

    try:
        learner.run(
            save_path=info.model_path,
            topics=topic_list,
            on_status=lambda msg: _safe_echo(f"  {msg}"),
        )
    finally:
        signal.signal(signal.SIGINT, original_handler)

    s = learner.summary()
    info.training_steps += s.get("cycles", 0) * 10  # approximate
    info.training_sources += s.get("cycles", 0)
    info.updated = time.time()
    if s.get("best_perplexity", 0) > 0:
        info.quality_score = s["best_perplexity"]
    registry.register(info)

    click.echo()
    click.echo(f"  Done. {s.get('cycles', 0)} cycles, best ppl: {s.get('best_perplexity', 0):.1f}")


@specialist.command("ask")
@click.argument("query")
@click.option("--top-k", "-k", default=2, help="Number of specialists to query")
def specialist_ask(query: str, top_k: int):
    """Ask a question — routes to the best specialist(s).

    Examples:
      gdf specialist ask "What causes atrial fibrillation?"
      gdf specialist ask "How do I set up Flask routing?"
    """
    registry = SpecialistRegistry()
    if not registry.list_all():
        click.echo("  No specialists registered. Create some first.")
        return

    click.echo(f"  Routing query: \"{query}\"")
    click.echo()

    results = query_specialists(query, registry, top_k=top_k)

    if not results:
        click.echo("  No matching specialists found.")
        return

    for r in results:
        click.echo(f"  [{r['specialist']}] (domain: {r['domain']}, relevance: {r['score']:.1f})")
        _safe_echo(f"    {r['response']}")
        click.echo()


@specialist.command("domains")
def specialist_domains():
    """Show suggested specialist domains to create."""
    click.echo()
    click.echo("  Suggested specialist domains:")
    click.echo()
    for domain, info in SUGGESTED_DOMAINS.items():
        click.echo(f"  {domain}")
        click.echo(f"    {info['description']}")
        click.echo(f"    Keywords: {', '.join(info['keywords'][:5])}")
        click.echo(f"    Data: {info['data_sources']}")
        click.echo()
    click.echo(f"  Create one: gdf specialist create <name> -d <domain>")
    click.echo(f"  Or invent your own domain — these are just suggestions.")


# ── Legacy aliases ───────────────────────────────────────────────────────────

@cli.command(hidden=True)
@click.argument("path", required=False)
@click.option("--interactive", "-i", is_flag=True)
@click.option("--text", "-t", default=None)
@click.option("--file", "-f", "file_path", default=None)
@click.option("--epochs", "-e", default=5)
def train(path: str | None, interactive: bool, text: str | None, file_path: str | None, epochs: int):
    """Legacy train command — use 'learn' or 'chat' instead."""
    model_path = resolve_model(path)
    bm = GDFModel.load(model_path)

    if file_path:
        click.echo(f"  Tip: use 'gdf learn {file_path}' instead")
        result = bm.train_file(file_path, epochs=epochs, on_step=_progress_bar)
        click.echo()
        click.echo(f"  Done! {result['steps']} steps | Loss: {result['first_loss']:.4f} -> {result['final_loss']:.4f}")
        bm.save(model_path)
        return

    if text:
        result = bm.train(text)
        bm.save(model_path)
        click.echo(f"  Learned (loss: {result['loss']:.4f})")
        return

    if interactive:
        click.echo("  Tip: use 'gdf chat' instead")
        # Redirect to chat-like behavior
        ctx = click.get_current_context()
        ctx.invoke(chat, model=model_path)
        return

    click.echo("  Use: gdf learn <source> or gdf chat")


@cli.command(hidden=True)
@click.argument("path")
def info(path: str):
    """Legacy info command — use 'status' instead."""
    click.echo("  Tip: use 'gdf status' instead")
    ctx = click.get_current_context()
    ctx.invoke(status, model=path)


if __name__ == "__main__":
    cli()

"""Torrent-style model distribution.

Splits models into chunks that peers share with each other,
so the hub doesn't need to serve every download.

How it works:
    1. Model is split into chunks (one per layer/tensor group)
    2. A manifest lists all chunks with their SHA-256 hashes
    3. The hub seeds the initial chunks and tracks who has what
    4. Peers download chunks from the hub AND from other peers
    5. Once a peer has chunks, it serves them to others
    6. Result: hub bandwidth = seed once, not serve everyone

Math for a 7B model with 10k peers:
    Centralized:  14 GB × 10,000 = 140 TB from hub
    Torrent:      14 GB × ~10    = 140 GB from hub (seed to first 10)
                  The other 9,990 peers get chunks from each other

Components:
    Manifest     — describes the model chunks and their hashes
    Seeder       — lightweight HTTP server that serves chunks (runs on every peer)
    Tracker      — hub endpoint that tracks which peers have which chunks
    Downloader   — pulls chunks from multiple sources in parallel
"""

from __future__ import annotations

import io
import json
import hashlib
import threading
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Callable
import urllib.request
import urllib.error

import torch

from .model import TinyTransformer, ModelConfig
from .serialization import load_model, save_model


# ── Chunking ────────────────────────────────────────────────────────────────

CHUNK_PREFIX = "chunk_"


@dataclass
class ChunkInfo:
    """Metadata for a single chunk of the model."""
    index: int
    keys: list[str]          # which state_dict keys are in this chunk
    sha256: str              # hash of the chunk data
    size_bytes: int          # size of the serialized chunk


@dataclass
class Manifest:
    """Describes a chunked model — like a .torrent file.

    Small enough to send in a single request (~1KB).
    Contains everything needed to verify and reassemble the model.
    """
    config: dict                          # ModelConfig as dict
    chunks: list[ChunkInfo] = field(default_factory=list)
    total_size: int = 0
    model_hash: str = ""
    created: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, s: str) -> Manifest:
        d = json.loads(s)
        chunks = [ChunkInfo(**c) for c in d.get("chunks", [])]
        return cls(
            config=d["config"],
            chunks=chunks,
            total_size=d.get("total_size", 0),
            model_hash=d.get("model_hash", ""),
            created=d.get("created", 0),
        )


def create_chunks(model_path: str, output_dir: str) -> Manifest:
    """Split a saved model into chunks and create a manifest.

    Groups state_dict keys by layer to create natural chunk boundaries.
    Each chunk is independently loadable and verifiable.

    Returns the manifest describing all chunks.
    """
    model, _, meta = load_model(model_path, load_trainer=False)
    sd = model.state_dict()
    config = model.config.to_dict()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Group keys by layer/component for natural chunking
    groups = _group_keys(list(sd.keys()))

    chunks = []
    total_size = 0

    for i, keys in enumerate(groups):
        # Serialize this chunk
        chunk_data = {k: sd[k].cpu() for k in keys}
        buf = io.BytesIO()
        torch.save(chunk_data, buf)
        raw = buf.getvalue()

        # Hash it
        sha = hashlib.sha256(raw).hexdigest()

        # Write chunk file
        chunk_path = output / f"{CHUNK_PREFIX}{i:04d}.pt"
        chunk_path.write_bytes(raw)

        chunks.append(ChunkInfo(
            index=i,
            keys=keys,
            sha256=sha,
            size_bytes=len(raw),
        ))
        total_size += len(raw)

    manifest = Manifest(
        config=config,
        chunks=chunks,
        total_size=total_size,
        model_hash=meta.get("model_hash", ""),
    )

    # Write manifest
    (output / "manifest.json").write_text(manifest.to_json(), encoding="utf-8")

    return manifest


def reassemble_model(manifest: Manifest, chunks_dir: str, output_path: str) -> None:
    """Reassemble a model from verified chunks.

    Verifies each chunk's hash before loading.
    Raises ValueError if any chunk is corrupted or missing.
    """
    chunks_path = Path(chunks_dir)
    full_sd = {}

    for chunk_info in manifest.chunks:
        chunk_file = chunks_path / f"{CHUNK_PREFIX}{chunk_info.index:04d}.pt"
        if not chunk_file.exists():
            raise ValueError(f"Missing chunk {chunk_info.index}")

        raw = chunk_file.read_bytes()

        # Verify hash
        actual_hash = hashlib.sha256(raw).hexdigest()
        if actual_hash != chunk_info.sha256:
            raise ValueError(
                f"Chunk {chunk_info.index} corrupted: "
                f"expected {chunk_info.sha256[:16]}, got {actual_hash[:16]}"
            )

        # Load chunk
        buf = io.BytesIO(raw)
        chunk_data = torch.load(buf, map_location="cpu", weights_only=False)
        full_sd.update(chunk_data)

    # Rebuild model
    config = ModelConfig.from_dict(manifest.config)
    model = TinyTransformer(config)
    model.load_state_dict(full_sd)

    from .trainer import OnlineTrainer
    trainer = OnlineTrainer(model, device=torch.device("cpu"))
    save_model(output_path, model, trainer)


def _group_keys(keys: list[str]) -> list[list[str]]:
    """Group state_dict keys into chunks.

    Groups by top-level component (embeddings, each block, final layer norm, head).
    This creates natural boundaries that map to model layers.
    """
    groups: dict[str, list[str]] = {}

    for key in keys:
        # Parse the group from the key
        parts = key.split(".")
        if parts[0] == "blocks" and len(parts) > 1:
            # Group by block: blocks.0.*, blocks.1.*, etc.
            group = f"blocks.{parts[1]}"
        else:
            # Top-level: token_emb, pos_emb, ln_f, head
            group = parts[0]
        groups.setdefault(group, []).append(key)

    # Return in a stable order
    return [groups[k] for k in sorted(groups.keys())]


# ── Seeder (runs on every peer) ────────────────────────────────────────────

class ChunkSeeder:
    """Lightweight HTTP server that serves model chunks to other peers.

    Every peer runs this after downloading chunks. This is what makes
    the distribution exponential instead of linear.
    """

    def __init__(self, chunks_dir: str, port: int = 0):
        self.chunks_dir = Path(chunks_dir)
        self.port = port
        self._server = None
        self._thread = None

    @property
    def address(self) -> str | None:
        if self._server:
            return f"http://{self._server.server_address[0]}:{self._server.server_address[1]}"
        return None

    def start(self) -> int:
        """Start serving chunks in a background thread. Returns the port."""
        seeder = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def do_GET(self):
                # GET /chunk/0000 — serve a chunk
                if self.path.startswith("/chunk/"):
                    idx = self.path.split("/chunk/")[1]
                    chunk_file = seeder.chunks_dir / f"{CHUNK_PREFIX}{idx}.pt"
                    if chunk_file.exists():
                        data = chunk_file.read_bytes()
                        self.send_response(200)
                        self.send_header("Content-Type", "application/octet-stream")
                        self.send_header("Content-Length", str(len(data)))
                        self.end_headers()
                        self.wfile.write(data)
                    else:
                        self.send_error(404)
                # GET /have — list which chunks we have
                elif self.path == "/have":
                    chunks = []
                    for f in seeder.chunks_dir.glob(f"{CHUNK_PREFIX}*.pt"):
                        idx = f.stem.replace(CHUNK_PREFIX, "")
                        chunks.append(idx)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(chunks).encode())
                else:
                    self.send_error(404)

        self._server = HTTPServer(("0.0.0.0", self.port), Handler)
        actual_port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return actual_port

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server = None


# ── Tracker (runs on the hub) ──────────────────────────────────────────────

@dataclass
class PeerSeed:
    """A peer that is seeding chunks."""
    peer_id: str
    address: str          # http://ip:port where this peer serves chunks
    chunks: list[str]     # which chunk indices this peer has
    last_seen: float = 0


class Tracker:
    """Tracks which peers have which chunks.

    Integrated into the Hub. When a peer wants to download, the tracker
    tells it which other peers have the chunks it needs.
    """

    def __init__(self):
        self.seeds: dict[str, PeerSeed] = {}  # peer_id -> PeerSeed
        self._lock = threading.Lock()

    def register_seed(self, peer_id: str, address: str, chunks: list[str]) -> None:
        """Register a peer as a seeder for certain chunks."""
        with self._lock:
            self.seeds[peer_id] = PeerSeed(
                peer_id=peer_id,
                address=address,
                chunks=chunks,
                last_seen=time.time(),
            )

    def get_sources(self, chunk_idx: str) -> list[str]:
        """Get list of peer addresses that have a specific chunk."""
        with self._lock:
            sources = []
            cutoff = time.time() - 300  # only consider peers seen in last 5 min
            for seed in self.seeds.values():
                if seed.last_seen > cutoff and chunk_idx in seed.chunks:
                    sources.append(seed.address)
            return sources

    def get_all_sources(self) -> dict[str, list[str]]:
        """Get all chunk -> sources mapping."""
        with self._lock:
            result: dict[str, list[str]] = {}
            cutoff = time.time() - 300
            for seed in self.seeds.values():
                if seed.last_seen > cutoff:
                    for idx in seed.chunks:
                        result.setdefault(idx, []).append(seed.address)
            return result

    def stats(self) -> dict:
        cutoff = time.time() - 300
        active = [s for s in self.seeds.values() if s.last_seen > cutoff]
        return {
            "active_seeders": len(active),
            "total_registered": len(self.seeds),
        }


# ── Downloader (runs on peers) ─────────────────────────────────────────────

def download_chunks(
    manifest: Manifest,
    output_dir: str,
    hub_url: str,
    token: str,
    tracker_sources: dict[str, list[str]] | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> list[int]:
    """Download model chunks from hub and/or peers.

    Tries peer sources first (faster, distributes load).
    Falls back to hub for any chunks peers don't have.

    Args:
        manifest: The model manifest.
        output_dir: Where to save chunks.
        hub_url: Hub URL for fallback downloads.
        token: Auth token for hub.
        tracker_sources: chunk_idx -> list of peer URLs (from tracker).
        on_progress: Callback(chunk_idx, total, source_type).

    Returns:
        List of chunk indices that were downloaded.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    total = len(manifest.chunks)
    downloaded = []
    sources = tracker_sources or {}

    for chunk_info in manifest.chunks:
        idx_str = f"{chunk_info.index:04d}"
        chunk_path = output / f"{CHUNK_PREFIX}{idx_str}.pt"

        # Skip if we already have it and it's valid
        if chunk_path.exists():
            raw = chunk_path.read_bytes()
            if hashlib.sha256(raw).hexdigest() == chunk_info.sha256:
                if on_progress:
                    on_progress(chunk_info.index, total, "cached")
                continue

        # Try peer sources first
        got_from_peer = False
        peer_urls = sources.get(idx_str, [])
        for peer_url in peer_urls:
            try:
                req = urllib.request.Request(f"{peer_url}/chunk/{idx_str}")
                with urllib.request.urlopen(req, timeout=30) as resp:
                    raw = resp.read()
                # Verify
                if hashlib.sha256(raw).hexdigest() == chunk_info.sha256:
                    chunk_path.write_bytes(raw)
                    got_from_peer = True
                    if on_progress:
                        on_progress(chunk_info.index, total, "peer")
                    break
            except Exception:
                continue

        # Fall back to hub
        if not got_from_peer:
            try:
                req = urllib.request.Request(
                    f"{hub_url}/chunk/{idx_str}",
                    headers={"X-GDF-Token": token},
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    raw = resp.read()
                if hashlib.sha256(raw).hexdigest() == chunk_info.sha256:
                    chunk_path.write_bytes(raw)
                    if on_progress:
                        on_progress(chunk_info.index, total, "hub")
                else:
                    raise ValueError(f"Chunk {idx_str} hash mismatch from hub")
            except Exception as e:
                raise RuntimeError(f"Failed to download chunk {idx_str}: {e}")

        downloaded.append(chunk_info.index)

    return downloaded

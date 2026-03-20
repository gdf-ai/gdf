"""Distributed training at scale.

Designed to work from 2 peers to 10,000+. Three key innovations over
naive federated learning:

1. DELTA COMPRESSION
   Peers don't send the full model. They send only what changed
   (model - base), compressed. For a 7B model where 5% of weights
   changed significantly, this reduces transfer from 14GB to ~700MB.

2. HIERARCHICAL MERGING
   Instead of one hub merging 10k models at once (which destroys info),
   we use a tree of hubs:

       Root Hub
      /    |    \\
   Hub1  Hub2  Hub3    (regional hubs, each handles ~100 peers)
   /|\\   /|\\   /|\\
   peers peers peers

   Each level merges small groups (3-10), preserving more knowledge.

3. ASYNC ROUNDS
   Peers don't need to synchronize. Each trains at their own speed,
   pushes when ready. The hub merges when it has enough contributions.
   Slow peers don't block fast ones.

Architecture:
    gdf hub              → run a hub (root or regional)
    gdf peer <hub-url>   → join as a training worker
"""

from __future__ import annotations

import io
import json
import random
import time
import zlib
import hashlib
import threading
from pathlib import Path
from dataclasses import dataclass, field
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Callable
import urllib.request
import urllib.error

import torch

from .model import TinyTransformer, ModelConfig
from .serialization import save_model, load_model, _compute_hash
from .merging import merge_models
from .trainer import OnlineTrainer
from .device import detect_device
from .torrent import (
    Manifest, Tracker, ChunkSeeder,
    create_chunks, reassemble_model, download_chunks,
)


# ── Delta compression ───────────────────────────────────────────────────────

def compute_delta(current: dict, base: dict, threshold: float = 1e-6) -> dict:
    """Compute the difference between current and base model weights.

    Only includes parameters that changed more than threshold.
    This is what gets sent over the network instead of the full model.

    A 7B model where 5% of weights changed significantly:
        Full model:  ~14 GB
        Delta only:  ~700 MB
        Compressed:  ~200-400 MB
    """
    delta = {}
    for key in current:
        if key in base:
            diff = current[key].cpu().float() - base[key].cpu().float()
            # Only include if the change is significant
            if diff.abs().max().item() > threshold:
                # Sparsify: zero out tiny changes to improve compression
                mask = diff.abs() > threshold
                sparse_diff = diff * mask.float()
                delta[key] = sparse_diff.half()  # fp16 to save space
        else:
            delta[key] = current[key].cpu().half()
    return delta


def apply_delta(base: dict, delta: dict) -> dict:
    """Apply a delta back onto a base model to reconstruct the trained model."""
    result = {}
    for key in base:
        if key in delta:
            result[key] = (base[key].float() + delta[key].float()).to(base[key].dtype)
        else:
            result[key] = base[key]
    return result


def compress_delta(delta: dict) -> bytes:
    """Serialize and compress a delta dict for network transfer."""
    buf = io.BytesIO()
    torch.save(delta, buf)
    raw = buf.getvalue()
    compressed = zlib.compress(raw, level=6)
    return compressed


def decompress_delta(data: bytes) -> dict:
    """Decompress and deserialize a delta dict."""
    raw = zlib.decompress(data)
    buf = io.BytesIO(raw)
    return torch.load(buf, map_location="cpu", weights_only=False)


def delta_stats(full_size: int, delta_bytes: int) -> dict:
    """Calculate compression statistics."""
    return {
        "full_model_bytes": full_size,
        "delta_bytes": delta_bytes,
        "compression_ratio": full_size / max(delta_bytes, 1),
        "savings_pct": (1 - delta_bytes / max(full_size, 1)) * 100,
    }


# ── Peer info ───────────────────────────────────────────────────────────────

@dataclass
class PeerInfo:
    """Information about a connected peer."""
    peer_id: str
    last_seen: float
    contributions: int = 0
    device_info: dict = field(default_factory=dict)
    gpu_name: str = ""
    vram_gb: float = 0.0


# ── Hub (coordination server) ──────────────────────────────────────────────

def generate_token() -> str:
    """Generate a random auth token for the hub."""
    import secrets
    return secrets.token_urlsafe(32)


class Hub:
    """Coordination server that collects and merges peer contributions.

    Supports both full model and delta-based communication.
    Can operate as a root hub or regional hub in a hierarchy.
    Protected by a shared token — only peers with the token can connect.
    """

    def __init__(
        self,
        model_path: str,
        merge_strategy: str = "fedavg",
        merge_every: int = 3,
        host: str = "0.0.0.0",
        port: int = 7677,
        token: str | None = None,        # auth token (auto-generated if None)
        parent_hub: str | None = None,
        push_to_parent_every: int = 5,
        seeds_file: str | None = None,
    ):
        self.model_path = model_path
        self.merge_strategy = merge_strategy
        self.merge_every = merge_every
        self.host = host
        self.port = port
        self.token = token or generate_token()
        self.parent_hub = parent_hub
        self.push_to_parent_every = push_to_parent_every

        self.peers: dict[str, PeerInfo] = {}
        self.pending_deltas: list[tuple[str, bytes]] = []
        self.pending_full_models: list[tuple[str, bytes]] = []
        self.generation = 0
        self.total_merges = 0
        self.merges_since_parent_push = 0
        self._lock = threading.Lock()
        self._on_log: Callable[[str], None] | None = None

        # Delta validation
        self._max_delta_norm = 10.0

        # P2P distribution
        self._tracker = Tracker()
        self._chunks_dir = str(Path(model_path).parent / "chunks")
        self._manifest: Manifest | None = None

        # Seed URLs for peer assignment (peers crawl autonomously)
        self._seed_urls: list[str] = []
        if seeds_file:
            try:
                for line in Path(seeds_file).read_text().strip().splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        self._seed_urls.append(line)
            except Exception:
                pass

        # Cache the base model weights for delta computation
        self._base_weights: dict | None = None
        self._load_base()

    def _load_base(self) -> None:
        """Load and cache the current base model weights, and chunk for P2P."""
        try:
            model, _, _ = load_model(self.model_path, load_trainer=False)
            self._base_weights = {k: v.clone() for k, v in model.state_dict().items()}
            self._manifest = create_chunks(self.model_path, self._chunks_dir)
        except Exception:
            self._base_weights = None

    def _log(self, msg: str) -> None:
        if self._on_log:
            self._on_log(msg)

    def _check_auth(self, handler) -> bool:
        """Verify the request has the correct auth token."""
        token = handler.headers.get("X-GDF-Token", "")
        if token != self.token:
            handler.send_error(403, "Invalid or missing token")
            self._log(f"  REJECTED: bad token from {handler.client_address[0]}")
            return False
        return True

    def _validate_delta(self, delta: dict) -> str | None:
        """Return error string if delta is suspicious, None if OK."""
        for key, tensor in delta.items():
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                return f"rejected: NaN/Inf in {key}"
            if tensor.abs().max().item() > self._max_delta_norm:
                return f"rejected: {key} change too large ({tensor.abs().max().item():.2f} > {self._max_delta_norm})"
        return None

    def start(self, on_log: Callable[[str], None] | None = None) -> None:
        """Start the hub server (blocking)."""
        self._on_log = on_log
        hub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def do_GET(self):
                if not hub._check_auth(self):
                    return
                if self.path == "/model":
                    hub._handle_pull(self)
                elif self.path == "/status":
                    hub._handle_status(self)
                elif self.path == "/base-hash":
                    hub._handle_base_hash(self)
                elif self.path == "/manifest":
                    hub._handle_manifest(self)
                elif self.path.startswith("/chunk/"):
                    hub._handle_chunk(self)
                elif self.path == "/tracker/sources":
                    hub._handle_tracker_sources(self)
                else:
                    self.send_error(404)

            def do_POST(self):
                if not hub._check_auth(self):
                    return
                if self.path == "/model":
                    hub._handle_push_full(self)
                elif self.path == "/delta":
                    hub._handle_push_delta(self)
                elif self.path == "/text":
                    hub._handle_push_text(self)
                elif self.path == "/register":
                    hub._handle_register(self)
                elif self.path == "/tracker/register":
                    hub._handle_tracker_register(self)
                else:
                    self.send_error(404)

        self._log(f"Hub starting on {self.host}:{self.port}")
        self._log(f"Strategy: {self.merge_strategy} | merge every {self.merge_every}")
        if self.parent_hub:
            self._log(f"Parent hub: {self.parent_hub} (push every {self.push_to_parent_every} merges)")
        self._log(f"Model: {self.model_path}")
        self._log("")

        server = HTTPServer((self.host, self.port), Handler)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            self._log("Hub shutting down.")
            server.shutdown()

    def _handle_pull(self, handler) -> None:
        """Peer requests the current global model."""
        try:
            model_bytes = Path(self.model_path).read_bytes()
            handler.send_response(200)
            handler.send_header("Content-Type", "application/octet-stream")
            handler.send_header("X-Generation", str(self.generation))
            handler.send_header("X-Base-Hash", _compute_hash(self._base_weights) if self._base_weights else "none")
            handler.end_headers()
            handler.wfile.write(model_bytes)
            self._log(f"  -> Model sent ({len(model_bytes):,} bytes, gen {self.generation})")
        except Exception as e:
            handler.send_error(500, str(e))

    def _handle_base_hash(self, handler) -> None:
        """Return the hash of the current base model (for delta validation)."""
        h = _compute_hash(self._base_weights) if self._base_weights else "none"
        handler.send_response(200)
        handler.send_header("Content-Type", "text/plain")
        handler.end_headers()
        handler.wfile.write(h.encode())

    def _handle_push_delta(self, handler) -> None:
        """Peer pushes a compressed delta (only what changed)."""
        content_length = int(handler.headers.get("Content-Length", 0))
        peer_id = handler.headers.get("X-Peer-ID", "unknown")
        delta_bytes = handler.rfile.read(content_length)

        # Validate delta
        try:
            delta = decompress_delta(delta_bytes)
            error = self._validate_delta(delta)
            if error:
                self._log(f"  REJECTED delta from {peer_id}: {error}")
                handler.send_response(400)
                handler.send_header("Content-Type", "application/json")
                handler.end_headers()
                handler.wfile.write(json.dumps({"status": "rejected", "reason": error}).encode())
                return
        except Exception as e:
            self._log(f"  REJECTED delta from {peer_id}: bad data ({e})")
            handler.send_response(400)
            handler.send_header("Content-Type", "application/json")
            handler.end_headers()
            handler.wfile.write(json.dumps({"status": "rejected", "reason": str(e)}).encode())
            return

        with self._lock:
            self.pending_deltas.append((peer_id, delta_bytes))

            if peer_id in self.peers:
                self.peers[peer_id].contributions += 1
                self.peers[peer_id].last_seen = time.time()

            total_pending = len(self.pending_deltas) + len(self.pending_full_models)
            self._log(f"  <- Delta from {peer_id} ({len(delta_bytes):,} bytes)")
            self._log(f"     Pending: {total_pending}/{self.merge_every}")

            if total_pending >= self.merge_every:
                self._do_merge()

        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.end_headers()
        handler.wfile.write(json.dumps({
            "status": "accepted",
            "type": "delta",
            "generation": self.generation,
        }).encode())

    def _handle_push_full(self, handler) -> None:
        """Peer pushes a full model (fallback when delta isn't possible)."""
        content_length = int(handler.headers.get("Content-Length", 0))
        peer_id = handler.headers.get("X-Peer-ID", "unknown")
        model_bytes = handler.rfile.read(content_length)

        # Validate: compare against base to check for poisoning
        if self._base_weights is not None:
            try:
                buf = io.BytesIO(model_bytes)
                data = torch.load(buf, map_location="cpu", weights_only=False)
                pushed_weights = data.get("weights", {})
                if pushed_weights:
                    delta = compute_delta(pushed_weights, self._base_weights)
                    error = self._validate_delta(delta)
                    if error:
                        self._log(f"  REJECTED full model from {peer_id}: {error}")
                        handler.send_response(400)
                        handler.send_header("Content-Type", "application/json")
                        handler.end_headers()
                        handler.wfile.write(json.dumps({"status": "rejected", "reason": error}).encode())
                        return
            except Exception:
                pass  # Can't validate — accept anyway (might be different format)

        with self._lock:
            self.pending_full_models.append((peer_id, model_bytes))

            if peer_id in self.peers:
                self.peers[peer_id].contributions += 1
                self.peers[peer_id].last_seen = time.time()

            total_pending = len(self.pending_deltas) + len(self.pending_full_models)
            self._log(f"  <- Full model from {peer_id} ({len(model_bytes):,} bytes)")
            self._log(f"     Pending: {total_pending}/{self.merge_every}")

            if total_pending >= self.merge_every:
                self._do_merge()

        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.end_headers()
        handler.wfile.write(json.dumps({
            "status": "accepted",
            "type": "full",
            "generation": self.generation,
        }).encode())

    def _handle_register(self, handler) -> None:
        """Peer registers itself."""
        content_length = int(handler.headers.get("Content-Length", 0))
        body = json.loads(handler.rfile.read(content_length).decode())
        peer_id = body.get("peer_id", "unknown")
        dev = body.get("device_info", {})

        gpu_name = ""
        vram = 0.0
        gpus = dev.get("cuda_devices", [])
        if gpus:
            gpu_name = gpus[0].get("name", "")
            vram = gpus[0].get("vram_gb", 0.0)

        with self._lock:
            self.peers[peer_id] = PeerInfo(
                peer_id=peer_id,
                last_seen=time.time(),
                device_info=dev,
                gpu_name=gpu_name,
                vram_gb=vram,
            )

        # Pick a seed URL for this peer + send the full pool for reseeding
        if self._seed_urls:
            seed_url = random.choice(self._seed_urls)
        else:
            seed_url = "https://en.wikipedia.org/wiki/Special:Random"

        self._log(f"  Peer registered: {peer_id} ({gpu_name or dev.get('device', 'cpu')}, {vram}GB)")

        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.end_headers()
        handler.wfile.write(json.dumps({
            "status": "registered",
            "generation": self.generation,
            "seed_url": seed_url,
            "reseed_urls": self._seed_urls,
        }).encode())

    def _handle_status(self, handler) -> None:
        """Return hub status."""
        with self._lock:
            total_vram = sum(p.vram_gb for p in self.peers.values())
            total_contributions = sum(p.contributions for p in self.peers.values())

            status = {
                "generation": self.generation,
                "total_merges": self.total_merges,
                "active_peers": len(self.peers),
                "total_contributions": total_contributions,
                "total_vram_gb": round(total_vram, 1),
                "pending_deltas": len(self.pending_deltas),
                "pending_full": len(self.pending_full_models),
                "merge_strategy": self.merge_strategy,
                "parent_hub": self.parent_hub,
                "peers": [
                    {
                        "peer_id": p.peer_id,
                        "gpu": p.gpu_name or "cpu",
                        "vram_gb": p.vram_gb,
                        "contributions": p.contributions,
                        "last_seen_ago": f"{time.time() - p.last_seen:.0f}s",
                    }
                    for p in sorted(self.peers.values(),
                                    key=lambda p: p.contributions, reverse=True)
                ],
            }

        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.end_headers()
        handler.wfile.write(json.dumps(status, indent=2).encode())

    def _handle_manifest(self, handler) -> None:
        """Return the model manifest for P2P download."""
        if not self._manifest:
            handler.send_error(404, "No manifest available")
            return
        data = self._manifest.to_json().encode()
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.end_headers()
        handler.wfile.write(data)

    def _handle_chunk(self, handler) -> None:
        """Serve a model chunk (hub as initial seed)."""
        idx = handler.path.split("/chunk/")[1]
        chunk_file = Path(self._chunks_dir) / f"chunk_{idx}.pt"
        if not chunk_file.exists():
            handler.send_error(404, f"Chunk {idx} not found")
            return
        data = chunk_file.read_bytes()
        handler.send_response(200)
        handler.send_header("Content-Type", "application/octet-stream")
        handler.send_header("Content-Length", str(len(data)))
        handler.end_headers()
        handler.wfile.write(data)

    def _handle_tracker_sources(self, handler) -> None:
        """Return peer→chunk mapping from tracker."""
        sources = self._tracker.get_all_sources()
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.end_headers()
        handler.wfile.write(json.dumps(sources).encode())

    def _handle_tracker_register(self, handler) -> None:
        """Register a peer as a seeder."""
        content_length = int(handler.headers.get("Content-Length", 0))
        body = json.loads(handler.rfile.read(content_length).decode())
        peer_id = body.get("peer_id", "unknown")
        address = body.get("address", "")
        chunks = body.get("chunks", [])

        # Validate: peer must be registered
        if peer_id not in self.peers:
            handler.send_error(403, "Peer not registered")
            return

        # Validate: chunk indices must exist in manifest
        if self._manifest:
            valid_indices = {f"{c.index:04d}" for c in self._manifest.chunks}
            invalid = [c for c in chunks if c not in valid_indices]
            if invalid:
                handler.send_error(400, f"Invalid chunk indices: {invalid}")
                return

        self._tracker.register_seed(peer_id, address, chunks)
        self._log(f"  Seeder registered: {peer_id} ({len(chunks)} chunks at {address})")

        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.end_headers()
        handler.wfile.write(json.dumps({"status": "registered"}).encode())

    def _handle_push_text(self, handler) -> None:
        """Peer pushes training text — hub trains on it directly."""
        content_length = int(handler.headers.get("Content-Length", 0))
        body = json.loads(handler.rfile.read(content_length).decode())
        peer_id = body.get("peer_id", "unknown")
        text = body.get("text", "")
        source = body.get("source", "unknown")

        if not text or len(text.strip()) < 10:
            handler.send_response(400)
            handler.send_header("Content-Type", "application/json")
            handler.end_headers()
            handler.wfile.write(json.dumps({"status": "rejected", "reason": "text too short"}).encode())
            return

        try:
            from .api import GDFModel
            bm = GDFModel.load(self.model_path)
            bm.train_text(text, epochs=3)
            bm.save(self.model_path)
        except Exception as e:
            self._log(f"  Training on text from {peer_id} failed: {e}")
            handler.send_response(500)
            handler.send_header("Content-Type", "application/json")
            handler.end_headers()
            handler.wfile.write(json.dumps({"status": "error", "reason": str(e)}).encode())
            return

        with self._lock:
            self.generation += 1

            if peer_id in self.peers:
                self.peers[peer_id].contributions += 1
                self.peers[peer_id].last_seen = time.time()

        self._log(f"  <- Text from {peer_id} ({len(text):,} chars, source: {source})")

        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.end_headers()
        handler.wfile.write(json.dumps({
            "status": "ok",
            "text_length": len(text),
            "generation": self.generation,
        }).encode())

    def _do_merge(self) -> None:
        """Merge all pending contributions with the global model."""
        total = len(self.pending_deltas) + len(self.pending_full_models)
        self._log(f"  Merging {total} contributions...")

        # Load current global model
        global_model, _, meta = load_model(self.model_path, load_trainer=False)
        models = [global_model]
        base_sd = {k: v.clone() for k, v in global_model.state_dict().items()}

        # Reconstruct models from deltas
        for peer_id, delta_bytes in self.pending_deltas:
            try:
                delta = decompress_delta(delta_bytes)
                reconstructed_sd = apply_delta(base_sd, delta)
                config = global_model.config
                m = TinyTransformer(config)
                m.load_state_dict(reconstructed_sd)
                models.append(m)
            except Exception as e:
                self._log(f"    Bad delta from {peer_id}: {e}")

        # Load full models
        for peer_id, model_bytes in self.pending_full_models:
            try:
                buf = io.BytesIO(model_bytes)
                data = torch.load(buf, map_location="cpu", weights_only=False)
                config = ModelConfig.from_dict(data["config"])
                m = TinyTransformer(config)
                m.load_state_dict(data["weights"])
                models.append(m)
            except Exception as e:
                self._log(f"    Bad model from {peer_id}: {e}")

        if len(models) < 2:
            self._log("    Not enough valid contributions.")
            self.pending_deltas.clear()
            self.pending_full_models.clear()
            return

        # Merge all
        merged, new_hash = merge_models(models, strategy=self.merge_strategy)
        trainer = OnlineTrainer(merged, device=torch.device("cpu"))
        base_hash = meta.get("base_model_hash")
        save_model(self.model_path, merged, trainer, base_hash)

        # Update base weights cache
        self._base_weights = {k: v.clone() for k, v in merged.state_dict().items()}

        self.generation += 1
        self.total_merges += 1
        self.pending_deltas.clear()
        self.pending_full_models.clear()

        # Re-chunk for P2P distribution
        try:
            self._manifest = create_chunks(self.model_path, self._chunks_dir)
        except Exception as e:
            self._log(f"    Warning: failed to re-chunk: {e}")

        self._log(f"    Merged! Gen {self.generation} "
                  f"({merged.count_parameters():,} params, hash={new_hash})")

        # Push to parent hub if configured
        if self.parent_hub:
            self.merges_since_parent_push += 1
            if self.merges_since_parent_push >= self.push_to_parent_every:
                self._push_to_parent()
                self.merges_since_parent_push = 0

    def _push_to_parent(self) -> None:
        """Push the current merged model to the parent hub."""
        self._log(f"  Pushing to parent hub {self.parent_hub}...")
        try:
            model_bytes = Path(self.model_path).read_bytes()
            req = urllib.request.Request(
                f"{self.parent_hub}/model",
                data=model_bytes,
                headers={
                    "Content-Type": "application/octet-stream",
                    "X-Peer-ID": f"hub-{self.port}",
                    "Content-Length": str(len(model_bytes)),
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode())
            self._log(f"    Pushed to parent (gen {result.get('generation', '?')})")
        except Exception as e:
            self._log(f"    Failed to push to parent: {e}")


# ── Peer (worker) ───────────────────────────────────────────────────────────

class Peer:
    """A distributed training peer that connects to a Hub.

    Uses delta compression by default — only sends what changed.
    Falls back to full model transfer when the base model has changed.
    Authenticates with the hub using a shared token.
    """

    def __init__(
        self,
        hub_url: str,
        token: str,
        peer_id: str | None = None,
        local_model_path: str = "peer_model.pt",
    ):
        self.hub_url = hub_url.rstrip("/")
        self.token = token
        self.peer_id = peer_id or self._generate_peer_id()
        self.local_model_path = local_model_path
        self._base_weights: dict | None = None
        self._base_hash: str | None = None
        self._seeder: ChunkSeeder | None = None

    def _auth_headers(self) -> dict:
        """Return headers with auth token."""
        return {"X-GDF-Token": self.token}

    def _generate_peer_id(self) -> str:
        """Generate a unique peer ID."""
        import socket
        hostname = socket.gethostname()
        return hashlib.sha256(f"{hostname}-{time.time()}".encode()).hexdigest()[:12]

    def register(self) -> dict:
        """Register with the hub. Stores the seed_url for autonomous crawling."""
        from .device import device_info
        body = json.dumps({
            "peer_id": self.peer_id,
            "device_info": device_info(),
        }).encode()

        req = urllib.request.Request(
            f"{self.hub_url}/register",
            data=body,
            headers={"Content-Type": "application/json", **self._auth_headers()},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())

        self.seed_url = result.get("seed_url", "https://en.wikipedia.org/wiki/Special:Random")
        return result

    def pull_model(self) -> tuple[str, dict]:
        """Pull the latest model from the hub.

        Tries P2P chunked download first, falls back to direct download.
        Returns (local_path, pull_info).
        Caches the base weights for delta computation on push.
        """
        # Try P2P chunked download first
        try:
            return self._pull_chunked()
        except Exception:
            pass

        # Fallback: direct download
        req = urllib.request.Request(f"{self.hub_url}/model", headers=self._auth_headers())
        with urllib.request.urlopen(req, timeout=120) as resp:
            model_bytes = resp.read()
            generation = resp.headers.get("X-Generation", "?")
            base_hash = resp.headers.get("X-Base-Hash", "none")

        Path(self.local_model_path).write_bytes(model_bytes)

        # Cache the base weights for delta on push
        model, _, _ = load_model(self.local_model_path, load_trainer=False)
        self._base_weights = {k: v.clone() for k, v in model.state_dict().items()}
        self._base_hash = base_hash

        return self.local_model_path, {
            "generation": generation,
            "size_bytes": len(model_bytes),
            "params": model.count_parameters(),
        }

    def _pull_chunked(self) -> tuple[str, dict]:
        """Pull model via P2P chunks."""
        # Fetch manifest
        req = urllib.request.Request(
            f"{self.hub_url}/manifest", headers=self._auth_headers(),
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            manifest = Manifest.from_json(resp.read().decode())

        # Get tracker sources
        try:
            req = urllib.request.Request(
                f"{self.hub_url}/tracker/sources", headers=self._auth_headers(),
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                tracker_sources = json.loads(resp.read().decode())
        except Exception:
            tracker_sources = None

        # Download chunks
        chunks_dir = str(Path(self.local_model_path).parent / "peer_chunks")
        download_chunks(
            manifest, chunks_dir, self.hub_url, self.token,
            tracker_sources=tracker_sources,
        )

        # Reassemble
        reassemble_model(manifest, chunks_dir, self.local_model_path)

        # Cache base weights
        model, _, _ = load_model(self.local_model_path, load_trainer=False)
        self._base_weights = {k: v.clone() for k, v in model.state_dict().items()}
        self._base_hash = _compute_hash(self._base_weights)

        # Start seeding
        self._start_seeding(chunks_dir)

        return self.local_model_path, {
            "generation": "?",
            "size_bytes": manifest.total_size,
            "params": model.count_parameters(),
            "transfer": "p2p",
        }

    def _start_seeding(self, chunks_dir: str) -> None:
        """Start seeding chunks and register with tracker."""
        if self._seeder:
            self._seeder.stop()

        self._seeder = ChunkSeeder(chunks_dir, port=0)
        port = self._seeder.start()

        # Register with tracker
        import socket
        local_ip = "127.0.0.1"
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
        except Exception:
            pass

        # List chunks we have
        chunks = []
        for f in Path(chunks_dir).glob("chunk_*.pt"):
            chunks.append(f.stem.replace("chunk_", ""))

        try:
            body = json.dumps({
                "peer_id": self.peer_id,
                "address": f"http://{local_ip}:{port}",
                "chunks": chunks,
            }).encode()
            req = urllib.request.Request(
                f"{self.hub_url}/tracker/register",
                data=body,
                headers={"Content-Type": "application/json", **self._auth_headers()},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=10)
        except Exception:
            pass  # Non-critical — still works without tracker

    def push_model(self, model_path: str | None = None) -> dict:
        """Push trained model to hub. Uses delta compression when possible.

        Returns push result dict with compression stats.
        """
        path = model_path or self.local_model_path
        model, _, _ = load_model(path, load_trainer=False)
        current_weights = model.state_dict()

        # Try delta compression if we have a cached base
        if self._base_weights is not None:
            # Verify hub still has the same base
            try:
                req = urllib.request.Request(f"{self.hub_url}/base-hash",
                                            headers=self._auth_headers())
                with urllib.request.urlopen(req, timeout=10) as resp:
                    hub_hash = resp.read().decode().strip()

                if hub_hash == self._base_hash:
                    return self._push_delta(current_weights)
            except Exception:
                pass  # Fall through to full push

        # Fallback: send full model
        return self._push_full(path)

    def _push_delta(self, current_weights: dict) -> dict:
        """Push only the delta (what changed) — much smaller."""
        delta = compute_delta(current_weights, self._base_weights)
        compressed = compress_delta(delta)

        # Calculate stats
        full_buf = io.BytesIO()
        torch.save(current_weights, full_buf)
        full_size = len(full_buf.getvalue())
        stats = delta_stats(full_size, len(compressed))

        req = urllib.request.Request(
            f"{self.hub_url}/delta",
            data=compressed,
            headers={
                "Content-Type": "application/octet-stream",
                "X-Peer-ID": self.peer_id,
                "Content-Length": str(len(compressed)),
                **self._auth_headers(),
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode())

        result["transfer_type"] = "delta"
        result["compression"] = stats
        return result

    def _push_full(self, path: str) -> dict:
        """Push the full model (fallback)."""
        model_bytes = Path(path).read_bytes()

        req = urllib.request.Request(
            f"{self.hub_url}/model",
            data=model_bytes,
            headers={
                "Content-Type": "application/octet-stream",
                "X-Peer-ID": self.peer_id,
                "Content-Length": str(len(model_bytes)),
                **self._auth_headers(),
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode())

        result["transfer_type"] = "full"
        result["bytes_sent"] = len(model_bytes)
        return result

    def push_text(self, text: str, source: str = "unknown") -> dict:
        """Push training text to the hub (hub trains on it)."""
        body = json.dumps({
            "peer_id": self.peer_id,
            "text": text,
            "source": source,
        }).encode()

        req = urllib.request.Request(
            f"{self.hub_url}/text",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
                **self._auth_headers(),
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode())

    def hub_status(self) -> dict:
        """Get the hub's current status."""
        req = urllib.request.Request(f"{self.hub_url}/status", headers=self._auth_headers())
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())

    def train_and_push(
        self,
        cycles: int = 5,
        on_status: Callable[[str], None] | None = None,
        topics: list[str] | None = None,
    ) -> dict:
        """Crawl text and push to hub for training.

        Args:
            cycles: Number of articles to fetch and send.
            on_status: Status callback.
            topics: Optional topic list for Wikipedia fetching.

        Returns:
            Dict with results.
        """
        from .selflearn import fetch_wikipedia_random, fetch_wikipedia_topic

        def status(msg: str):
            if on_status:
                on_status(msg)

        articles_sent = 0
        total_chars = 0
        topic_idx = 0

        for i in range(cycles):
            # Pick topic
            topic = None
            if topics:
                topic = topics[topic_idx % len(topics)]
                topic_idx += 1

            # Fetch
            status(f"  Cycle {i + 1}/{cycles}: Fetching...")
            try:
                if topic:
                    articles = fetch_wikipedia_topic(topic)
                    if articles:
                        title, text = articles[0]
                    else:
                        title, text = fetch_wikipedia_random()
                else:
                    title, text = fetch_wikipedia_random()
            except Exception as e:
                status(f"  Fetch failed: {e}")
                continue

            if len(text.strip()) < 50:
                status(f"  Skipping '{title}' — too short")
                continue

            # Truncate very long articles
            if len(text) > 50000:
                text = text[:50000]

            # Push text to hub
            status(f"  Sending '{title}' ({len(text):,} chars)...")
            try:
                result = self.push_text(text, source=f"wikipedia:{title}")
                gen = result.get("generation", "?")
                status(f"  Accepted. Hub generation: {gen}")
                articles_sent += 1
                total_chars += len(text)
            except Exception as e:
                status(f"  Push failed: {e}")

        return {
            "cycles": cycles,
            "articles_sent": articles_sent,
            "total_chars": total_chars,
        }

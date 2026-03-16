"""Tests for P2P torrent-style model distribution."""

import json
import tempfile
from pathlib import Path

import torch
import pytest

from gdf.torrent import (
    Manifest, ChunkInfo, Tracker,
    create_chunks, reassemble_model, download_chunks,
)
from gdf.model import TinyTransformer, ModelConfig
from gdf.serialization import save_model, load_model
from gdf.trainer import OnlineTrainer


@pytest.fixture
def tmp_model(tmp_path):
    """Create a small model and save it."""
    config = ModelConfig(d_model=64, n_heads=2, n_layers=2, d_ff=128)
    model = TinyTransformer(config)
    trainer = OnlineTrainer(model, device=torch.device("cpu"))
    model_path = str(tmp_path / "test_model.pt")
    save_model(model_path, model, trainer)
    return model_path, model


class TestChunking:
    def test_create_and_reassemble_roundtrip(self, tmp_model, tmp_path):
        """create_chunks + reassemble_model should produce identical weights."""
        model_path, original = tmp_model
        chunks_dir = str(tmp_path / "chunks")
        output_path = str(tmp_path / "reassembled.pt")

        manifest = create_chunks(model_path, chunks_dir)
        assert len(manifest.chunks) > 0
        assert manifest.total_size > 0

        reassemble_model(manifest, chunks_dir, output_path)

        reassembled, _, _ = load_model(output_path, load_trainer=False)
        orig_sd = original.state_dict()
        new_sd = reassembled.state_dict()

        assert set(orig_sd.keys()) == set(new_sd.keys())
        for key in orig_sd:
            assert torch.equal(orig_sd[key], new_sd[key]), f"Mismatch in {key}"

    def test_chunks_are_verifiable(self, tmp_model, tmp_path):
        """Each chunk file should match its hash in the manifest."""
        model_path, _ = tmp_model
        chunks_dir = str(tmp_path / "chunks")

        manifest = create_chunks(model_path, chunks_dir)

        import hashlib
        for chunk in manifest.chunks:
            chunk_file = Path(chunks_dir) / f"chunk_{chunk.index:04d}.pt"
            assert chunk_file.exists()
            actual_hash = hashlib.sha256(chunk_file.read_bytes()).hexdigest()
            assert actual_hash == chunk.sha256

    def test_corrupted_chunk_detected(self, tmp_model, tmp_path):
        """Reassembly should fail if a chunk is corrupted."""
        model_path, _ = tmp_model
        chunks_dir = str(tmp_path / "chunks")
        output_path = str(tmp_path / "bad.pt")

        manifest = create_chunks(model_path, chunks_dir)

        # Corrupt first chunk
        first_chunk = Path(chunks_dir) / f"chunk_{manifest.chunks[0].index:04d}.pt"
        first_chunk.write_bytes(b"corrupted data")

        with pytest.raises(ValueError, match="corrupted"):
            reassemble_model(manifest, chunks_dir, output_path)


class TestManifest:
    def test_json_roundtrip(self):
        """Manifest should survive JSON serialization."""
        manifest = Manifest(
            config={"d_model": 64, "n_heads": 2},
            chunks=[
                ChunkInfo(index=0, keys=["a", "b"], sha256="abc123", size_bytes=100),
                ChunkInfo(index=1, keys=["c"], sha256="def456", size_bytes=200),
            ],
            total_size=300,
            model_hash="xyz789",
        )

        json_str = manifest.to_json()
        restored = Manifest.from_json(json_str)

        assert restored.config == manifest.config
        assert len(restored.chunks) == 2
        assert restored.chunks[0].keys == ["a", "b"]
        assert restored.chunks[1].sha256 == "def456"
        assert restored.total_size == 300
        assert restored.model_hash == "xyz789"


class TestTracker:
    def test_register_and_get_sources(self):
        """Registered seeders should appear in source queries."""
        tracker = Tracker()
        tracker.register_seed("peer1", "http://1.2.3.4:8000", ["0000", "0001"])
        tracker.register_seed("peer2", "http://5.6.7.8:8000", ["0001", "0002"])

        sources_0 = tracker.get_sources("0000")
        assert "http://1.2.3.4:8000" in sources_0
        assert len(sources_0) == 1

        sources_1 = tracker.get_sources("0001")
        assert len(sources_1) == 2

        sources_2 = tracker.get_sources("0002")
        assert "http://5.6.7.8:8000" in sources_2

    def test_expired_peers_excluded(self):
        """Peers not seen in 5 minutes should be excluded."""
        tracker = Tracker()
        tracker.register_seed("old_peer", "http://old:8000", ["0000"])

        # Manually expire
        import time
        tracker.seeds["old_peer"].last_seen = time.time() - 400

        sources = tracker.get_sources("0000")
        assert len(sources) == 0

    def test_get_all_sources(self):
        """get_all_sources should return complete mapping."""
        tracker = Tracker()
        tracker.register_seed("p1", "http://a:8000", ["0000", "0001"])
        tracker.register_seed("p2", "http://b:8000", ["0001"])

        all_sources = tracker.get_all_sources()
        assert "0000" in all_sources
        assert "0001" in all_sources
        assert len(all_sources["0001"]) == 2


class TestDownloadChunks:
    def test_hub_only_fallback(self, tmp_model, tmp_path):
        """download_chunks should work with hub-only (no peer sources)."""
        model_path, _ = tmp_model
        chunks_dir = str(tmp_path / "src_chunks")
        manifest = create_chunks(model_path, chunks_dir)

        # Start a simple HTTP server to serve chunks
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import threading

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.headers.get("X-GDF-Token") != "test-token":
                    self.send_error(403)
                    return
                if self.path.startswith("/chunk/"):
                    idx = self.path.split("/chunk/")[1]
                    chunk_file = Path(chunks_dir) / f"chunk_{idx}.pt"
                    if chunk_file.exists():
                        data = chunk_file.read_bytes()
                        self.send_response(200)
                        self.send_header("Content-Type", "application/octet-stream")
                        self.end_headers()
                        self.wfile.write(data)
                    else:
                        self.send_error(404)
                else:
                    self.send_error(404)

        server = HTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        try:
            dl_dir = str(tmp_path / "downloaded")
            downloaded = download_chunks(
                manifest, dl_dir,
                hub_url=f"http://127.0.0.1:{port}",
                token="test-token",
            )
            assert len(downloaded) == len(manifest.chunks)

            # Verify reassembly works
            output = str(tmp_path / "reassembled.pt")
            reassemble_model(manifest, dl_dir, output)
        finally:
            server.shutdown()

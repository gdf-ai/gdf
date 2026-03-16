"""Tests for the model registry."""

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from gdf.registry import (
    ModelEntry, fetch_registry, get_model,
    CACHE_PATH, CACHE_TTL,
)


class TestModelEntry:
    def test_from_dict(self):
        """ModelEntry should parse from dict."""
        d = {
            "name": "test-model",
            "description": "A test model",
            "hub_url": "http://localhost:7677",
            "token": "abc123",
            "size": "1GB",
            "status": "active",
        }
        entry = ModelEntry.from_dict(d)
        assert entry.name == "test-model"
        assert entry.description == "A test model"
        assert entry.hub_url == "http://localhost:7677"
        assert entry.token == "abc123"
        assert entry.size == "1GB"
        assert entry.status == "active"

    def test_from_dict_defaults(self):
        """Missing optional fields should get defaults."""
        d = {
            "name": "minimal",
            "hub_url": "http://localhost:7677",
            "token": "xyz",
        }
        entry = ModelEntry.from_dict(d)
        assert entry.description == ""
        assert entry.size == "?"
        assert entry.status == "active"


class TestFetchRegistry:
    def test_parse_registry_json(self, tmp_path):
        """Registry JSON should parse into ModelEntry list."""
        models = [
            {
                "name": "general-7b",
                "description": "General-purpose 7B",
                "hub_url": "http://hub1:7677",
                "token": "pub-token",
                "size": "14GB",
                "status": "active",
            },
            {
                "name": "code-3b",
                "description": "Code model",
                "hub_url": "http://hub2:7677",
                "token": "pub-token-2",
                "size": "6GB",
                "status": "paused",
            },
        ]

        cache_data = {"fetched_at": time.time(), "models": models}
        cache_file = tmp_path / "registry.json"
        cache_file.write_text(json.dumps(cache_data))

        with patch("gdf.registry.CACHE_PATH", cache_file):
            entries = fetch_registry()

        assert len(entries) == 2
        assert entries[0].name == "general-7b"
        assert entries[1].status == "paused"

    def test_get_model_lookup(self, tmp_path):
        """get_model should find by name."""
        models = [
            {"name": "alpha", "hub_url": "http://a:7677", "token": "t1"},
            {"name": "beta", "hub_url": "http://b:7677", "token": "t2"},
        ]
        cache_data = {"fetched_at": time.time(), "models": models}
        cache_file = tmp_path / "registry.json"
        cache_file.write_text(json.dumps(cache_data))

        with patch("gdf.registry.CACHE_PATH", cache_file):
            found = get_model("beta")
            assert found is not None
            assert found.hub_url == "http://b:7677"

            missing = get_model("gamma")
            assert missing is None

    def test_cache_fallback_on_network_error(self, tmp_path):
        """Should fall back to expired cache when network fails."""
        models = [{"name": "cached", "hub_url": "http://c:7677", "token": "t"}]
        cache_data = {"fetched_at": time.time() - CACHE_TTL - 100, "models": models}
        cache_file = tmp_path / "registry.json"
        cache_file.write_text(json.dumps(cache_data))

        def fail_fetch(*args, **kwargs):
            raise ConnectionError("no network")

        with patch("gdf.registry.CACHE_PATH", cache_file):
            with patch("gdf.registry.LOCAL_REGISTRY", tmp_path / "nonexistent.json"):
                with patch("urllib.request.urlopen", side_effect=fail_fetch):
                    entries = fetch_registry()

        assert len(entries) == 1
        assert entries[0].name == "cached"

"""Tests for the model registry."""

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from gdf.registry import (
    ModelInfo, ModelRegistry, fetch_registry, get_model,
    CACHE_PATH, CACHE_TTL, LOCAL_MODELS_PATH,
    _apply_env_overrides,
)


class TestModelInfo:
    def test_from_dict(self):
        """ModelInfo should parse from dict."""
        d = {
            "name": "test-model",
            "description": "A test model",
            "hub_url": "http://localhost:7677",
            "token": "abc123",
            "size": "1GB",
            "status": "active",
        }
        entry = ModelInfo.from_dict(d)
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
        entry = ModelInfo.from_dict(d)
        assert entry.description == ""
        assert entry.size == "?"
        assert entry.status == "active"


class TestFetchRegistry:
    def test_parse_registry_json(self, tmp_path):
        """Registry JSON should parse into ModelInfo list."""
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
        cache_file = tmp_path / "registry_cache.json"
        cache_file.write_text(json.dumps(cache_data))

        local_models = tmp_path / "models.json"

        with patch("gdf.registry.CACHE_PATH", cache_file), \
             patch("gdf.registry.LOCAL_MODELS_PATH", local_models):
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
        cache_file = tmp_path / "registry_cache.json"
        cache_file.write_text(json.dumps(cache_data))

        local_models = tmp_path / "models.json"

        with patch("gdf.registry.CACHE_PATH", cache_file), \
             patch("gdf.registry.LOCAL_MODELS_PATH", local_models):
            found = get_model("beta")
            assert found is not None
            assert found.hub_url == "http://b:7677"

            missing = get_model("gamma")
            assert missing is None

    def test_cache_fallback_on_network_error(self, tmp_path):
        """Should fall back to expired cache when network fails."""
        models = [{"name": "cached", "hub_url": "http://c:7677", "token": "t"}]
        cache_data = {"fetched_at": time.time() - CACHE_TTL - 100, "models": models}
        cache_file = tmp_path / "registry_cache.json"
        cache_file.write_text(json.dumps(cache_data))

        local_models = tmp_path / "models.json"

        def fail_fetch(*args, **kwargs):
            raise ConnectionError("no network")

        with patch("gdf.registry.CACHE_PATH", cache_file), \
             patch("gdf.registry.LOCAL_REGISTRY", tmp_path / "nonexistent.json"), \
             patch("gdf.registry.LOCAL_MODELS_PATH", local_models), \
             patch("urllib.request.urlopen", side_effect=fail_fetch):
            entries = fetch_registry()

        assert len(entries) == 1
        assert entries[0].name == "cached"


class TestEnvOverrides:
    def test_global_env_overrides(self):
        """GDF_HUB_URL and GDF_HUB_TOKEN override all models."""
        models = [
            ModelInfo(name="general", hub_url="http://localhost:7677", token="local-dev"),
        ]
        env = {"GDF_HUB_URL": "https://hub.gdf.ai", "GDF_HUB_TOKEN": "prod-token"}
        with patch.dict("os.environ", env, clear=False):
            result = _apply_env_overrides(models)
        assert result[0].hub_url == "https://hub.gdf.ai"
        assert result[0].token == "prod-token"

    def test_model_specific_env_overrides(self):
        """Model-specific env vars take precedence over global."""
        models = [
            ModelInfo(name="general", hub_url="http://localhost:7677", token="local-dev"),
        ]
        env = {
            "GDF_HUB_URL": "https://default.gdf.ai",
            "GDF_HUB_TOKEN": "default-token",
            "GDF_GENERAL_HUB_URL": "https://general.gdf.ai",
            "GDF_GENERAL_HUB_TOKEN": "general-token",
        }
        with patch.dict("os.environ", env, clear=False):
            result = _apply_env_overrides(models)
        assert result[0].hub_url == "https://general.gdf.ai"
        assert result[0].token == "general-token"

    def test_no_env_keeps_defaults(self):
        """Without env vars, original values are preserved."""
        models = [
            ModelInfo(name="general", hub_url="http://localhost:7677", token="local-dev"),
        ]
        # Clear any GDF env vars that might exist
        env_clear = {k: "" for k in ["GDF_HUB_URL", "GDF_HUB_TOKEN",
                                      "GDF_GENERAL_HUB_URL", "GDF_GENERAL_HUB_TOKEN"]}
        with patch.dict("os.environ", {}, clear=False):
            # Remove GDF vars if they exist
            import os
            saved = {}
            for k in env_clear:
                if k in os.environ:
                    saved[k] = os.environ.pop(k)
            try:
                result = _apply_env_overrides(models)
            finally:
                os.environ.update(saved)
        assert result[0].hub_url == "http://localhost:7677"
        assert result[0].token == "local-dev"


class TestModelRegistry:
    def test_register_and_get(self, tmp_path):
        """Should register and retrieve a model."""
        registry = ModelRegistry(local_path=tmp_path / "models.json")
        info = ModelInfo(name="test", domain="code", keywords=["python"])
        registry.register(info)

        got = registry.get("test")
        assert got is not None
        assert got.domain == "code"

    def test_unregister(self, tmp_path):
        """Should remove a model."""
        registry = ModelRegistry(local_path=tmp_path / "models.json")
        info = ModelInfo(name="test", domain="code")
        registry.register(info)
        assert registry.unregister("test") is True
        assert registry.get("test") is None

    def test_list_routable(self, tmp_path):
        """Only models with keywords and model_path are routable."""
        registry = ModelRegistry(local_path=tmp_path / "models.json")
        registry.register(ModelInfo(name="a", keywords=["x"], model_path="/a.pt"))
        registry.register(ModelInfo(name="b", keywords=[], model_path="/b.pt"))
        registry.register(ModelInfo(name="c", keywords=["y"]))

        routable = registry.list_routable()
        assert len(routable) == 1
        assert routable[0].name == "a"

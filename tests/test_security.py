"""Tests for delta validation (rejecting poisoned contributions)."""

import torch
import pytest

from gdf.distributed import Hub, compute_delta
from gdf.model import TinyTransformer, ModelConfig
from gdf.serialization import save_model
from gdf.trainer import OnlineTrainer


@pytest.fixture
def hub_with_model(tmp_path):
    """Create a Hub with a small model."""
    config = ModelConfig(d_model=64, n_heads=2, n_layers=2, d_ff=128)
    model = TinyTransformer(config)
    trainer = OnlineTrainer(model, device=torch.device("cpu"))
    model_path = str(tmp_path / "hub_model.pt")
    save_model(model_path, model, trainer)
    return Hub(model_path=model_path, port=0), model


class TestDeltaValidation:
    def test_nan_rejected(self, hub_with_model):
        """Delta containing NaN should be rejected."""
        hub, model = hub_with_model
        delta = {"some_key": torch.tensor([1.0, float("nan"), 3.0])}
        error = hub._validate_delta(delta)
        assert error is not None
        assert "NaN" in error or "Inf" in error

    def test_inf_rejected(self, hub_with_model):
        """Delta containing Inf should be rejected."""
        hub, model = hub_with_model
        delta = {"some_key": torch.tensor([1.0, float("inf"), 3.0])}
        error = hub._validate_delta(delta)
        assert error is not None
        assert "Inf" in error or "NaN" in error

    def test_negative_inf_rejected(self, hub_with_model):
        """Delta containing -Inf should be rejected."""
        hub, model = hub_with_model
        delta = {"some_key": torch.tensor([1.0, float("-inf"), 3.0])}
        error = hub._validate_delta(delta)
        assert error is not None

    def test_large_delta_rejected(self, hub_with_model):
        """Delta with values exceeding max norm should be rejected."""
        hub, model = hub_with_model
        delta = {"some_key": torch.tensor([0.1, 0.2, 50.0])}  # 50 > default 10
        error = hub._validate_delta(delta)
        assert error is not None
        assert "too large" in error

    def test_normal_delta_accepted(self, hub_with_model):
        """Small, normal delta should be accepted."""
        hub, model = hub_with_model
        delta = {"some_key": torch.tensor([0.001, -0.002, 0.005])}
        error = hub._validate_delta(delta)
        assert error is None

    def test_zero_delta_accepted(self, hub_with_model):
        """Zero delta should be accepted."""
        hub, model = hub_with_model
        delta = {"some_key": torch.zeros(10)}
        error = hub._validate_delta(delta)
        assert error is None

    def test_boundary_delta_accepted(self, hub_with_model):
        """Delta exactly at max norm should be accepted."""
        hub, model = hub_with_model
        delta = {"some_key": torch.tensor([10.0])}  # exactly at limit
        error = hub._validate_delta(delta)
        assert error is None

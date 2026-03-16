"""Tests for model merging."""

import torch
from gdf.model import TinyTransformer, ModelConfig
from gdf.merging import fedavg, task_arithmetic, ties, merge_models
from gdf.trainer import OnlineTrainer


def _make_trained_model(text: str, steps: int = 5) -> TinyTransformer:
    """Create and train a model on some text."""
    model = TinyTransformer()
    trainer = OnlineTrainer(model)
    for _ in range(steps):
        trainer.train_step(text, feedback="good")
    model.cpu()  # ensure on CPU for merging tests
    return model


def test_fedavg_two_models():
    m1 = _make_trained_model("hello world")
    m2 = _make_trained_model("goodbye world")
    merged = fedavg([m1, m2])
    # Merged params should be between the two
    for key in m1.state_dict():
        p1 = m1.state_dict()[key].cpu().float()
        p2 = m2.state_dict()[key].cpu().float()
        pm = merged.state_dict()[key].cpu().float()
        # Check merged is roughly the average
        expected = (p1 + p2) / 2
        assert torch.allclose(pm, expected, atol=1e-5)


def test_fedavg_single_model():
    m1 = _make_trained_model("test")
    merged = fedavg([m1])
    for key in m1.state_dict():
        assert torch.allclose(m1.state_dict()[key].cpu(), merged.state_dict()[key].cpu())


def test_task_arithmetic():
    base = TinyTransformer()
    m1 = _make_trained_model("hello world")
    m2 = _make_trained_model("goodbye world")
    merged = task_arithmetic(base, [m1, m2], scaling=0.5)
    # Merged model should exist and have valid params
    x = torch.tensor([[72, 101, 108]], dtype=torch.long)  # "Hel"
    logits = merged(x)
    assert logits.shape == (1, 3, 256)
    assert not torch.isnan(logits).any()


def test_ties():
    base = TinyTransformer()
    m1 = _make_trained_model("hello world")
    m2 = _make_trained_model("goodbye world")
    merged = ties(base, [m1, m2], density=0.2)
    x = torch.tensor([[72, 101, 108]], dtype=torch.long)
    logits = merged(x)
    assert logits.shape == (1, 3, 256)
    assert not torch.isnan(logits).any()


def test_merge_models_fedavg():
    m1 = _make_trained_model("aaa")
    m2 = _make_trained_model("bbb")
    merged, hash_val = merge_models([m1, m2], strategy="fedavg")
    assert hash_val is not None
    assert len(hash_val) == 16


def test_merge_models_ties_no_base_falls_back():
    m1 = _make_trained_model("aaa")
    m2 = _make_trained_model("bbb")
    # ties without base should fall back to fedavg
    merged, hash_val = merge_models([m1, m2], strategy="ties", base=None)
    assert hash_val is not None

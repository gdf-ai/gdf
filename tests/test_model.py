"""Tests for TinyTransformer model."""

import torch
from gdf.model import TinyTransformer, ModelConfig
from gdf.tokenizer import encode, VOCAB_SIZE


def test_config_defaults():
    config = ModelConfig()
    assert config.vocab_size == VOCAB_SIZE
    assert config.d_model == 128
    assert config.n_heads == 4
    assert config.n_layers == 2


def test_config_roundtrip():
    config = ModelConfig(d_model=64, n_heads=2)
    d = config.to_dict()
    config2 = ModelConfig.from_dict(d)
    assert config == config2


def test_model_forward():
    model = TinyTransformer()
    tokens = encode("hello world")
    x = torch.tensor([tokens], dtype=torch.long)
    logits = model(x)
    assert logits.shape == (1, len(tokens), VOCAB_SIZE)


def test_model_backward():
    model = TinyTransformer()
    tokens = encode("test")
    x = torch.tensor([tokens[:-1]], dtype=torch.long)
    y = torch.tensor([tokens[1:]], dtype=torch.long)
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
    loss.backward()
    # Check gradients exist
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None


def test_parameter_count():
    model = TinyTransformer()
    params = model.count_parameters()
    # Should be around 396K
    assert 300_000 < params < 500_000

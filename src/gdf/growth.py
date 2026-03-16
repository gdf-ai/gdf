"""Model growth: expand a trained model to a larger architecture.

When the distributed network decides it's time to scale up, this module
lets you take a trained small model and "grow" it into a larger one,
preserving everything it already learned.

Uses the Net2Net approach:
- Net2Wider: expand layer widths (d_model, d_ff) by copying neurons
- Net2Deeper: add new layers initialized as identity functions

This means growth is lossless — the larger model produces the exact
same outputs as the small one immediately after growth, then can be
trained further to use the extra capacity.
"""

from __future__ import annotations

import copy
import math
from collections import OrderedDict

import torch
import torch.nn as nn

from .model import TinyTransformer, ModelConfig


def grow_wider(
    model: TinyTransformer,
    new_d_model: int | None = None,
    new_d_ff: int | None = None,
    new_n_heads: int | None = None,
) -> TinyTransformer:
    """Grow a model wider: increase d_model, d_ff, and/or n_heads.

    The new model will produce identical outputs to the old one initially.
    Extra capacity is initialized by copying existing neurons with noise.

    Args:
        model: Source model.
        new_d_model: New embedding/attention dimension (must be >= current).
        new_d_ff: New feed-forward dimension (must be >= current).
        new_n_heads: New number of attention heads (must divide new_d_model).

    Returns:
        New, wider TinyTransformer.
    """
    old_cfg = model.config
    d_model = new_d_model or old_cfg.d_model
    d_ff = new_d_ff or old_cfg.d_ff
    n_heads = new_n_heads or old_cfg.n_heads

    if d_model < old_cfg.d_model:
        raise ValueError(f"Can't shrink d_model from {old_cfg.d_model} to {d_model}")
    if d_ff < old_cfg.d_ff:
        raise ValueError(f"Can't shrink d_ff from {old_cfg.d_ff} to {d_ff}")
    if d_model % n_heads != 0:
        raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")

    new_cfg = ModelConfig(
        vocab_size=old_cfg.vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=old_cfg.n_layers,
        d_ff=d_ff,
        max_seq_len=old_cfg.max_seq_len,
        dropout=old_cfg.dropout,
    )
    new_model = TinyTransformer(new_cfg)

    old_sd = model.state_dict()
    new_sd = new_model.state_dict()

    # For each parameter, copy old values and fill extra with small noise
    grown_sd = OrderedDict()
    for key in new_sd:
        if key in old_sd:
            grown_sd[key] = _grow_tensor(old_sd[key], new_sd[key].shape)
        else:
            grown_sd[key] = new_sd[key]

    new_model.load_state_dict(grown_sd)
    return new_model


def grow_deeper(
    model: TinyTransformer,
    extra_layers: int = 1,
) -> TinyTransformer:
    """Grow a model deeper: add new transformer layers.

    New layers are initialized as approximate identity functions
    (zero output projections) so the model behaves the same initially.

    Args:
        model: Source model.
        extra_layers: Number of layers to add.

    Returns:
        New, deeper TinyTransformer.
    """
    old_cfg = model.config
    new_cfg = ModelConfig(
        vocab_size=old_cfg.vocab_size,
        d_model=old_cfg.d_model,
        n_heads=old_cfg.n_heads,
        n_layers=old_cfg.n_layers + extra_layers,
        d_ff=old_cfg.d_ff,
        max_seq_len=old_cfg.max_seq_len,
        dropout=old_cfg.dropout,
    )
    new_model = TinyTransformer(new_cfg)
    new_sd = new_model.state_dict()
    old_sd = model.state_dict()

    grown_sd = OrderedDict()

    for key in new_sd:
        # Parse block index from key like "blocks.0.attn.qkv.weight"
        parts = key.split(".")
        if parts[0] == "blocks" and len(parts) > 1:
            block_idx = int(parts[1])
            if block_idx < old_cfg.n_layers:
                # Copy from old model
                grown_sd[key] = old_sd[key].clone()
            else:
                # New layer — initialize to near-identity
                grown_sd[key] = _identity_init(key, new_sd[key])
        elif key in old_sd:
            grown_sd[key] = old_sd[key].clone()
        else:
            grown_sd[key] = new_sd[key]

    new_model.load_state_dict(grown_sd)
    return new_model


def grow_model(
    model: TinyTransformer,
    target_config: ModelConfig,
) -> TinyTransformer:
    """Grow a model to match a target configuration.

    Applies wider + deeper growth as needed.
    """
    result = model

    # First grow wider if needed
    old_cfg = result.config
    if (target_config.d_model > old_cfg.d_model or
            target_config.d_ff > old_cfg.d_ff or
            target_config.n_heads > old_cfg.n_heads):
        result = grow_wider(
            result,
            new_d_model=target_config.d_model,
            new_d_ff=target_config.d_ff,
            new_n_heads=target_config.n_heads,
        )

    # Then grow deeper if needed
    if target_config.n_layers > result.config.n_layers:
        extra = target_config.n_layers - result.config.n_layers
        result = grow_deeper(result, extra_layers=extra)

    return result


# ── Presets for progressive growth ──────────────────────────────────────────

# Suggested growth stages — each can be merged with others at the same stage
GROWTH_STAGES = {
    "micro":  ModelConfig(d_model=128,  n_heads=4,  n_layers=2,  d_ff=256),
    "tiny":   ModelConfig(d_model=256,  n_heads=4,  n_layers=4,  d_ff=512),
    "small":  ModelConfig(d_model=384,  n_heads=6,  n_layers=6,  d_ff=1024),
    "medium": ModelConfig(d_model=512,  n_heads=8,  n_layers=8,  d_ff=2048),
    "large":  ModelConfig(d_model=768,  n_heads=12, n_layers=12, d_ff=3072),
}

def suggest_next_stage(config: ModelConfig) -> str | None:
    """Suggest the next growth stage for a model."""
    stages = list(GROWTH_STAGES.items())
    for i, (name, cfg) in enumerate(stages):
        if (config.d_model <= cfg.d_model and
                config.n_layers <= cfg.n_layers):
            # Current model fits in this stage or smaller
            if i + 1 < len(stages):
                return stages[i + 1][0]
            return None
    return None


# ── Internal helpers ────────────────────────────────────────────────────────

def _grow_tensor(old: torch.Tensor, new_shape: tuple) -> torch.Tensor:
    """Grow a tensor to a new shape, copying old values and filling extra with noise."""
    if old.shape == new_shape:
        return old.clone()

    new = torch.zeros(new_shape, dtype=old.dtype)
    noise_std = old.std().item() * 0.01  # tiny noise to break symmetry

    # Fill with small noise first
    new.normal_(0, noise_std)

    # Copy old values into the new tensor
    slices = tuple(slice(0, min(o, n)) for o, n in zip(old.shape, new_shape))
    new[slices] = old[slices]

    return new


def _identity_init(key: str, tensor: torch.Tensor) -> torch.Tensor:
    """Initialize a parameter for a new layer to approximate identity.

    The key insight: if the output projection of attention and FFN are zero,
    the residual connection x + attn(x) ≈ x + 0 = x, making the layer a no-op.
    """
    # Zero out projection weights so residual connections pass through
    if "proj.weight" in key or "ff.2.weight" in key:
        return torch.zeros_like(tensor)
    if "proj.bias" in key or "ff.2.bias" in key:
        return torch.zeros_like(tensor)

    # LayerNorm: initialize to identity (weight=1, bias=0)
    if "ln" in key and "weight" in key:
        return torch.ones_like(tensor)
    if "ln" in key and "bias" in key:
        return torch.zeros_like(tensor)

    # Everything else: small random init
    t = torch.empty_like(tensor)
    nn.init.normal_(t, mean=0.0, std=0.02)
    return t

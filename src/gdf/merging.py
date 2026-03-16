"""Model merging strategies: FedAvg, Task Arithmetic, TIES."""

from __future__ import annotations

import copy
from collections import OrderedDict

import torch

from .model import TinyTransformer, ModelConfig
from .serialization import _compute_hash


def fedavg(
    models: list[TinyTransformer],
    weights: list[float] | None = None,
) -> TinyTransformer:
    """Federated averaging: weighted average of model parameters.

    Works without a shared base model.
    """
    if not models:
        raise ValueError("Need at least one model to merge")

    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    else:
        total = sum(weights)
        weights = [w / total for w in weights]

    config = copy.deepcopy(models[0].config)
    merged = TinyTransformer(config)
    merged_sd = OrderedDict()

    ref_sd = models[0].state_dict()
    for key in ref_sd:
        merged_sd[key] = torch.zeros_like(ref_sd[key], dtype=torch.float32, device="cpu")
        for model, w in zip(models, weights):
            merged_sd[key] += w * model.state_dict()[key].cpu().float()
        merged_sd[key] = merged_sd[key].to(ref_sd[key].dtype)

    merged.load_state_dict(merged_sd)
    return merged


def task_arithmetic(
    base: TinyTransformer,
    models: list[TinyTransformer],
    scaling: float = 1.0,
) -> TinyTransformer:
    """Task arithmetic: sum deltas from a shared base, then apply to base.

    merged = base + scaling * sum(model_i - base)
    """
    if not models:
        raise ValueError("Need at least one model to merge")

    config = copy.deepcopy(base.config)
    merged = TinyTransformer(config)
    base_sd = base.state_dict()
    merged_sd = OrderedDict()

    for key in base_sd:
        base_val = base_sd[key].cpu().float()
        delta_sum = torch.zeros_like(base_val)
        for model in models:
            delta_sum += model.state_dict()[key].cpu().float() - base_val
        merged_sd[key] = (base_val + scaling * delta_sum).to(base_sd[key].dtype)

    merged.load_state_dict(merged_sd)
    return merged


def ties(
    base: TinyTransformer,
    models: list[TinyTransformer],
    density: float = 0.2,
    scaling: float = 1.0,
) -> TinyTransformer:
    """TIES merging: Trim, Elect sign, merge agreeing values.

    Steps per parameter tensor:
    1. Compute deltas from base
    2. Trim: zero out the smallest (1-density) fraction of each delta
    3. Elect sign: majority vote on sign per element
    4. Disjoint merge: average values that agree with elected sign, zero others
    5. Apply merged delta to base
    """
    if not models:
        raise ValueError("Need at least one model to merge")

    config = copy.deepcopy(base.config)
    merged = TinyTransformer(config)
    base_sd = base.state_dict()
    merged_sd = OrderedDict()

    for key in base_sd:
        base_param = base_sd[key].cpu().float()
        deltas = []
        for model in models:
            delta = model.state_dict()[key].cpu().float() - base_param
            deltas.append(delta)

        # Step 1: Trim — zero out smallest values in each delta
        trimmed = []
        for delta in deltas:
            flat = delta.abs().flatten()
            if flat.numel() == 0:
                trimmed.append(delta)
                continue
            k = max(1, int(density * flat.numel()))
            threshold = torch.topk(flat, k).values[-1]
            mask = delta.abs() >= threshold
            trimmed.append(delta * mask.float())

        # Step 2: Elect sign — majority vote
        stacked = torch.stack(trimmed)  # (N, *shape)
        sign_votes = torch.sign(stacked)
        # Sum of signs: positive sum → positive, negative sum → negative
        elected_sign = torch.sign(sign_votes.sum(dim=0))

        # Step 3: Disjoint merge — average values that agree with elected sign
        merged_delta = torch.zeros_like(base_param)
        count = torch.zeros_like(base_param)
        for t in trimmed:
            agree = (torch.sign(t) == elected_sign) & (t != 0)
            merged_delta += t * agree.float()
            count += agree.float()
        count = count.clamp(min=1)
        merged_delta = merged_delta / count

        merged_sd[key] = (base_param + scaling * merged_delta).to(base_sd[key].dtype)

    merged.load_state_dict(merged_sd)
    return merged


def merge_models(
    models: list[TinyTransformer],
    base: TinyTransformer | None = None,
    strategy: str = "ties",
    base_model_hash: str | None = None,
    **kwargs,
) -> tuple[TinyTransformer, str]:
    """High-level merge function.

    Returns (merged_model, new_base_model_hash).
    """
    if strategy == "fedavg":
        merged = fedavg(models, **kwargs)
    elif strategy == "task_arithmetic":
        if base is None:
            raise ValueError("task_arithmetic requires a base model")
        merged = task_arithmetic(base, models, **kwargs)
    elif strategy == "ties":
        if base is None:
            # Fall back to fedavg if no shared base
            merged = fedavg(models)
        else:
            merged = ties(base, models, **kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    new_hash = _compute_hash(merged.state_dict())
    return merged, new_hash

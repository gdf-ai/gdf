"""Device detection and management — CPU, CUDA, MPS (Apple Silicon)."""

from __future__ import annotations

import torch


def detect_device() -> torch.device:
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_info() -> dict:
    """Get detailed info about available compute devices."""
    info: dict = {
        "device": str(detect_device()),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
    }

    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_devices"] = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["cuda_devices"].append({
                "name": props.name,
                "vram_gb": round(props.total_memory / 1e9, 1),
                "compute_capability": f"{props.major}.{props.minor}",
            })
        info["bf16_supported"] = torch.cuda.is_bf16_supported()
    else:
        info["cuda_device_count"] = 0
        info["cuda_devices"] = []
        info["bf16_supported"] = False

    # Estimate max model size based on available VRAM
    if info["cuda_devices"]:
        vram = info["cuda_devices"][0]["vram_gb"]
        # Rule of thumb: fp16 model uses ~2 bytes/param, training needs ~4x for gradients+optimizer
        # So training budget ≈ VRAM / 8 bytes per param
        info["est_max_params_training"] = int(vram * 1e9 / 8)
        info["est_max_params_inference"] = int(vram * 1e9 / 2)
    else:
        info["est_max_params_training"] = 0
        info["est_max_params_inference"] = 0

    return info


def format_device_info(info: dict) -> str:
    """Format device info for display."""
    lines = [f"Device: {info['device']}"]

    if info["cuda_devices"]:
        for i, gpu in enumerate(info["cuda_devices"]):
            lines.append(f"  GPU {i}: {gpu['name']} ({gpu['vram_gb']} GB VRAM)")
        max_train = info["est_max_params_training"]
        max_infer = info["est_max_params_inference"]
        lines.append(f"  Max model (training): ~{max_train / 1e6:.0f}M params")
        lines.append(f"  Max model (inference): ~{max_infer / 1e6:.0f}M params")
        lines.append(f"  BF16 supported: {info['bf16_supported']}")
    elif info["mps_available"]:
        lines.append("  Apple Silicon GPU (MPS)")
    else:
        lines.append("  CPU only — training will be slow")
        lines.append("  Consider using a machine with a GPU for faster learning")

    return "\n".join(lines)

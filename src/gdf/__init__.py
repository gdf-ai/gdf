"""gdf — Distributed Federated LLM Training."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("gdf")
except PackageNotFoundError:
    __version__ = "0.1.0"

from .model import TinyTransformer, ModelConfig
from .tokenizer import encode, decode, VOCAB_SIZE
from .trainer import OnlineTrainer, TrainerConfig
from .serialization import save_model, load_model, get_model_info
from .merging import fedavg, task_arithmetic, ties, merge_models
from .api import GDFModel
from .config import get_default_model, set_default_model, resolve_model
from .fetcher import fetch_url, is_url
from .crawler import discover_files, crawl_and_train
from .registry import ModelInfo, ModelRegistry, Router

__all__ = [
    "__version__",
    "TinyTransformer",
    "ModelConfig",
    "encode",
    "decode",
    "VOCAB_SIZE",
    "OnlineTrainer",
    "TrainerConfig",
    "save_model",
    "load_model",
    "get_model_info",
    "fedavg",
    "task_arithmetic",
    "ties",
    "merge_models",
    "GDFModel",
    "get_default_model",
    "set_default_model",
    "resolve_model",
    "fetch_url",
    "is_url",
    "discover_files",
    "crawl_and_train",
    "ModelInfo",
    "ModelRegistry",
    "Router",
]

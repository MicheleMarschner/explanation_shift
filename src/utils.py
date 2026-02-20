import os
import random
import numpy as np
import torch
from pathlib import Path
import pandas as pd

from src.configs.global_config import Paths, PATHS


def prefix_keys(d: dict, prefix: str) -> dict:
    return {f"{prefix}{k}": v for k, v in d.items()}


def cpu(x):
    return x.detach().cpu() if torch.is_tensor(x) else x


def set_seeds(seed: int = 51, deterministic: bool = True) -> None:
    """Sets seeds for complete reproducibility across all libraries and operations"""
    # Python hashing (affects iteration order in some cases)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        # CUDA matmul determinism (PyTorch recommends setting this env var)
        # Only needed for some CUDA versions/ops; harmless otherwise.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # set early

    random.seed(seed)           # Python random module
    np.random.seed(seed)        # NumPy
    torch.manual_seed(seed)     # PyTorch CPU
    # PyTorch GPU (all devices)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)    # For multi-GPU setups
    
    # CUDA deterministic operations
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic

    if deterministic:
        # Force deterministic algorithms when available
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as e:
            print(f"Deterministic algorithms not fully enforced: {e}")

    print(f"Seeds set to {seed} (deterministic={deterministic})")


def create_file_path(run_dir: Path, stage_dir: str, prefix: str, corruption: str | None = None, severity: int | None = None) -> Path:
    d = run_dir / stage_dir
    ensure_dir(d)

    if corruption is None or severity is None:
        return d / f"{prefix}.pt"

    return d / f"{prefix}__{corruption}__sev{severity}.pt"


def ensure_dir(p: Path) -> None:
    """Create directory and parent directories if it doesn't already exist"""
    p.mkdir(parents=True, exist_ok=True)


def ensure_dirs(paths: Paths) -> None:
    """Create all directories that should exist"""
    for p in [paths.data_clean, paths.data_corr, paths.results, paths.runs, paths.checkpoints]:
        ensure_dir(p)


def to_np_idx(idx):
    return idx.cpu().numpy() if torch.is_tensor(idx) else idx


@torch.no_grad()
def collect_x_from_loader(dataloader):
    xs = []
    for xb, _ in dataloader:
        xs.append(xb.cpu())
    return torch.cat(xs, dim=0)  # [N,3,32,32] CPU


def collect_labels_from_loader(dataloader):
    ys = []
    for _, yb in dataloader:
        ys.append(yb.cpu())
    return torch.cat(ys, dim=0)  # [N]


# aggregate metric values with an optional boolean mask (e.g., stable-prediction subset)
def mean_std_over_mask(values, mask=None):
    """
    Mean/std over values[mask] if mask is given, else over all values.
    Returns (nan, nan) if the selected subset is empty.
    """
    if mask is None:
        subset = values
    else:
        if mask.sum() == 0:
            return float("nan"), float("nan")
        subset = values[mask]

    if subset.numel() == 0:
        return float("nan"), float("nan")

    return subset.mean().item(), subset.std(unbiased=False).item()


def to_cpu_f16(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to("cpu").to(torch.float16).contiguous()


def py_scalar(v):
    """Convert numpy scalars to Python scalars; leave strings as-is."""
    if isinstance(v, (np.generic,)):
        return v.item()
    return v

def as_np_int64_1d(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy().reshape(-1).astype(np.int64)
    p = exp_dir / f"artifacts_{corruption}_sev{severity}.pt"
    art = torch.load(p, map_location="cpu")
    cc = art["corrupt_reference"]

    cc["sal_corr"] = cc["sal_corr"].float()
    cc["logits_corr"] = cc["logits_corr"].float()
    cc["E_corr"] = cc["E_corr"].float()
    cc["entropy_corr"] = cc["entropy_corr"].float()
    return art
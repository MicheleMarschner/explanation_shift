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


def save_results(results, path):
    """
    Append results to a CSV at `path`. If it doesn't exist, create it.

    Args:
        results: dict (one row) or list[dict] (many rows)
        path: str | Path to .csv
    """
    new_df = pd.DataFrame([results])

    if path.exists():
        old_df = pd.read_csv(path)

        # union of columns, keep old order first
        all_cols = list(old_df.columns)
        for c in new_df.columns:
            if c not in all_cols:
                all_cols.append(c)

        # align both frames to same columns
        old_df = old_df.reindex(columns=all_cols)
        new_df = new_df.reindex(columns=all_cols)

        out_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        out_df = new_df

    out_df.to_csv(path, index=False)
    

def save_experiment_artifacts(corruption, severity, conf_drift_vec, exp_drift_vec, pred_corr, stable, dir):
  
  artifact = {
    "pred_corr":  pred_corr.detach().cpu(),
    "stable":     stable.detach().cpu(),
    "conf": {k: v.detach().cpu() for k, v in conf_drift_vec.items()},
    "drift": {k: v.detach().cpu() for k, v in exp_drift_vec.items()},
  }

  path = PATHS.runs / dir

  torch.save(artifact, f"{path}/vectors_corr{corruption}_sev{severity}.pt")


def print_results(row):

    print(f"  Acc corr={row['acc_corr']:.3f}  StableRate={row['stable_rate']:.3f}")
    print(f"  ΔE cosine  all={row['cos_mean']:.3f}±{row['cos_sd']:.3f}  " f"stable={row['cos_mean_stable']:.3f}±{row['cos_sd_stable']:.3f}")
    print(f"  ΔE IoU@5%  all={row['iou_mean']:.3f}±{row['iou_sd']:.3f}  " f"stable={row['iou_mean_stable']:.3f}±{row['iou_sd_stable']:.3f}")
    print(f"  |Δp|      all={row['p_shift_mean']:.3f}±{row['p_shift_sd']:.3f}  "  f"stable={row['p_shift_mean_stable']:.3f}±{row['p_shift_sd_stable']:.3f}")
    print(f"  |Δmargin| all={row['margin_shift_mean']:.3f}±{row['margin_shift_sd']:.3f}  " f"stable={row['margin_shift_mean_stable']:.3f}±{row['margin_shift_sd_stable']:.3f}")


def to_cpu_f16(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to("cpu").to(torch.float16).contiguous()


def update_results_inplace(results: dict, path: Path, key_cols=("corruption", "severity")):
    """
    Update the row matching key_cols in an existing CSV by filling/overwriting columns
    from `results`. Keeps all existing columns (e.g., acc_corr) and adds new columns at end.
    If no matching row exists, appends as a new row.
    """
    new_df = pd.DataFrame([results])

    if path.exists():
        df = pd.read_csv(path)

        # add missing columns at the end
        for c in new_df.columns:
            if c not in df.columns:
                df[c] = np.nan

        # try to find a matching row
        can_match = all(k in df.columns for k in key_cols)
        if can_match:
            mask = np.ones(len(df), dtype=bool)
            for k in key_cols:
                mask &= (df[k].astype(str) == str(results[k]))

            if mask.sum() == 0:
                # append new row aligned to df columns
                df = pd.concat([df, new_df.reindex(columns=df.columns)], ignore_index=True)
            else:
                # update FIRST match
                idx = df.index[mask][0]
                for k, v in results.items():
                    df.at[idx, k] = v
        else:
            # cannot match -> append
            df = pd.concat([df, new_df.reindex(columns=df.columns)], ignore_index=True)
    else:
        df = new_df

    df.to_csv(path, index=False)


def load_clean_reference(exp_dir: Path) -> dict:
    ref = torch.load(exp_dir / "experiment_reference.pt", map_location="cpu")
    cr = ref["clean_reference"]

    cr["sal_clean"] = cr["sal_clean"].float()
    cr["logits_clean"] = cr["logits_clean"].float()
    cr["E_clean"] = cr["E_clean"].float()
    cr["entropy_clean"] = cr["entropy_clean"].float()
    return ref


def load_corr_artifact(exp_dir: Path, corruption: str, severity: int) -> dict:
    p = exp_dir / f"artifacts_{corruption}_sev{severity}.pt"
    art = torch.load(p, map_location="cpu")
    cc = art["corrupt_reference"]

    cc["sal_corr"] = cc["sal_corr"].float()
    cc["logits_corr"] = cc["logits_corr"].float()
    cc["E_corr"] = cc["E_corr"].float()
    cc["entropy_corr"] = cc["entropy_corr"].float()
    return art
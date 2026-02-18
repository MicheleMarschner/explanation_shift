from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import torch

from src.configs.global_config import PATHS, DEVICE
from src.utils import ensure_dir, collect_x_from_loader
from src.data import get_clean_data, get_corrupted_data
from src.experiment_stages.helper import _upsert_row_to_csv

# -----------------------
# helpers: formatting
# -----------------------

def _as_np_int64_1d(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy().reshape(-1).astype(np.int64)

def _to_float_tensor(x) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.float()
    return torch.as_tensor(x).float()

def _sal_to_abatch(sal: torch.Tensor) -> np.ndarray:
    """
    Quantus expects attribution maps a_batch.
    We'll provide (N, 1, H, W) float32.
    Accepts:
      - (N, C, H, W) or (N, H, W)
    """
    sal = _to_float_tensor(sal).detach().cpu()

    if sal.ndim == 4:         # (N,C,H,W)
        sal = sal.abs().sum(dim=1, keepdim=True)   # -> (N,1,H,W)
    elif sal.ndim == 3:       # (N,H,W)
        sal = sal.abs().unsqueeze(1)               # -> (N,1,H,W)
    else:
        raise ValueError(f"Unexpected saliency shape: {tuple(sal.shape)}")

    return sal.numpy().astype(np.float32)

def _x_to_xbatch(x_t: torch.Tensor) -> np.ndarray:
    """
    Provide x_batch as (N,C,H,W) float32 numpy.
    Note: x_t is already normalized (your transform includes Normalize).
    """
    if not torch.is_tensor(x_t):
        x_t = torch.as_tensor(x_t)
    return x_t.detach().cpu().numpy().astype(np.float32)

# -----------------------
# main stage function
# -----------------------

def run_quantus_metrics(
    clean_path: Path,
    artifact_path: Optional[Path],
    out_path: Path,
    exp_config: Any,
    model,
    transform,
    mode: Literal["clean", "corr"] = "corr",
) -> Dict[str, Any]:
    """
    Computes Quantus metrics for either:
      - mode='clean': uses clean data + sal_clean
      - mode='corr' : uses corrupted data + sal_corr (from artifact)

    Protocol:
      y_batch = pred_clean (fixed target)
      x_batch = domain inputs (clean or corrupted)
      a_batch = domain attribution maps (sal_clean or sal_corr)

    Saves:
      out_path (.pt): payload {row, metrics, meta}
      03__quantus_results.csv: upsert by (corruption,severity)
    """
    
    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    # Load stage00 reference
    ref = torch.load(clean_path, map_location="cpu", weights_only=False)
    cr = ref["clean_reference"]

    pred_clean = cr["pred_clean"].long()       # torch tensor [N]
    y_batch = _as_np_int64_1d(pred_clean)      # numpy [N]
    pair_idx = ref.get("pair_idx", None)
    if pair_idx is None:
        raise KeyError("clean reference missing top-level 'pair_idx'.")
    if torch.is_tensor(pair_idx):
        pair_idx = pair_idx.detach().cpu().numpy()
    pair_idx = np.asarray(pair_idx).reshape(-1)

    # Decide domain inputs + attributions
    if mode == "clean":
        corruption = "clean"
        severity = 0

        clean_loader, _, _ = get_clean_data(path=PATHS.data_clean, idx=pair_idx, transform=transform)
        X_t = collect_x_from_loader(clean_loader)     # torch [N,C,H,W]
        x_batch = _x_to_xbatch(X_t)

        a_batch = _sal_to_abatch(cr["sal_clean"])

    else:
        if artifact_path is None:
            raise ValueError("mode='corr' requires artifact_path.")
        art = torch.load(artifact_path, map_location="cpu", weights_only=False)
        cc = art["corrupt_reference"]

        corruption = str(art["corruption"])
        severity = int(art["severity"])

        corr_loader, _, _ = get_corrupted_data(
            idx=pair_idx,
            path=PATHS.data_corr,
            transform=transform,
            corruption=corruption,
            severity=severity,
        )
        X_t = collect_x_from_loader(corr_loader)
        x_batch = _x_to_xbatch(X_t)

        a_batch = _sal_to_abatch(cc["sal_corr"])

    # Sanity
    if x_batch.shape[0] != y_batch.shape[0] or a_batch.shape[0] != y_batch.shape[0]:
        raise ValueError(
            f"Batch size mismatch: x={x_batch.shape[0]}, a={a_batch.shape[0]}, y={y_batch.shape[0]}"
        )

    # -----------------------
    # HERE: Quantus metrics
    # -----------------------
    # Keep it modular: compute a dict of scalar results and optionally vector scores.
    # Example placeholders (replace with real Quantus calls):
    quantus_scalars: Dict[str, float] = {}
    quantus_vectors: Dict[str, Any] = {}

    # TODO: insert Quantus metric calls.
    # quantus_scalars["deletion_auc"] = float(...)
    # quantus_scalars["avg_sensitivity"] = float(...)
    # quantus_vectors["deletion_scores"] = <np array or torch> (optional)

    row = {
        "corruption": corruption,
        "severity": severity,
        **quantus_scalars,
    }

    payload = {
        "row": row,
        "vectors": quantus_vectors,
        "meta": {
            "mode": mode,
            "y_batch_policy": "pred_clean",
            "x_domain": "clean" if mode == "clean" else "corrupted",
            "a_domain": "clean" if mode == "clean" else "corrupted",
            "x_shape": tuple(x_batch.shape),
            "a_shape": tuple(a_batch.shape),
            "y_shape": tuple(y_batch.shape),
        },
    }

    torch.save(payload, out_path)

    # CSV
    csv_path = out_path.parent / "03__quantus_results.csv"
    _upsert_row_to_csv(csv_path, row, keys=("corruption", "severity"))

    return row

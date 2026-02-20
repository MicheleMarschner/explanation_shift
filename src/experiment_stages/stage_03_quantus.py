from __future__ import annotations

import torch
import quantus
from pathlib import Path
import numpy as np

from typing import Any, Dict, Literal, Optional

from src.configs.global_config import PATHS, DEVICE, IG_STEPS
from src.utils import cpu, as_np_int64_1d, collect_x_from_loader
from src.data import get_clean_data, get_corrupted_data
from src.experiment_stages.helper import save_quantus_metrics
from src.metrics import build_quantus_metrics



def to_scalar(x):
    # torch scalar
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()

    # numpy scalar
    if isinstance(x, (np.generic,)):
        return float(x)

    # list/tuple/ndarray
    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asarray(x).astype(float)
        # if it's a single number like [0.6] -> 0.6
        if arr.size == 1:
            return float(arr.reshape(-1)[0])
        # if it's per-sample scores -> store mean (or median)
        return float(arr.mean())

    # python number
    if isinstance(x, (int, float)):
        return float(x)

    # fallback: string
    return str(x)


def run_quantus_metrics(
    pair_idx,
    corruption,
    severity,
    clean_path: Path,
    artifact_path: Optional[Path],
    save_path: Path,
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

    # Load stage00 reference
    ref = torch.load(clean_path, map_location="cpu", weights_only=False)
    cr = ref["clean_reference"]

    pred_clean = cr["pred_clean"].long()       # torch tensor [N]
    y_batch = as_np_int64_1d(pred_clean)


    # Decide domain inputs + attributions
    if mode == "clean":
        clean_loader, _, _ = get_clean_data(path=PATHS.data_clean, idx=pair_idx, transform=transform)
        X_clean_t = collect_x_from_loader(clean_loader)     # torch [N,C,H,W]
        x_batch = cpu(X_clean_t).numpy()

        #sal_clean = cr["sal_clean"].float()
        #a_batch = cpu(sal_clean.unsqueeze(1)).numpy()  # numpy [N,1,H,W]

    else:
        corruption = str(corruption)
        severity = int(severity)

        corr_loader, _, _ = get_corrupted_data(
            idx=pair_idx,
            path=PATHS.data_corr,
            transform=transform,
            corruption=corruption,
            severity=severity,
        )
        X_corr_t = collect_x_from_loader(corr_loader)
        x_batch = cpu(X_corr_t).numpy()

        #art = torch.load(artifact_path, map_location="cpu", weights_only=False)
        #cc = art["corrupt_reference"]
        #sal_corr = cc["sal_corr"].float()
        #a_batch = cpu(sal_corr.unsqueeze(1)).numpy()  # numpy [N,1,H,W]

    assert x_batch.shape[0] == y_batch.shape[0]
    assert y_batch.ndim == 1
    #assert a_batch.shape[0] == x_batch.shape[0] == y_batch.shape[0]
    #assert a_batch.shape[-2:] == x_batch.shape[-2:]  # (H,W)

    # -----------------------
    # Quantus metrics
    # -----------------------
    metrics = build_quantus_metrics()

    # Quantus runs forward passes 
    model.eval()

    results = {}
    for metric, metric_func in metrics.items():
        scores = metric_func(
            model=model, 
            x_batch=x_batch, 
            y_batch=y_batch, 
            a_batch=None,
            s_batch=None,
            explain_func=quantus.explain,   
            explain_func_kwargs={
                "method": "IntegratedGradients",
                "n_steps": int(IG_STEPS),
                "device": DEVICE,
            },
        )
        results[metric] = to_scalar(scores)

    row = {"corruption": corruption, "severity": severity, **results}
    
    save_quantus_metrics(save_path, row, mode)

    return row

from __future__ import annotations

import torch
import quantus
from pathlib import Path
import gc
import numpy as np

from typing import Any, Dict, Literal, Optional

from src.configs.global_config import PATHS, DEVICE, IG_STEPS, BATCH_SIZE_EXPLAINER, BATCH_SIZE
from src.utils import cpu, as_np_int64_1d, collect_x_from_loader
from src.data import get_clean_data, get_corrupted_data
from src.experiment_stages.helper import save_quantus_metrics
from src.metrics import build_quantus_metrics
from src.explainers import mask_invariant, mask_correct, compute_saliency_maps


def to_scalar(x):
    print(type(x))

    return (float(x[0]))


def run_quantus_metrics(
    pair_idx,
    corruption,
    severity,
    clean_path: Path,
    artifact_path: Optional[Path],
    save_path: Path,
    model,
    transform,
    explainer_name: str,
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

    Note:
    Quantus should see inputs exactly as the model sees inputs during inference.
    a_batch shape/type is what Quantus expects. / Use Quantus’ explain only if you want a quick sanity run or if you’re struggling with shape/device quirks.
    s_batch is segmentation masks only needed for Localization
    abs = absolute relevance (treat negative and positive relevance as “importance magnitude”). -> abs=True: importance = magnitude, ignores sign
    normalise = Whether Quantus should normalize the attribution map before computing the metric.
    return_nan_when_prediction_changes = used in metrics where the protocol assumes you evaluate explanations for a fixed decision. If prediction changes during perturbations
    """

    # Load stage00 reference
    ref = torch.load(clean_path, map_location="cpu", weights_only=False)
    cr = ref["clean_reference"]

    pred_clean = cr["pred_clean"].long()       # torch tensor [N]
    y_batch = as_np_int64_1d(pred_clean)

    def quantus_explain_func(model, inputs, targets, **kwargs):
        """
        Quantus-compatible explanation function.
        Returns absolute saliency maps as numpy [N, 1, H, W].
        """
        x_t = torch.as_tensor(inputs, dtype=torch.float32)
        t_t = torch.as_tensor(targets, dtype=torch.long)

        sal = compute_saliency_maps(
            x_t,
            target=t_t,
            explainer_name=explainer_name,
            model=model,
            device=DEVICE,
            steps=IG_STEPS,
            internal_bs=BATCH_SIZE_EXPLAINER,
            batch_size=BATCH_SIZE,
        )

        a = sal.abs().unsqueeze(1)   # [N,1,H,W]
        return cpu(a).numpy()


    # Decide domain inputs + attributions
    if mode == "clean":
        clean_loader, _, _ = get_clean_data(path=PATHS.data_clean, idx=pair_idx, transform=transform)
        X_clean_t = collect_x_from_loader(clean_loader)     # torch [N,C,H,W]
        x_batch = cpu(X_clean_t).numpy()

        sal_clean = cr["sal_clean"].float()
        a_batch = cpu(sal_clean.abs().unsqueeze(1)).numpy()  # numpy [N,1,H,W]

        masks = {
            "all": np.ones_like(pred_clean, dtype=bool)
        }

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

        y_true = cr["y_true"]

        art = torch.load(artifact_path, map_location="cpu", weights_only=False)
        cc = art["corrupt_reference"]
        pred_corr = cc["pred_corr"]

        sal_corr = cc["sal_corr"].float()
        a_batch = cpu(sal_corr.abs().unsqueeze(1)).numpy()  # numpy [N,1,H,W]

        masks = {
            "all": np.ones_like(pred_clean, dtype=bool),
            "inv": mask_invariant(pred_clean, pred_corr).bool(),
            "both_corr": mask_correct(pred_clean, pred_corr, y_true).bool()
        }

        n_invariant = masks['inv'].sum().item()
        print(f" labels corresponding both domains {n_invariant} from {len(pred_clean)}")
        n_both_correct = masks['both_corr'].sum().item()
        print(f" labels correct in both domains {n_both_correct} from {len(pred_clean)}")

    assert x_batch.shape[0] == y_batch.shape[0]
    assert y_batch.ndim == 1
    assert a_batch.shape[0] == x_batch.shape[0] == y_batch.shape[0]
    assert a_batch.shape[-2:] == x_batch.shape[-2:]  # (H,W)

    # -----------------------
    # Quantus metrics
    # -----------------------
    metrics = build_quantus_metrics()

    # Quantus runs forward passes 
    model.eval()

    results = {}
    for slice_name, m in masks.items():
        idx = np.where(m)[0]

        # always record slice size
        results[f"n__{slice_name}"] = int(idx.size)

        # handle empty slice
        if idx.size == 0:
            # store NaN if slice is empty
            for metric_name in metrics:
                results[f"{metric_name}__{slice_name}"] = float("nan")
            continue

        x_s = x_batch[idx]
        y_s = y_batch[idx]
        a_s = a_batch[idx]

        for metric, metric_func in metrics.items():
            print(f"Evaluating {metric}.")
            gc.collect()
            torch.cuda.empty_cache()

            scores = metric_func(
                model=model,
                x_batch=x_s,
                y_batch=y_s,
                a_batch=a_s,
                s_batch=None,
                device=DEVICE,
                explain_func=quantus_explain_func,
            )
            results[f"{metric}__{slice_name}"] = to_scalar(scores)

    row = {"corruption": corruption, "severity": severity, **results}
    
    save_quantus_metrics(save_path, row, mode)

    # Empty cache.
    gc.collect()
    torch.cuda.empty_cache()

    return row
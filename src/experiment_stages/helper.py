import json
from pathlib import Path
import torch
import csv
import pandas as pd

from typing import Any, Dict, Sequence

from src.utils import to_cpu_f16, cpu
from src.configs.global_config import BATCH_SIZE_EXPLAINER, IG_STEPS, CIFAR10_MEAN, CIFAR10_SD, TARGET_POLICY


def save_experiment_reference(
    save_path: Path,
    seed: int,
    pair_idx: Sequence[int],
    exp_config: Any,  # ExperimentTemplate or dict-like
    clean_ref: Dict[str, Any],
) -> None:
    """
    Writes:
      - {save_path}                (pt)
      - {save_path.with_suffix('.json')}  (small metadata json)
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    ref_pt = {
        "seed": int(seed),
        "N_pairs": int(len(pair_idx)),
        "pair_idx": cpu(torch.as_tensor(pair_idx).long()),
        "corruptions": list(exp_config.CORRUPTIONS),
        "severities": [int(s) for s in exp_config.SEVERITIES],
        "preprocess": {
            "cifar_mean": tuple(map(float, CIFAR10_MEAN)),
            "cifar_std": tuple(map(float, CIFAR10_SD)),
        },
        "ig_config": {
            "ig_steps": int(IG_STEPS),
            "internal_bs": int(BATCH_SIZE_EXPLAINER),
            "baseline": "zeros_like_input (normalized space)",
            "target_policy": TARGET_POLICY,
        },
        "clean_reference": {
            "logits_clean": cpu(clean_ref["logits"]),
            "pred_clean": cpu(clean_ref["pred"]),
            "proba_clean": cpu(clean_ref["proba"]),
            "acc_clean": float(clean_ref["acc"]),
            "entropy_clean": cpu(clean_ref["entropy"]),
            "E_clean": cpu(clean_ref["E"]),
            "sal_clean": to_cpu_f16(clean_ref["sal"]),
            "sigma_ref": float(clean_ref["sigma"]),
            "y_true": cpu(clean_ref["y"].long())
        },
    }

    torch.save(ref_pt, save_path)

    meta = {
        "seed": int(seed),
        "N_pairs": int(len(pair_idx)),
        "corruptions": list(exp_config.CORRUPTIONS),
        "severities": [int(s) for s in exp_config.SEVERITIES],
        "preprocess": ref_pt["preprocess"],
        "ig_config": ref_pt["ig_config"],
        "clean_reference_scalars": {
            "acc_clean": ref_pt["clean_reference"]["acc_clean"],
            "sigma_ref": ref_pt["clean_reference"]["sigma_ref"],
        },
        "files": {"reference_pt": save_path.name},
    }
    save_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))


def _append_row_csv_stage01(csv_path: Path, row: dict) -> None:
    """
    Append `row` to csv_path. If file doesn't exist, create it + write header.
    Keeps a stable column order.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["corruption", "severity", "time_sec", "acc_corr"]

    file_exists = csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        # only keep relevant keys (prevents accidental extra cols)
        writer.writerow({k: row.get(k) for k in fieldnames})


def save_artifacts(
        save_path: Path,
        corruption: str,
        severity: int, 
        time: float, 
        corr_ref: Dict[str, Any]
    ):
    
    artifact = {
        "corruption": corruption,
        "severity": severity,
        "time_sec": time,
        "corrupt_reference": {
            "logits_corr": cpu(corr_ref["logits"]),
            "pred_corr": cpu(corr_ref["pred"]),
            "proba_corr": cpu(corr_ref["proba"]),
            "acc_corr": float(corr_ref["acc"]),
            "entropy_corr": cpu(corr_ref["entropy"]),
            "E_corr": cpu(corr_ref["E"]),
            "sal_corr": to_cpu_f16(corr_ref["sal"]),
        },
    }
    torch.save(artifact, save_path)

    row = {
        "corruption": corruption, 
        "severity": severity, 
        "time_sec": time,
        "acc_corr": float(corr_ref["acc"]),
    }

    results_csv = save_path.parent / "01__artifact_results.csv"
    _append_row_csv_stage01(results_csv, row)


def _ensure_stage02_csv(stage01_csv: Path, stage02_csv: Path) -> None:
    """
    Ensure stage02 exists and contains at least all rows/columns from stage01.
    If stage02 exists, we *sync* (union) stage01 into it.
    """
    stage01_csv = Path(stage01_csv)
    stage02_csv = Path(stage02_csv)
    stage02_csv.parent.mkdir(parents=True, exist_ok=True)

    if not stage01_csv.exists():
        raise FileNotFoundError(f"Stage01 CSV not found: {stage01_csv}")

    df1 = pd.read_csv(stage01_csv)

    if not stage02_csv.exists():
        df1.to_csv(stage02_csv, index=False)
        return

    df2 = pd.read_csv(stage02_csv)

    # ensure all stage01 columns exist in stage02
    for c in df1.columns:
        if c not in df2.columns:
            df2[c] = pd.NA

    # ensure keys exist
    for k in ["corruption", "severity"]:
        if k not in df2.columns:
            df2[k] = pd.NA

    # append missing stage01 rows (by key)
    def key_series(df):
        return df["corruption"].astype(str).str.strip() + "__" + pd.to_numeric(df["severity"], errors="coerce").astype("Int64").astype(str)

    k1 = key_series(df1)
    k2 = set(key_series(df2).tolist())

    missing = df1.loc[~k1.isin(k2)].copy()
    if len(missing) > 0:
        # add any extra columns that exist in stage02 but not in stage01
        for c in df2.columns:
            if c not in missing.columns:
                missing[c] = pd.NA
        df2 = pd.concat([df2, missing[df2.columns]], ignore_index=True)

    df2.to_csv(stage02_csv, index=False)


def _upsert_row_to_csv(csv_path: Path, row: dict, keys=["corruption", "severity"]) -> None:
    """
    Update matching row by keys, or append if missing.
    Adds new columns if needed.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    row_df = pd.DataFrame([row])

    # ensure key columns exist
    for k in keys:
        if k not in df.columns:
            df[k] = pd.NA
        if k not in row_df.columns:
            raise ValueError(f"Row missing key '{k}': {row}")

    # ensure new columns exist
    for c in row_df.columns:
        if c not in df.columns:
            df[c] = pd.NA

    # find match
    mask = pd.Series(True, index=df.index)
    for k in keys:
        val = row_df.loc[0, k]
        if k == "severity":
            mask &= (pd.to_numeric(df[k], errors="coerce") == int(val))
        else:
            mask &= (df[k].astype(str) == str(val))

    if mask.any():
        idx = df.index[mask][0]
        for c in row_df.columns:
            df.at[idx, c] = row_df.loc[0, c]
    else:
        df = pd.concat([df, row_df], ignore_index=True)

    # nice column order: keys first
    cols = keys + [c for c in df.columns if c not in keys]
    df = df[cols]

    df.to_csv(csv_path, index=False)


def save_drift_metrics(save_path: Path, row: dict, vectors: dict) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {"row": row, "vectors": vectors}
    torch.save(payload, save_path)

    run_dir = save_path.parents[1]  # .../experiment__n...__seed...
    stage01_csv = run_dir / "01__artifacts" / "01__artifact_results.csv"
    stage02_csv = run_dir / "02__drift" / "02__drift_results.csv"

    _ensure_stage02_csv(stage01_csv, stage02_csv)
    _upsert_row_to_csv(stage02_csv, row, keys=["corruption", "severity"])


import json
from pathlib import Path
import torch

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


def save_drift_metrics(save_path: Path, row: dict, vectors: dict) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {"row": row, "vectors": vectors}
    torch.save(payload, save_path)


def save_quantus_metrics(save_path: Path, row: dict, mode: str) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "row": row,
        "meta": {
            "mode": mode,
            "y_batch_policy": "pred_clean",
            "x_domain": "clean" if mode == "clean" else "corrupted",
            "a_domain": "clean" if mode == "clean" else "corrupted",
        },
    }
    torch.save(payload, save_path)
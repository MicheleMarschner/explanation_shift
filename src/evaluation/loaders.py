import re
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional, List

# --- Helpers ---

def parse_corr_sev_from_path(p: Path) -> Tuple[Optional[str], Optional[int]]:
    """Extracts corruption and severity from filenames or parent directories."""
    s = p.as_posix()
    name = p.name

    # Pattern: *__<corr>__sev<k>*
    m = re.search(r"__(artifacts|drift)__([A-Za-z0-9_]+)__sev(\d+)", s)
    if m:
        return m.group(2), int(m.group(3))

    # Pattern: corr<name>...sev<k>
    m = re.search(r"corr([A-Za-z0-9_]+).*sev(\d+)", name)
    if m:
        return m.group(1), int(m.group(2))
    
    # Pattern: corruption=<name>...severity=<k>
    m = re.search(r"corruption[=/]([A-Za-z0-9_]+)", s)
    n = re.search(r"severity[=/](\d+)", s)
    if m and n:
        return m.group(1), int(n.group(1))

    return None, None

def to_np(x):
    if x is None: return None
    if torch.is_tensor(x): return x.detach().cpu().numpy()
    return np.asarray(x)

def load_msp_clean_and_pair_idx(ref_pt: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loads clean reference data."""
    ref = torch.load(ref_pt, map_location="cpu")
    # Adjust keys based on your specific structure inside clean_ref.pt
    # Assuming standard structure based on your snippet
    pair_idx = to_np(ref.get("pair_idx"))
    
    # Handle different saving structures for clean proba
    if "clean_reference" in ref:
        proba = ref["clean_reference"]["proba_clean"]
        y_clean = to_np(ref["clean_reference"]["y_clean"])
    else:
        proba = ref["proba_clean"]
        y_clean = to_np(ref["y_clean"])
        
    proba = to_np(proba)
    msp_clean = proba.max(axis=1).astype(float).reshape(-1)
    
    return pair_idx.astype(int), msp_clean, y_clean

def load_artifact_pred(artifact_pt: Path):
    art = torch.load(artifact_pt, map_location="cpu")
    # Handle nesting
    base = art.get("corrupt_reference", art)
    
    pred = to_np(base["pred_corr"]).astype(int).reshape(-1)
    proba = to_np(base.get("proba_corr"))
    msp = None
    if proba is not None:
        msp = proba.max(axis=1).astype(float).reshape(-1)
    return pred, msp

def load_drift_vectors(drift_pt: Path):
    obj = torch.load(drift_pt, map_location="cpu")
    v = obj.get("vectors", obj) # Handle nesting

    # Safe extraction with defaults if keys differ slightly
    dH = to_np(v.get("dH", v.get("entropy_diff"))).astype(float).reshape(-1)
    inv = to_np(v.get("invariant", v.get("mask_invariant"))).astype(bool).reshape(-1)
    bc = to_np(v.get("both_correct", v.get("mask_both_correct"))).astype(bool).reshape(-1)
    
    cos = to_np(v.get("exp__cosine_sim", v.get("cosine_sim"))).astype(float).reshape(-1)
    iou = to_np(v.get("exp__iou_topk", v.get("iou"))).astype(float).reshape(-1)
    rho = to_np(v.get("exp__spearman_rho", v.get("spearman_rho"))).astype(float).reshape(-1)

    return dH, inv, bc, cos, iou, rho

# --- Main Loading Functions ---

def load_aggregated_results(csv_path: Path) -> pd.DataFrame:
    """Loads the per-severity aggregated table (e.g. 02__drift_results.csv)."""
    df = pd.read_csv(csv_path)
    # Normalize
    if "severity" in df.columns:
        df["severity"] = pd.to_numeric(df["severity"], errors="coerce").fillna(0).astype(int)
    if "corruption" in df.columns:
        df["corruption"] = df["corruption"].astype(str)
    return df

def build_pairs_table(
    exp_dir: Path, 
    save_path: Optional[Path] = None,
    force_rebuild: bool = False
) -> pd.DataFrame:
    """
    Scans folders, loads .pt files, and creates the master pairs CSV.
    If save_path exists and not force_rebuild, loads from disk.
    """
    if save_path and save_path.exists() and not force_rebuild:
        print(f"Loading existing pairs table from {save_path}")
        return pd.read_csv(save_path)

    print("Building pairs table from .pt files...")
    
    # 1. Load Reference
    ref_pt = exp_dir / "00__reference" / "00__clean_ref.pt"
    if not ref_pt.exists():
        # Fallback search
        ref_pt = list((exp_dir / "00__reference").glob("*.pt"))[0]
        
    pair_idx_ref, msp_clean_ref, y_true_ref = load_msp_clean_and_pair_idx(ref_pt)
    
    # Map pair_idx to clean stats for quick lookup
    # Note: Assuming pair_idx is unique and consistent across files
    idx_to_msp = dict(zip(pair_idx_ref, msp_clean_ref))
    idx_to_y = dict(zip(pair_idx_ref, y_true_ref))

    # 2. Map Files
    artifacts_dir = exp_dir / "01__artifacts"
    drift_dir = exp_dir / "02__drift"
    
    # Helper to map (corr, sev) -> path
    def map_files(root):
        mapping = {}
        for p in root.rglob("*.pt"):
            c, s = parse_corr_sev_from_path(p)
            if c is not None and s is not None:
                mapping[(c, s)] = p
        return mapping

    art_map = map_files(artifacts_dir)
    drift_map = map_files(drift_dir)
    
    common_keys = sorted(set(art_map.keys()) & set(drift_map.keys()))
    
    rows = []
    for (corr, sev) in common_keys:
        pred_corr, msp_corr = load_artifact_pred(art_map[(corr, sev)])
        dH, inv, bc, cos, iou, rho = load_drift_vectors(drift_map[(corr, sev)])
        
        # We assume the .pt files are ordered by the original pair_idx 
        # (Usually standard in these pipelines). 
        # If not, you need 'pair_idx' stored in every .pt file to join on.
        # Here assuming strict ordering matching clean_ref:
        
        n_samples = len(pred_corr)
        
        # Reconstruct DataFrame
        df_batch = pd.DataFrame({
            "corruption": [corr] * n_samples,
            "severity": [sev] * n_samples,
            "pair_idx": pair_idx_ref[:n_samples], # Assuming same size
            "y_true": y_true_ref[:n_samples],
            "pred_corr": pred_corr,
            "msp_corr": msp_corr if msp_corr is not None else np.nan,
            "msp_clean": msp_clean_ref[:n_samples],
            
            "dH": dH,
            "abs_dH": np.abs(dH),
            "invariant": inv,
            "both_correct": bc,
            
            "cos": cos,
            "iou": iou,
            "rho": rho,
            
            # Pre-calculate drift metrics
            "drift_1m_cos": 1.0 - cos,
            "drift_1m_iou": 1.0 - iou,
            "drift_1m_rho": 1.0 - rho
        })
        
        df_batch["correct_corr"] = (df_batch["pred_corr"] == df_batch["y_true"])
        rows.append(df_batch)

    if not rows:
        raise ValueError("No matching files found to build table.")

    df_final = pd.concat(rows, ignore_index=True)
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df_final.to_csv(save_path, index=False)
        print(f"Saved pairs table to {save_path}")
        
    return df_final
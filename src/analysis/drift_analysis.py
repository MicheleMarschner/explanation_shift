"""
Plot B1–B3 (severity vs explanation similarity, one line per corruption)

These plots show, for each corruption type, how explanation similarity between clean and corrupted inputs changes as severity 
increases. Each line is one corruption, so you can compare which corruptions preserve similar explanations versus which ones 
cause explanations to diverge quickly. The different versions (cosine / IoU / Spearman ρ) are just different similarity 
measures for the same idea: higher means more stable explanations, lower means more drift.



Plot B4 (severity vs explanation drift magnitude, one line per corruption)

This plot shows the same trend as above, but converted into an explicit drift magnitude (e.g., 1 − ρ). That makes the 
direction intuitive: higher values mean stronger explanation drift. It highlights which corruptions produce large 
explanation changes at low severity and how drift grows with severity across corruption types.

"""


# src/analysis/plot_stage02_dE.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.configs.global_config import PATHS


# -------------------------
# IO
# -------------------------
def load_stage_table(path: Path) -> pd.DataFrame:
    """Load stage results table (CSV/Parquet) and normalize key dtypes."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    for col in ("corruption", "severity"):
        if col not in df.columns:
            raise ValueError(f"Missing required column: '{col}'")

    df = df.copy()
    df["corruption"] = df["corruption"].astype(str)
    df["severity"] = pd.to_numeric(df["severity"], errors="raise").astype(int)
    return df


# -------------------------
# Baseline handling (clean severity=0 shared across all)
# -------------------------
def attach_clean_baseline(
    df: pd.DataFrame,
    corruption: str,
    severity_col: str = "severity",
    corruption_col: str = "corruption",
) -> pd.DataFrame:
    """
    For a given corruption curve, prepend the shared clean baseline row (severity==0).

    - If multiple clean rows exist, averages numeric columns.
    - Sets the clean row's corruption label to the given corruption so it plots as one curve.
    """
    df = df.copy()
    df[severity_col] = pd.to_numeric(df[severity_col], errors="raise").astype(int)

    d_corr = df[df[corruption_col] == corruption].copy()
    if d_corr.empty:
        raise ValueError(
            f"No rows for corruption='{corruption}'. "
            f"Available: {sorted(df[corruption_col].unique())}"
        )

    d_clean = df[df[severity_col] == 0].copy()
    if d_clean.empty:
        # If you truly don't have a sev0 row, we just return the corruption rows.
        return d_corr.sort_values(severity_col)

    # robust: average numeric cols across all clean rows
    num_cols = d_clean.select_dtypes(include=[np.number]).columns.tolist()
    non_num_cols = [c for c in d_clean.columns if c not in num_cols]

    clean_row = pd.DataFrame([d_clean[num_cols].mean(numeric_only=True)])
    clean_row[severity_col] = 0
    clean_row[corruption_col] = corruption

    for c in non_num_cols:
        if c not in (severity_col, corruption_col):
            clean_row[c] = d_clean[c].iloc[0]

    out = pd.concat([clean_row, d_corr], ignore_index=True)
    out = out.sort_values(severity_col).drop_duplicates(subset=[severity_col], keep="first")
    return out


# -------------------------
# Plotting helpers
# -------------------------

def plot_severity_vs_metric_by_corruption(
    df: pd.DataFrame,
    y_col: str,
    severities: Sequence[int] = (1, 2, 3, 5),
    include_clean_baseline: bool = True,
    exclude: Iterable[str] = ("clean",),
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> None:
    """
    One line per corruption: x=severity, y=y_col.
    Values should already be mean over (stable) pairs in your table.
    """
    if y_col not in df.columns:
        raise ValueError(
            f"Column '{y_col}' not found. "
            f"Available columns include: {sorted(list(df.columns))[:50]} ..."
        )

    df = df.copy()
    df["severity"] = pd.to_numeric(df["severity"], errors="raise").astype(int)

    fig, ax = plt.subplots(figsize=(9.5, 5.2))

    corrs = sorted(df["corruption"].unique())
    for corr in corrs:
        if corr in set(exclude):
            continue

        d = df[df["corruption"] == corr].copy()
        if include_clean_baseline:
            d = attach_clean_baseline(df, corruption=corr)

        d = d[d["severity"].isin(list(severities))].sort_values("severity")
        if d.empty:
            continue

        ax.plot(d["severity"], d[y_col], marker="o", label=corr)

    ax.set_xlabel("Severity")
    ax.set_xticks(list(severities))
    ax.set_ylabel(ylabel or y_col)
    ax.grid(True, alpha=0.25)

    if title is None:
        title = f"Severity vs {y_col} (one line per corruption)"
    ax.set_title(title)

    ax.legend(loc="best", ncol=2)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)

    plt.show()


# -------------------------
# Configure your Stage02 ΔE columns here
# -------------------------
def resolve_dE_columns(
    df: pd.DataFrame,
    slice_name: str = "inv",                 # "inv", "all", "both_corr", ...
    strict: bool = True,                     # True = error if missing
) -> dict[str, str]:
    """
    Pick ΔE columns for a specific slice explicitly.
    Expects Stage02 wide columns like:
      exp_{slice}__cos_mean
      exp_{slice}__iou_mean
      exp_{slice}__rho_mean

    If strict=False, falls back to 'all' if requested slice is missing.
    """
    want = {
        "cosine": f"exp_{slice_name}__cos_mean",
        "iou":    f"exp_{slice_name}__iou_mean",
        "rho":    f"exp_{slice_name}__rho_mean",
    }

    out = {}
    missing = []
    for k, col in want.items():
        if col in df.columns:
            out[k] = col
        else:
            missing.append(col)

    if missing:
        if (not strict) and slice_name != "all":
            # fallback to all
            fallback = resolve_dE_columns(df, slice_name="all", strict=True)
            out.update(fallback)
            print(f"[resolve_dE_columns] slice '{slice_name}' missing {missing} -> falling back to 'all'")
        else:
            # helpful error
            available = [c for c in df.columns if c.startswith("exp_") and c.endswith("_mean")]
            raise ValueError(
                f"Requested slice='{slice_name}' but missing columns: {missing}\n"
                f"Available exp_*__*_mean columns:\n" + "\n".join(sorted(available))
            )

    return out



# -------------------------
# Example main
# -------------------------
if __name__ == "__main__":

    stage02_path = PATHS.runs / "experiment__n250__IG__seed51" / "02__drift" / "02__drift_results.csv"
    df2 = load_stage_table(stage02_path)

    SLICE = "both_corr"        # <- choose: "inv", "all", "both_corr", ...
    cols = resolve_dE_columns(df2, slice_name=SLICE, strict=True)

    cos_col = cols["cosine"]
    iou_col = cols["iou"]
    rho_col = cols["rho"]

    print("Chosen slice:", SLICE)
    print("Plotting columns:", cols)

    plot_severity_vs_metric_by_corruption(
        df2,
        y_col=cos_col,
        title=f"Plot B1: Severity vs ΔE (Cosine) — slice={SLICE}",
        ylabel="Cosine similarity (higher = more similar)",
        save_path=PATHS.results / f"plotB1_dE_cosine__{SLICE}.png",
    )

    plot_severity_vs_metric_by_corruption(
        df2,
        y_col=iou_col,
        title=f"Plot B2: Severity vs ΔE (IoU) — slice={SLICE}",
        ylabel="IoU% (higher = more overlap)",
        save_path=PATHS.results / f"plotB2_dE_iou__{SLICE}.png",
    )

    plot_severity_vs_metric_by_corruption(
        df2,
        y_col=rho_col,
        title=f"Plot B3: Severity vs ΔE (Spearman ρ) — slice={SLICE}",
        ylabel="Spearman ρ (higher = more similar, lower = more drift)",
        save_path=PATHS.results / f"plotB3_dE_rho__{SLICE}.png",
    )

    # Drift magnitude version
    df2b = df2.copy()
    df2b[f"drift_1m_rho__{SLICE}"] = 1.0 - df2b[rho_col]

    plot_severity_vs_metric_by_corruption(
        df2b,
        y_col=f"drift_1m_rho__{SLICE}",
        title=f"Plot B4: Severity vs Explanation Drift (1 − ρ) — slice={SLICE}",
        ylabel="1 − Spearman ρ (higher = more drift)",
        save_path=PATHS.results / f"plotB4_drift_1minus_rho__{SLICE}.png",
    )
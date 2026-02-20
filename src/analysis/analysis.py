'''
## Plot 1: acc_vs_exp folder

Description:
This plot shows how model performance and explanation stability change as corruption severity increases. 
For a fixed corruption type, it tracks accuracy on corrupted images and, at the same time, tracks how similar 
the explanations remain between clean and corrupted inputs (for different subsets like all pairs vs. invariant 
pairs). The goal is to see whether accuracy drops together with explanation similarity—or whether explanations 
can drift even when accuracy stays relatively stable.

## Plot 2: plot_deltaP_vs_deltaE file

Description:
This plot shows whether performance degradation (ΔP) and explanation change (ΔE) move together across different 
corruption settings. Each point represents one (corruption, severity) condition: the x-axis is a proxy for how 
much performance worsens (e.g., 1 − accuracy or Δ entropy), and the y-axis is a proxy for how much the explanation 
drifts (1 − similarity). By coloring points by corruption and annotating severity, it becomes visible which 
corruptions cause large explanation drift with little performance drop (decoupling) versus cases where both 
degrade together. The optional Spearman ρ summarizes the overall monotonic association between ΔP and ΔE across 
all conditions.


## Plot 3: metric_vs_severity_faith_fog file

This plot shows how a Quantus faithfulness score changes as corruption severity increases for a fixed corruption type. 
It includes a clean baseline at severity 0, so the curve can be read as “how faithfulness deteriorates (or stays stable) 
relative to clean data.” Optionally, it overlays accuracy vs. severity in the same figure, making it easy to see whether 
faithfulness drops together with performance or whether faithfulness changes even when accuracy remains relatively stable.

'''



from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.configs.global_config import PATHS

# ----------------------------
# IO + helpers
# ----------------------------
def load_results(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # normalize types
    if "severity" in df.columns:
        df["severity"] = pd.to_numeric(df["severity"], errors="coerce").astype("Int64")
    if "acc_corr" in df.columns:
        df["acc_corr"] = pd.to_numeric(df["acc_corr"], errors="coerce")
    if "corruption" in df.columns:
        df["corruption"] = df["corruption"].astype(str).str.strip()
    return df


def _require_cols(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}\nAvailable columns: {list(df.columns)}")


def _select_severities(df: pd.DataFrame, severities: Optional[Sequence[int]] = None) -> pd.DataFrame:
    if severities is None:
        # default: your typical set
        severities = [1, 2, 3, 5]
    df = df[df["severity"].isin(severities)].copy()
    return df


# ----------------------------
# Plot 1: Acc vs Explanation similarity over severity
# ----------------------------
def plot_acc_vs_expl_similarity(
    df: pd.DataFrame,
    corruption: str,
    severities: Sequence[int] = (1, 2, 3, 5),
    expl_metric: str = "rho_mean",   # or "cos_mean"
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot 1:
      x-axis: severity
      y-axis left: acc_corr
      y-axis right: explanation similarity (3 curves: all/inv/both_corr)
    """

    cols = [
        "corruption", "severity", "acc_corr",
        f"exp_all__{expl_metric}",
        f"exp_inv__{expl_metric}",
        f"exp_both_corr__{expl_metric}",
    ]
    _require_cols(df, cols)

    d = df[df["corruption"] == corruption].copy()
    d = _select_severities(d, severities)
    d = d.sort_values("severity")

    if d.empty:
        raise ValueError(f"No rows for corruption='{corruption}' and severities={list(severities)}")

    x = d["severity"].astype(int).to_numpy()
    acc = d["acc_corr"].to_numpy(dtype=float)

    e_all = pd.to_numeric(d[f"exp_all__{expl_metric}"], errors="coerce").to_numpy()
    e_inv = pd.to_numeric(d[f"exp_inv__{expl_metric}"], errors="coerce").to_numpy()
    e_cor = pd.to_numeric(d[f"exp_both_corr__{expl_metric}"], errors="coerce").to_numpy()

    fig, ax1 = plt.subplots(figsize=(8.5, 4.8))
    ax2 = ax1.twinx()

    # left axis: accuracy
    ax1.plot(x, acc, marker="o", label="acc_corr")

    # right axis: explanation similarity curves (slices)
    ax2.plot(x, e_all, marker="o", linestyle="--", label=f"exp_all__{expl_metric}")
    ax2.plot(x, e_inv, marker="o", linestyle="--", label=f"exp_inv__{expl_metric}")
    ax2.plot(x, e_cor, marker="o", linestyle="--", label=f"exp_both_corr__{expl_metric}")

    ax1.set_xlabel("Severity")
    ax1.set_ylabel("Accuracy (acc_corr)")
    ax2.set_ylabel(f"Explanation similarity ({expl_metric})")

    ax1.set_xticks(list(x))
    ax1.set_title(f"{corruption}: Accuracy vs Explanation Similarity over Severity")

    # single combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")

    ax1.grid(True, alpha=0.25)

    plt.tight_layout()
    print(save_path)
    if save_path is not None:
        file_path = save_path / f"acc_vs_expl_similarity_{corruption}"
        plt.savefig(file_path)
    plt.show()


# ----------------------------
# Plot 2: ΔP vs ΔE scatter
# ----------------------------
def plot_deltaP_vs_deltaE_scatter(
    df: pd.DataFrame,
    severities: Sequence[int] = (1, 2, 3, 5),
    # x-axis options:
    x_proxy: str = "one_minus_acc",     # "one_minus_acc" or "mean_abs_delta_entropy"
    # y-axis slice choice:
    y_slice: str = "inv",               # "all" | "inv" | "both_corr"
    expl_metric: str = "rho_mean",      # or "cos_mean"
    save_path: Optional[Path] = None,
    compute_spearman: bool = True,
) -> None:
    """
    Plot 2:
      x = ΔP proxy (1-acc_corr) or mean_abs_delta_entropy
      y = ΔE magnitude proxy (1 - similarity)
      one point per (corruption, severity)
      colored by corruption (matplotlib default color cycle)
    """

    _require_cols(df, ["corruption", "severity", "acc_corr", "mean_abs_delta_entropy"])
    y_col = f"exp_{y_slice}__{expl_metric}"
    _require_cols(df, [y_col])

    d = _select_severities(df.copy(), severities)
    d = d.dropna(subset=["corruption", "severity"]).copy()
    d["severity"] = d["severity"].astype(int)

    # x
    if x_proxy == "one_minus_acc":
        x = 1.0 - pd.to_numeric(d["acc_corr"], errors="coerce")
        x_label = "ΔP proxy = 1 - acc_corr"
    elif x_proxy == "mean_abs_delta_entropy":
        x = pd.to_numeric(d["mean_abs_delta_entropy"], errors="coerce")
        x_label = "ΔP proxy = mean_abs_delta_entropy"
    else:
        raise ValueError("x_proxy must be 'one_minus_acc' or 'mean_abs_delta_entropy'")

    # y (convert similarity -> drift magnitude)
    sim = pd.to_numeric(d[y_col], errors="coerce")
    y = 1.0 - sim
    y_label = f"ΔE proxy = 1 - {y_col}"

    d["x"] = x
    d["y"] = y
    d = d.dropna(subset=["x", "y"]).copy()

    if d.empty:
        raise ValueError("No valid rows after dropping NaNs for x/y. Check your columns and values.")

    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    # Scatter by corruption
    for corr, g in d.groupby("corruption", sort=True):
        ax.scatter(g["x"], g["y"], label=str(corr), alpha=0.9)
        # annotate with severity (small, optional)
        for _, r in g.iterrows():
            ax.annotate(str(int(r["severity"])), (r["x"], r["y"]), fontsize=8, alpha=0.75)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"ΔP vs ΔE scatter (slice='{y_slice}', metric='{expl_metric}')")
    ax.grid(True, alpha=0.25)
    ax.legend(title="corruption", loc="best")

    # (optional) correlation across all points
    if compute_spearman:
        # rank correlation without scipy: use pandas rank + pearson on ranks
        xr = d["x"].rank(method="average")
        yr = d["y"].rank(method="average")
        rho = np.corrcoef(xr.to_numpy(), yr.to_numpy())[0, 1]
        ax.text(
            0.02, 0.98,
            f"Spearman ρ ≈ {rho:.3f}  (all points)",
            transform=ax.transAxes,
            va="top",
            fontsize=10,
        )

    plt.tight_layout()
    if save_path is not None:
        file_path = save_path / "deltaP_vs_deltaE_scatter"
        plt.savefig(file_path)
    plt.show()


def load_stage03_table(path: Path) -> pd.DataFrame:
    """Load Stage03 results (CSV/Parquet) and normalize key dtypes."""
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
# Baseline handling
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
        raise ValueError("No clean baseline row found with severity==0.")

    # Average numeric columns across all clean rows (robust)
    num_cols = d_clean.select_dtypes(include=[np.number]).columns.tolist()
    non_num_cols = [c for c in d_clean.columns if c not in num_cols]

    clean_row = pd.DataFrame([d_clean[num_cols].mean(numeric_only=True)])
    clean_row[severity_col] = 0
    clean_row[corruption_col] = corruption

    # keep non-numeric fields from first clean row (optional)
    for c in non_num_cols:
        if c not in (severity_col, corruption_col):
            clean_row[c] = d_clean[c].iloc[0]

    out = pd.concat([clean_row, d_corr], ignore_index=True)
    out = out.sort_values(severity_col).drop_duplicates(subset=[severity_col], keep="first")
    return out


# -------------------------
# Plotting
# -------------------------
def plot_faithfulness_vs_severity(
    df: pd.DataFrame,
    corruption: str,
    faith_col: str = "faithfulness__corr",  # your Stage03 column
    severities: Iterable[int] = (0, 1, 2, 3, 5),
    overlay_acc: bool = True,
    acc_col: str = "acc_corr",
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot Quantus faithfulness correlation vs severity, including clean baseline severity=0.
    Optionally overlays accuracy on the left y-axis.
    """
    if faith_col not in df.columns:
        raise ValueError(
            f"Faithfulness column '{faith_col}' not found. "
            f"Available faith columns: {[c for c in df.columns if 'faith' in c.lower()]}"
        )

    d = attach_clean_baseline(df, corruption=corruption)
    d = d[d["severity"].isin(list(severities))].sort_values("severity")

    fig, ax1 = plt.subplots(figsize=(8.5, 4.8))

    # Left axis: accuracy (optional)
    if overlay_acc:
        if acc_col not in d.columns:
            raise ValueError(f"overlay_acc=True but '{acc_col}' not in df columns.")
        ax1.plot(d["severity"], d[acc_col], marker="o", label=acc_col)
        ax1.set_ylabel("Accuracy")
        ax2 = ax1.twinx()
        ax2.set_ylabel("FaithfulnessCorrelation")
    else:
        ax2 = ax1
        ax2.set_ylabel("FaithfulnessCorrelation")

    # Right axis: faithfulness
    ax2.plot(d["severity"], d[faith_col], marker="o", label=faith_col)

    ax1.set_xlabel("Severity")
    ax1.set_xticks(list(severities))

    if title is None:
        title = f"{corruption}: faithfulness vs severity (incl. clean sev0)"
    ax1.set_title(title)

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")

    ax1.grid(True, alpha=0.25)
    plt.tight_layout()


    if save_path is not None:
        save_path = Path(save_path)
        file_path = save_path / f"metric_vs_severity_faith_{corruption}"
        plt.savefig(file_path)
        plt.savefig(save_path, dpi=200)
    plt.show()


# ----------------------------
# Example usage
# ----------------------------

# Adjust to your actual file:
csv_path = Path("experiments/experiment__n250__IG__seed51/02__drift/02__drift_results.csv")
save_path = PATHS.results
df = load_results(csv_path)

# Plot 1 for one corruption
plot_acc_vs_expl_similarity(
    df,
    corruption="gaussian_noise",
    severities=(1, 2, 3, 5),
    expl_metric="rho_mean",   # or "cos_mean"
    save_path=save_path,
)

# Plot 2 (all corruptions together)
plot_deltaP_vs_deltaE_scatter(
    df,
    severities=(1, 2, 3, 5),
    x_proxy="one_minus_acc",  # or "mean_abs_delta_entropy"
    y_slice="inv",            # "all" | "inv" | "both_corr"
    expl_metric="rho_mean",   # or "cos_mean"
    save_path=save_path,
    compute_spearman=True,
)


stage03_file = PATHS.runs / "experiment__n250__IG__seed51" / "03__quantus" / "03__quantus_results.csv"  # <- adapt

df3 = load_stage03_table(stage03_file)

plot_faithfulness_vs_severity(
    df3,
    corruption="fog",
    faith_col="faithfulness__corr",
    overlay_acc=True,           # optional: show acc_corr too
    save_path=PATHS.results
)
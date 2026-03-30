"""
## Plot 1: spearman_entropy / trust_zones_stacked file

Description:

This analysis turns each sample into a simple “trust zone” based on two signals: did the model get the corrupted 
image correct? and did the explanation drift a lot? (high drift = top quantile). It then shows, per severity, what 
fraction of samples fall into four interpretable cases: Robust (correct + low drift), Silent Drift (correct + high 
drift: warning sign even though accuracy looks fine), Expected Failure (wrong + high drift: failure with an alarm), 
and Stubborn Failure (wrong + low drift: failure without an obvious drift signal). The stacked bar plot visualizes 
how these zones shift as corruption gets stronger, highlighting where drift provides useful warnings beyond 
accuracy alone.

3. The "Silent Failure" Quantile Plot
This categorizes your samples into four "Trust Zones."
Plot: A stacked bar chart per severity level showing the percentage of samples in these categories:
Robust: Correct Prediction + Low Drift (High Similarity).
Silent Drift: Correct Prediction + High Drift (The "Dangerous" zone).
Expected Failure: Incorrect Prediction + High Drift.
Stubborn Failure: Incorrect Prediction + Low Drift (The model is wrong but very confident/stable in its wrongness).
Value: This quantifies exactly how many "Correct" labels are actually "Right for the wrong reasons."

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_trust_zones_stacked(
    tab: pd.DataFrame,
    severities=(1, 2, 3, 5),
    title: str = "Silent Failure Trust Zones",
    file_path: str | Path | None = None,
):
    zones = ["Robust", "Silent Drift", "Expected Failure", "Stubborn Failure"]

    # pivot to severity x zone
    pivot = tab.pivot(index="severity", columns="zone", values="pct").reindex(severities)

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    bottom = np.zeros(len(severities), dtype=float)

    for z in zones:
        vals = pivot[z].to_numpy()
        ax.bar([str(s) for s in severities], vals, bottom=bottom, label=z)
        bottom += vals

    ax.set_xlabel("Severity")
    ax.set_ylabel("Share of samples (%)")
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    plt.tight_layout()

    if file_path is not None:
        plt.savefig(file_path, dpi=200)

    plt.show()


def trust_zone_table(
    df_pairs: pd.DataFrame,
    drift_col: str = "drift",
    correct_col: str = "correct_corr",
    severities=(1, 2, 3, 5),
    q: float = 0.75,                 # high drift = top 25%
    threshold_mode: str = "global",  # "global" or "per_severity"
    corruption: str | None = None,   # optionally filter one corruption
):
    d = df_pairs.copy()
    d = d[d["severity"].isin(severities)].dropna(subset=[drift_col, correct_col]).copy()
    if corruption is not None:
        d = d[d["corruption"] == corruption].copy()

    # threshold for "high drift"
    if threshold_mode == "global":
        thr = float(d[drift_col].quantile(q))
        d["high_drift"] = d[drift_col] >= thr
    elif threshold_mode == "per_severity":
        d["high_drift"] = False
        for sev, g in d.groupby("severity"):
            thr = float(g[drift_col].quantile(q))
            d.loc[g.index, "high_drift"] = g[drift_col] >= thr
    else:
        raise ValueError("threshold_mode must be 'global' or 'per_severity'")

    correct = d[correct_col].astype(bool)
    high = d["high_drift"].astype(bool)

    d["zone"] = np.select(
        [
            correct & (~high),
            correct & high,
            (~correct) & high,
            (~correct) & (~high),
        ],
        [
            "Robust",
            "Silent Drift",
            "Expected Failure",
            "Stubborn Failure",
        ],
        default="(unknown)",
    )

    tab = d.groupby(["severity", "zone"]).size().rename("n").reset_index()
    totals = d.groupby("severity").size().rename("N").reset_index()
    tab = tab.merge(totals, on="severity", how="left")
    tab["pct"] = tab["n"] / tab["N"] * 100.0

    # ensure all zones exist for all severities
    zones = ["Robust", "Silent Drift", "Expected Failure", "Stubborn Failure"]
    idx = pd.MultiIndex.from_product([severities, zones], names=["severity", "zone"])
    tab = tab.set_index(["severity", "zone"]).reindex(idx, fill_value=0).reset_index()
    return tab


def run_trust_zones_analysis(exp_dir, save_dir, severities, expl_metric="cos", corruption=None):

    # pairs_table_from_pt needs to be created
    # severity, corruption, table: pairs_table_from_pt.csv
    # just cosine?

    df_pairs = pd.read_csv(save_dir / "pairs_table_from_pt.csv")
    df_pairs["severity"] = df_pairs["severity"].astype(int)

    # drift from cosine similarity
    df_pairs["drift"] = 1.0 - df_pairs[expl_metric]

    # If you stored pred_corr + y_true:
    df_pairs["correct_corr"] = (df_pairs["pred_corr"] == df_pairs["y_true"])

    tab = trust_zone_table(
        df_pairs,
        drift_col="drift",
        correct_col="correct_corr",
        severities=severities,
        q=0.75,
        threshold_mode="global",
        corruption=corruption,   # or e.g. "brightness"
    )

    plot_trust_zones_stacked(
        tab,
        severities=severities,
        title=f"Trust Zones by severity (high drift = top 25% of 1−{expl_metric})",
        file_path=save_dir / "trust_zones_stacked.png",
    )

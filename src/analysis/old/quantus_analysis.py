import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Iterable, Optional, Sequence

from src.configs.global_config import PATHS


def plot_Q_vs_severity_with_acc_overlay(
    df: pd.DataFrame,
    q_col: str,                              # e.g. "faithfulness_corr__inv" or "avg_sensitivity__all"
    severities: Sequence[int] = (1, 2, 3, 4, 5),
    include_clean_baseline: bool = False,     # set True if you have sev=0 in df
    overlay_acc: bool = True,
    acc_col: str = "acc_corr",
    exclude_corruptions: Iterable[str] = ("clean",),
    title: Optional[str] = None,
    ylabel_q: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> None:
    """
    One line per corruption:
      x = severity
      y2 = Quantus score (q_col)
    Optional overlay:
      y1 = accuracy (acc_col) on the left axis (single curve = pooled over corruptions)
      OR you can overlay per-corruption accuracy by enabling it in the loop (see comment below).
    """
    if "corruption" not in df.columns or "severity" not in df.columns:
        raise ValueError("df must contain columns: 'corruption', 'severity'")

    if q_col not in df.columns:
        raise ValueError(f"q_col='{q_col}' not found. Available: {list(df.columns)[:50]} ...")

    df = df.copy()
    df["corruption"] = df["corruption"].astype(str)
    df["severity"] = pd.to_numeric(df["severity"], errors="raise").astype(int)

    fig, ax1 = plt.subplots(figsize=(9.8, 5.3))

    # Left axis: accuracy (optional)
    if overlay_acc:
        if acc_col not in df.columns:
            raise ValueError(f"overlay_acc=True but acc_col='{acc_col}' not found in df.")
        ax2 = ax1.twinx()
        ax1.set_ylabel("Accuracy")
        ax2.set_ylabel(ylabel_q or q_col)
    else:
        ax2 = ax1
        ax2.set_ylabel(ylabel_q or q_col)

    # Plot one line per corruption for Quantus metric
    for corr in sorted(df["corruption"].unique()):
        if corr in set(exclude_corruptions):
            continue

        d = df[df["corruption"] == corr].copy()
        d = d[d["severity"].isin(list(severities))].sort_values("severity")
        if d.empty:
            continue

        ax2.plot(d["severity"], d[q_col], marker="o", label=f"{corr}")

        # If you want per-corruption accuracy lines too, uncomment:
        # if overlay_acc:
        #     ax1.plot(d["severity"], d[acc_col], marker="x", linestyle=":", alpha=0.6)

    # Optional: overlay a single “mean accuracy across corruptions” curve
    if overlay_acc:
        d_acc = df[df["severity"].isin(list(severities))].copy()
        d_acc = d_acc.groupby("severity", as_index=False)[acc_col].mean(numeric_only=True).sort_values("severity")
        ax1.plot(d_acc["severity"], d_acc[acc_col], marker="s", linestyle="--", label="acc_corr (mean)")

    ax1.set_xlabel("Severity")
    ax1.set_xticks(list(severities))
    ax1.grid(True, alpha=0.25)

    if title is None:
        title = f"Q vs severity ({q_col})" + (" + accuracy overlay" if overlay_acc else "")
    ax1.set_title(title)

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best", ncol=2)

    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()


exp_dir = PATHS.runs / "experiment__n250__IG__seed51"
df_q = pd.read_csv(exp_dir / "03__quantus" / "03__quantus_results.csv")

plot_Q_vs_severity_with_acc_overlay(
    df_q,
    q_col="faithfulness_corr__inv",          # or avg_sensitivity__inv, sparseness__inv
    severities=(1,2,3,5),
    overlay_acc=True,
    acc_col="acc_corr",
    title="FaithfulnessCorrelation vs severity (slice=inv) + accuracy",
    save_path=PATHS.results / "quantus" / "plot_Q_faith_inv_vs_sev.png"
)
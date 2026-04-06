"""
RQ1 — Existiert Explanation Shift systematisch?
================================================

Aggregiert die pro Seed und Explainer gespeicherten Drift-Ergebnisse
(experiments/experiment__n{N}__{EXPLAINER}__seed{SEED}/02__drift/02__drift_results.csv)
und erzeugt zwei Figures:

Figure 1 — Primary: Spearman 1 - ρ vs. Severity, eine Linie pro Korruption,
           ein Panel pro Explainer (GradCAM, IG).
           Beantwortet: tritt Drift auf, skaliert sie, ist sie korruption-spezifisch?

Figure 2 — Vergleich der drei Ähnlichkeitsmaße (Spearman, Cosine, Top-k IoU).
           3 Spalten (Maße) × N Zeilen (Explainer). Zeigt, ob das Drift-Muster
           robust gegenüber der Wahl des Similarity-Maßes ist.

Aggregation erfolgt über Seeds: pro (explainer, corruption, severity) werden die
per-Seed sample-means gemittelt und deren Seed-Streuung als SD ausgewiesen
(als Shaded Band in den Plots).

Alle Slices werden geplottet: `all` (alle Samples), `inv` (invariant predictions),
`both_corr` (both correct clean & shifted).
"""

from pathlib import Path
from typing import Iterable
import matplotlib.pyplot as plt
import pandas as pd

from analysis.analysis_helper import (
    DRIFT_LABELS, 
    PRIMARY_MEASURE, 
    SIM_MEASURES, 
    SLICES, 
    corruption_palette, 
    draw_corruption_lines, 
    similarity_to_drift_agg, 
    x_label, 
    load_drift_results
)



# ----------------------------------------------------------------------------
# Aggregation across seeds
# ----------------------------------------------------------------------------

def aggregate_over_seeds(df: pd.DataFrame, slice_key: str = "all") -> pd.DataFrame:
    """For a given slice (`all` / `inv` / `both_corr`), compute mean and SD
    across seeds for Spearman ρ, Cosine, Top-k IoU (and the corresponding
    confidence shift metrics for future use).

    SD is the between-seed standard deviation of per-seed sample-means —
    i.e. it reflects the reproducibility of the drift measurement, not
    the within-seed sample variability.
    """
    if slice_key not in SLICES:
        raise ValueError(f"Unknown slice {slice_key!r}, must be one of {list(SLICES)}")

    exp_prefix = f"exp_{slice_key}__"
    conf_prefix = f"conf_{slice_key}__"

    rename = {
        f"{exp_prefix}rho_mean": "rho",
        f"{exp_prefix}cos_mean": "cos",
        f"{exp_prefix}iou_mean": "iou",
        f"{conf_prefix}p_shift_mean": "p_shift",
        f"{conf_prefix}margin_shift_mean": "margin_shift",
    }
    missing = [c for c in rename if c not in df.columns]
    if missing:
        raise KeyError(f"Columns missing for slice {slice_key!r}: {missing}")

    sub = df[
        ["explainer", "corruption", "severity", "seed",
         "max_mean_discrepancy", "mean_abs_delta_entropy",
         *rename.keys()]
    ].rename(columns=rename)

    group = sub.groupby(["explainer", "corruption", "severity"], as_index=False)
    agg = group.agg(
        rho_mean=("rho", "mean"),
        rho_sd=("rho", "std"),
        cos_mean=("cos", "mean"),
        cos_sd=("cos", "std"),
        iou_mean=("iou", "mean"),
        iou_sd=("iou", "std"),
        p_shift_mean=("p_shift", "mean"),
        p_shift_sd=("p_shift", "std"),
        margin_shift_mean=("margin_shift", "mean"),
        margin_shift_sd=("margin_shift", "std"),
        mmd_mean=("max_mean_discrepancy", "mean"),
        mmd_sd=("max_mean_discrepancy", "std"),
        dE_abs_mean=("mean_abs_delta_entropy", "mean"),
        dE_abs_sd=("mean_abs_delta_entropy", "std"),
        n_seeds=("seed", "nunique"),
    )

    # fillna(0) on SDs makes plotting bands degenerate-safe if only 1 seed present.
    sd_cols = [c for c in agg.columns if c.endswith("_sd")]
    agg[sd_cols] = agg[sd_cols].fillna(0.0)
    return agg.sort_values(["explainer", "corruption", "severity"]).reset_index(drop=True)


# ----------------------------------------------------------------------------
# Figure 1 — Primary (1 - Spearman) vs severity, panel per explainer
# ----------------------------------------------------------------------------
def plot_figure1(
    agg: pd.DataFrame,
    slice_key: str,
    x_axis: str = "severity",
    output_path: str | Path | None = None,
) -> plt.Figure:
    agg_plot = similarity_to_drift_agg(agg)

    explainers = sorted(agg_plot["explainer"].unique())
    palette = corruption_palette(agg_plot["corruption"].unique())

    fig, axes = plt.subplots(
        1, len(explainers),
        figsize=(5.2 * len(explainers), 4.3),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]

    for ax, explainer in zip(axes, explainers):
        panel = agg_plot[agg_plot["explainer"] == explainer]
        draw_corruption_lines(ax, panel, PRIMARY_MEASURE, x_axis, palette)
        ax.set_title(explainer, fontsize=12, fontweight="bold")
        ax.set_xlabel(x_label(x_axis))
        ax.grid(True, alpha=0.3, linewidth=0.6)
        ax.set_ylim(0, 1.02)
        if x_axis == "severity":
            sevs = sorted(agg_plot["severity"].unique())
            ax.set_xticks(sevs)

    axes[0].set_ylabel(r"$\Delta E = 1 - \rho$")

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=min(len(labels), 6),
        frameon=False,
        bbox_to_anchor=(0.5, -0.05),
        fontsize=9,
    )
    fig.suptitle(
        f"Figure 1 — Explanation drift vs. severity  ·  slice: {SLICES[slice_key]}",
        y=1.02, fontsize=13,
    )
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=200)
        fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=200)
    return fig


# ----------------------------------------------------------------------------
# Figure 2 — three similarity measures side by side, per explainer row
# ----------------------------------------------------------------------------

def plot_figure2(
    agg: pd.DataFrame,
    slice_key: str,
    x_axis: str = "severity",
    output_path: str | Path | None = None,
) -> plt.Figure:
    agg_plot = similarity_to_drift_agg(agg)

    explainers = sorted(agg_plot["explainer"].unique())
    palette = corruption_palette(agg_plot["corruption"].unique())

    n_rows = len(explainers)
    n_cols = len(SIM_MEASURES)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.4 * n_cols, 3.8 * n_rows),
        sharex=True,
        sharey="col",
        squeeze=False,
    )

    for i, explainer in enumerate(explainers):
        panel = agg_plot[agg_plot["explainer"] == explainer]
        for j, (key, label) in enumerate(SIM_MEASURES):
            ax = axes[i, j]
            draw_corruption_lines(ax, panel, key, x_axis, palette)
            ax.grid(True, alpha=0.3, linewidth=0.6)
            ax.set_ylim(0, 1.02)
            if i == 0:
                ax.set_title(DRIFT_LABELS.get(key, label), fontsize=12, fontweight="bold")
            if i == n_rows - 1:
                ax.set_xlabel(x_label(x_axis))
                if x_axis == "severity":
                    ax.set_xticks(sorted(agg_plot["severity"].unique()))
            if j == 0:
                ax.set_ylabel(f"{explainer}\ndrift", fontsize=11)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=min(len(labels), 6),
        frameon=False,
        bbox_to_anchor=(0.5, -0.03),
        fontsize=9,
    )
    fig.suptitle(
        f"Figure 2 — Robustness across drift measures  ·  slice: {SLICES[slice_key]}",
        y=1.01, fontsize=13,
    )
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=200)
        fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=200)
    return fig


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def run_explanation_shift_analysis(
    experiments_dir: str | Path,
    output_root: str | Path,
    n: int | None = 1000,
    x_axis: str = "severity",
    slices: Iterable[str] = tuple(SLICES.keys()),
) -> None:
    output_root = Path(output_root)
    output_dir = output_root / "explanation_shift"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_drift_results(experiments_dir, n=n)

    for slice_key in slices:
        agg = aggregate_over_seeds(df, slice_key=slice_key)
        agg.to_csv(output_dir / f"aggregated__{slice_key}.csv", index=False)

        plot_figure1(
            agg, slice_key=slice_key, x_axis=x_axis,
            output_path=output_dir / f"fig1_primary_spearman__{slice_key}.pdf",
        )
        plot_figure2(
            agg, slice_key=slice_key, x_axis=x_axis,
            output_path=output_dir / f"fig2_similarity_comparison__{slice_key}.pdf",
        )
        plt.close("all")
        print(f"[{slice_key:10s}] figures + aggregated CSV written to {output_dir}")


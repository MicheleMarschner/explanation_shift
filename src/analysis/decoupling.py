from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from analysis.analysis_helper import (
    DP_OPTIONS,
    SIMILARITY_SPECS,
    SLICE_STYLES,
    SLICES,
    corruption_label,
    corruption_palette,
    load_drift_results,
)


# ----------------------------------------------------------------------------
# Core derivation: attach delta_e / delta_p columns per seed-level row
# ----------------------------------------------------------------------------

def compute_deltas(
    df: pd.DataFrame,
    de: str = "rho",
    dp: str = "flip_rate",
    slice_key: str = "all",
) -> pd.DataFrame:
    """Return a copy of ``df`` with ``delta_e`` and ``delta_p`` columns derived
    according to the chosen operationalisation.

    * ``delta_e`` is evaluated on the requested slice
      (so one dataframe answers "ΔE on both_corr").
    * ``delta_p`` is always evaluated on the full population (slice=all),
      because it is a single global performance signal — slicing it defeats
      the purpose of the decoupling argument.
    """
    if de not in SIMILARITY_SPECS:
        raise ValueError(f"unknown --de {de!r}, choose from {list(SIMILARITY_SPECS)}")
    if dp not in DP_OPTIONS:
        raise ValueError(f"unknown --dp {dp!r}, choose from {list(DP_OPTIONS)}")
    if slice_key not in SLICES:
        raise ValueError(f"unknown slice {slice_key!r}")

    out = df.copy()

    # ΔE on the requested slice
    exp_col = f"exp_{slice_key}__{de}_mean"
    if exp_col not in out.columns:
        raise KeyError(f"missing column {exp_col!r} in drift results")
    out["delta_e"] = 1.0 - out[exp_col]

    # ΔP always evaluated population-wide
    if dp == "flip_rate":
        out["delta_p"] = 1.0 - out["invariant_rate"]
    elif dp == "err_rate":
        out["delta_p"] = 1.0 - out["both_correct_rate"]
    else:
        col = DP_OPTIONS[dp][1]
        if col not in out.columns:
            raise KeyError(f"missing column {col!r} for --dp {dp}")
        # make ΔP non-negative for scatter intuition
        out["delta_p"] = out[col].abs()

    return out


# ----------------------------------------------------------------------------
# Figure 3 — ΔE across the three slices, per (explainer, corruption) panel
# ----------------------------------------------------------------------------

def plot_explanation_drift_slices(
    df: pd.DataFrame,
    de: str = "rho",
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Lines-per-slice, grid = explainers × corruptions, x = severity."""
    # Build a long-format table with delta_e computed on each slice
    chunks = []
    for slice_key in SLICES:
        sub = compute_deltas(df, de=de, dp="flip_rate", slice_key=slice_key)
        agg = (
            sub.groupby(["explainer", "corruption", "severity"], as_index=False)
               .agg(delta_e_mean=("delta_e", "mean"),
                    delta_e_sd=("delta_e", "std"),
                    n_seeds=("seed", "nunique"))
        )
        agg["slice"] = slice_key
        chunks.append(agg)
    full = pd.concat(chunks, ignore_index=True)
    full["delta_e_sd"] = full["delta_e_sd"].fillna(0.0)

    explainers = sorted(full["explainer"].unique())
    corruptions = sorted(full["corruption"].unique())
    severities = sorted(full["severity"].unique())
    n_rows, n_cols = len(explainers), len(corruptions)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.4 * n_cols, 3.1 * n_rows),
        sharex=True, sharey=True, squeeze=False,
    )

    for i, explainer in enumerate(explainers):
        for j, corruption in enumerate(corruptions):
            ax = axes[i, j]
            for slice_key, style in SLICE_STYLES.items():
                s = full[
                    (full["explainer"] == explainer)
                    & (full["corruption"] == corruption)
                    & (full["slice"] == slice_key)
                ].sort_values("severity")
                if s.empty:
                    continue
                x = s["severity"].to_numpy()
                y = s["delta_e_mean"].to_numpy()
                err = s["delta_e_sd"].to_numpy()
                ax.plot(x, y, linewidth=1.8, **{k: v for k, v in style.items() if k != "label"},
                        label=style["label"] if (i == 0 and j == 0) else None)
                ax.fill_between(x, y - err, y + err, color=style["color"], alpha=0.12,
                                linewidth=0)
            if i == 0:
                ax.set_title(corruption_label(corruption), fontsize=11, fontweight="bold")
            if i == n_rows - 1:
                ax.set_xlabel("Severity")
                ax.set_xticks(severities)
            if j == 0:
                ax.set_ylabel(f"{explainer}\n{SIMILARITY_SPECS[de]['drift_axis']}", fontsize=10, linespacing=1.6)
            ax.grid(True, alpha=0.3, linewidth=0.6)

    # Global y-limits: 0 to (max + headroom), propagated via sharey.
    y_hi = (full["delta_e_mean"] + full["delta_e_sd"]).max()
    axes[0, 0].set_ylim(0, y_hi * 1.08)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=len(SLICES), frameon=False,
        bbox_to_anchor=(0.5, -0.06), fontsize=10,
    )
    fig.suptitle(
        "Explanation drift across slices",
        y=1.01, fontsize=12,
    )
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=200)
        fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=200)
    return fig


# ----------------------------------------------------------------------------
# Figure 4 — ΔP vs. ΔE scatter, one point per (corruption × severity × seed)
# ----------------------------------------------------------------------------

def plot_deltaP_deltaE_scatter(
    df: pd.DataFrame,
    de: str = "rho",
    dp: str = "flip_rate",
    slice_key: str = "all",
    output_path: str | Path | None = None,
) -> plt.Figure:
    sub = compute_deltas(df, de=de, dp=dp, slice_key=slice_key)
    explainers = sorted(sub["explainer"].unique())
    corruptions = sorted(sub["corruption"].unique())
    severities = sorted(sub["severity"].unique())
    palette = corruption_palette(corruptions)

    # Marker size encodes severity
    size_min, size_max = 30.0, 150.0
    if len(severities) > 1:
        sev_sizes = {
            s: size_min + (size_max - size_min) * i / (len(severities) - 1)
            for i, s in enumerate(severities)
        }
    else:
        sev_sizes = {severities[0]: (size_min + size_max) / 2}

    fig, axes = plt.subplots(
        1, len(explainers),
        figsize=(5.5 * len(explainers), 5.0),
        sharex=True, sharey=True, squeeze=False,
    )
    axes = axes[0]

    for ax, explainer in zip(axes, explainers):
        panel = sub[sub["explainer"] == explainer]

        # scatter per corruption × severity
        for corruption in corruptions:
            color = palette[corruption]
            for sev in severities:
                pts = panel[(panel["corruption"] == corruption)
                            & (panel["severity"] == sev)]
                if pts.empty:
                    continue
                ax.scatter(
                    pts["delta_p"], pts["delta_e"],
                    color=color, s=sev_sizes[sev],
                    alpha=0.72, edgecolor="white", linewidth=0.7,
                    label=corruption if sev == severities[0] else None,
                    zorder=3,
                )

        # Best-fit line + Pearson r for this panel
        if len(panel) >= 2 and panel["delta_p"].std() > 0 and panel["delta_e"].std() > 0:
            r = panel[["delta_p", "delta_e"]].corr().iloc[0, 1]
            slope, intercept = np.polyfit(panel["delta_p"], panel["delta_e"], deg=1)
            xs = np.linspace(panel["delta_p"].min(), panel["delta_p"].max(), 50)
            ax.plot(xs, slope * xs + intercept, color="#222", linewidth=1.2,
                    linestyle=":", zorder=2, alpha=0.8)
            ax.text(
                0.04, 0.96,
                f"Pearson $r$ = {r:.2f}\n$n$ = {len(panel)}",
                transform=ax.transAxes, va="top", ha="left", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.35",
                          facecolor="white", edgecolor="#cccccc", alpha=0.92),
            )

        ax.set_title(explainer, fontsize=12, fontweight="bold")
        ax.set_xlabel(f"ΔP = {DP_OPTIONS[dp][0]}")
        ax.grid(True, alpha=0.3, linewidth=0.6)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    axes[0].set_ylabel(SIMILARITY_SPECS[de]["drift_axis"])

    # Two legends: corruption colors + severity sizes
    corr_handles = [
        Line2D([], [], marker="o", linestyle="", color=palette[c],
               markersize=9, markeredgecolor="white", label=corruption_label(c))
        for c in corruptions
    ]
    sev_handles = [
        Line2D([], [], marker="o", linestyle="", color="gray",
               markersize=np.sqrt(sev_sizes[s]), markeredgecolor="white",
               label=f"Severity {s}")
        for s in severities
    ]
    leg1 = fig.legend(
        handles=corr_handles, loc="lower center",
        bbox_to_anchor=(0.3, -0.06), ncol=len(corruptions),
        frameon=False, fontsize=9,
    )
    fig.add_artist(leg1)
    fig.legend(
        handles=sev_handles, loc="lower center",
        bbox_to_anchor=(0.75, -0.06), ncol=len(severities),
        frameon=False, fontsize=9,
    )

    fig.suptitle(
        f"Performance drift vs. explanation drift  ·  slice: {SLICES[slice_key]}  ·  one marker per seed",
        y=1.02, fontsize=12,
    )
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=200)
        fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=200)
    return fig

# ----------------------------------------------------------------------------
# Aggregated CSV export (for later stats / tables)
# ----------------------------------------------------------------------------

def export_aggregated_csv(
    df: pd.DataFrame,
    output_dir: Path,
    de: str,
    dp: str,
) -> None:
    """Write one CSV per slice with mean/SD of delta_e, delta_p across seeds."""
    for slice_key in SLICES:
        sub = compute_deltas(df, de=de, dp=dp, slice_key=slice_key)
        agg = (
            sub.groupby(["explainer", "corruption", "severity"], as_index=False)
               .agg(
                   delta_e_mean=("delta_e", "mean"),
                   delta_e_sd=("delta_e", "std"),
                   delta_p_mean=("delta_p", "mean"),
                   delta_p_sd=("delta_p", "std"),
                   n_seeds=("seed", "nunique"),
               )
               .fillna(0.0)
        )
        agg["slice"] = slice_key
        agg["de_metric"] = de
        agg["dp_metric"] = dp
        agg.to_csv(output_dir / f"decoupling_deltas__{slice_key}.csv", index=False)


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def run_decoupling_analysis(
    experiments_dir: str | Path,
    output_root: str | Path,
    n: int | None = 1000,
    de: str = "rho",
    dp: str = "flip_rate",
    scatter_slice: str = "all",
) -> None:
    output_root = Path(output_root)
    output_dir = output_root / "decoupling"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_drift_results(experiments_dir, n=n)

    plot_explanation_drift_slices(
        df, de=de,
        output_path=output_dir / f"fig3_de_across_slices_{de}.pdf",
    )
    plot_deltaP_deltaE_scatter(
        df, de=de, dp=dp, slice_key=scatter_slice,
        output_path=output_dir / f"fig4_dp_vs_de_scatter_{scatter_slice}_{dp}_{de}.pdf",
    )
    export_aggregated_csv(df, output_dir, de=de, dp=dp)

    plt.close("all")
    print(f"RQ2 figures + CSVs written to {output_dir}")




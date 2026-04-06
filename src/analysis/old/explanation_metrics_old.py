"""
RQ3 — Trackt ΔQ das ΔE?
=======================

Reads the per-experiment Quantus result CSVs

    experiments/experiment__n{N}__{EXPLAINER}__seed{SEED}/03__quantus/
        03__quantus_results__clean.csv   (slices: only __all)
        03__quantus_results__corr.csv    (slices: __all, __inv, __both_corr)

and produces three figures that sit next to the RQ1/RQ2 outputs:

Figure 6 — ΔQ vs. ΔE scatter, one panel per metric.
           One marker per (corruption × severity × seed).
           Pearson r per panel — if |r| is small, Q does not track E.

Figure 7 — ΔQ across the three slices (all / inv / both_corr), per metric.
           Analogous to RQ2 Figure 3 but for Q. Non-zero ΔQ on `both_corr`
           is the protocol-artefact signature: model behaviour was stable by
           construction, yet the quality metric moved.
          
! TODO: Drop
Figure 8 — Paired severity curves of ΔQ and ΔE, z-scored per explainer.
           Grid = explainers × corruptions, one figure per metric, two lines
           per panel. Makes tracking vs. non-tracking visually obvious.

ΔQ definition
-------------
Because the clean-baseline CSV only stores the `__all` slice, we define

    ΔQ[slice]  :=  | Q_shifted[slice]  −  Q_clean[all] |

for every slice. On `both_corr`, model behaviour is stable by construction,
so a non-zero ΔQ there is the protocol-artefact signature we care about.

Use ``--signed`` to switch from |ΔQ| to signed ΔQ = Q_shifted − Q_clean.
The scatter and paired-curves figures pair ΔQ with |ΔE| (which is already
non-negative), so |ΔQ| is the default for visual comparability.

Metrics that are all-NaN in the CSVs (e.g. an unfinished AvgSensitivity run)
are skipped with a warning instead of crashing.

Reuses ``load_drift_results``, ``SLICES``, ``corruption_palette`` from
``drift_analysis`` and ``compute_deltas`` from ``decoupling_analysis``.

Usage
-----
    python quantus_analysis.py \\
        --experiments-dir ./experiments \\
        --output-root ./analysis_output \\
        --n 1000 \\
        --de rho \\
        --scatter-slice all

    # Files land in ./analysis_output/explanation_shift/
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from explanation_shift import (
    SLICES,
    EXPERIMENT_DIR_RE,
    load_drift_results,
    corruption_palette,
)
from decoupling import (
    DE_OPTIONS,
    compute_deltas,
)


# ----------------------------------------------------------------------------
# Metric parsing / config
# ----------------------------------------------------------------------------

METRIC_COL_RE = re.compile(
    r"^(?P<category>[a-z]+)__(?P<name>[a-z0-9_]+)__(?P<slice>all|inv|both_corr)$"
)

CATEGORY_LABELS: dict[str, str] = {
    "faithfulness": "Faithfulness",
    "complexity":   "Complexity / Sparseness",
    "robustness":   "Robustness",
}

METRIC_LABELS: dict[str, str] = {
    "faithfulness__corr":          "Faithfulness correlation",
    "complexity__sparseness":      "Sparseness",
    "robustness__avg_sensitivity": "Avg. sensitivity",
}

def metric_display(cat: str, name: str) -> str:
    """Human-readable label for a (category, metric) pair."""
    key = f"{cat}__{name}"
    if key in METRIC_LABELS:
        return METRIC_LABELS[key]
    return f"{CATEGORY_LABELS.get(cat, cat.title())}: {name.replace('_', ' ')}"


SLICE_STYLES: dict[str, dict] = {
    "all":       dict(color="#444444", ls="-",  marker="o", label="all samples"),
    "inv":       dict(color="#1f77b4", ls="--", marker="s", label="invariant"),
    "both_corr": dict(color="#d62728", ls="-.", marker="^", label="both correct"),
}


# ----------------------------------------------------------------------------
# Loader
# ----------------------------------------------------------------------------

def _discover_metric_columns(df: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Return list of (category, name, slice) triples found in the CSV."""
    out: list[tuple[str, str, str]] = []
    for col in df.columns:
        m = METRIC_COL_RE.match(col)
        if m:
            out.append((m["category"], m["name"], m["slice"]))
    return out


def load_quantus_results(
    experiments_dir: str | Path,
    n: int | None = None,
) -> pd.DataFrame:
    """Walk experiments dir, load clean + corr CSVs, return one long frame
    with one row per (explainer, seed, corruption, severity, slice, category, metric)
    carrying ``Q_clean``, ``Q_shifted`` and ``delta_q`` (signed).

    Rows where a metric is NaN (either clean or shifted) are dropped.
    """
    experiments_dir = Path(experiments_dir)
    if not experiments_dir.is_dir():
        raise FileNotFoundError(f"Experiments directory not found: {experiments_dir}")

    rows: list[dict] = []
    skipped_metrics: set[str] = set()

    for sub in sorted(experiments_dir.iterdir()):
        if not sub.is_dir():
            continue
        m = EXPERIMENT_DIR_RE.match(sub.name)
        if m is None:
            continue
        if n is not None and int(m["n"]) != n:
            continue

        qdir = sub / "03__quantus"
        clean_csv = qdir / "03__quantus_results__clean.csv"
        corr_csv = qdir / "03__quantus_results__corr.csv"
        if not clean_csv.exists() or not corr_csv.exists():
            print(f"[warn] missing quantus CSV(s) in {qdir}")
            continue

        clean = pd.read_csv(clean_csv)
        corr = pd.read_csv(corr_csv)
        if clean.empty:
            print(f"[warn] empty clean CSV: {clean_csv}")
            continue

        explainer = m["explainer"]
        seed = int(m["seed"])

        # Clean baseline is a single row with slice=all only.
        clean_row = clean.iloc[0]

        for (cat, name, slice_key) in _discover_metric_columns(corr):
            shifted_col = f"{cat}__{name}__{slice_key}"
            clean_col = f"{cat}__{name}__all"  # baseline is always __all
            if clean_col not in clean_row.index:
                continue
            q_clean = clean_row[clean_col]
            if pd.isna(q_clean):
                skipped_metrics.add(f"{cat}__{name}")
                continue

            for _, corr_row in corr.iterrows():
                q_shift = corr_row[shifted_col]
                if pd.isna(q_shift):
                    continue
                rows.append({
                    "explainer": explainer,
                    "seed": seed,
                    "corruption": corr_row["corruption"],
                    "severity": int(corr_row["severity"]),
                    "slice": slice_key,
                    "category": cat,
                    "metric": name,
                    "q_clean": float(q_clean),
                    "q_shifted": float(q_shift),
                    "delta_q_signed": float(q_shift) - float(q_clean),
                    "delta_q_abs": abs(float(q_shift) - float(q_clean)),
                })

    if not rows:
        raise FileNotFoundError(
            f"No quantus rows could be loaded from {experiments_dir} (n={n})"
        )

    df = pd.DataFrame(rows)

    if skipped_metrics:
        print(f"[info] skipped all-NaN clean baselines: {sorted(skipped_metrics)}")

    # Report which (cat, name) survived
    surviving = (
        df[["category", "metric"]].drop_duplicates()
        .sort_values(["category", "metric"])
        .agg(lambda r: f"{r['category']}__{r['metric']}", axis=1)
        .tolist()
    )
    print(
        f"Loaded {len(df)} quantus rows | "
        f"explainers={sorted(df['explainer'].unique())} | "
        f"metrics={surviving}"
    )
    return df


# ----------------------------------------------------------------------------
# Merge ΔQ with ΔE (per-seed, per-condition, per-slice)
# ----------------------------------------------------------------------------

def merge_with_drift(
    quantus_df: pd.DataFrame,
    drift_df: pd.DataFrame,
    de: str = "rho",
) -> pd.DataFrame:
    """Attach the matching ΔE to each ΔQ row. ΔE is computed on the same slice
    as the ΔQ row, using the chosen similarity basis.
    """
    frames = []
    for slice_key in SLICES:
        sl_drift = compute_deltas(drift_df, de=de, dp="flip_rate", slice_key=slice_key)
        sl_drift = sl_drift[["explainer", "seed", "corruption", "severity", "delta_e"]]
        sl_quant = quantus_df[quantus_df["slice"] == slice_key]
        merged = sl_quant.merge(
            sl_drift,
            on=["explainer", "seed", "corruption", "severity"],
            how="left",
        )
        frames.append(merged)
    out = pd.concat(frames, ignore_index=True)
    n_missing = out["delta_e"].isna().sum()
    if n_missing:
        print(f"[warn] {n_missing} quantus rows could not be matched to a drift row")
        out = out.dropna(subset=["delta_e"])
    return out


# ----------------------------------------------------------------------------
# Figure 6 — ΔQ vs. ΔE scatter per metric
# ----------------------------------------------------------------------------

def plot_figure6_scatter(
    merged: pd.DataFrame,
    slice_key: str = "all",
    signed: bool = False,
    output_path: str | Path | None = None,
) -> plt.Figure:
    sub = merged[merged["slice"] == slice_key].copy()
    if sub.empty:
        raise ValueError(f"no rows for slice={slice_key!r}")

    dq_col = "delta_q_signed" if signed else "delta_q_abs"
    dq_symbol = r"$\Delta Q$" if signed else r"$|\Delta Q|$"

    explainers = sorted(sub["explainer"].unique())
    cat_metrics = (
        sub[["category", "metric"]]
        .drop_duplicates()
        .sort_values(["category", "metric"])
        .itertuples(index=False, name=None)
    )
    cat_metrics = list(cat_metrics)
    palette = corruption_palette(sub["corruption"].unique())
    severities = sorted(sub["severity"].unique())

    size_min, size_max = 26.0, 140.0
    if len(severities) > 1:
        sev_sizes = {
            s: size_min + (size_max - size_min) * i / (len(severities) - 1)
            for i, s in enumerate(severities)
        }
    else:
        sev_sizes = {severities[0]: (size_min + size_max) / 2}

    n_rows, n_cols = len(explainers), len(cat_metrics)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 4.0 * n_rows),
        squeeze=False,
    )

    for i, explainer in enumerate(explainers):
        for j, (cat, name) in enumerate(cat_metrics):
            ax = axes[i, j]
            panel = sub[
                (sub["explainer"] == explainer)
                & (sub["category"] == cat)
                & (sub["metric"] == name)
            ]
            for corruption, cdf in panel.groupby("corruption"):
                color = palette[corruption]
                for sev, sdf in cdf.groupby("severity"):
                    ax.scatter(
                        sdf["delta_e"], sdf[dq_col],
                        color=color, s=sev_sizes[sev], alpha=0.72,
                        edgecolor="white", linewidth=0.6, zorder=3,
                    )

            # Pearson r + best fit
            if len(panel) >= 2 and panel["delta_e"].std() > 0 and panel[dq_col].std() > 0:
                r = panel[["delta_e", dq_col]].corr().iloc[0, 1]
                slope, intercept = np.polyfit(panel["delta_e"], panel[dq_col], deg=1)
                xs = np.linspace(panel["delta_e"].min(), panel["delta_e"].max(), 50)
                ax.plot(xs, slope * xs + intercept, color="#222", linewidth=1.2,
                        linestyle=":", alpha=0.85, zorder=2)
                ax.text(
                    0.04, 0.96,
                    f"Pearson $r$ = {r:.2f}\n$n$ = {len(panel)}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.32",
                              facecolor="white", edgecolor="#cccccc", alpha=0.92),
                )

            if i == 0:
                ax.set_title(metric_display(cat, name), fontsize=11, fontweight="bold")
            if i == n_rows - 1:
                ax.set_xlabel(r"$\Delta E$ (explanation drift)")
            if j == 0:
                ax.set_ylabel(f"{explainer}\n{dq_symbol}", fontsize=10)
            ax.grid(True, alpha=0.3, linewidth=0.6)
            ax.set_xlim(left=0)

    corr_handles = [
        Line2D([], [], marker="o", linestyle="", color=palette[c],
               markersize=9, markeredgecolor="white", label=c)
        for c in sorted(sub["corruption"].unique())
    ]
    sev_handles = [
        Line2D([], [], marker="o", linestyle="", color="gray",
               markersize=np.sqrt(sev_sizes[s]), markeredgecolor="white",
               label=f"severity {s}")
        for s in severities
    ]
    leg1 = fig.legend(
        handles=corr_handles, loc="lower center",
        bbox_to_anchor=(0.5, -0.035), ncol=len(corr_handles),
        frameon=False, fontsize=9, title="Corruption",
    )
    fig.add_artist(leg1)
    fig.legend(
        handles=sev_handles, loc="lower center",
        bbox_to_anchor=(0.5, -0.095), ncol=len(sev_handles),
        frameon=False, fontsize=9, title="Severity",
    )

    fig.suptitle(
        f"Figure 6 — Does ΔQ track ΔE?  ·  slice: {SLICES[slice_key]}",
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
# Figure 7 — ΔQ across the three slices, per metric
# ----------------------------------------------------------------------------

def plot_figure7_slices(
    quantus_df: pd.DataFrame,
    signed: bool = False,
    output_path: str | Path | None = None,
) -> plt.Figure:
    dq_col = "delta_q_signed" if signed else "delta_q_abs"
    dq_symbol = r"$\Delta Q$" if signed else r"$|\Delta Q|$"

    agg = (
        quantus_df
        .groupby(["explainer", "category", "metric", "corruption", "severity", "slice"],
                 as_index=False)
        .agg(dq_mean=(dq_col, "mean"),
             dq_sd=(dq_col, "std"),
             n_seeds=("seed", "nunique"))
    )
    agg["dq_sd"] = agg["dq_sd"].fillna(0.0)

    explainers = sorted(agg["explainer"].unique())
    cat_metrics = list(
        agg[["category", "metric"]].drop_duplicates()
        .sort_values(["category", "metric"])
        .itertuples(index=False, name=None)
    )
    corruptions = sorted(agg["corruption"].unique())
    severities = sorted(agg["severity"].unique())

    # One figure: rows = (explainer, metric), cols = corruptions.
    # This is dense but keeps all info visible at once.
    n_rows = len(explainers) * len(cat_metrics)
    n_cols = len(corruptions)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.3 * n_cols, 2.7 * n_rows),
        sharex=True, squeeze=False,
    )

    row = 0
    for explainer in explainers:
        for (cat, name) in cat_metrics:
            # Per-row y-limits: unified across corruptions of this (explainer, metric).
            row_mask = (
                (agg["explainer"] == explainer)
                & (agg["category"] == cat)
                & (agg["metric"] == name)
            )
            row_data = agg[row_mask]
            if row_data.empty:
                row += 1
                continue
            y_lo_row = (row_data["dq_mean"] - row_data["dq_sd"]).min()
            y_hi_row = (row_data["dq_mean"] + row_data["dq_sd"]).max()
            if signed:
                pad = max(0.05 * (y_hi_row - y_lo_row), 1e-3)
                y_lo_row -= pad
                y_hi_row += pad
            else:
                y_lo_row = 0.0
                y_hi_row = max(y_hi_row * 1.08, 1e-3)

            for j, corruption in enumerate(corruptions):
                ax = axes[row, j]
                for slice_key, style in SLICE_STYLES.items():
                    s = row_data[
                        (row_data["corruption"] == corruption)
                        & (row_data["slice"] == slice_key)
                    ].sort_values("severity")
                    if s.empty:
                        continue
                    x = s["severity"].to_numpy()
                    y = s["dq_mean"].to_numpy()
                    err = s["dq_sd"].to_numpy()
                    ax.plot(x, y, linewidth=1.8,
                            **{k: v for k, v in style.items() if k != "label"})
                    ax.fill_between(x, y - err, y + err,
                                    color=style["color"], alpha=0.12, linewidth=0)
                if signed:
                    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.6)
                if row == 0:
                    ax.set_title(corruption, fontsize=10, fontweight="bold")
                if row == n_rows - 1:
                    ax.set_xlabel("Severity")
                    ax.set_xticks(severities)
                if j == 0:
                    ax.set_ylabel(
                        f"{explainer}\n{metric_display(cat, name)}\n{dq_symbol}",
                        fontsize=9,
                    )
                ax.grid(True, alpha=0.3, linewidth=0.6)
                ax.set_ylim(y_lo_row, y_hi_row)
            row += 1

    slice_handles = [
        Line2D([], [],
               color=style["color"], linestyle=style["ls"],
               marker=style["marker"], markersize=7, linewidth=1.8,
               label=style["label"])
        for style in SLICE_STYLES.values()
    ]
    fig.legend(
        handles=slice_handles,
        loc="lower center", ncol=len(SLICE_STYLES), frameon=False,
        bbox_to_anchor=(0.5, -0.015), fontsize=10,
    )
    fig.suptitle(
        "Figure 7 — ΔQ across slices  "
        "(non-zero on both-correct ⇒ protocol-artefact signature)",
        y=1.005, fontsize=12,
    )
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=200)
        fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=200)
    return fig


# !TODO: Drop
# ----------------------------------------------------------------------------
# Figure 8 — paired severity curves: ΔQ (per metric) vs. ΔE, z-scored
# ----------------------------------------------------------------------------

def plot_figure8_paired(
    merged: pd.DataFrame,
    slice_key: str = "all",
    signed: bool = False,
    output_path: str | Path | None = None,
) -> plt.Figure:
    sub = merged[merged["slice"] == slice_key].copy()
    if sub.empty:
        raise ValueError(f"no rows for slice={slice_key!r}")

    dq_col = "delta_q_signed" if signed else "delta_q_abs"

    agg = (
        sub.groupby(["explainer", "category", "metric", "corruption", "severity"],
                    as_index=False)
           .agg(dq=(dq_col, "mean"),
                dq_sd=(dq_col, "std"),
                de=("delta_e", "mean"),
                de_sd=("delta_e", "std"))
           .fillna({"dq_sd": 0.0, "de_sd": 0.0})
    )

    # z-score ΔE per explainer (global across corruption × severity), and
    # z-score ΔQ per (explainer, metric) — different metrics live on different
    # scales, so each metric gets its own standardisation.
    for explainer in agg["explainer"].unique():
        emask = agg["explainer"] == explainer
        mu_e, sd_e = agg.loc[emask, "de"].mean(), agg.loc[emask, "de"].std() or 1.0
        agg.loc[emask, "de_z"] = (agg.loc[emask, "de"] - mu_e) / sd_e
        agg.loc[emask, "de_z_sd"] = agg.loc[emask, "de_sd"] / sd_e

        for (cat, name) in agg[emask][["category", "metric"]].drop_duplicates().itertuples(index=False, name=None):
            mask = emask & (agg["category"] == cat) & (agg["metric"] == name)
            mu_q = agg.loc[mask, "dq"].mean()
            sd_q = agg.loc[mask, "dq"].std() or 1.0
            agg.loc[mask, "dq_z"] = (agg.loc[mask, "dq"] - mu_q) / sd_q
            agg.loc[mask, "dq_z_sd"] = agg.loc[mask, "dq_sd"] / sd_q

    explainers = sorted(agg["explainer"].unique())
    cat_metrics = list(
        agg[["category", "metric"]].drop_duplicates()
        .sort_values(["category", "metric"])
        .itertuples(index=False, name=None)
    )
    corruptions = sorted(agg["corruption"].unique())
    severities = sorted(agg["severity"].unique())

    n_rows = len(explainers) * len(cat_metrics)
    n_cols = len(corruptions)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.3 * n_cols, 2.6 * n_rows),
        sharex=True, sharey=True, squeeze=False,
    )

    dq_color, de_color = "#1f77b4", "#d62728"
    row = 0
    for explainer in explainers:
        for (cat, name) in cat_metrics:
            for j, corruption in enumerate(corruptions):
                ax = axes[row, j]
                s = agg[
                    (agg["explainer"] == explainer)
                    & (agg["category"] == cat)
                    & (agg["metric"] == name)
                    & (agg["corruption"] == corruption)
                ].sort_values("severity")
                if s.empty:
                    continue
                x = s["severity"].to_numpy()

                yq = s["dq_z"].to_numpy()
                yq_err = s["dq_z_sd"].to_numpy()
                ax.plot(x, yq, marker="s", ls="--", color=dq_color, linewidth=1.9,
                        label="ΔQ (z)")
                ax.fill_between(x, yq - yq_err, yq + yq_err,
                                color=dq_color, alpha=0.12, linewidth=0)

                ye = s["de_z"].to_numpy()
                ye_err = s["de_z_sd"].to_numpy()
                ax.plot(x, ye, marker="o", ls="-", color=de_color, linewidth=1.9,
                        label="ΔE (z)")
                ax.fill_between(x, ye - ye_err, ye + ye_err,
                                color=de_color, alpha=0.12, linewidth=0)

                ax.axhline(0, color="gray", linewidth=0.6, alpha=0.6)
                if row == 0:
                    ax.set_title(corruption, fontsize=10, fontweight="bold")
                if row == n_rows - 1:
                    ax.set_xlabel("Severity")
                    ax.set_xticks(severities)
                if j == 0:
                    ax.set_ylabel(
                        f"{explainer}\n{metric_display(cat, name)}\nz-score",
                        fontsize=9,
                    )
                ax.grid(True, alpha=0.3, linewidth=0.6)
            row += 1

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=2, frameon=False,
        bbox_to_anchor=(0.5, -0.015), fontsize=10,
    )
    fig.suptitle(
        "Figure 8 — Paired ΔQ & ΔE severity curves (z-scored)  ·  "
        f"slice: {SLICES[slice_key]}",
        y=1.005, fontsize=12,
    )
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=200)
        fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=200)
    return fig


# ----------------------------------------------------------------------------
# CSV export (aggregated ΔQ per slice, including matched ΔE for convenience)
# ----------------------------------------------------------------------------

def export_aggregated_csv(
    merged: pd.DataFrame,
    output_dir: Path,
) -> None:
    for slice_key in SLICES:
        sub = merged[merged["slice"] == slice_key]
        if sub.empty:
            continue
        agg = (
            sub.groupby(["explainer", "category", "metric",
                         "corruption", "severity"], as_index=False)
               .agg(
                   q_clean_mean=("q_clean", "mean"),
                   q_shifted_mean=("q_shifted", "mean"),
                   delta_q_signed_mean=("delta_q_signed", "mean"),
                   delta_q_signed_sd=("delta_q_signed", "std"),
                   delta_q_abs_mean=("delta_q_abs", "mean"),
                   delta_q_abs_sd=("delta_q_abs", "std"),
                   delta_e_mean=("delta_e", "mean"),
                   delta_e_sd=("delta_e", "std"),
                   n_seeds=("seed", "nunique"),
               )
               .fillna(0.0)
        )
        agg["slice"] = slice_key
        agg.to_csv(output_dir / f"quantus_deltas__{slice_key}.csv", index=False)


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def run_quantus_analysis(
    experiments_dir: str | Path,
    output_root: str | Path,
    n: int | None = 1000,
    de: str = "rho",
    scatter_slice: str = "all",
    signed: bool = False,
) -> None:
    if de not in DE_OPTIONS:
        raise ValueError(f"unknown --de {de!r}")
    if scatter_slice not in SLICES:
        raise ValueError(f"unknown --scatter-slice {scatter_slice!r}")

    output_root = Path(output_root)
    output_dir = output_root / "quantus"
    output_dir.mkdir(parents=True, exist_ok=True)

    drift_df = load_drift_results(experiments_dir, n=n)
    quantus_df = load_quantus_results(experiments_dir, n=n)
    merged = merge_with_drift(quantus_df, drift_df, de=de)

    plot_figure6_scatter(
        merged, slice_key=scatter_slice, signed=signed,
        output_path=output_dir / f"fig6_dq_vs_de_scatter_{scatter_slice}.pdf",
    )
    plot_figure7_slices(
        quantus_df, signed=signed,
        output_path=output_dir / "fig7_dq_across_slices.pdf",
    )
    plot_figure8_paired(
        merged, slice_key=scatter_slice, signed=signed,
        output_path=output_dir / f"fig8_paired_dq_de_severity_{scatter_slice}.pdf",
    )
    export_aggregated_csv(merged, output_dir)

    plt.close("all")
    print(f"RQ3 figures + CSVs written to {output_dir}")


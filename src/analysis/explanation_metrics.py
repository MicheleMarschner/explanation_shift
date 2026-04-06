import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from analysis.analysis_helper import (
    SLICES,
    EXPERIMENT_DIR_RE,
    SIMILARITY_SPECS,
    corruption_label,
    corruption_palette,
    filter_excluded_corruptions,
    load_drift_results,
)
from analysis.decoupling import compute_deltas


# ----------------------------------------------------------------------------
# Active metric config (single source of truth)
# ----------------------------------------------------------------------------

ACTIVE_QUANTUS_METRICS: list[tuple[str, str]] = [
    ("complexity", "sparseness"),
    ("faithfulness", "corr"),
]

SPARSENESS_METRIC: tuple[str, str] = ("complexity", "sparseness")


def _is_active_metric(category: str, metric: str) -> bool:
    return (category, metric) in ACTIVE_QUANTUS_METRICS


def filter_active_quantus_metrics(
    df: pd.DataFrame,
    sparseness_only: bool = False,
) -> pd.DataFrame:
    """Keep only active Quantus metrics."""
    allowed = [SPARSENESS_METRIC] if sparseness_only else ACTIVE_QUANTUS_METRICS

    mask = pd.Series(False, index=df.index)
    for category, metric in allowed:
        mask |= (df["category"] == category) & (df["metric"] == metric)

    return df[mask].copy()


def active_cat_metrics(
    df: pd.DataFrame,
    sparseness_only: bool = False,
) -> list[tuple[str, str]]:
    """Return active (category, metric) pairs in canonical order."""
    allowed = [SPARSENESS_METRIC] if sparseness_only else ACTIVE_QUANTUS_METRICS
    present = set(
        df[["category", "metric"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    return [cm for cm in allowed if cm in present]


# ----------------------------------------------------------------------------
# Metric parsing / labels
# ----------------------------------------------------------------------------

METRIC_COL_RE = re.compile(
    r"^(?P<category>[a-z]+)__(?P<name>[a-z0-9_]+)__(?P<slice>all|inv|both_corr)$"
)

CATEGORY_LABELS: dict[str, str] = {
    "faithfulness": "Faithfulness",
    "complexity": "Complexity / Sparseness",
}

METRIC_LABELS: dict[str, str] = {
    "faithfulness__corr": "Faithfulness correlation",
    "complexity__sparseness": "Sparseness",
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

    Only active metrics are loaded.
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
        clean_row = clean.iloc[0]  # clean baseline is a single row

        for (cat, name, slice_key) in _discover_metric_columns(corr):
            if not _is_active_metric(cat, name):
                continue

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
            f"No active quantus rows could be loaded from {experiments_dir} (n={n})"
        )

    df = pd.DataFrame(rows)
    df = filter_excluded_corruptions(df, column="corruption")

    if skipped_metrics:
        print(f"[info] skipped all-NaN clean baselines: {sorted(skipped_metrics)}")

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
    quantus_df = filter_active_quantus_metrics(quantus_df, sparseness_only=False)

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

def plot_metrics_drift_scatter(
    merged: pd.DataFrame,
    de: str = "rho",
    slice_key: str = "all",
    signed: bool = False,
    sparseness_only: bool = False,
    output_path: str | Path | None = None,
) -> plt.Figure:
    sub = merged[merged["slice"] == slice_key].copy()
    sub = filter_active_quantus_metrics(sub, sparseness_only=sparseness_only)
    if sub.empty:
        raise ValueError(f"no rows for slice={slice_key!r}")

    dq_col = "delta_q_signed" if signed else "delta_q_abs"
    dq_symbol = r"$\Delta Q$" if signed else r"$|\Delta Q|$"

    explainers = sorted(sub["explainer"].unique())
    cat_metrics = active_cat_metrics(sub, sparseness_only=sparseness_only)
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

    # ---- layout ----
    if sparseness_only and len(cat_metrics) == 1:
        n_rows, n_cols = 1, len(explainers)
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(4.8 * n_cols, 4.2),
            squeeze=False,
        )

        for j, explainer in enumerate(explainers):
            ax = axes[0, j]
            cat, name = cat_metrics[0]
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

            ax.set_title(explainer, fontsize=11, fontweight="bold")
            ax.set_xlabel(SIMILARITY_SPECS[de]["drift_axis"])
            if j == 0:
                ax.set_ylabel(f"{metric_display(cat, name)}\n{dq_symbol}", fontsize=10, linespacing=1.5, labelpad=10,)
            ax.grid(True, alpha=0.3, linewidth=0.6)
            ax.set_xlim(left=0)
            if signed:
                ax.axhline(0, color="gray", linewidth=0.5, alpha=0.6)
            else:
                ax.set_ylim(bottom=0)

    else:
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
                    ax.set_xlabel(SIMILARITY_SPECS[de]["drift_axis"], labelpad=6)
                if j == 0:
                    ax.set_ylabel(f"{explainer}\n{dq_symbol}", fontsize=10, linespacing=1.5, labelpad=6)
                ax.grid(True, alpha=0.3, linewidth=0.6)
                ax.set_xlim(left=0)
                if signed:
                    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.6)
                else:
                    ax.set_ylim(bottom=0)

    corr_handles = [
        Line2D([], [], marker="o", linestyle="", color=palette[c],
            markersize=9, markeredgecolor="white", label=corruption_label(c))
        for c in sorted(sub["corruption"].unique())
    ]
    sev_handles = [
        Line2D([], [], marker="o", linestyle="", color="gray",
            markersize=np.sqrt(sev_sizes[s]), markeredgecolor="white",
            label=f"Severity {s}")
        for s in severities
    ]

    legend_handles = corr_handles + sev_handles
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.045),
        ncol=len(legend_handles),
        frameon=False,
        fontsize=9,
    )

    fig_name = "Does metric change track explanation drift?"
    if sparseness_only:
        fig_name += " (Sparseness only)"

    fig.suptitle(
        f"{fig_name}  ·  slice: {SLICES[slice_key]}",
        y=1.0, fontsize=12,
    )
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=200)
        fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=200)
    return fig


def plot_metrics_drift_scatter_sparseness(
    merged: pd.DataFrame,
    de: str = "rho",
    slice_key: str = "both_corr",
    signed: bool = False,
    output_path: str | Path | None = None,
) -> plt.Figure:
    return plot_metrics_drift_scatter(
        merged=merged,
        de=de,
        slice_key=slice_key,
        signed=signed,
        sparseness_only=True,
        output_path=output_path,
    )


# ----------------------------------------------------------------------------
# Figure 7 — ΔQ across the three slices, per metric
# ----------------------------------------------------------------------------

def plot_metric_change_slices(
    quantus_df: pd.DataFrame,
    signed: bool = False,
    sparseness_only: bool = False,
    output_path: str | Path | None = None,
) -> plt.Figure:
    quantus_df = filter_active_quantus_metrics(quantus_df, sparseness_only=sparseness_only)
    if quantus_df.empty:
        raise ValueError("no rows left for Figure 7 after filtering metrics")

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
    cat_metrics = active_cat_metrics(agg, sparseness_only=sparseness_only)
    corruptions = sorted(agg["corruption"].unique())
    severities = sorted(agg["severity"].unique())

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
                    ax.set_title(corruption_label(corruption), fontsize=10, fontweight="bold")
                if row == n_rows - 1:
                    ax.set_xlabel("Severity")
                    ax.set_xticks(severities)
                if j == 0:
                    ax.set_ylabel(
                        f"{explainer}\n{metric_display(cat, name)} {dq_symbol}",
                        fontsize=9,
                        linespacing=1.4,
                        labelpad=6,
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
        bbox_to_anchor=(0.5, -0.03), fontsize=10,
    )

    fig_name = "Metric change across slices"
    if sparseness_only:
        fig_name += " (Sparseness only)"

    fig.suptitle(
        fig_name,
        y=1.005, fontsize=12,
    )
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=200)
        fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=200)
    return fig


def plot_metric_change_slices_sparseness(
    quantus_df: pd.DataFrame,
    signed: bool = False,
    output_path: str | Path | None = None,
) -> plt.Figure:
    return plot_metric_change_slices(
        quantus_df=quantus_df,
        signed=signed,
        sparseness_only=True,
        output_path=output_path,
    )


# ----------------------------------------------------------------------------
# CSV export (aggregated ΔQ per slice, including matched ΔE for convenience)
# ----------------------------------------------------------------------------

def export_aggregated_csv(
    merged: pd.DataFrame,
    output_dir: Path,
) -> None:
    merged = filter_active_quantus_metrics(merged, sparseness_only=False)

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
    if de not in SIMILARITY_SPECS:
        raise ValueError(f"unknown --de {de!r}")
    if scatter_slice not in SLICES:
        raise ValueError(f"unknown --scatter-slice {scatter_slice!r}")

    output_root = Path(output_root)
    output_dir = output_root / "quantus"
    output_dir.mkdir(parents=True, exist_ok=True)

    drift_df = load_drift_results(experiments_dir, n=n)
    quantus_df = load_quantus_results(experiments_dir, n=n)
    merged = merge_with_drift(quantus_df, drift_df, de=de)

    # Standard outputs
    plot_metrics_drift_scatter(
        merged, de=de, slice_key=scatter_slice, signed=signed,
        output_path=output_dir / f"fig6_dq_vs_de_scatter_{scatter_slice}.pdf",
    )
    plot_metric_change_slices(
        quantus_df, signed=signed,
        output_path=output_dir / "fig7_dq_across_slices.pdf",
    )

    plot_metrics_drift_scatter_sparseness(
        merged=merged,
        de=de,
        slice_key="both_corr",
        signed=signed,
        output_path=output_dir / "fig6_sparseness_both_corr.pdf",
    )
    plot_metric_change_slices_sparseness(
        quantus_df=quantus_df,
        signed=signed,
        output_path=output_dir / "fig7_sparseness_across_slices.pdf",
    )

    export_aggregated_csv(merged, output_dir)

    plt.close("all")
    print(f"RQ3 figures + CSVs written to {output_dir}")
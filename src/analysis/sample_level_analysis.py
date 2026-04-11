import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore
    _HAS_TORCH = False

from analysis.analysis_helper import (
    CORRELATION_METRICS,
    EXPERIMENT_DIR_RE,
    SIMILARITY_SPECS,
    SLICES,
    TENSOR_KEYS,
    ZONE_COLORS,
    ZONE_ORDER,
    corruption_label,
    corruption_palette,
    EXCLUDED_CORRUPTIONS,
    filter_excluded_corruptions,
)


def _augment_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns (1 − sim, |ΔH|) that the heatmap consumes."""
    out = df.copy()
    out["drift_1m_cos"] = (1.0 - out["cos"]).clip(lower=0.0)
    out["drift_1m_iou"] = (1.0 - out["iou"]).clip(lower=0.0)
    out["drift_1m_rho"] = (1.0 - out["rho"]).clip(lower=0.0)
    out["abs_dH"] = out["dH"].abs()
    return out

# ----------------------------------------------------------------------------
# Loader — walks .pt payloads into a long DataFrame
# ----------------------------------------------------------------------------

_DRIFT_FILE_RE = re.compile(
    r"^02__drift__(?P<corruption>[a-z_]+)__sev(?P<sev>\d+)\.pt$"
)


def _tensor_to_numpy(t) -> np.ndarray:
    """torch.Tensor / numpy / list → 1-D numpy array."""
    if _HAS_TORCH and isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy().ravel()
    return np.asarray(t).ravel()


def _load_payload(path: Path):
    """Load a drift payload file. Uses torch.load if torch is available
    (needed for real .pt files saved via torch.save); falls back to pickle
    otherwise (sufficient for payloads that were saved as pure-python dicts
    of numpy arrays)."""
    if _HAS_TORCH:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            # older torch without weights_only kw
            return torch.load(path, map_location="cpu")
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def _unwrap_payload(payload, wanted_keys: Iterable[str]) -> dict | None:
    """Return the inner dict that actually contains ``wanted_keys``.

    Drift payloads produced by the pipeline come in two shapes:
        {<tensor_keys>...}                              — flat
        {"vectors": {<tensor_keys>...}, "meta": {...}}  — nested

    Clean-reference payloads use yet another nesting:
        {"seed": ..., "clean_reference": {"proba_clean": ..., ...}, ...}

    This helper looks at the top level first, then tries common wrapper
    keys (``vectors``, ``row``, ``clean_reference``, ``data``, ``payload``)
    and returns whichever dict actually holds at least one of
    ``wanted_keys``. Returns ``None`` if nothing matches.
    """
    if not isinstance(payload, dict):
        return None
    wanted = set(wanted_keys)
    if wanted & set(payload.keys()):
        return payload
    for wrapper in ("vectors", "row", "clean_reference", "corrupt_reference", "data", "payload"):
        inner = payload.get(wrapper)
        if isinstance(inner, dict) and (wanted & set(inner.keys())):
            return inner
    return None


def load_sample_level_drift(
    experiments_dir: str | Path,
    n: int | None = None,
) -> pd.DataFrame:
    """Walk all drift .pt payloads and return a long DataFrame with one row
    per (explainer, seed, corruption, severity, sample_idx).

    Columns: explainer, seed, corruption, severity, sample_idx,
             invariant, both_correct, dH, rho, cos, iou, p_shift, margin_shift.
    """
    experiments_dir = Path(experiments_dir)
    if not experiments_dir.is_dir():
        raise FileNotFoundError(f"Experiments directory not found: {experiments_dir}")

    rows: list[pd.DataFrame] = []
    for sub in sorted(experiments_dir.iterdir()):
        if not sub.is_dir():
            continue
        m = EXPERIMENT_DIR_RE.match(sub.name)
        if m is None:
            continue
        if n is not None and int(m["n"]) != n:
            continue
        explainer = m["explainer"]
        seed = int(m["seed"])

        drift_dir = sub / "02__drift"
        if not drift_dir.is_dir():
            continue

        for pt_file in sorted(drift_dir.iterdir()):
            mm = _DRIFT_FILE_RE.match(pt_file.name)
            if mm is None:
                continue
            corruption = mm["corruption"]
            severity = int(mm["sev"])

            if corruption in EXCLUDED_CORRUPTIONS:
                continue

            payload = _load_payload(pt_file)

            inner = _unwrap_payload(payload, TENSOR_KEYS)
            if inner is None:
                top_keys = list(payload.keys()) if isinstance(payload, dict) else "(not a dict)"
                print(f"[warn] {pt_file.name}: no tensor keys found. "
                      f"Top-level keys: {top_keys}")
                continue
            missing = [k for k in TENSOR_KEYS if k not in inner]
            if missing:
                print(f"[warn] {pt_file.name} missing keys: {missing}")
                continue

            arrs = {k: _tensor_to_numpy(inner[k]) for k in TENSOR_KEYS}
            length = len(arrs["invariant"])
            if any(len(a) != length for a in arrs.values()):
                print(f"[warn] inconsistent tensor lengths in {pt_file.name}, skipping")
                continue

            df = pd.DataFrame({
                "explainer":    explainer,
                "seed":         seed,
                "corruption":   corruption,
                "severity":     severity,
                "sample_idx":   np.arange(length),
                "invariant":    arrs["invariant"].astype(bool),
                "both_correct": arrs["both_correct"].astype(bool),
                "dH":           arrs["dH"].astype(float),
                "rho":          arrs["exp__spearman_rho"].astype(float),
                "cos":          arrs["exp__cosine_sim"].astype(float),
                "iou":          arrs["exp__iou_topk"].astype(float),
                "p_shift":      arrs["conf__p_shift_abs"].astype(float),
                "margin_shift": arrs["conf__margin_shift_abs"].astype(float),
            })
            rows.append(df)

    if not rows:
        raise FileNotFoundError(
            f"No drift .pt payloads found under {experiments_dir} (n={n})"
        )
    
    out = pd.concat(rows, ignore_index=True)
    out = filter_excluded_corruptions(out, column="corruption")

    print(
        f"Loaded {len(out):,} sample-level rows | "
        f"explainers={sorted(out['explainer'].unique())} | "
        f"seeds={sorted(out['seed'].unique())} | "
        f"corruptions={sorted(out['corruption'].unique())} | "
        f"severities={sorted(out['severity'].unique())}"
    )
    return out


def slice_mask(df: pd.DataFrame, slice_key: str) -> pd.Series:
    """Boolean selector for the chosen slice."""
    if slice_key == "all":
        return pd.Series(True, index=df.index)
    if slice_key == "inv":
        return df["invariant"]
    if slice_key == "both_corr":
        return df["both_correct"]
    raise ValueError(f"unknown slice {slice_key!r}")


def add_delta_e(df: pd.DataFrame, basis: str = "cos") -> pd.DataFrame:
    """Attach a `delta_e` column based on the requested similarity basis."""
    if basis not in SIMILARITY_SPECS:
        raise ValueError(f"unknown --de {basis!r}, choose from {list(SIMILARITY_SPECS)}")
    sim_col = SIMILARITY_SPECS[basis]["tensor_col"]
    out = df.copy()
    out["delta_e"] = (1.0 - out[sim_col]).clip(lower=0.0)
    return out


# ----------------------------------------------------------------------------
# Figure 9 — Trust Zones
# ----------------------------------------------------------------------------

def compute_trust_zones(
    df: pd.DataFrame,
    de_basis: str = "cos",
    threshold_quantile: float = 0.75,
) -> tuple[pd.DataFrame, float]:
    """For every (explainer, corruption, severity, seed) compute the share of
    samples in each of the four zones. Returns (long_df, threshold_value).

    Drift threshold is a *global* quantile across all shifted samples (i.e.
    the same threshold for every severity), so sev=1 lands mostly in Robust
    while sev=5 shifts mass into Silent/Expected failures.
    """
    df = add_delta_e(df, basis=de_basis)
    thr = float(df["delta_e"].quantile(threshold_quantile))

    high_drift = df["delta_e"] >= thr
    correct = df["both_correct"]

    zone = np.where(
        correct & ~high_drift, "Robust",
        np.where(correct & high_drift, "Silent Drift",
                 np.where(~correct & high_drift, "Visible Failure",
                          "Hidden Failure"))
    )
    tmp = df[["explainer", "corruption", "severity", "seed"]].copy()
    tmp["zone"] = zone

    counts = (
        tmp.groupby(["explainer", "corruption", "severity", "seed", "zone"])
           .size().rename("n").reset_index()
    )
    totals = (
        tmp.groupby(["explainer", "corruption", "severity", "seed"])
           .size().rename("n_total").reset_index()
    )
    merged = counts.merge(totals,
                          on=["explainer", "corruption", "severity", "seed"])
    merged["share"] = merged["n"] / merged["n_total"]
    # Aggregate across seeds
    agg = (
        merged.groupby(["explainer", "corruption", "severity", "zone"],
                       as_index=False)
              .agg(share_mean=("share", "mean"),
                   share_sd=("share", "std"),
                   n_seeds=("seed", "nunique"))
              .fillna({"share_sd": 0.0})
    )
    return agg, thr


def plot_trust_zones(
    df: pd.DataFrame,
    de_basis: str = "cos",
    threshold_quantile: float = 0.75,
    output_path: str | Path | None = None,
) -> plt.Figure:
    agg, thr = compute_trust_zones(df, de_basis=de_basis,
                                   threshold_quantile=threshold_quantile)

    explainers = sorted(agg["explainer"].unique())
    corruptions = sorted(agg["corruption"].unique())
    severities = sorted(agg["severity"].unique())
    n_rows, n_cols = len(explainers), len(corruptions)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.2 * n_cols, 2.9 * n_rows),
        sharey=True, squeeze=False,
    )

    x_positions = np.arange(len(severities))
    bar_width = 0.72

    for i, explainer in enumerate(explainers):
        for j, corruption in enumerate(corruptions):
            ax = axes[i, j]
            panel = agg[
                (agg["explainer"] == explainer)
                & (agg["corruption"] == corruption)
            ]
            bottom = np.zeros(len(severities))
            for zone in ZONE_ORDER:
                heights = []
                for sev in severities:
                    row = panel[(panel["severity"] == sev) & (panel["zone"] == zone)]
                    heights.append(row["share_mean"].iloc[0] * 100 if not row.empty else 0.0)
                heights = np.array(heights)
                bars = ax.bar(
                    x_positions, heights, width=bar_width, bottom=bottom,
                    color=ZONE_COLORS[zone], edgecolor="white", linewidth=0.6,
                )

                labels = [f"{h:.0f}%" if h >= 5 else "" for h in heights]
                ax.bar_label(
                    bars,
                    labels=labels,
                    label_type="center",
                    fontsize=8,
                    color="white",
                )

                bottom += heights
            ax.set_xticks(x_positions)
            ax.set_xticklabels([str(s) for s in severities])
            ax.set_ylim(0, 100)
            if i == 0:
                ax.set_title(corruption_label(corruption), fontsize=11, fontweight="bold")
            if i == n_rows - 1:
                ax.set_xlabel("Severity")
            if j == 0:
                ax.set_ylabel(f"{explainer}\nshare of samples (%)", fontsize=10, linespacing=1.6)
            ax.grid(True, axis="y", alpha=0.3, linewidth=0.6)
            ax.set_axisbelow(True)

    legend_handles = [Patch(facecolor=ZONE_COLORS[z], edgecolor="white",
                             label=z) for z in ZONE_ORDER]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=len(ZONE_ORDER),
        frameon=False, bbox_to_anchor=(0.5, -0.06), fontsize=10,
    )
    basis_label = SIMILARITY_SPECS[de_basis]["drift_axis"]
    fig.suptitle(
        f"Prediction–drift regimes  ·  high drift: "
        f"{basis_label}  $\\geq$ q$_{{{threshold_quantile:.2f}}}$ = {thr:.3f}",
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
# Figure 10 — Violins of per-sample ρ (or cos / iou)
# ----------------------------------------------------------------------------

def plot_violins(
    df: pd.DataFrame,
    slice_key: str = "inv",
    similarity: str = "rho",
    output_path: str | Path | None = None,
) -> plt.Figure:
    if similarity not in ("rho", "cos", "iou"):
        raise ValueError(f"unknown similarity {similarity!r}")

    sub = df[slice_mask(df, slice_key)].copy()
    explainers = sorted(sub["explainer"].unique())
    corruptions = sorted(sub["corruption"].unique())
    severities = sorted(sub["severity"].unique())

    n_rows, n_cols = len(explainers), len(corruptions)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.3 * n_cols, 3.0 * n_rows),
        sharey=True, squeeze=False,
    )

    palette = corruption_palette(corruptions)

    for i, explainer in enumerate(explainers):
        for j, corruption in enumerate(corruptions):
            ax = axes[i, j]
            color = palette[corruption]
            data = []
            labels = []
            ns = []
            for sev in severities:
                vals = sub[
                    (sub["explainer"] == explainer)
                    & (sub["corruption"] == corruption)
                    & (sub["severity"] == sev)
                ][similarity].dropna().to_numpy()
                data.append(vals if len(vals) else np.array([np.nan]))
                labels.append(str(sev))
                ns.append(len(vals))
            parts = ax.violinplot(data, positions=np.arange(len(severities)),
                                   showmeans=False, showmedians=True, widths=0.8)
            for body in parts["bodies"]:
                body.set_facecolor(color)
                body.set_edgecolor("#333333")
                body.set_alpha(0.55)
            for key in ("cmedians", "cmins", "cmaxes", "cbars"):
                if key in parts:
                    parts[key].set_edgecolor("#333333")
                    parts[key].set_linewidth(0.9)
            ax.set_xticks(np.arange(len(severities)))
            ax.set_xticklabels(labels)
            ax.set_ylim(-0.02, 1.05)
            ax.grid(True, axis="y", alpha=0.3, linewidth=0.6)
            ax.set_axisbelow(True)

            # sample-count annotation, placed below the axis in axes coords
            trans = blended_transform_factory(ax.transData, ax.transAxes)
            for k, nk in enumerate(ns):
                ax.text(k, -0.18, f"n={nk}", ha="center", va="top",
                        fontsize=7, color="#555555", transform=trans)

            if i == 0:
                ax.set_title(corruption_label(corruption), fontsize=11, fontweight="bold")
            if i == n_rows - 1:
                ax.set_xlabel("Severity", labelpad=24)
            if j == 0:
                ax.set_ylabel(f"{explainer}\nper-sample {similarity}", fontsize=10, linespacing=1.6)

    fig.suptitle(
        f"Distribution of per-sample {SIMILARITY_SPECS[similarity]['similarity']} by severity  ·  "
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
# Clean reference loader — provides MSP on clean inputs for Fig 13
# ----------------------------------------------------------------------------

CLEAN_REF_FILENAME = "00__clean_ref.pt"

def load_clean_references(
    experiments_dir: str | Path,
    n: int | None = None,
) -> pd.DataFrame:
    """Walk experiment dirs and load each ``00__reference/00__clean_ref.pt``.
    Returns a long DataFrame:
        (explainer, seed, sample_idx, msp_clean, clean_correct)

    Requires:
        - proba_clean
        - pred_clean
        - y_clean

    MSP is max(proba_clean, axis=1).
    clean_correct is (pred_clean == y_clean).
    Missing files are skipped with a warning.
    """
    experiments_dir = Path(experiments_dir)
    frames: list[pd.DataFrame] = []
    missing: list[str] = []

    for sub in sorted(experiments_dir.iterdir()):
        if not sub.is_dir():
            continue
        m = EXPERIMENT_DIR_RE.match(sub.name)
        if m is None:
            continue
        if n is not None and int(m["n"]) != n:
            continue

        ref_file = sub / "00__reference" / CLEAN_REF_FILENAME
        if not ref_file.exists():
            missing.append(sub.name)
            continue

        payload = _load_payload(ref_file)
        inner = _unwrap_payload(payload, ["proba_clean", "pred_clean", "y_true"])
        if inner is None:
            top_keys = list(payload.keys()) if isinstance(payload, dict) else "(not a dict)"
            print(f"[warn] {ref_file}: no clean-reference keys found. "
                  f"Top-level keys: {top_keys}")
            continue

        required = ["proba_clean", "pred_clean", "y_true"]
        missing_keys = [k for k in required if k not in inner]
        if missing_keys:
            print(f"[warn] {ref_file}: missing keys {missing_keys}, skipping")
            continue

        proba = inner["proba_clean"]
        pred_clean = inner["pred_clean"]
        y_clean = inner["y_true"]

        if _HAS_TORCH and hasattr(proba, "detach"):
            proba = proba.detach().cpu().numpy()
        if _HAS_TORCH and hasattr(pred_clean, "detach"):
            pred_clean = pred_clean.detach().cpu().numpy()
        if _HAS_TORCH and hasattr(y_clean, "detach"):
            y_clean = y_clean.detach().cpu().numpy()

        proba = np.asarray(proba)
        pred_clean = np.asarray(pred_clean).ravel()
        y_clean = np.asarray(y_clean).ravel()

        if proba.ndim != 2:
            print(f"[warn] {ref_file}: proba_clean has unexpected shape {proba.shape}")
            continue
        if len(pred_clean) != len(y_clean) or len(pred_clean) != len(proba):
            print(
                f"[warn] {ref_file}: inconsistent lengths "
                f"(proba={len(proba)}, pred_clean={len(pred_clean)}, y_clean={len(y_clean)})"
            )
            continue

        msp = proba.max(axis=1).astype(float).ravel()
        clean_correct = (pred_clean == y_clean)

        frames.append(pd.DataFrame({
            "explainer":     m["explainer"],
            "seed":          int(m["seed"]),
            "sample_idx":    np.arange(len(msp)),
            "msp_clean":     msp,
            "clean_correct": clean_correct,
        }))

    if missing:
        print(f"[info] no clean reference in {len(missing)} experiment dir(s): "
              f"{missing[:3]}{'…' if len(missing) > 3 else ''}")
    if not frames:
        return pd.DataFrame(columns=["explainer", "seed", "sample_idx", "msp_clean", "clean_correct"])

    out = pd.concat(frames, ignore_index=True)
    print(f"Loaded clean metadata for {len(out):,} sample rows across "
          f"{out[['explainer','seed']].drop_duplicates().shape[0]} experiments")
    return out


def attach_clean_msp(
    df: pd.DataFrame,
    clean_ref: pd.DataFrame,
) -> pd.DataFrame:
    """Left-merge clean metadata onto the sample-level drift frame by
    (explainer, seed, sample_idx).

    Adds:
        - msp_clean
        - clean_correct
    """
    if clean_ref.empty:
        out = df.copy()
        out["msp_clean"] = np.nan
        out["clean_correct"] = np.nan
        return out

    return df.merge(
        clean_ref,
        on=["explainer", "seed", "sample_idx"],
        how="left",
    )

# ----------------------------------------------------------------------------
# Figure 13 — Clean confidence vs. vulnerability (per-sample null-result test)
# ----------------------------------------------------------------------------

def plot_clean_msp_vs_vulnerability(
    df: pd.DataFrame,
    slice_key: str = "all",
    de_basis: str = "rho",
    max_points_per_panel: int = 15_000,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Scatter per-sample clean MSP vs. vulnerability (ΔE = 1 − similarity).
    Grid = Explainer × Severity, points pooled across corruptions and seeds.
    A Spearman ρ annotation per panel quantifies whether clean confidence
    predicts explanation drift. Expectation: ρ ≈ 0 (null result = key claim).

    Requires that ``df`` already has a ``msp_clean`` column (produced by
    ``attach_clean_msp``). Panels with all-NaN msp_clean are rendered empty
    with a hint.
    """
    if "msp_clean" not in df.columns:
        raise KeyError("df has no 'msp_clean' column — call attach_clean_msp first")

    sub = df[slice_mask(df, slice_key)].copy()
    sub = add_delta_e(sub, basis=de_basis)
    sub = sub.dropna(subset=["msp_clean", "delta_e"])

    explainers = sorted(sub["explainer"].unique())
    severities = sorted(sub["severity"].unique())
    corruptions = sorted(sub["corruption"].unique())
    palette = corruption_palette(corruptions)

    n_rows, n_cols = len(explainers), len(severities)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.3 * n_cols, 3.1 * n_rows),
        sharex=True, sharey=True, squeeze=False,
    )

    rng = np.random.default_rng(0)

    for i, explainer in enumerate(explainers):
        for j, sev in enumerate(severities):
            ax = axes[i, j]
            panel = sub[(sub["explainer"] == explainer) & (sub["severity"] == sev)]
            if panel.empty:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", color="#888")
                continue

            # Optional downsample for plotting (Spearman still computed on full)
            if len(panel) > max_points_per_panel:
                idx = rng.choice(len(panel), size=max_points_per_panel, replace=False)
                panel_plot = panel.iloc[idx]
            else:
                panel_plot = panel

            for corruption, cdf in panel_plot.groupby("corruption"):
                ax.scatter(
                    cdf["msp_clean"], cdf["delta_e"],
                    s=7, alpha=0.32, color=palette[corruption],
                    linewidths=0,
                )

            # Pearson on the full (unsubsampled) panel
            if len(panel) >= 2 and panel["msp_clean"].std() > 0 \
                    and panel["delta_e"].std() > 0:
                r_val = float(panel[["msp_clean", "delta_e"]].corr().iloc[0, 1])
                ax.text(
                    0.04, 0.96,
                    f"Pearson $r$ = {r_val:+.3f}\n$n$ = {len(panel):,}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.32",
                            facecolor="white", edgecolor="#cccccc", alpha=0.92),
                )

            if i == 0:
                ax.set_title(f"Severity {sev}", fontsize=11, fontweight="bold")
            if i == n_rows - 1:
                ax.set_xlabel("Clean MSP")
            if j == 0:
                ax.set_ylabel(
                    f"{explainer}\n{SIMILARITY_SPECS[de_basis]['drift_axis']}",
                    fontsize=9, linespacing=1.6
                )
            ax.grid(True, alpha=0.3, linewidth=0.6)
            ax.set_xlim(0, 1.02)

    # Global y-limits from data (0 to slightly above observed max)
    y_hi = float(sub["delta_e"].quantile(0.995))
    axes[0, 0].set_ylim(0, max(y_hi * 1.05, 0.1))

    # Corruption colour legend at the bottom
    legend_handles = [
        Line2D([], [], marker="o", linestyle="", color=palette[c],
               markersize=8, label=corruption_label(c))
        for c in corruptions
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=len(corruptions),
        frameon=False, bbox_to_anchor=(0.5, -0.06), fontsize=9,
    )
    fig.suptitle(
        f"Clean confidence vs. explanation drift  ·  "
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
# Figure 14 — Metric correlation heatmap
# ----------------------------------------------------------------------------

def plot_metric_correlation(
    df: pd.DataFrame,
    mode: str = "by_corruption",
    slice_key: str = "all",
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Spearman correlation heatmap of all drift/uncertainty metrics.

    Parameters
    ----------
    mode : "pooled" | "by_corruption"
        ``pooled`` produces one heatmap per explainer, pooled over corruptions
        and severities. ``by_corruption`` produces a grid
        (explainer × corruption) so that corruption-specific coupling is visible.
    """
    if mode not in ("pooled", "by_corruption"):
        raise ValueError(f"mode must be 'pooled' or 'by_corruption', got {mode!r}")
    
    mode_display = mode.replace("_", " ")

    sub = df[slice_mask(df, slice_key)].copy()
    sub = _augment_metric_columns(sub)
    metric_cols = [c for c, _ in CORRELATION_METRICS]
    metric_labels = [label for _, label in CORRELATION_METRICS]

    explainers = sorted(sub["explainer"].unique())
    corruptions = sorted(sub["corruption"].unique())

    if mode == "pooled":
        panels = [(expl, None) for expl in explainers]
        n_rows, n_cols = 1, len(explainers)
    else:
        panels = [(expl, c) for expl in explainers for c in corruptions]
        n_rows, n_cols = len(explainers), len(corruptions)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.4 * n_cols, 3.2 * n_rows),
        squeeze=False,
    )

    im = None
    for idx, (explainer, corruption) in enumerate(panels):
        i, j = idx // n_cols, idx % n_cols
        ax = axes[i, j]

        panel = sub[sub["explainer"] == explainer]
        if corruption is not None:
            panel = panel[panel["corruption"] == corruption]

        if len(panel) < 2:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="#888")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        corr = panel[metric_cols].corr(method="spearman").to_numpy()
        im = ax.imshow(corr, cmap="viridis", vmin=-1, vmax=1, aspect="equal")

        ax.set_xticks(range(len(metric_labels)))
        ax.set_xticklabels(metric_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(metric_labels)))
        ax.set_yticklabels(metric_labels, fontsize=8)

        for ii in range(len(metric_labels)):
            for jj in range(len(metric_labels)):
                val = corr[ii, jj]
                # white text on dark cells (low |val| = mid viridis = dark)
                text_color = "white" if val < 0.4 else "black"
                ax.text(jj, ii, f"{val:.2f}", ha="center", va="center",
                        color=text_color, fontsize=7)

        title = explainer if corruption is None else f"{explainer} · {corruption_label(corruption)}"
        ax.set_title(title, fontsize=10, fontweight="bold")

    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(),
                            shrink=0.7, label=r"Spearman $\rho$",
                            pad=0.02)

    fig.suptitle(
        f"Metric correlation  ·  slice: {SLICES[slice_key]}  ·  mode: {mode_display}",
        y=0.98, fontsize=12,
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=200)
        fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=200)
    return fig


# ----------------------------------------------------------------------------
# Trust-zone exemplar selector
# ----------------------------------------------------------------------------

def _pick_exemplar(
    mask: np.ndarray,
    similarity: np.ndarray,
    abs_dH: np.ndarray,
    sample_idx: np.ndarray,
    want_stable: bool,
) -> int | None:
    """Return the sample_idx inside ``mask`` that best illustrates either the
    stable case (high sim + low |ΔH|) or the unstable case (low sim + high |ΔH|).

    Rank-based combination: both criteria contribute equally, ties broken by
    |ΔH|. Returns None if the mask is empty.
    """
    if not mask.any():
        return None

    if want_stable:
        # Want: HIGH similarity (ascending=False → high sim = rank 1)
        #       LOW  |ΔH|       (ascending=True  → low  dH  = rank 1)
        sim_rank = pd.Series(similarity).rank(method="average", ascending=False).to_numpy()
        dh_rank = pd.Series(abs_dH).rank(method="average", ascending=True).to_numpy()
    else:
        # Want: LOW  similarity, HIGH |ΔH|
        sim_rank = pd.Series(similarity).rank(method="average", ascending=True).to_numpy()
        dh_rank = pd.Series(abs_dH).rank(method="average", ascending=False).to_numpy()

    combined = (sim_rank + dh_rank).astype(float)
    combined[~mask] = np.inf
    return int(sample_idx[int(np.argmin(combined))])


def find_trust_zone_exemplars(
    df: pd.DataFrame,
    explainer: str,
    seed: int,
    corruption: str,
    severity: int,
    de_basis: str = "cos",
) -> dict[str, int | None]:
    """Find one representative sample_idx per Trust Zone for a specific
    (explainer, seed, corruption, severity) condition.

    Zone definitions (same as Figure 9 in sample_level_analysis):

    - **Robust**: both_correct ∧ stable explanation (high sim, low |ΔH|)
    - **Silent Drift**: both_correct ∧ unstable explanation — the decoupling signal
    - **Hidden Failure**: ¬both_correct ∧ stable explanation — wrong but the
      saliency didn't even react
    - **Visible Failure**: ¬both_correct ∧ unstable explanation

    Returns
    -------
    dict of zone name → sample_idx (or None if the zone is empty)
    """
    if de_basis not in SIMILARITY_SPECS:
        raise ValueError(f"unknown de_basis {de_basis!r}")
    sim_col = SIMILARITY_SPECS[de_basis]["tensor_col"]

    sub = df[
        (df["explainer"] == explainer)
        & (df["seed"] == seed)
        & (df["corruption"] == corruption)
        & (df["severity"] == severity)
    ]
    if sub.empty:
        return {z: None for z in ZONE_ORDER}

    similarity = sub[sim_col].to_numpy()
    abs_dH = sub["dH"].abs().to_numpy()
    both_correct = sub["both_correct"].to_numpy()
    sample_idx = sub["sample_idx"].to_numpy()

    return {
        "Robust":
            _pick_exemplar(both_correct, similarity, abs_dH, sample_idx, want_stable=True),
        "Silent Drift":
            _pick_exemplar(both_correct, similarity, abs_dH, sample_idx, want_stable=False),
        "Hidden Failure":
            _pick_exemplar(~both_correct, similarity, abs_dH, sample_idx, want_stable=True),
        "Visible Failure":
            _pick_exemplar(~both_correct, similarity, abs_dH, sample_idx, want_stable=False),
    }


def export_trust_zone_exemplars(
    df: pd.DataFrame,
    output_dir: Path,
    de_basis: str = "cos",
) -> pd.DataFrame:
    """Build a long-format CSV with one row per (condition × zone),
    including the metric values of the picked sample so the downstream
    rendering pipeline has full context.
    """
    if de_basis not in SIMILARITY_SPECS:
        raise ValueError(f"unknown de_basis {de_basis!r}")
    sim_col = SIMILARITY_SPECS[de_basis]["tensor_col"]

    rows: list[dict] = []
    keys = ["explainer", "seed", "corruption", "severity"]
    for (expl, seed, corr, sev), cond in df.groupby(keys):
        exs = find_trust_zone_exemplars(df, expl, seed, corr, sev, de_basis=de_basis)
        for zone, idx in exs.items():
            if idx is None:
                rows.append({
                    "explainer": expl, "seed": int(seed),
                    "corruption": corr, "severity": int(sev),
                    "zone": zone, "sample_idx": None,
                    "similarity": None, "abs_dH": None,
                    "invariant": None, "both_correct": None,
                })
                continue
            sample = cond[cond["sample_idx"] == idx].iloc[0]
            rows.append({
                "explainer": expl, "seed": int(seed),
                "corruption": corr, "severity": int(sev),
                "zone": zone, "sample_idx": int(idx),
                "similarity": float(sample[sim_col]),
                "abs_dH":     float(abs(sample["dH"])),
                "invariant":    bool(sample["invariant"]),
                "both_correct": bool(sample["both_correct"]),
            })

    out = pd.DataFrame(rows)
    out_path = output_dir / "trust_zone_exemplars.csv"
    out.to_csv(out_path, index=False)
    n_picked = int(out["sample_idx"].notna().sum())
    print(f"Wrote {out_path.name} with {len(out)} rows ({n_picked} non-null exemplars)")
    return out


# ----------------------------------------------------------------------------
# CSV export
# ----------------------------------------------------------------------------

def export_aggregated_csvs(
    df: pd.DataFrame,
    output_dir: Path,
    de_basis: str = "cos",
    threshold_quantile: float = 0.75,
) -> None:
    zones, _ = compute_trust_zones(df, de_basis=de_basis,
                                   threshold_quantile=threshold_quantile)
    zones.to_csv(output_dir / "trust_zones.csv", index=False)


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def run_sample_level_analysis(
    experiments_dir: str | Path,
    output_root: str | Path,
    n: int | None = 1000,
    de_basis: str = "cos",
    threshold_quantile: float = 0.75,
    violin_slice: str = "inv",
    violin_similarity: str = "rho",
    scatter_slice: str = "inv",
    roc_conditions: list[tuple[str, int]] | None = None,
    vulnerability_de: str = "rho",
    vulnerability_slice: str = "all",
    heatmap_mode: str = "by_corruption",
    heatmap_slice: str = "all",
) -> None:
    output_root = Path(output_root)
    output_dir = output_root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_sample_level_drift(experiments_dir, n=n)

    # Attach clean MSP if the reference files exist (Fig 13 is opt-in).
    clean_ref = load_clean_references(experiments_dir, n=n)
    has_clean_msp = not clean_ref.empty
    if has_clean_msp:
        df = attach_clean_msp(df, clean_ref)

    plot_trust_zones(
        df, de_basis=de_basis, threshold_quantile=threshold_quantile,
        output_path=output_dir / f"fig9_trust_zones_{de_basis}_q{threshold_quantile}.pdf",
    )
    plot_violins(
        df, slice_key=violin_slice, similarity=violin_similarity,
        output_path=output_dir / f"fig10_similarity_violins_{violin_slice}_{violin_similarity}.pdf",
    )
    if has_clean_msp:
        plot_clean_msp_vs_vulnerability(
            df, slice_key=vulnerability_slice, de_basis=vulnerability_de,
            output_path=output_dir / f"fig13_clean_msp_vs_vulnerability_{vulnerability_slice}_{vulnerability_de}.pdf",
        )
    else:
        print("[info] Fig 13 skipped: no 00__reference/00__clean_ref.pt files found")
    export_aggregated_csvs(df, output_dir,
                           de_basis=de_basis,
                           threshold_quantile=threshold_quantile)
    
    plot_metric_correlation(
        df, mode=heatmap_mode, slice_key=heatmap_slice,
        output_path=output_dir / f"fig14_metric_correlation_{heatmap_slice}_{heatmap_mode}.pdf",
    )

    export_trust_zone_exemplars(df, output_dir, de_basis=de_basis)

    plt.close("all")
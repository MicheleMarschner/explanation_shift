from pathlib import Path
import re
from typing import Iterable
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

SLICES: dict[str, str] = {
    "all": "all samples",
    "inv": "invariant predictions",
    "both_corr": "both correct (clean & shifted)",
}

SLICE_STYLES: dict[str, dict] = {
    "all":       dict(color="#444444", ls="-",  marker="o", label="all samples"),
    "inv":       dict(color="#1f77b4", ls="--", marker="s", label="invariant"),
    "both_corr": dict(color="#d62728", ls="-.", marker="^", label="both correct"),
}

SIM_MEASURES: list[tuple[str, str]] = [
    ("rho", r"Spearman $\rho$"),
    ("cos", "Cosine similarity"),
    ("iou", "Top-$k$ IoU"),
]

PRIMARY_MEASURE = "rho"
PRIMARY_LABEL = r"Spearman $\rho$ (explanation similarity)"

EXPERIMENT_DIR_RE = re.compile(
    r"experiment__n(?P<n>\d+)__(?P<explainer>[A-Za-z0-9]+)__seed(?P<seed>\d+)$"
)

DRIFT_LABELS = {
    "rho": r"$1 - \rho$",
    "cos": r"$1 - \cos$",
    "iou": r"$1 - \mathrm{IoU}$",
}

# ΔE is defined as 1 − similarity. User picks which similarity to use.
DE_OPTIONS: dict[str, tuple[str, str]] = {
    # key: (label for plots, column stem in CSV)
    "rho": (r"$1-\rho$ (Spearman)", "rho"),
    "cos": (r"$1-\cos$ (cosine)", "cos"),
    "iou": (r"$1-\mathrm{IoU}$ (top-$k$)", "iou"),
}

# ΔP has four reasonable operationalisations. Default is flip_rate because it
# ties cleanly to the "both-correct ⇒ ΔP = 0 by construction" argument.
DP_OPTIONS: dict[str, tuple[str, str]] = {
    "flip_rate":   ("1 − invariant rate", "__derived__"),
    "err_rate":    ("1 − both-correct rate", "__derived__"),
    "p_shift":     (r"$|\Delta p_{\mathrm{pred}}|$", "conf_all__p_shift_mean"),
    "margin_shift":(r"$|\Delta\,\mathrm{margin}|$", "conf_all__margin_shift_mean"),
}

TENSOR_KEYS = (
    "invariant", "both_correct", "dH",
    "exp__spearman_rho", "exp__cosine_sim", "exp__iou_topk",
    "conf__p_shift_abs", "conf__margin_shift_abs",
)

DE_BASIS: dict[str, tuple[str, str]] = {
    # key: (tensor column, plot label)
    "rho": ("rho", r"$\Delta E = 1 - \rho$"),
    "cos": ("cos", r"$\Delta E = 1 - \cos$"),
    "iou": ("iou", r"$\Delta E = 1 - \mathrm{IoU}$"),
}

# Trust-zone colours: robust / silent / expected / stubborn
ZONE_COLORS = {
    "Robust":           "#1f77b4",
    "Silent Drift":     "#ff7f0e",
    "Expected Failure": "#2ca02c",
    "Stubborn Failure": "#d62728",
}
ZONE_ORDER = ["Robust", "Silent Drift", "Expected Failure", "Stubborn Failure"]

CORRELATION_METRICS: list[tuple[str, str]] = [
    ("drift_1m_cos", r"$1-\cos$"),
    ("drift_1m_iou", r"$1-\mathrm{IoU}$"),
    ("drift_1m_rho", r"$1-\rho$"),
    ("abs_dH",       r"$|\Delta H|$"),
    ("p_shift",      r"$|\Delta p|$"),
    ("margin_shift", r"$|\Delta\mathrm{margin}|$"),
]

DE_SIM_COLS = {"cos": "cos", "rho": "rho", "iou": "iou"}

# ----------------------------------------------------------------------------
# Global corruption filter
# ----------------------------------------------------------------------------

EXCLUDED_CORRUPTIONS: frozenset[str] = frozenset({
    "defocus_blur",
})


def filter_excluded_corruptions(
    df: pd.DataFrame,
    column: str = "corruption",
) -> pd.DataFrame:
    """Remove globally excluded corruptions from a dataframe.

    This is the single central switch for hiding redundant corruptions
    (currently: defocus / defocus_blur) from all downstream analyses.
    """
    if column not in df.columns:
        return df

    mask = ~df[column].isin(EXCLUDED_CORRUPTIONS)
    return df.loc[mask].copy()


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def load_drift_results(experiments_dir: str | Path, n: int | None = None) -> pd.DataFrame:
    """Walk `experiments_dir`, pick up every `02__drift/02__drift_results.csv`,
    tag it with (explainer, seed), concatenate.

    Parameters
    ----------
    experiments_dir : Path-like
        Directory containing `experiment__n{N}__{EXPLAINER}__seed{SEED}/` folders.
    n : int, optional
        If given, only experiments with this sample count are loaded.

    Returns
    -------
    DataFrame in long format: original CSV columns + `explainer`, `seed`.
    """
    experiments_dir = Path(experiments_dir)
    if not experiments_dir.is_dir():
        raise FileNotFoundError(f"Experiments directory not found: {experiments_dir}")

    frames: list[pd.DataFrame] = []
    for sub in sorted(experiments_dir.iterdir()):
        if not sub.is_dir():
            continue
        m = EXPERIMENT_DIR_RE.match(sub.name)
        if m is None:
            continue
        if n is not None and int(m.group("n")) != n:
            continue
        csv_path = sub / "02__drift" / "02__drift_results.csv"
        if not csv_path.exists():
            print(f"[warn] missing CSV: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        df["explainer"] = m.group("explainer")
        df["seed"] = int(m.group("seed"))
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No drift_results.csv files found under {experiments_dir} "
            f"(n filter = {n})"
        )

    out = pd.concat(frames, ignore_index=True)
    out = filter_excluded_corruptions(out, column="corruption")

    print(
        f"Loaded {len(out)} rows | "
        f"explainers={sorted(out['explainer'].unique())} | "
        f"seeds={sorted(out['seed'].unique())} | "
        f"corruptions={sorted(out['corruption'].unique())} | "
        f"severities={sorted(out['severity'].unique())}"
    )
    return out



# ----------------------------------------------------------------------------
# Plotting helpers
# ----------------------------------------------------------------------------

def corruption_palette(corruptions: Iterable[str]) -> dict[str, tuple]:
    cmap = plt.get_cmap("tab10")
    return {c: cmap(i % 10) for i, c in enumerate(sorted(corruptions))}


def _x_values(frame: pd.DataFrame, x_axis: str) -> np.ndarray:
    if x_axis == "severity":
        return frame["severity"].to_numpy(dtype=float)
    if x_axis == "mmd":
        return frame["mmd_mean"].to_numpy(dtype=float)
    raise ValueError(f"Unknown x_axis {x_axis!r}, expected 'severity' or 'mmd'.")


def x_label(x_axis: str) -> str:
    return "Corruption severity" if x_axis == "severity" else r"max MMD$^2$ (feature-space)"


def draw_corruption_lines(
    ax: plt.Axes,
    panel_df: pd.DataFrame,
    metric_key: str,
    x_axis: str,
    palette: dict[str, tuple],
) -> None:
    for corr, cdf in panel_df.groupby("corruption"):
        cdf = cdf.sort_values("severity")
        x = _x_values(cdf, x_axis)
        order = np.argsort(x)
        x = x[order]
        y = cdf[f"{metric_key}_mean"].to_numpy()[order]
        err = cdf[f"{metric_key}_sd"].to_numpy()[order]
        color = palette[corr]
        ax.plot(x, y, marker="o", linewidth=1.8, color=color, label=corr, zorder=3)
        ax.fill_between(x, y - err, y + err, color=color, alpha=0.15, linewidth=0, zorder=2)


def similarity_to_drift_agg(agg: pd.DataFrame) -> pd.DataFrame:
    """Return a plotting copy where similarity means are converted to drift means:
       drift = 1 - similarity.
       SD stays unchanged.
    """
    out = agg.copy()

    for key, _ in SIM_MEASURES:
        mean_col = f"{key}_mean"
        if mean_col in out.columns:
            out[mean_col] = (1.0 - out[mean_col]).clip(lower=0.0, upper=1.0)

    return out


EXPERIMENT_DIR_RE = re.compile(
    r"experiment__n(?P<n>\d+)__(?P<explainer>[A-Za-z0-9]+)__seed(?P<seed>\d+)$"
)


def find_experiment_dir(
    experiments_dir: Path, explainer: str, seed: int
) -> Path | None:
    """Locate the experiment dir for a given (explainer, seed). Matches any N."""
    for sub in experiments_dir.iterdir():
        if not sub.is_dir():
            continue
        m = EXPERIMENT_DIR_RE.match(sub.name)
        if m and m["explainer"] == explainer and int(m["seed"]) == seed:
            return sub
    return None
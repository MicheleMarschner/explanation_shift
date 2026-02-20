"""
Correlation heatmap (pooled)

This figure shows how your different monitoring signals relate to each other across all pairs. It computes the Spearman 
correlation between drift metrics (1−cos, 1−IoU, 1−ρ), the prediction-change proxy (|ΔEntropy|), and confidence on the 
corrupted input (MSP). The goal is to see whether these signals are largely redundant (high correlations) or capture 
different aspects of shift (low/moderate correlations).


Correlation heatmaps by severity / by corruption (commented loops)

These variants repeat the same correlation analysis within one severity level or within one corruption type, to check 
whether relationships between signals are stable or change depending on how strong the shift is or what kind of shift it is.


Clean confidence vs vulnerability scatter

This plot tests whether samples the model is very confident about on clean data are also more robust under corruption. It 
compares clean MSP (x-axis) to a vulnerability measure (y-axis: explanation drift at a fixed severity, e.g. 1−Spearman ρ). 
If the trend is negative, high clean confidence corresponds to lower vulnerability; if not, clean confidence is not a 
reliable indicator of robustness.



"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import torch

from src.configs.global_config import PATHS


def spearman_corr_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    sub = df[cols].dropna(axis=0, how="any").copy()
    ranked = sub.rank(method="average")          # Spearman = Pearson on ranks
    return ranked.corr(method="pearson")


def plot_corr_heatmap(corr: pd.DataFrame, title: str, save_path: str | Path | None = None):
    n = len(corr.columns)
    fig, ax = plt.subplots(figsize=(0.75 * n + 3, 0.75 * n + 2.5))
    im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)

    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spearman ρ")

    vals = corr.to_numpy()
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{vals[i, j]:.2f}", ha="center", va="center", fontsize=9)

    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
    plt.show()


# ---- load your per-pair table ----
csv_path = PATHS.results / "spearman_entropy" / "pairs_table_from_pt.csv"
df = pd.read_csv(csv_path)

# choose metrics only (exclude labels/preds)
metric_cols = [
    "drift_1m_cos",
    "drift_1m_iou",
    "drift_1m_rho",
    "abs_dH",
    "msp_corr",
]

corr = spearman_corr_matrix(df, metric_cols)
plot_corr_heatmap(
    corr,
    title="Metric-vs-Metric Spearman Correlation (per pair, pooled)",
    save_path=PATHS.results / "spearman_entropy" / "metric_corr_heatmap.png",
)


'''
for sev, g in df.groupby("severity"):
    c = spearman_corr_matrix(g, metric_cols)
    plot_corr_heatmap(
        c,
        title=f"Metric Spearman Correlation (severity={sev})",
        save_path=PATHS.results / "spearman_entropy" / "metric_corr_by_severity" / f"heatmap_sev{sev}.png",
    )


for corr_name, g in df.groupby("corruption"):
    c = spearman_corr_matrix(g, metric_cols)
    plot_corr_heatmap(
        c,
        title=f"Metric Spearman Correlation ({corr_name})",
        save_path=PATHS.results / "spearman_entropy" / "metric_corr_by_corruption" / f"heatmap_{corr_name}.png",
    )
'''


def load_msp_clean_and_pair_idx(ref_pt: Path):
    ref = torch.load(ref_pt, map_location="cpu")
    pair_idx = ref["pair_idx"].detach().cpu().numpy().astype(int)                 # [N]
    proba_clean = ref["clean_reference"]["proba_clean"]
    proba_clean = proba_clean.detach().cpu().numpy() if torch.is_tensor(proba_clean) else np.asarray(proba_clean)
    msp_clean = proba_clean.max(axis=1).astype(float)                              # [N]
    return pair_idx, msp_clean


def spearman_r(x, y) -> float:
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def plot_clean_conf_vs_vulnerability(
    df: pd.DataFrame,
    severity: int = 3,
    x_col: str = "msp_clean",
    y_col: str = "drift_1m_rho",   # vulnerability = 1 - rho
    corruption: str | None = None,
    max_points: int = 8000,
    save_path: Path | None = None,
):
    d = df.copy()
    d["severity"] = d["severity"].astype(int)
    d = d[d["severity"] == severity].dropna(subset=[x_col, y_col]).copy()
    if corruption is not None:
        d = d[d["corruption"] == corruption].copy()

    if len(d) > max_points:
        d = d.sample(max_points, random_state=0)

    rho = spearman_r(d[x_col].to_numpy(), d[y_col].to_numpy())

    plt.figure(figsize=(7.6, 5.4))
    plt.scatter(d[x_col], d[y_col], s=10, alpha=0.35)
    plt.xlabel("Clean confidence (MSP on clean)")
    plt.ylabel("Vulnerability = 1 − Spearman ρ (severity=3)")
    title = f"Clean confidence vs vulnerability (sev={severity}), Spearman ρ={rho:.3f}, n={len(d)}"
    if corruption:
        title += f"\ncorruption={corruption}"
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)

    plt.show()


def main():
    exp_dir = PATHS.runs / "experiment__n250__IG__seed51"
    ref_pt  = exp_dir / "00__reference" / "00__clean_ref.pt"

    pairs_csv = PATHS.results / "spearman_entropy" / "pairs_table_from_pt.csv"
    df = pd.read_csv(pairs_csv)

    # add msp_clean via pair_idx mapping
    pair_idx_ref, msp_clean = load_msp_clean_and_pair_idx(ref_pt)
    msp_map = {int(pair_idx_ref[i]): float(msp_clean[i]) for i in range(len(pair_idx_ref))}
    df["pair_idx"] = df["pair_idx"].astype(int)
    df["msp_clean"] = df["pair_idx"].map(msp_map)

    # pooled plot
    plot_clean_conf_vs_vulnerability(
        df,
        severity=3,
        corruption=None,
        save_path=PATHS.results / "spearman_entropy" / "clean_conf_vs_vulnerability_sev3.png",
    )

    # optional: per corruption (uncomment)
    # for corr in sorted(df["corruption"].unique()):
    #     plot_clean_conf_vs_vulnerability(df, severity=3, corruption=corr)

if __name__ == "__main__":
    main()
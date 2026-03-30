import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Sequence, Optional, Union

# --- Helpers ---

def spearman_r(x, y) -> float:
    """Manual Spearman calc to avoid scipy dependency if preferred."""
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) < 2: return np.nan
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    if np.std(rx) == 0 or np.std(ry) == 0: return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])

def _attach_clean_baseline(df: pd.DataFrame, corruption: str) -> pd.DataFrame:
    """Prepends severity=0 row to a corruption slice."""
    df = df.copy()
    d_corr = df[df["corruption"] == corruption].copy()
    d_clean = df[df["severity"] == 0].copy()
    
    if d_clean.empty: return d_corr
    
    # Average numeric cols for clean baseline
    num_cols = d_clean.select_dtypes(include=[np.number]).columns
    clean_row = pd.DataFrame([d_clean[num_cols].mean(numeric_only=True)])
    clean_row["severity"] = 0
    clean_row["corruption"] = corruption
    
    # Concat and sort
    return pd.concat([clean_row, d_corr], ignore_index=True).sort_values("severity")

# --- Plot Functions ---

def plot_correlation_heatmap(
    df: pd.DataFrame, 
    cols: list, 
    title: str, 
    save_path: Optional[Path] = None
):
    """Generates Spearman correlation heatmap for given columns."""
    sub = df[cols].dropna()
    ranked = sub.rank(method="average")
    corr = ranked.corr(method="pearson")
    
    n = len(corr.columns)
    fig, ax = plt.subplots(figsize=(0.75 * n + 4, 0.75 * n + 3))
    im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    ax.set_title(title)
    
    plt.colorbar(im, ax=ax, label="Spearman ρ")
    
    # Text annotations
    vals = corr.to_numpy()
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{vals[i, j]:.2f}", ha="center", va="center", fontsize=9)
            
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=200)
    plt.close() # Close to free memory

def plot_clean_conf_vs_vulnerability(
    df: pd.DataFrame,
    severity: int,
    y_col: str,
    x_col: str = "msp_clean",
    corruption: Optional[str] = None,
    save_path: Optional[Path] = None
):
    """Scatter: Clean Confidence vs Vulnerability."""
    d = df[df["severity"] == severity].copy()
    if corruption:
        d = d[d["corruption"] == corruption]
    
    d = d.dropna(subset=[x_col, y_col])
    if len(d) > 8000: d = d.sample(8000, random_state=42)
    
    rho = spearman_r(d[x_col], d[y_col])
    
    plt.figure(figsize=(7, 5))
    plt.scatter(d[x_col], d[y_col], s=10, alpha=0.3)
    plt.xlabel("Clean Confidence (MSP)")
    plt.ylabel(f"Vulnerability ({y_col})")
    plt.title(f"Clean Conf vs Vuln (sev={severity})\nρ={rho:.3f}, n={len(d)}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=200)
    plt.close()

def plot_metric_vs_severity_lines(
    df_agg: pd.DataFrame,
    y_col: str,
    corruptions: list[str],
    ylabel: str,
    title: str,
    save_path: Optional[Path] = None
):
    """Line plot: Metric vs Severity, one line per corruption."""
    plt.figure(figsize=(9, 6))
    
    for corr in corruptions:
        d = _attach_clean_baseline(df_agg, corr)
        # Filter severities if needed, here taking all available
        plt.plot(d["severity"], d[y_col], marker="o", label=corr)
        
    plt.xlabel("Severity")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=200)
    plt.close()

def plot_deltaP_vs_deltaE(
    df_agg: pd.DataFrame,
    x_col: str,
    y_col: str,
    severities: list,
    save_path: Optional[Path] = None
):
    """Scatter: Delta Performance vs Delta Explanation (Aggregated)."""
    d = df_agg[df_agg["severity"].isin(severities)].copy()
    
    plt.figure(figsize=(8, 6))
    for corr, g in d.groupby("corruption"):
        plt.scatter(g[x_col], g[y_col], label=corr, alpha=0.8)
        # Annotate severity
        for _, row in g.iterrows():
            plt.annotate(str(int(row["severity"])), (row[x_col], row[y_col]), fontsize=8)
            
    # Overall correlation
    d_clean = d.dropna(subset=[x_col, y_col])
    rho = spearman_r(d_clean[x_col], d_clean[y_col])
    
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"ΔP vs ΔE (All corruptions)\nGlobal Spearman ρ = {rho:.3f}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=200)
    plt.close()

def plot_acc_vs_expl_dual_axis(
    df_agg: pd.DataFrame,
    corruption: str,
    expl_cols: dict, # {"Label": "col_name"}
    save_path: Optional[Path] = None
):
    """Dual axis plot: Accuracy (left) vs Explanation Similarity (right)."""
    d = _attach_clean_baseline(df_agg, corruption)
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    
    # Left: Accuracy
    p1 = ax1.plot(d["severity"], d["acc_corr"], 'k-o', label="Accuracy", linewidth=2)
    ax1.set_ylabel("Accuracy")
    ax1.set_xlabel("Severity")
    ax1.set_ylim(0, 1.05)
    
    # Right: Explanations
    lines = []
    for label, col in expl_cols.items():
        l = ax2.plot(d["severity"], d[col], linestyle='--', marker='x', label=label)
        lines.extend(l)
        
    ax2.set_ylabel("Explanation Similarity")
    
    # Legend
    lns = p1 + lines
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="lower left")
    
    plt.title(f"{corruption}: Accuracy vs Explanation Stability")
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=200)
    plt.close()

def plot_trust_zones_stacked(
    df_pairs: pd.DataFrame,
    severities: list,
    drift_col: str,
    save_path: Optional[Path] = None
):
    """Stacked bar chart of Trust Zones."""
    # Logic: 
    # High Drift = top 25% quantile of drift_col
    # Zones: Robust (Corr+LowDrift), Silent (Corr+HighDrift), Expected (Wrong+High), Stubborn (Wrong+Low)
    
    # Calculate global threshold
    d = df_pairs[df_pairs["severity"].isin(severities)].copy()
    threshold = d[drift_col].quantile(0.75)
    
    d["high_drift"] = d[drift_col] >= threshold
    d["is_correct"] = d["correct_corr"]
    
    # Assign zones
    conds = [
        (d["is_correct"] & ~d["high_drift"]),
        (d["is_correct"] & d["high_drift"]),
        (~d["is_correct"] & d["high_drift"]),
        (~d["is_correct"] & ~d["high_drift"])
    ]
    choices = ["Robust", "Silent Drift", "Expected Failure", "Stubborn Failure"]
    d["zone"] = np.select(conds, choices)
    
    # Aggregate
    grouped = d.groupby(["severity", "zone"]).size().unstack(fill_value=0)
    # Convert to percentage
    grouped = grouped.div(grouped.sum(axis=1), axis=0) * 100
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    grouped = grouped.reindex(columns=choices) # Ensure order
    
    bottom = np.zeros(len(grouped))
    for zone in choices:
        if zone in grouped.columns:
            ax.bar(grouped.index.astype(str), grouped[zone], bottom=bottom, label=zone)
            bottom += grouped[zone].values
            
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("Severity")
    ax.set_title(f"Trust Zones (Drift Threshold={threshold:.3f})")
    ax.legend(bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=200)
    plt.close()

def plot_violin_by_severity(
    df_pairs: pd.DataFrame,
    y_col: str,
    slice_col: str,
    corruption: str,
    save_path: Optional[Path] = None
):
    """Violin plot of a metric across severities for a specific corruption."""
    d = df_pairs[(df_pairs["corruption"] == corruption) & (df_pairs[slice_col])].copy()
    
    data = []
    positions = []
    
    sevs = sorted(d["severity"].unique())
    for s in sevs:
        vals = d[d["severity"] == s][y_col].dropna().values
        if len(vals) > 0:
            data.append(vals)
            positions.append(s)
            
    if not data: return

    plt.figure(figsize=(8, 5))
    plt.violinplot(data, positions=positions, showmedians=True)
    plt.xlabel("Severity")
    plt.ylabel(y_col)
    plt.title(f"Distribution of {y_col} over Severity\n({corruption}, {slice_col})")
    plt.grid(True, axis='y', alpha=0.25)
    
    if save_path: plt.savefig(save_path, dpi=200)
    plt.close()
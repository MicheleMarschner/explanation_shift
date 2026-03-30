import pandas as pd
from pathlib import Path
from src.configs.global_config import PATHS # Assuming this exists as per your snippet
from . import loaders
from . import plotting

# --- Configuration ---
CONFIG = {
    "severities": [1, 2, 3, 5],
    "metrics_heatmap": [
        "drift_1m_cos", "drift_1m_iou", "drift_1m_rho", 
        "abs_dH", "msp_corr"
    ],
    "expl_slice": "invariant", # "invariant", "both_correct", or "all" (if logic handles 'all')
    "plot_max_points": 8000
}

def main():
    # 1. Setup Paths
    exp_name = "experiment__n250__IG__seed51"
    exp_dir = PATHS.runs / exp_name
    
    # Output structure
    results_dir = PATHS.results / exp_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Data Loading
    
    # A. Aggregated Table (Stage 2 Results)
    # This usually exists from the drift run
    agg_csv = exp_dir / "02__drift" / "02__drift_results.csv"
    if not agg_csv.exists():
        print(f"Warning: Aggregated results not found at {agg_csv}")
        df_agg = pd.DataFrame() # Empty to skip dependent plots
    else:
        df_agg = loaders.load_aggregated_results(agg_csv)
        print(f"Loaded aggregated results: {len(df_agg)} rows")

    # B. Pairs Table (Granular Data)
    # Rebuild from .pt files or load cached CSV
    pairs_csv_path = results_dir / "pairs_table.csv"
    df_pairs = loaders.build_pairs_table(
        exp_dir, 
        save_path=pairs_csv_path, 
        force_rebuild=False
    )
    print(f"Loaded pairs table: {len(df_pairs)} rows")
    
    corruptions = sorted(df_pairs["corruption"].unique())
    
    # ---------------------------------------------------------
    # 3. Generate Plots
    # ---------------------------------------------------------
    
    print("Generating plots...")

    # --- Group A: Granular Analysis (Heatmaps, Scatters) ---
    
    # A1. Correlation Heatmap (Pooled)
    plotting.plot_correlation_heatmap(
        df_pairs,
        cols=CONFIG["metrics_heatmap"],
        title="Metric Correlation (Pooled)",
        save_path=results_dir / "heatmap_pooled.png"
    )
    
    # A2. Clean Confidence vs Vulnerability
    plotting.plot_clean_conf_vs_vulnerability(
        df_pairs,
        severity=3,
        y_col="drift_1m_rho",
        save_path=results_dir / "scatter_clean_conf_vs_vuln_sev3.png"
    )
    
    # A3. Trust Zones (Pooled severities)
    plotting.plot_trust_zones_stacked(
        df_pairs,
        severities=CONFIG["severities"],
        drift_col="drift_1m_cos", # Using 1-cosine as drift
        save_path=results_dir / "trust_zones_stacked.png"
    )

    # A4. Violin Plots (Per Corruption)
    violin_dir = results_dir / "violins"
    violin_dir.mkdir(exist_ok=True)
    for corr in corruptions:
        plotting.plot_violin_by_severity(
            df_pairs,
            y_col="rho",
            slice_col="invariant",
            corruption=corr,
            save_path=violin_dir / f"violin_rho_{corr}.png"
        )

    # --- Group B: Aggregated Analysis (Trends over Severity) ---
    
    if not df_agg.empty:
        drift_plots_dir = results_dir / "drift_trends"
        drift_plots_dir.mkdir(exist_ok=True)

        # B1. Severity vs Metric (Line plots)
        # Assuming column names match Stage 2 output e.g., 'exp_inv__rho_mean'
        metric_key = "rho_mean"
        col_name = f"exp_{CONFIG['expl_slice']}__{metric_key}"
        
        if col_name in df_agg.columns:
            plotting.plot_metric_vs_severity_lines(
                df_agg,
                y_col=col_name,
                corruptions=corruptions,
                ylabel=f"Similarity ({metric_key})",
                title=f"Severity vs Explanation Similarity ({CONFIG['expl_slice']})",
                save_path=drift_plots_dir / "severity_vs_rho.png"
            )
            
            # B2. Delta P vs Delta E
            # Assuming 'mean_abs_delta_entropy' exists or calculation of 1-acc
            if "mean_abs_delta_entropy" in df_agg.columns:
                plotting.plot_deltaP_vs_deltaE(
                    df_agg,
                    x_col="mean_abs_delta_entropy",
                    y_col=col_name, # Note: function expects similarity? or drift?
                    # If function expects drift, we might need 1-x. 
                    # Let's assume user wants to plot Raw Similarity vs Entropy change
                    severities=CONFIG["severities"],
                    save_path=drift_plots_dir / "deltaP_entropy_vs_sim.png"
                )

        # B3. Accuracy vs Explanation (Dual Axis)
        for corr in corruptions:
            plotting.plot_acc_vs_expl_dual_axis(
                df_agg,
                corruption=corr,
                expl_cols={
                    "Inv Cosine": "exp_inv__cos_mean", 
                    "Inv Rank": "exp_inv__rho_mean"
                },
                save_path=drift_plots_dir / f"acc_vs_expl_{corr}.png"
            )

    print(f"Analysis complete. Results saved to {results_dir}")
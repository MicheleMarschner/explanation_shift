import numpy as np

from analysis.explanation_shift import run_explanation_shift_analysis
from analysis.decoupling import run_decoupling_analysis
from analysis.explanation_metrics import run_quantus_analysis
from analysis.sample_level_analysis import run_sample_level_analysis
from configs.global_config import PATHS
from data import get_clean_data, get_corrupted_data
from analysis.qualitative_rendering import render_trust_zone_exemplars
from analysis.analysis_helper import SLICES

if __name__ == "__main__":

    experiments_dir = PATHS.runs
    output_dir = PATHS.results
    n = 1000
    de = "iou"              # "cos", "iou", "rho"
    dp = "flip_rate"        # "p_shift", "flip_rate", "err_rate", "margin_shift"
    slice = "both_corr"           # "all", "inv", "both_corr"

    # optional
    x_axis = "severity"     # "mmd", "severity"
    signed = "store_true"   # !TODO und für nicht???
    threshold_quantile = 0.75       # Global ΔE quantile defining 'high drift' in Fig 9
    heatmap_mode = "by_corruption"      # "pooled", "by_corruption"

    run_explanation_shift_analysis(
        experiments_dir=experiments_dir,
        output_root=output_dir,
        n=n,
        x_axis=x_axis,
        slices=SLICES
    )

    #python src/evaluation/run_analysis.py --experiments-dir experiments --output-dir results --de rho --dp flip_rate 
    run_decoupling_analysis(
        experiments_dir=experiments_dir,
        output_root=output_dir,
        n=n,
        de=de,
        dp=dp,
        scatter_slice=slice,
    )
    
    run_quantus_analysis(
        experiments_dir=experiments_dir,
        output_root=output_dir,
        n=n,
        de=de,
        scatter_slice=slice,
        signed=signed,
    )

    run_sample_level_analysis(
        experiments_dir=experiments_dir,
        output_root=output_dir,
        n=n,
        de_basis=de,
        threshold_quantile=threshold_quantile,
        violin_slice=slice,
        violin_similarity=de,
        scatter_slice=slice,
        vulnerability_de=de,
        vulnerability_slice=slice,
        heatmap_mode=heatmap_mode,
        heatmap_slice=slice,
    )

    def clean_loader(pair_idx: np.ndarray):
        _, X_clean, _ = get_clean_data(
            path=PATHS.data_clean, idx=pair_idx, transform=None,
        )
        return X_clean


    def corr_loader(pair_idx: np.ndarray, corruption: str, severity: int):
        _, X_corr, _ = get_corrupted_data(
            idx=pair_idx, path=PATHS.data_corr, transform=None,
            corruption=corruption, severity=severity,
        )
        return X_corr


    render_trust_zone_exemplars(
        exemplars_csv=PATHS.results / "analysis" / "trust_zone_exemplars.csv",
        experiments_dir=PATHS.runs,
        output_dir=PATHS.results / "qualitative_imgs",
        image_loader_clean=clean_loader,
        image_loader_corr=corr_loader,
        # recommended filters for a focused set of paper figures:
        zones=("silent_drift", "stubborn_failure"),
        corruptions=["fog", "gaussian_noise"],
        severities=[3, 5],
        seeds=[7],  # any single representative seed
    )
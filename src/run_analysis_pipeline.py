from pathlib import Path
import numpy as np

from src.analysis.aggregate import aggregate_all_experiments
from src.analysis.analysis_helper import EXPERIMENT_DIR_RE, SLICES
from src.analysis.decoupling import run_decoupling_analysis
from src.analysis.explanation_metrics import run_quantus_analysis
from src.analysis.explanation_shift import run_explanation_shift_analysis
from src.analysis.qualitative_rendering import render_trust_zone_exemplars
from src.analysis.sample_level_analysis import run_sample_level_analysis
from src.configs.global_config import PATHS, PROJECT_ROOT
from src.data import get_clean_data, get_corrupted_data
from src.scripts import download_experiments


def find_experiment_dirs(experiments_dir: Path) -> list[Path]:
    if not experiments_dir.exists() or not experiments_dir.is_dir():
        return []

    return sorted(
        p for p in experiments_dir.iterdir()
        if p.is_dir() and EXPERIMENT_DIR_RE.match(p.name)
    )


def first_experiment_has_results_csv(experiment_dirs: list[Path]) -> bool:
    if not experiment_dirs:
        return False

    first_experiment_dir = experiment_dirs[0]
    return any(first_experiment_dir.rglob("*_results.csv"))


def run_analysis_pipeline(experiments_dir):
    experiment_dirs = find_experiment_dirs(experiments_dir)

    if not experiment_dirs:
        download_experiments(PROJECT_ROOT)
        experiment_dirs = find_experiment_dirs(experiments_dir)

    if not experiment_dirs:
        raise FileNotFoundError(
            f"No experiment directories matching "
            f"'experiment__n{{samples}}__{{explainer}}__seed{{seed}}' found in {experiments_dir}"
        )

    if not first_experiment_has_results_csv(experiment_dirs):
        aggregate_all_experiments(experiments_dir)
    else:
        print(
            f"[info] Skipping aggregation: found existing '*_results.csv' "
            f"in {experiment_dirs[0]}"
        )

    output_dir = PATHS.results
    n = 1000
    de = "rho"                      # "cos", "iou", "rho"
    dp = "flip_rate"                # "p_shift", "flip_rate", "err_rate", "margin_shift"
    slice = "all"             # "all", "inv", "both_corr"

    # optional
    x_axis = "severity"             # "mmd", "severity"
    signed = "store_true"   
    threshold_quantile = 0.75       # Global ΔE quantile defining 'high drift' in Fig 9
    heatmap_mode = "pooled"         # "pooled", "by_corruption"

    run_explanation_shift_analysis(
        experiments_dir=experiments_dir,
        output_root=output_dir,
        n=n,
        x_axis=x_axis,
        slices=SLICES
    )

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
        zones=("Silent Drift", "Stubborn Failure"),
        corruptions=["fog", "gaussian_noise"],
        severities=[3, 5],
        seeds=[7],
    )
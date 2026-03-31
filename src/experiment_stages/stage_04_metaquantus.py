from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import gc
import numpy as np
import torch
import quantus
import metaquantus

from src.configs.global_config import DEVICE, IG_STEPS, BATCH_SIZE_EXPLAINER, BATCH_SIZE, PATHS
from src.data import get_clean_data
from src.models import load_model
from src.utils import collect_x_from_loader, cpu, to_np_idx
from src.explainers import compute_saliency_maps


def build_metaquantus_estimators() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    MetaQuantus-style estimator dictionary for all metrics used in this project.

    score_direction:
        True  -> lower is better
        False -> higher is better
    """
    return {
        "Complexity": {
            "Sparseness": {
                "init": quantus.Sparseness(
                    abs=True,
                    normalise=True,
                    normalise_func=quantus.normalise_func.normalise_by_max,
                    return_aggregate=False,
                    aggregate_func=np.mean,
                    disable_warnings=True,
                    display_progressbar=False,
                ),
                "score_direction": False,   # higher better
            },
        },
        "Robustness": {
            "AvgSensitivity": {
                "init": quantus.AvgSensitivity(
                    nr_samples=50,
                    lower_bound=0.2,
                    norm_numerator=quantus.norm_func.fro_norm,
                    norm_denominator=quantus.norm_func.fro_norm,
                    perturb_func=quantus.functions.perturb_func.batch_uniform_noise,
                    perturb_func_kwargs={"perturb_mean": 0.0, "perturb_std": 0.15},
                    similarity_func=quantus.similarity_func.difference,
                    abs=True,
                    return_nan_when_prediction_changes=True,
                    normalise=True,
                    normalise_func=quantus.normalise_func.normalise_by_max,
                    return_aggregate=False,
                    aggregate_func=np.mean,
                    disable_warnings=False,
                    display_progressbar=False,
                ),
                "score_direction": True,    # lower better
            },
        },
        "Faithfulness": {
            "FaithfulnessCorrelation": {
                "init": quantus.FaithfulnessCorrelation(
                    nr_runs=100,
                    subset_size=300,
                    perturb_baseline="mean",
                    perturb_func=quantus.functions.perturb_func.batch_baseline_replacement_by_indices,
                    similarity_func=quantus.similarity_func.correlation_pearson,
                    abs=True,
                    normalise=True,
                    normalise_func=quantus.normalise_func.normalise_by_max,
                    return_aggregate=False,
                    aggregate_func=np.mean,
                    disable_warnings=False,
                    display_progressbar=False,
                ),
                "score_direction": False,   # higher better
            },
        },
    }


def build_metaquantus_test_suite() -> Dict[str, Any]:
    """
    Directly mirrors the package's intended test suite structure.
    """
    return {
        "Model Resilience Test": metaquantus.ModelPerturbationTest(
            noise_type="multiplicative",
            mean=1.0,
            std=0.001,
            type="Resilience",
        ),
        "Model Adversary Test": metaquantus.ModelPerturbationTest(
            noise_type="multiplicative",
            mean=1.0,
            std=2.0,
            type="Adversary",
        ),
        "Input Resilience Test": metaquantus.InputPerturbationTest(
            noise=0.001,
            type="Resilience",
        ),
        "Input Adversary Test": metaquantus.InputPerturbationTest(
            noise=5.0,
            type="Adversary",
        ),
    }


def build_xai_methods() -> Dict[str, Dict[str, Any]]:
    """
    MetaQuantus-style XAI method registry.
    The method name is passed through to metaquantus_explain_func via kwargs["method"].
    """
    return {
        "IG": {},
        "GradCAM": {},
    }


def metaquantus_explain_func(model, inputs, targets, **kwargs):
    """
    Adapter so MetaQuantus uses the project's own explanation pipeline.

    Returns numpy saliency maps with shape [N, 1, H, W].
    """
    method = kwargs["method"]

    x_t = torch.as_tensor(inputs, dtype=torch.float32)
    t_t = torch.as_tensor(targets, dtype=torch.long)

    sal = compute_saliency_maps(
        x_t,
        target=t_t,
        explainer_name=method,
        model=model,
        device=DEVICE,
        steps=IG_STEPS,
        internal_bs=BATCH_SIZE_EXPLAINER,
        batch_size=BATCH_SIZE,
    )

    a = sal.abs().unsqueeze(1)
    return cpu(a).numpy()


def run_metaquantus_stage(
    pair_idx,
    transform,
    save_path: Path,
    iterations: int = 2,
    nr_perturbations: int = 3,
) -> dict:
    """
    Runs MetaQuantus benchmarking for all metrics used in this project on the clean subset.
    Saves one independent .pt artifact.

    This stage is intentionally independent of Stage 03 artifacts.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model = load_model(device=DEVICE)

    clean_loader, _, y_clean = get_clean_data(
        path=PATHS.data_clean,
        idx=pair_idx,
        transform=transform, 
    )
    X_clean_t = collect_x_from_loader(clean_loader)
    x_batch = cpu(X_clean_t).numpy()
    y_batch = np.asarray(y_clean, dtype=np.int64)

    estimators = build_metaquantus_estimators()
    test_suite = build_metaquantus_test_suite()
    xai_methods = build_xai_methods()

    benchmark_results = {}

    for category, category_estimators in estimators.items():
        benchmark_results[category] = {}

        for metric_name, metric_info in category_estimators.items():
            estimator = metric_info["init"]
            score_direction = metric_info["score_direction"]

            evaluator = metaquantus.MetaEvaluation(
                test_suite=test_suite,
                xai_methods=xai_methods,
                iterations=iterations,
                nr_perturbations=nr_perturbations,
                explain_func=metaquantus_explain_func,
                write_to_file=False,
                print_results=False,
            )

            result_obj = evaluator(
                estimator=estimator,
                model=model,
                x_batch=x_batch,
                y_batch=y_batch,
                a_batch=None,
                s_batch=None,
                channel_first=True,
                softmax=False,
                device=DEVICE,
                model_predict_kwargs={},
                score_direction=score_direction,
            )

            benchmark_results[category][metric_name] = {
                "results_eval_scores": result_obj.get_results_eval_scores(),
                "results_eval_scores_perturbed": result_obj.get_results_eval_scores_perturbed(),
                "results_y_preds_perturbed": result_obj.get_results_y_preds_perturbed(),
                "results_indices_perturbed": result_obj.get_results_indices_perturbed(),
                "results_y_true": result_obj.get_results_y_true(),
                "results_y_preds": result_obj.get_results_y_preds(),
                "results_indices_correct": result_obj.get_results_indices_correct(),
                "results_intra_scores": result_obj.get_results_intra_scores(),
                "results_inter_scores": result_obj.get_results_inter_scores(),
                "results_meta_consistency_scores": result_obj.get_results_meta_consistency_scores(),
                "results_consistency_scores": result_obj.get_results_consistency_scores(),
            }

            gc.collect()
            torch.cuda.empty_cache()

    payload = {
        "meta": {
            "iterations": int(iterations),
            "nr_perturbations": int(nr_perturbations),
            "xai_methods": list(xai_methods.keys()),
            "estimator_categories": list(estimators.keys()),
            "pair_idx": [int(i) for i in pair_idx],
        },
        "results": benchmark_results,
    }

    torch.save(payload, save_path)
    return payload
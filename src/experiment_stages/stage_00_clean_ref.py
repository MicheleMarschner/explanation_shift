from pathlib import Path
from typing import Any, Dict
import torch

from src.configs.global_config import BATCH_SIZE_EXPLAINER, DEVICE, IG_STEPS, PATHS
from src.experiment_stages.helper import save_experiment_reference
from src.data import get_clean_data
from src.distribution_shift import estimate_sigma
from src.explainers import compute_saliency_maps
from src.models import (
    entropy_from_logits,
    predict_logits_and_accuracy,
    predict_resnet_embeddings,
    transform_logits_to_preds,
    transform_logits_to_probs,
)
from src.utils import collect_x_from_loader


def compute_clean_reference(
    pair_idx,
    exp_config: Any,
    save_path: Path,
    model,
    transform,
    explainer_name: str,
    seed: int,
) -> Dict[str, Any]:
    """
    Stage 00: compute clean predictions + IG saliency + embedding reference.
    Saves a PT artifact at save_path and returns the same dict (loaded-friendly format).
    """
    # Load clean subset
    clean_loader, X_clean, y_clean = get_clean_data(path=PATHS.data_clean, idx=pair_idx, transform=transform)
    assert X_clean.shape[0] == len(pair_idx)
    
    # Predictions on clean
    logits_clean, acc_clean = predict_logits_and_accuracy(model, clean_loader)
    pred_clean = transform_logits_to_preds(logits_clean)
    proba_clean = transform_logits_to_probs(logits_clean)
    entropy_clean = entropy_from_logits(logits_clean)

    # Embeddings (for sigma reference)
    E_clean = predict_resnet_embeddings(model, clean_loader)
    sigma_ref = estimate_sigma(E_clean)

    
    # Saliency on clean
    X_clean_t = collect_x_from_loader(clean_loader)
    target = pred_clean

    sal_clean = compute_saliency_maps(
        X_clean_t,
        target=target,
        explainer_name=explainer_name,
        model=model,
        device=DEVICE,
        steps=IG_STEPS,
        internal_bs=BATCH_SIZE_EXPLAINER,
        batch_size=BATCH_SIZE_EXPLAINER,
    )

    clean_ref = {
        "logits": logits_clean,
        "pred": pred_clean,
        "proba": proba_clean,
        "acc": acc_clean,
        "entropy": entropy_clean,
        "E": E_clean,
        "sal": sal_clean,
        "sigma": sigma_ref,
        "y": torch.as_tensor(y_clean),
    }

    save_experiment_reference(
        save_path=save_path,
        seed=seed,
        pair_idx=pair_idx,
        exp_config=exp_config,
        clean_ref=clean_ref,
    )

    return torch.load(save_path, map_location="cpu")

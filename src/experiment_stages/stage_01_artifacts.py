import torch
import numpy as np
import time

from src.configs.global_config import BATCH_SIZE_EXPLAINER, DEVICE, IG_STEPS, PATHS, TargetPolicy, TARGET_POLICY
from src.experiment_stages.helper import save_artifacts
from src.utils import collect_x_from_loader
from src.data import get_corrupted_data
from src.models import predict_resnet_embeddings, transform_logits_to_preds, predict_logits_and_accuracy, entropy_from_logits, transform_logits_to_probs
from src.explainers import ig_saliency_batched


def run_experiment(
        pair_idx, 
        corruption, 
        severity, 
        clean_path, 
        save_path,
        model, 
        transform,
        explainer,
    ):

    ref = torch.load(clean_path, map_location="cpu", weights_only=False)
    cr = ref['clean_reference']

    # IG explanations (timed)
    torch.cuda.synchronize() if DEVICE == "cuda" else None
    t0 = time.time()

    pred_clean = cr['pred_clean']
    y_true = cr['y_true']
    y_true_np = y_true.detach().cpu().numpy()
    
    corr_dataloader, X_corr, y_corr = get_corrupted_data(
        idx=pair_idx, 
        path=PATHS.data_corr,
        transform=transform, 
        corruption=corruption, 
        severity=severity)

    assert X_corr.shape[0] == len(pair_idx)
    assert np.array_equal(y_true_np, y_corr)

    # predictions
    logits_corr, acc_corr = predict_logits_and_accuracy(model, corr_dataloader)
    pred_corr = transform_logits_to_preds(logits_corr)
    proba_corr = transform_logits_to_probs(logits_corr)
    entropy_corr  = entropy_from_logits(logits_corr)    # [B]

    E_corr = predict_resnet_embeddings(model, corr_dataloader)

    # IG saliency
    X_corr_t = collect_x_from_loader(corr_dataloader)
    target = pred_clean.to(DEVICE) if TARGET_POLICY == TargetPolicy.PRED_CLEAN else pred_corr.to(DEVICE) 
    sal_corr  = ig_saliency_batched(
        X_corr_t, 
        target=target, 
        device=DEVICE, 
        explainer=explainer, 
        steps=IG_STEPS, 
        internal_bs=BATCH_SIZE_EXPLAINER, 
        batch_size=BATCH_SIZE_EXPLAINER
    )

    corr_ref = {
        "logits": logits_corr,
        "pred": pred_corr,
        "proba": proba_corr,
        "acc": acc_corr,
        "entropy": entropy_corr,
        "E": E_corr,
        "sal": sal_corr,
    }

    torch.cuda.synchronize() if DEVICE == "cuda" else None
    t_ig = time.time() - t0

    save_artifacts(
        save_path= save_path,
        corruption = corruption,
        severity = severity,
        time = t_ig,
        corr_ref=corr_ref,
    )

    print(f"\n[{corruption} sev{severity}] IG time: {t_ig:.2f}s for N={len(pair_idx)} (steps={IG_STEPS})")
    
    return {
        "corruption": corruption, 
        "severity": severity, 
        "time_sec": t_ig,
        "acc_corr": acc_corr,    
    }
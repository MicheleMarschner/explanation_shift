from pathlib import Path
import torch

from src.experiment_stages.helper import save_drift_metrics
from src.explainers import mask_invariant, mask_correct, compute_explanation_drift_metrics
from src.distribution_shift import compute_confidence_shift_metrics, compute_shift_strength_mmd
from src.utils import prefix_keys


def compute_drift_metrics_unpaired(clean_path: Path, artifact_path: Path, save_path: Path) -> dict:
    ref = torch.load(clean_path, map_location="cpu", weights_only=False)
    art = torch.load(artifact_path, map_location="cpu", weights_only=False)

    cr = ref["clean_reference"]
    cc = art["corrupt_reference"]

    H_clean = cr["entropy_clean"].float()
    H_corr = cc["entropy_corr"].float()

    logits_clean = cr["logits_clean"]
    logits_corr = cc["logits_corr"]

    E_clean = cr["E_clean"].float()
    E_corr = cc["E_corr"].float()
    sigma_ref = float(cr["sigma_ref"])

    sal_clean = cr["sal_clean"].float()
    sal_corr = cc["sal_corr"].float()

    mmd2, _ = compute_shift_strength_mmd(E_clean, E_corr, sigma_ref)
    mmd2 = max(0.0, float(mmd2))

    conf_clean = torch.softmax(logits_clean, dim=1).max(dim=1).values
    conf_corr = torch.softmax(logits_corr, dim=1).max(dim=1).values

    sal_mass_clean = sal_clean.abs().reshape(sal_clean.size(0), -1).mean(dim=1)
    sal_mass_corr = sal_corr.abs().reshape(sal_corr.size(0), -1).mean(dim=1)

    row = {
        "corruption": art["corruption"],
        "severity": int(art["severity"]),
        "max_mean_discrepancy": mmd2,
        "mean_entropy_clean": float(H_clean.mean().item()),
        "mean_entropy_corr": float(H_corr.mean().item()),
        "delta_mean_entropy_unpaired": float(H_corr.mean().item() - H_clean.mean().item()),
        "mean_conf_clean": float(conf_clean.mean().item()),
        "mean_conf_corr": float(conf_corr.mean().item()),
        "delta_mean_conf_unpaired": float(conf_corr.mean().item() - conf_clean.mean().item()),
        "mean_sal_mass_clean": float(sal_mass_clean.mean().item()),
        "mean_sal_mass_corr": float(sal_mass_corr.mean().item()),
        "delta_mean_sal_mass_unpaired": float(sal_mass_corr.mean().item() - sal_mass_clean.mean().item()),
    }

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"row": row}, save_path)

    return row



def compute_drift_metrics(clean_path: Path, artifact_path: Path, save_path: Path) -> tuple[dict, dict]:
    ref = torch.load(clean_path, map_location="cpu", weights_only=False)
    art = torch.load(artifact_path, map_location="cpu", weights_only=False)
    
    cr = ref["clean_reference"]
    cc = art["corrupt_reference"]

    logits_clean = cr["logits_clean"]
    pred_clean   = cr["pred_clean"].long()
    H_clean      = cr["entropy_clean"].float()
    E_clean      = cr["E_clean"].float()
    sal_clean    = cr["sal_clean"].float()
    sigma_ref    = float(cr["sigma_ref"])
    y_true       = cr["y_true"].long()

    logits_corr  = cc["logits_corr"]
    pred_corr    = cc["pred_corr"].long()
    H_corr       = cc["entropy_corr"].float()
    E_corr       = cc["E_corr"].float()
    sal_corr     = cc["sal_corr"].float()

    # slice 1: prediction_invariance mask + rate
    invariant = mask_invariant(pred_clean, pred_corr).bool()
    n_invariant = invariant.sum().item()
    print(f" labels corresponding both domains {n_invariant} from {len(pred_clean)}")
    invariant_rate = float(invariant.float().mean().item())

    # slice 2: both_correct (only if labels exist)
    both_correct = mask_correct(pred_clean, pred_corr, y_true).bool()
    n_both_correct = both_correct.sum().item()
    print(f" labels correct in both domains {n_both_correct} from {len(pred_clean)}")
    both_correct_rate = float(both_correct.float().mean().item())

    # shift strength
    mmd2, _ = compute_shift_strength_mmd(E_clean, E_corr, sigma_ref)
    mmd2 = max(0.0, float(mmd2))    # shouldn't be negative, but can happen slightly due to finite sample size

    # entropy drift (ΔH)
    # Measures how the model's prediction uncertainty changes from clean to corrupted inputs.
    # Entropy is computed from the full softmax distribution over classes.
    # Positive ΔH means higher uncertainty under corruption; negative ΔH means lower uncertainty.
    dH = (H_corr - H_clean)
    mean_dH = float(dH.mean().item())
    mean_abs_dH = float(dH.abs().mean().item())

    row = {
        "corruption": art["corruption"],
        "severity": int(art["severity"]),
        "invariant_rate": invariant_rate,
        "n_invariant": int(n_invariant),
        "both_correct_rate": both_correct_rate,
        "n_both_correct": int(n_both_correct),
        "max_mean_discrepancy": mmd2,
        "mean_delta_entropy": mean_dH,
        "mean_abs_delta_entropy": mean_abs_dH,
    }

    vectors = {
        "invariant": invariant.cpu(),
        "both_correct": both_correct.cpu(),
        "dH": dH.cpu()
    }

    # explanation drift and confidence drift for 3 slices: all, prediction invariance (pred_clean == pred_corr) and both correct ((pred_clean==y) & (pred_corr==y))
    # ALL (no mask)
    exp_vec_all, exp_sum_all = compute_explanation_drift_metrics(sal_clean, sal_corr, mask=None)
    conf_vec_all, conf_sum_all = compute_confidence_shift_metrics(logits_clean, logits_corr, mask=None)

    row.update(prefix_keys(exp_sum_all, "exp_all__"))
    row.update(prefix_keys(conf_sum_all, "conf_all__"))
    vectors.update(prefix_keys(exp_vec_all, "exp__"))
    vectors.update(prefix_keys(conf_vec_all, "conf__"))

    # INVARIANT
    _, exp_sum_inv = compute_explanation_drift_metrics(sal_clean, sal_corr, mask=invariant)
    _, conf_sum_inv = compute_confidence_shift_metrics(logits_clean, logits_corr, mask=invariant)

    row.update(prefix_keys(exp_sum_inv, "exp_inv__"))
    row.update(prefix_keys(conf_sum_inv, "conf_inv__"))

    # BOTH_CORRECT
    _, exp_sum_both_correct = compute_explanation_drift_metrics(sal_clean, sal_corr, mask=both_correct)
    _, conf_sum_both_correct = compute_confidence_shift_metrics(logits_clean, logits_corr, mask=both_correct)
    
    row.update(prefix_keys(exp_sum_both_correct, "exp_both_corr__"))
    row.update(prefix_keys(conf_sum_both_correct, "conf_both_corr__"))

    save_drift_metrics(save_path, row, vectors)

    return row, vectors
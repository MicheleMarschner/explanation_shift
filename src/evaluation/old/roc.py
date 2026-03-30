
'''
## Plot 1: roc folder

Description
This experiment shows that explanation drift under corruption can be used as a warning signal for when the model will fail.

It takes images the model was correct on when clean, checks whether they become wrong after corruption, and compares 
different warning signals (uncertainty, confidence drop, explanation drift). The ROC/AUC results show that explanation 
drift separates “will fail” vs “will not fail” best, and the bootstrap ΔAUC confirms this improvement over uncertainty 
is robust, not just noise.

Drift as a "Failure Predictor" (ROC/PR Curves)
This is a very strong academic angle. You can test if Explanation Drift is a better predictor of the model being wrong than 
Maximum Softmax Probability (MSP).
Plot: An ROC curve where you use "1 - Cosine Similarity" as a "score" to predict whether a sample will be misclassified.
Value: If the AUC is high, it proves that the explanation drift is a valid Out-of-Distribution (OOD) detection signal.
Scientific Claim: "Explanation drift can detect model failure even when the model's internal confidence (Softmax) remains high."

Explanation Drift as an OOD Detector (ROC Curve)
Research Question: Can we use the "Drift" score to detect if an image is corrupted before the model makes a mistake?
The Plot: An ROC Curve where you try to distinguish "Clean" from "Severity 3" images.
Line 1: Using Softmax Entropy as the signal.
Line 2: Using Explanation Drift (1 - Rho) as the signal.
Value for Report: If the "Drift" line is higher than the "Entropy" line, you have discovered a New Monitoring Tool. You can 
claim: "Explanations are better sensors for environmental change than the model's own confidence scores."
------

Generalization table/plot across conditions
For each corruption × severity: report

AUC(uncertainty), AUC(uncertainty+drift), ΔAUC
Then show a heatmap or small multiples.

Bootstrap CI for ΔAUC per condition (or at least aggregated)
Even just showing median ΔAUC and the fraction of conditions where the CI excludes 0 is very convincing.

Aggregate evidence like this:

For each condition c: compute ΔAUC and CI via bootstrap.

Report:
median ΔAUC across conditions
% of conditions with ΔAUC > 0
% of conditions with CI entirely > 0
'''



import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import torch
from scipy.stats import spearmanr

from src.configs.global_config import PATHS


exp_dir = PATHS.runs / "experiment__n250__IG__seed51"
out_dir = PATHS.results / "roc"
out_dir.mkdir(parents=True, exist_ok=True)

def load_clean_ref(keys):
    ref_pt  = exp_dir / "00__reference" / "00__clean_ref.pt"
    ref = torch.load(ref_pt, map_location="cpu")

    cr = ref["clean_reference"]
    res = {}
    for k in keys:
        if k not in cr:
            raise KeyError(f"Missing key '{k}' in clean_reference. Available: {list(cr.keys())[:20]} ...")
        res[k] = cr[k]
    return res


def load_corr_ref(corr, sev, keys):
    corr_pt = exp_dir / "01__artifacts" / f"01__artifacts__{corr}__sev{sev}.pt"
    art = torch.load(corr_pt, map_location="cpu", weights_only=False)

    cc = art["corrupt_reference"]
    res = {}
    for k in keys:
        if k not in cc:
            raise KeyError(f"Missing key '{k}' in corrupt_reference. Available: {list(cc.keys())[:20]} ...")
        res[k] = cc[k]
    return res


def load_drift_artifacts(corr, sev, keys):
    drift_pt = exp_dir / "02__drift" / f"02__drift_{corr}__sev{sev}.pt"
    ref = torch.load(drift_pt, map_location="cpu")

    res = []
    for key in keys: 
        res.append(ref[f"{key}"])
    
    return res
    
# ==========================================
# 1. DATA PREPARATION
# ==========================================

corruption = "fog"
severity = 3

clean_keys = ["pred_clean", "proba_clean", "E_clean", "y_clean"]
clean_ref = load_clean_ref(clean_keys)

corr_keys = ["pred_corr", "proba_corr", "E_corr"]
corr_ref = load_corr_ref(corr=corruption, sev=severity, keys=corr_keys)


# Ground Truth Labels (N,)
y_true = clean_ref['y_clean']

preds_clean = clean_ref['pred_clean']
conf_clean = clean_ref['proba_clean']
E_clean = clean_ref['E_clean']

preds_corr = corr_ref['pred_corr']
conf_corr = corr_ref['proba_corr']
E_corr = corr_ref['E_corr']

p_hat_clean = conf_clean.gather(1, preds_clean.view(-1, 1)).squeeze(1)  # [N]
p_hat_corr = conf_corr.gather(1, preds_corr.view(-1, 1)).squeeze(1)

drift_scores = 1.0 - torch.nn.functional.cosine_similarity(E_clean, E_corr, dim=1)

# ==========================================
# 2. FILTERING: Condition on Clean Correctness
# ==========================================
# Scientific Standard: We only care about failures on concepts the model 
# actually "knew" before corruption.
clean_correct_mask = (preds_clean == y_true)

print(f"Total Samples: {len(y_true)}")
print(f"Clean Correct Samples (Subset for Analysis): {clean_correct_mask.sum().item()}")

# Apply mask to everything
y_true_valid       = y_true[clean_correct_mask]
preds_corr_valid   = preds_corr[clean_correct_mask]
conf_clean_valid   = conf_clean[clean_correct_mask]
conf_corr_valid    = conf_corr[clean_correct_mask]
drift_scores_valid = drift_scores[clean_correct_mask]

p_hat_clean_valid = p_hat_clean[clean_correct_mask]
p_hat_corr_valid  = p_hat_corr[clean_correct_mask]

# ==========================================
# 3. DEFINE THE TARGET: "Did it Fail?"
# ==========================================
# 1 = Model Failed (Wrong Prediction)
# 0 = Model Succeeded (Correct Prediction)
# We use the filtered subset, so 'Failure' implies corruption broke the model.
y_failure_target = (preds_corr_valid != y_true_valid).to(torch.int64)

# Safety Check
if len(np.unique(y_failure_target)) < 2:
    raise ValueError("Error: The model is either 100% correct or 100% wrong on the subset. Cannot compute ROC.")

# ==========================================
# 4. CALCULATE THE THREE SCORES
# ==========================================

# A. Absolute Uncertainty (The Weak Baseline)
# Hypothesis: "If the model is unsure (Low Conf on corr), it's likely wrong."
# Formula: 1.0 - Confidence_corr
score_abs_uncertainty = 1.0 - p_hat_corr_valid

# B. Confidence Drift (The Strong Baseline)
# Hypothesis: "If the confidence dropped compared to clean (even if still high), it's likely wrong."
# Formula: Confidence_Clean - Confidence_corr
score_conf_drift = p_hat_clean_valid - p_hat_corr_valid

# C. Explanation Drift (Your Method)
# Hypothesis: "If the explanation changed, the reasoning is broken."
score_exp_drift = drift_scores_valid

# ==========================================
# 5. COMPUTE AUC & PLOT
# ==========================================
y_failure_np   = y_failure_target.detach().cpu().numpy()
abs_unc_np     = score_abs_uncertainty.detach().cpu().numpy()
conf_drift_np  = score_conf_drift.detach().cpu().numpy()
exp_drift_np   = score_exp_drift.detach().cpu().numpy()

print("y_failure_np shape:", y_failure_np.shape, "unique:", np.unique(y_failure_np)[:20])
print("abs_unc_np shape:", abs_unc_np.shape, "dtype:", abs_unc_np.dtype)
print("conf_corr_valid shape:", conf_corr_valid.shape)

auc_abs  = roc_auc_score(y_failure_np, abs_unc_np)
auc_conf = roc_auc_score(y_failure_np, conf_drift_np)
auc_exp  = roc_auc_score(y_failure_np, exp_drift_np)

print("\n--- Failure Prediction Results ---")
print(f"1. Explanation Drift (Yours):  AUC = {auc_exp:.4f}")
print(f"2. Confidence Drift (Baseline): AUC = {auc_conf:.4f}")
print(f"3. Absolute Conf (Baseline):    AUC = {auc_abs:.4f}")
print(f"4. Δ AUC = {auc_exp-auc_abs:.4f}")

# Calculate Curves
fpr_exp, tpr_exp, _   = roc_curve(y_failure_np, exp_drift_np)
fpr_conf, tpr_conf, _ = roc_curve(y_failure_np, conf_drift_np)
fpr_abs, tpr_abs, _   = roc_curve(y_failure_np, abs_unc_np)


out_path = out_dir / f"roc_failure_detection__{corruption}__sev{severity}.png"

# Plot
plt.figure(figsize=(10, 8))
plt.plot(fpr_exp, tpr_exp, color='darkorange', lw=2.5, label=f'Explanation Drift (AUC={auc_exp:.3f})')
plt.plot(fpr_conf, tpr_conf, color='green', lw=2, linestyle='--', label=f'Confidence Drift (AUC={auc_conf:.3f})')
plt.plot(fpr_abs, tpr_abs, color='blue', lw=2, linestyle=':', label=f'Absolute Uncertainty (AUC={auc_abs:.3f})')

plt.plot([0, 1], [0, 1], color='gray', linestyle='-', alpha=0.3)
plt.xlabel('False Positive Rate (Flagging Correct Predictions as Failures)', fontsize=12)
plt.ylabel('True Positive Rate (Detecting Actual Failures)', fontsize=12)
plt.title(f'Detection of Corruption-Induced Failures ({corruption}, sev={severity})', fontsize=14)
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print("Saved ROC to:", out_path)



# ----------------------------
# 2) SPEARMAN CORRELATION CHECKS
# ----------------------------
# Use the VALID (masked) vectors (torch) and convert to numpy 1D
drift_np = drift_scores_valid.detach().cpu().numpy().reshape(-1)          # explanation drift
unc_np   = (1.0 - p_hat_corr_valid).detach().cpu().numpy().reshape(-1)    # 1 - p_hat_corr
cd_np    = (p_hat_clean_valid - p_hat_corr_valid).detach().cpu().numpy().reshape(-1)  # p_hat_clean - p_hat_corr

rho_drift_unc, p_drift_unc = spearmanr(drift_np, unc_np)
rho_drift_cd,  p_drift_cd  = spearmanr(drift_np, cd_np)

print("\n--- Spearman correlation checks (on clean-correct subset) ---")
print(f"Spearman(drift, 1 - p_hat_corr):        rho={rho_drift_unc:.4f}, p={p_drift_unc:.2e}")
print(f"Spearman(drift, p_hat_clean - p_hat_corr): rho={rho_drift_cd:.4f}, p={p_drift_cd:.2e}")


# ----------------------------
# 2) !TODO Decide if this bootstrap test should stay
# ----------------------------


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def bootstrap_delta_auc(y, unc, drift, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)

    auc_unc = []
    auc_both = []
    delta = []

    X_unc  = unc.reshape(-1, 1)
    X_both = np.column_stack([unc, drift])

    for _ in range(B):
        idx = rng.integers(0, n, size=n)  # resample indices

        y_b = y[idx]
        X_unc_b  = X_unc[idx]
        X_both_b = X_both[idx]

        # skip degenerate bootstrap samples (all 0 or all 1)
        if np.unique(y_b).size < 2:
            continue

        clf1 = LogisticRegression(max_iter=1000).fit(X_unc_b, y_b)
        clf2 = LogisticRegression(max_iter=1000).fit(X_both_b, y_b)

        p1 = clf1.predict_proba(X_unc_b)[:, 1]
        p2 = clf2.predict_proba(X_both_b)[:, 1]

        a1 = roc_auc_score(y_b, p1)
        a2 = roc_auc_score(y_b, p2)

        auc_unc.append(a1)
        auc_both.append(a2)
        delta.append(a2 - a1)

    def ci(x):
        x = np.asarray(x)
        return float(np.mean(x)), (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))

    mean1, ci1 = ci(auc_unc)
    mean2, ci2 = ci(auc_both)
    meand, cid = ci(delta)

    return (mean1, ci1), (mean2, ci2), (meand, cid)

# usage with your arrays:
# y = y_failure_np.astype(int)
# unc = unc_np
# drift = drift_np
(auc1, ci1), (auc2, ci2), (dmean, dci) = bootstrap_delta_auc(y_failure_np.astype(int), unc_np, drift_np)

print(f"AUC(unc only): mean={auc1:.4f}, 95% CI=({ci1[0]:.4f}, {ci1[1]:.4f})")
print(f"AUC(unc+drift): mean={auc2:.4f}, 95% CI=({ci2[0]:.4f}, {ci2[1]:.4f})")
print(f"ΔAUC: mean={dmean:.4f}, 95% CI=({dci[0]:.4f}, {dci[1]:.4f})")
import torch

from src.models import transform_logits_to_probs
from src.utils import mean_std_over_mask


def estimate_sigma(E, max_points=2000, eps=1e-8):
    """
    Median heuristic for RBF kernel bandwidth.
    X: [N,D] embeddings (torch tensor, CPU ok)
    Returns sigma (float).
    """
    E = E.float()
    N = E.size(0)

    # deterministic subsample (stride) for speed
    K = min(max_points, N)
    step = max(1, N // K)
    Es = E[torch.arange(0, N, step)[:K]]  # [K,D]

    d2 = torch.cdist(Es, Es).pow(2)
    med = torch.median(d2[d2 > 0])
    sigma = torch.sqrt(med + eps).item()
    return sigma


# MMD² (Maximum Mean Discrepancy squared):
# Measures how different the clean and corrupted sample distributions are.
# Computed in feature / embedding space (E_clean vs. E_corr), not pixel space.
# Low values mean weak shift; high values mean stronger domain shift.
def compute_shift_strength_mmd(X, Y, sigma=None, eps=1e-8):
    X = X.float()
    Y = Y.float()
    Z = torch.cat([X, Y], dim=0)

    if sigma is None:
        idx = torch.randperm(Z.size(0))[:min(2000, Z.size(0))]
        Zs = Z[idx]
        d2 = torch.cdist(Zs, Zs).pow(2)
        med = torch.median(d2[d2 > 0])
        sigma = torch.sqrt(med + eps).item()

    gamma = 1.0 / (2 * (sigma ** 2) + eps)
    Kxx = torch.exp(-gamma * torch.cdist(X, X).pow(2))
    Kyy = torch.exp(-gamma * torch.cdist(Y, Y).pow(2))
    Kxy = torch.exp(-gamma * torch.cdist(X, Y).pow(2))

    n, m = X.size(0), Y.size(0)
    Kxx = Kxx - torch.diag(torch.diag(Kxx))
    Kyy = Kyy - torch.diag(torch.diag(Kyy))

    mmd2 = (Kxx.sum() / (n * (n - 1) + eps)
          + Kyy.sum() / (m * (m - 1) + eps)
          - 2 * Kxy.mean())
    return mmd2.item(), sigma



# Absolute probability shift:
# Measures how much the class probability changes from clean to corrupted input.
# Low values mean stable confidence; high values mean confidence changed strongly.
def _prob_shift_same_label(logits_a, logits_b):
    """
    Track confidence shift for the label predicted on A.
    Returns abs delta prob per sample.
    """
    pa = transform_logits_to_probs(logits_a)
    pb = transform_logits_to_probs(logits_b)
    pred_a = pa.argmax(dim=1)
    idx = torch.arange(pa.size(0))
    p_a = pa[idx, pred_a]
    p_b = pb[idx, pred_a]
    
    return (p_b - p_a).abs()  # [N]


# Absolute margin shift:
# Measures how much the gap between the top class and runner-up changes.
# Low values mean stable decision sharpness; high values mean the decision became much less/more decisive.
def _margin_shift_abs(logits_a, logits_b):
    """Absolute shift in top1-top2 logit margin."""
    a2 = logits_a.topk(2, dim=1).values
    b2 = logits_b.topk(2, dim=1).values
    m_a = a2[:, 0] - a2[:, 1]
    m_b = b2[:, 0] - b2[:, 1]
    return (m_b - m_a).abs()  # [N]


def compute_confidence_shift_metrics(logits_clean, logits_corr, mask=None):
    # per-sample vectors
    p_shift_abs      = _prob_shift_same_label(logits_clean, logits_corr)  # |Δp|
    margin_shift_abs = _margin_shift_abs(logits_clean, logits_corr)       # |Δmargin|

    vectors = {
        "p_shift_abs": p_shift_abs,
        "margin_shift_abs": margin_shift_abs,
    }

    if mask is None:
        # overall summary
        summary = {
            "p_shift_mean": p_shift_abs.mean().item(),
            "p_shift_sd": p_shift_abs.std(unbiased=False).item(),
            "margin_shift_mean": margin_shift_abs.mean().item(),
            "margin_shift_sd": margin_shift_abs.std(unbiased=False).item(),
        }
        return vectors, summary

    # masked summary
    p_mean, p_sd = mean_std_over_mask(p_shift_abs, mask)
    m_mean, m_sd = mean_std_over_mask(margin_shift_abs, mask)
    summary = {
        "p_shift_mean": p_mean,
        "p_shift_sd": p_sd,
        "margin_shift_mean": m_mean,
        "margin_shift_sd": m_sd,
    }
    
    return vectors, summary
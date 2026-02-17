import torch

from src.config import DEVICE
from src.utils import mean_std_over_mask


def ig_saliency(x: torch.Tensor, target: torch.Tensor, explainer, device=DEVICE, steps=16, internal_bs=32):
    """
    Integrated Gradients (IG) wrapper that returns a *single comparable saliency map per image*.

    Why this wrapper (vs. using ig.attribute directly)?
    - ig.attribute returns per-feature attributions with the SAME shape as the input:
        images -> [N, 3, H, W] (one attribution per pixel per channel), signed (+/-).
    - For downstream similarity metrics (cosine similarity, top-k IoU), we want ONE 2D map per image:
        [N, H, W], representing "importance magnitude" per pixel.
    This wrapper:
      1) runs ig.attribute to get [N,3,H,W] attributions for the chosen target logit,
      2) converts them to saliency by taking absolute value and summing over RGB channels,
         producing [N,H,W] maps that are easy to compare across conditions.

    Args:
        x: [N,3,32,32] normalized images on CPU.
        target: [N] class indices (on CPU). IG explains these logits (one per sample).
        steps: number of integration steps along baseline -> input path (more steps = smoother, slower).
        internal_bs: internal batching used by Captum across integration steps (efficiency / memory).

    Returns:
        sal: [N,32,32] saliency maps on CPU (|IG| summed over channels).
    """
    # Move to GPU/DEVICE and enable gradients w.r.t. input pixels
    x = x.to(device).requires_grad_(True)

    # Baseline for IG: zero in normalized space (≈ "mean image" after normalization)
    baseline = torch.zeros_like(x)

    # IG attribution: [N,3,32,32], per-pixel contribution to the chosen target logit
    attr = explainer.attribute(
        inputs=x,
        baselines=baseline,
        target=target.to(device),
        n_steps=steps,
        internal_batch_size=internal_bs,
    )

    # Collapse RGB channels -> one heatmap per image
    sal = attr.abs().sum(dim=1)  # [N,32,32]
    return sal.detach().cpu()


def ig_saliency_batched(X, target, explainer, device, steps=32, internal_bs=32, batch_size=32):
    outs = []
    for i in range(0, X.size(0), batch_size):
        xb = X[i:i+batch_size]
        tb = target[i:i+batch_size]
        outs.append(ig_saliency(xb, tb, explainer=explainer, device=device, steps=steps, internal_bs=internal_bs))
    return torch.cat(outs, dim=0)


def heatmap(attr):
    h = attr.abs().sum(dim=1, keepdim=True)          # [1,1,H,W]
    h = h / (h.amax(dim=(2,3), keepdim=True) + 1e-8)
    return h


def cosine_sim_maps(a, b, eps=1e-8):
    """
    Cosine similarity between two saliency maps per image.

    Args:
        a,b: [N,H,W] saliency maps

    Returns:
        sims: [N] cosine similarity in [-1,1] (1 = very similar patterns).
    """
    # Flatten each map to a vector
    a = a.reshape(a.shape[0], -1)
    b = b.reshape(b.shape[0], -1)

    # L2-normalize (ignore overall scale)
    a = a / (a.norm(p=2, dim=1, keepdim=True) + eps)
    b = b / (b.norm(p=2, dim=1, keepdim=True) + eps)

    # Dot product of normalized vectors = cosine similarity
    return (a * b).sum(dim=1)  # [N]


def topk_iou(a, b, topk_frac=0.05):
    """
    Top-k IoU between saliency maps: do they select the same "most important" pixels?

    Args:
        a,b: [N,H,W]
        topk_frac: fraction of pixels to keep as "important" (e.g. 0.05 = top 5%)

    Returns:
        ious: [N] IoU scores in [0,1]
    """
    N, H, W = a.shape
    k = max(1, int(topk_frac * H * W))

    a_flat = a.reshape(N, -1)
    b_flat = b.reshape(N, -1)

    # Indices of top-k salient pixels per image
    a_idx = torch.topk(a_flat, k=k, dim=1).indices
    b_idx = torch.topk(b_flat, k=k, dim=1).indices

    # IoU over sets of pixel indices (intersection / union)
    ious = []
    for i in range(N):
        sa = set(a_idx[i].tolist())
        sb = set(b_idx[i].tolist())
        inter = len(sa.intersection(sb))
        union = len(sa.union(sb))
        ious.append(inter / union if union > 0 else 0.0)

    return torch.tensor(ious)


def stable_pred_mask(pred_a, pred_b):
    """
    Label-wise stability mask: True where the predicted class did NOT change.

    Use this to analyze explanation drift only on samples where the decision stayed the same
    (e.g., clean vs corrupted prediction unchanged).
    """
    return (pred_a == pred_b)


def spearman_rho_maps(a, b, eps=1e-8):
    N = a.size(0)
    a = a.reshape(N, -1)
    b = b.reshape(N, -1)

    ra = a.argsort(dim=1).argsort(dim=1).float()
    rb = b.argsort(dim=1).argsort(dim=1).float()

    ra = ra - ra.mean(dim=1, keepdim=True)
    rb = rb - rb.mean(dim=1, keepdim=True)

    num = (ra * rb).sum(dim=1)
    den = (ra.norm(p=2, dim=1) * rb.norm(p=2, dim=1) + eps)
    return num / den  # [N]


def compute_explanation_drift_metrics(sal_clean, sal_corr, stable_mask):

    spearman_rho = spearman_rho_maps(sal_clean, sal_corr)
    cosine_sim = cosine_sim_maps(sal_clean, sal_corr)
    iou_topk = topk_iou(sal_clean, sal_corr, topk_frac=0.05)

    rho_mean_stable, rho_sd_stable = mean_std_over_mask(spearman_rho, stable_mask)
    cos_mean_stable, cos_sd_stable = mean_std_over_mask(cosine_sim, stable_mask)
    iou_mean_stable, iou_sd_stable = mean_std_over_mask(iou_topk, stable_mask)

    vectors = dict(
        spearman_rho=spearman_rho,
        cosine_sim=cosine_sim,
        iou_topk=iou_topk,
    )

    summary = dict(
        rho_mean=spearman_rho.mean().item(),
        rho_sd=spearman_rho.std(unbiased=False).item(),
        cos_mean=cosine_sim.mean().item(),
        cos_sd=cosine_sim.std(unbiased=False).item(),
        iou_mean=iou_topk.mean().item(),
        iou_sd=iou_topk.std(unbiased=False).item(),
        rho_mean_stable=rho_mean_stable,
        rho_sd_stable=rho_sd_stable,
        cos_mean_stable=cos_mean_stable,
        cos_sd_stable=cos_sd_stable,
        iou_mean_stable=iou_mean_stable,
        iou_sd_stable=iou_sd_stable
    )

    return vectors, summary
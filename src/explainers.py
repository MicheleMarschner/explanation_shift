import torch
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from captum.attr import IntegratedGradients

from src.configs.global_config import DEVICE
from src.utils import mean_std_over_mask
from src.models import get_gradcam_config


class ClassTarget:
    """
    Grad-CAM target for one class logit.
    """
    def __init__(self, class_idx: int):
        self.class_idx = int(class_idx)

    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        if model_output.ndim == 1:
            return model_output[self.class_idx]
        if model_output.ndim == 2:
            return model_output[:, self.class_idx]
        raise ValueError(
            f"Unexpected model_output shape in ClassTarget: {tuple(model_output.shape)}"
        )


def ig_saliency(x: torch.Tensor, target: torch.Tensor, explainer, device=DEVICE, steps=16, internal_bs=32):
    """Integrated Gradients (IG) wrapper that returns a *single comparable saliency map per image*"""
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
        method="riemann_trapezoid",
    )

    # Collapse RGB channels -> one heatmap per image
    sal = attr.sum(dim=1)  # [N,32,32]
    return sal.detach().cpu()


def ig_saliency_batched(X, target, explainer, device, steps=32, internal_bs=32, batch_size=32):
    outs = []
    for i in range(0, X.size(0), batch_size):
        xb = X[i:i+batch_size]
        tb = target[i:i+batch_size]
        outs.append(ig_saliency(xb, tb, explainer=explainer, device=device, steps=steps, internal_bs=internal_bs))
    return torch.cat(outs, dim=0)


def gradcam_saliency(
    x: torch.Tensor,
    target: torch.Tensor,
    model,
    device=DEVICE,
):
    """Grad-CAM wrapper returning one comparable saliency map per image"""
    model.eval()

    target_layer, reshape_transform = get_gradcam_config(model)

    x = x.to(device).requires_grad_(True)

    cam = GradCAM(
        model=model,
        target_layers=[target_layer],
        reshape_transform=reshape_transform,
    )

    targets = [ClassTarget(int(t.item())) for t in target]

    grayscale_cam = cam(input_tensor=x, targets=targets)   # numpy [N,h,w]
    sal = torch.as_tensor(grayscale_cam, dtype=torch.float32, device=x.device)

    H, W = x.shape[-2], x.shape[-1]
    if sal.shape[-2:] != (H, W):
        sal = F.interpolate(
            sal.unsqueeze(1),   # [N,1,h,w]
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)            # [N,H,W]

    return sal.detach().cpu()

def gradcam_saliency_batched(X, target, model, device, batch_size=32):
    outs = []
    for i in range(0, X.size(0), batch_size):
        xb = X[i:i+batch_size]
        tb = target[i:i+batch_size]
        outs.append(
            gradcam_saliency(
                xb,
                tb,
                model=model,
                device=device,
            )
        )
    return torch.cat(outs, dim=0)


def heatmap(attr):
    h = attr.abs().sum(dim=1, keepdim=True)          # [1,1,H,W]
    h = h / (h.amax(dim=(2,3), keepdim=True) + 1e-8)
    return h


# Cosine similarity:
# Measures the overall structural similarity between two saliency maps.
# High cosine similarity = similar explanation pattern; low = strong explanation change.
def cosine_sim_maps(a, b, eps=1e-8):
    """Cosine similarity in [-1,1] (1 = very similar patterns) between two saliency maps per image"""
    # Flatten each map to a vector
    a = a.reshape(a.shape[0], -1)
    b = b.reshape(b.shape[0], -1)

    # L2-normalize (ignore overall scale)
    a = a / (a.norm(p=2, dim=1, keepdim=True) + eps)
    b = b / (b.norm(p=2, dim=1, keepdim=True) + eps)

    # Dot product of normalized vectors = cosine similarity
    return (a * b).sum(dim=1)  # [N]


# Top-k IoU:
# Measures how much the most salient regions overlap between clean and corrupted explanations.
# High IoU = focus stays on the same key regions; low IoU = focus shifts.
def topk_iou(a, b, topk_frac=0.05):
    """Top-k IoU between saliency maps: do they select the same "most important" pixels? scores in [0,1]"""
    N, H, W = a.shape
    k = max(1, int(topk_frac * H * W))
    
    a_flat = a.reshape(N, -1)
    b_flat = b.reshape(N, -1)

    # Get the value at the k-th rank to create a binary mask
    # This is much faster than set operations
    val_a, _ = torch.topk(a_flat, k=k, dim=1)
    val_b, _ = torch.topk(b_flat, k=k, dim=1)
    
    # Create binary masks for top-k pixels
    # Use the k-th largest value as the threshold
    mask_a = (a_flat >= val_a[:, -1:].expand_as(a_flat))
    mask_b = (b_flat >= val_b[:, -1:].expand_as(b_flat))
    
    intersection = (mask_a & mask_b).float().sum(dim=1)
    union = (mask_a | mask_b).float().sum(dim=1)
    
    return intersection / (union + 1e-8)


def mask_invariant(pred_a, pred_b):
    """Label-wise stability mask: True where the predicted class did not change"""
    return (pred_a == pred_b)


def mask_correct(pred_a, pred_b, y_true):
    """Label-wise stability mask: True where predictions are both true"""
    return ((pred_a == y_true) & (pred_b == y_true))


# Spearman rho:
# Measures whether the ranking of important regions stays similar.
# High rho = similar importance ordering; low rho = ranking changed strongly.
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


def compute_explanation_drift_metrics(sal_clean, sal_corr, mask=None):

    # Pre-process: Absolute value and sum across color channels (if RGB)
    # Saliency maps are usually [N, H, W]
    # If saliency is [N,C,H,W], reduce channels; if [N,H,W], keep as is
    if sal_clean.ndim == 4:
        sal_clean = sal_clean.abs().sum(dim=1)
        sal_corr  = sal_corr.abs().sum(dim=1)
    else:
        sal_clean = sal_clean.abs()
        sal_corr  = sal_corr.abs()

    spearman_rho = spearman_rho_maps(sal_clean, sal_corr)   # [N]
    cosine_sim   = cosine_sim_maps(sal_clean, sal_corr)     # [N]
    iou_topk     = topk_iou(sal_clean, sal_corr, topk_frac=0.05)  # [N]

    vectors = {
        "spearman_rho": spearman_rho,
        "cosine_sim": cosine_sim,
        "iou_topk": iou_topk,
    }

    if mask is None:
        # summary all
        summary = {
            "rho_mean": spearman_rho.mean().item(),
            "rho_sd": spearman_rho.std(unbiased=False).item(),
            "cos_mean": cosine_sim.mean().item(),
            "cos_sd": cosine_sim.std(unbiased=False).item(),
            "iou_mean": iou_topk.mean().item(),
            "iou_sd": iou_topk.std(unbiased=False).item(),
        }
        return vectors, summary

    # summary masked
    rho_mean, rho_sd = mean_std_over_mask(spearman_rho, mask)
    cos_mean, cos_sd = mean_std_over_mask(cosine_sim, mask)
    iou_mean, iou_sd = mean_std_over_mask(iou_topk, mask)

    summary = {
        "rho_mean": rho_mean,
        "rho_sd": rho_sd,
        "cos_mean": cos_mean,
        "cos_sd": cos_sd,
        "iou_mean": iou_mean,
        "iou_sd": iou_sd,
    }
    return vectors, summary


def compute_saliency_maps(
    X: torch.Tensor,
    target: torch.Tensor,
    explainer_name: str,
    model,
    device=DEVICE,
    steps: int = 32,
    internal_bs: int = 32,
    batch_size: int = 32,
):
    """Unified saliency dispatcher: returns one saliency map per image [N,H,W] for the requested explainer"""
    if explainer_name == "IG":
        explainer = IntegratedGradients(model)
        return ig_saliency_batched(
            X,
            target,
            explainer=explainer,
            device=device,
            steps=steps,
            internal_bs=internal_bs,
            batch_size=batch_size,
        )

    if explainer_name == "GradCAM":
        return gradcam_saliency_batched(
            X,
            target,
            model=model,
            device=device,
            batch_size=batch_size,
        )

    raise ValueError(f"Unknown explainer: {explainer_name}")
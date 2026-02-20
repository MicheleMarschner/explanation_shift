import torch
import torch.nn.functional as F
import torchvision.transforms as T

from src.configs.global_config import PATHS, CIFAR10_MEAN, CIFAR10_SD
from src.data import get_clean_data, get_corrupted_data



exp_dir = PATHS.runs / "experiment__n250__IG__seed51"
out_dir = PATHS.results / "heatmaps"
out_dir.mkdir(parents=True, exist_ok=True)

def pick_index(mask: torch.Tensor, similarity: torch.Tensor, uncertainty: torch.Tensor) -> int:
    """
    Select index with lowest similarity AND highest uncertainty, but only within mask==True.
    Uses rank-sum; excludes out-of-mask by setting combined rank to +inf.
    """
    # ranks across the full tensor (fine), then exclude
    rank_sim = torch.argsort(torch.argsort(similarity, descending=False))      # low sim -> low rank
    rank_unc = torch.argsort(torch.argsort(uncertainty, descending=True))     # high unc -> low rank

    combined = (rank_sim + rank_unc).float()
    combined[~mask] = float("inf")

    if torch.isinf(combined).all():
        raise ValueError("No samples match the mask (empty candidate set).")

    return int(torch.argmin(combined).item())


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



## load data
# 
corruption = "gaussian_noise"
severity = 5
# gaussian_noise, 3: 55
# gaussian_noise, 5: 78

clean_keys = ["pred_clean", "logits_clean", "entropy_clean", "sal_clean", "y_clean"]
clean_ref = load_clean_ref(clean_keys)

corr_keys = ["pred_corr", "logits_corr", "entropy_corr", "sal_corr"]
corr_ref = load_corr_ref(corr=corruption, sev=severity, keys=corr_keys)


# ==========================================
# 1. CALCULATE NECESSARY METRICS (TENSORS)
# ==========================================

# A. Calculate Explanation Similarity (Cosine)
# Flatten (N, C, H, W) -> (N, Features)
flat_clean = clean_ref["sal_clean"].flatten(start_dim=1).float()
flat_corr = corr_ref["sal_corr"].flatten(start_dim=1).float()
similarity = F.cosine_similarity(flat_clean, flat_corr, dim=1) # Shape: (N,)

# B. Calculate |Delta Entropy|
ent_clean = clean_ref["entropy_clean"]
ent_corr = corr_ref["entropy_corr"]
abs_delta_entropy = (ent_corr - ent_clean).abs()

# C. Calculate |Margin Shift| (Alternative/Better Uncertainty metric)
def get_margin(logits):
    probs = F.softmax(logits, dim=1)
    top2 = torch.topk(probs, k=2, dim=1).values
    return top2[:, 0] - top2[:, 1] # P(Top1) - P(Top2)

margin_clean = get_margin(clean_ref["logits_clean"])
margin_corr = get_margin(corr_ref["logits_corr"])
abs_margin_shift = (margin_corr - margin_clean).abs()

# ==========================================
# 2. FIND THE "PERFECT BREAKDOWN" INDEX
# ==========================================

# We want: Lowest Similarity AND Highest Uncertainty Shift.
# We use Rank Sum: Rank indices by Sim (Ascending) + Rank indices by Uncertainty (Descending)
# The index with the lowest combined rank is the optimal intersection.

# 1. Rank by Similarity (Ascending: 0 = Lowest Sim)
rank_sim = torch.argsort(torch.argsort(similarity, descending=False))

# 2. Rank by Uncertainty (Descending: 0 = Highest Shift)
# Choose either Entropy or Margin based on preference
uncertainty_metric = abs_delta_entropy 
# uncertainty_metric = abs_margin_shift # Uncomment to use margin instead

rank_unc = torch.argsort(torch.argsort(uncertainty_metric, descending=True))

# 3. Find the 'Edge Case' Index
combined_rank = rank_sim + rank_unc
breakdown_idx = torch.argmin(combined_rank).item()

# ==========================================
# 3. PRINT & EXTRACT FOR VISUALIZATION
# ==========================================

print(f"\n--- VISUAL FAILURE ANALYSIS ---")
print(f"Target: Highest |ΔUncertainty| + Lowest Similarity")
print(f"Selected Index: {breakdown_idx}")
print(f"--------------------------------")
print(f"Similarity:       {similarity[breakdown_idx]:.4f} (Low = Alien)")
print(f"|ΔEntropy|:       {abs_delta_entropy[breakdown_idx]:.4f} (High = Unstable)")
print(f"|ΔMargin|:        {abs_margin_shift[breakdown_idx]:.4f}")
print(f"Pred Clean:       {clean_ref['pred_clean'][breakdown_idx].item()}")
print(f"Pred Corrupt:     {corr_ref['pred_corr'][breakdown_idx].item()}")
print(f"True Label:       {clean_ref['y_clean'][breakdown_idx].item()}")
print(f"--------------------------------")

# Store specific breakdown tensors for saving/plotting later
breakdown_data = {
    "index": breakdown_idx,
    "image_clean": None, # You can load the image here using pair_idx if available
    "image_corr": None,  # You can load the image here using pair_idx if available
    "sal_clean": clean_ref["sal_clean"][breakdown_idx],
    "sal_corr": corr_ref["sal_corr"][breakdown_idx]
}

# If you need to visualize this specific pair immediately in a notebook:
import matplotlib.pyplot as plt
plt.imshow(breakdown_data['sal_corr'].numpy())
plt.axis("off")
plt.show(block=True)


pred_clean = clean_ref["pred_clean"]
pred_corr  = corr_ref["pred_corr"]
y_true     = clean_ref["y_clean"]

mask_silent = (pred_clean == y_true) & (pred_corr == y_true) & (pred_clean == pred_corr)

# choose your uncertainty metric
uncertainty_metric = abs_delta_entropy     # or abs_margin_shift

silent_idx = pick_index(mask_silent, similarity, uncertainty_metric)


mask_stubborn = (pred_clean != y_true) & (pred_corr != y_true) & (pred_clean == pred_corr)

# Option A (dramatic): unstable + alien explanation
uncertainty_metric = abs_delta_entropy

stubborn_idx = pick_index(mask_stubborn, similarity, uncertainty_metric)

# Option B (truly stubborn): doesn't move, but explanation alien
# Convert to a 'stubbornness' score where higher is better:
stubbornness_metric = -abs_margin_shift   # higher = more stubborn (smaller shift)

stubborn_idx = pick_index(mask_stubborn, similarity, stubbornness_metric)

mask_flip = (pred_clean == y_true) & (pred_corr != y_true)  # correct -> wrong
flip_idx = pick_index(mask_flip, similarity, abs_delta_entropy)



def print_case(name: str, idx: int):
    print(f"\n--- {name.upper()} ---")
    print(f"Selected Index: {idx}")
    print(f"Similarity:       {similarity[idx]:.4f}")
    print(f"|ΔEntropy|:       {abs_delta_entropy[idx]:.4f}")
    print(f"|ΔMargin|:        {abs_margin_shift[idx]:.4f}")
    print(f"Pred Clean:       {pred_clean[idx].item()}")
    print(f"Pred Corrupt:     {pred_corr[idx].item()}")
    print(f"True Label:       {y_true[idx].item()}")
    print(f"------------------------------")

print_case("silent drift", silent_idx)
print_case("stubborn failure", stubborn_idx)
print_case("flip failure", flip_idx)


ref_pt  = exp_dir / "00__reference" / "00__clean_ref.pt"
ref = torch.load(ref_pt, map_location="cpu")
pair_idx = ref['pair_idx']

clean_loader, X_clean, y_clean = get_clean_data(path=PATHS.data_clean, idx=pair_idx, transform=None)
corr_dataloader, X_corr, y_corr = get_corrupted_data(
        idx=pair_idx, 
        path=PATHS.data_corr,
        transform=None, 
        corruption=corruption, 
        severity=severity
    )

import numpy as np
import torch
import matplotlib.pyplot as plt

def img_to_hwc01(img: torch.Tensor) -> np.ndarray:
    """img: (3,H,W) or (H,W,3) or (1,H,W) -> numpy (H,W,C) or (H,W) in [0,1]."""
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
    img = img.detach().cpu().float()

    # CHW -> HWC
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = img.permute(1, 2, 0)

    # scale if needed
    if img.max() > 1.5:
        img = img / 255.0

    # handle [-1,1]
    if img.min() < 0 and img.min() >= -1.01 and img.max() <= 1.01:
        img = (img + 1.0) / 2.0

    img = torch.clamp(img, 0.0, 1.0)
    arr = img.numpy()

    # if single-channel HWC with C=1, squeeze to (H,W) for nicer display
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]
    return arr

def sal_to_2d(sal: torch.Tensor) -> np.ndarray:
    """sal: (H,W) or (C,H,W) -> (H,W) normalized [0,1]."""
    sal = sal.detach().cpu().float()
    if sal.ndim == 3:
        sal = sal.squeeze(0) if sal.shape[0] == 1 else sal.abs().mean(dim=0)
    elif sal.ndim != 2:
        raise ValueError(f"Unexpected saliency shape: {tuple(sal.shape)}")

    sal = sal - sal.min()
    if float(sal.max()) > 0:
        sal = sal / sal.max()
    return sal.numpy()


def save_case_row(name: str, i: int, save_path):
    # images
    img_clean = img_to_hwc01(X_clean[i])
    img_corr  = img_to_hwc01(X_corr[i])

    # explanations (from your loaded refs)
    sal_clean_2d = sal_to_2d(clean_ref["sal_clean"][i])
    sal_corr_2d  = sal_to_2d(corr_ref["sal_corr"][i])

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))

    axes[0].imshow(img_clean)
    axes[0].set_title(f"{name}\nClean image")
    axes[0].axis("off")

    axes[1].imshow(img_clean)
    axes[1].imshow(sal_clean_2d, alpha=0.45)
    axes[1].set_title("Clean + explanation")
    axes[1].axis("off")

    axes[2].imshow(img_corr)
    axes[2].set_title("Corrupt image")
    axes[2].axis("off")

    axes[3].imshow(img_corr)
    axes[3].imshow(sal_corr_2d, alpha=0.45)
    axes[3].set_title("Corrupt + explanation")
    axes[3].axis("off")

    # optional: metrics line (uses tensors you already computed earlier)
    sim = float(similarity[i])
    dH  = float(abs_delta_entropy[i])
    dM  = float(abs_margin_shift[i])
    pc  = int(clean_ref["pred_clean"][i])
    pr  = int(corr_ref["pred_corr"][i])
    yt  = int(clean_ref["y_clean"][i])
    fig.suptitle(f"idx={i} | sim={sim:.3f} | |ΔH|={dH:.3f} | |Δmargin|={dM:.3f} | pred {pc}->{pr} | y={yt}", y=1.08)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", save_path)

silent_idx   = 48
stubborn_idx = 221
flip_idx     = 78

save_case_row("SILENT DRIFT",  silent_idx,   out_dir / f"qual_silent_idx{silent_idx}_{corruption}_sev{severity}.png")
save_case_row("STUBBORN FAIL", stubborn_idx, out_dir / f"qual_stubborn_idx{stubborn_idx}_{corruption}_sev{severity}.png")
save_case_row("FLIP FAIL",     flip_idx,     out_dir / f"qual_flip_idx{flip_idx}_{corruption}_sev{severity}.png")
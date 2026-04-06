from pathlib import Path
from typing import Callable, Iterable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from configs.global_config import PATHS
from analysis.analysis_helper import find_experiment_dir
from analysis.sample_level_analysis import _load_payload, _unwrap_payload

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore
    _HAS_TORCH = False


def _to_numpy(x) -> np.ndarray:
    """torch.Tensor / numpy / list → numpy array."""
    if _HAS_TORCH and torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)



# ----------------------------------------------------------------------------
# Image / saliency helpers
# ----------------------------------------------------------------------------

def _img_to_hwc01(img) -> np.ndarray:
    """Normalize an image to (H, W, C) in [0, 1], or (H, W) for single-channel."""
    arr = _to_numpy(img).astype(np.float32)

    # CHW → HWC
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))

    # Scale to [0, 1] if it looks like 0–255
    if arr.max() > 1.5:
        arr = arr / 255.0

    # Handle [-1, 1] input
    if arr.min() < 0 and arr.min() >= -1.01 and arr.max() <= 1.01:
        arr = (arr + 1.0) / 2.0

    arr = np.clip(arr, 0.0, 1.0)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]
    return arr


def _sal_to_2d(sal) -> np.ndarray:
    """Reduce a saliency tensor to (H, W), min-max normalized."""
    arr = _to_numpy(sal).astype(np.float32)
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        else:
            arr = np.mean(np.abs(arr), axis=0)
    elif arr.ndim != 2:
        raise ValueError(f"Unexpected saliency shape: {arr.shape}")
    arr = arr - arr.min()
    mx = float(arr.max())
    if mx > 0:
        arr = arr / mx
    return arr


# ----------------------------------------------------------------------------
# Experiment tree loaders
# ----------------------------------------------------------------------------


def _load_clean_arrays(exp_dir: Path) -> dict:
    """Return {'sal_clean', 'pair_idx', ...} from 00__reference/00__clean_ref.pt."""
    ref_file = exp_dir / "00__reference" / "00__clean_ref.pt"
    if not ref_file.exists():
        raise FileNotFoundError(f"clean reference not found: {ref_file}")
    payload = _load_payload(ref_file)
    if not isinstance(payload, dict):
        raise ValueError(f"{ref_file}: not a dict")

    # pair_idx lives at top level in the pipeline schema
    pair_idx = payload.get("pair_idx")

    # saliency + labels live under 'clean_reference'
    inner = _unwrap_payload(payload, ["sal_clean"])
    if inner is None:
        raise KeyError(
            f"{ref_file}: no 'sal_clean' found under any wrapper; "
            f"top-level keys: {list(payload.keys())}"
        )

    out = {"sal_clean": inner["sal_clean"]}
    for k in ("pred_clean", "y_clean", "y_true"):
        if k in inner:
            out[k] = inner[k]
    if pair_idx is not None:
        out["pair_idx"] = pair_idx
    elif "pair_idx" in inner:
        out["pair_idx"] = inner["pair_idx"]
    else:
        raise KeyError(
            f"{ref_file}: no 'pair_idx' at top-level or in clean_reference"
        )
    return out


def _load_artifact_arrays(
    exp_dir: Path, corruption: str, severity: int
) -> dict:
    """Return {'sal_corr', 'pred_corr', ...} from
    01__artifacts/01__artifacts__{corruption}__sev{sev}.pt."""
    art_file = (
        exp_dir / "01__artifacts" / f"01__artifacts__{corruption}__sev{severity}.pt"
    )
    if not art_file.exists():
        raise FileNotFoundError(f"artifact not found: {art_file}")
    payload = _load_payload(art_file)
    inner = _unwrap_payload(payload, ["sal_corr"])
    if inner is None:
        top = list(payload.keys()) if isinstance(payload, dict) else "(not a dict)"
        raise KeyError(
            f"{art_file}: no 'sal_corr' found under any wrapper; top-level: {top}"
        )
    out = {"sal_corr": inner["sal_corr"]}
    for k in ("pred_corr", "proba_corr"):
        if k in inner:
            out[k] = inner[k]
    return out


# ----------------------------------------------------------------------------
# Figure composition
# ----------------------------------------------------------------------------

ZONE_COLORS = {
    "robust":            "#1f77b4",
    "silent_drift":      "#ff7f0e",
    "expected_failure":  "#2ca02c",
    "stubborn_failure":  "#d62728",
}

ZONE_LABELS = {
    "robust":            "Robust",
    "silent_drift":      "Silent Drift",
    "expected_failure":  "Expected Failure",
    "stubborn_failure":  "Stubborn Failure",
}


def _save_exemplar_figure(
    zone: str,
    row: pd.Series,
    img_clean_arr: np.ndarray,
    img_corr_arr: np.ndarray,
    sal_clean_2d: np.ndarray,
    sal_corr_2d: np.ndarray,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), dpi=300)

    axes[0].imshow(img_clean_arr, interpolation="lanczos")
    axes[0].set_title("Clean", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(img_clean_arr, interpolation="lanczos")
    axes[1].imshow(sal_clean_2d, alpha=0.5, cmap="jet")
    axes[1].set_title("Clean + explanation", fontsize=9)
    axes[1].axis("off")

    axes[2].imshow(img_corr_arr, interpolation="lanczos")
    axes[2].set_title("Corrupt", fontsize=9)
    axes[2].axis("off")

    axes[3].imshow(img_corr_arr, interpolation="lanczos")
    axes[3].imshow(sal_corr_2d, alpha=0.5, cmap="jet")
    axes[3].set_title("Corrupt + explanation", fontsize=9)
    axes[3].axis("off")

    zone_color = ZONE_COLORS.get(zone, "#333333")
    zone_label = ZONE_LABELS.get(zone, zone)

    meta = (
        f"{row['explainer']} · {row['corruption']} · sev={int(row['severity'])} · "
        f"seed={int(row['seed'])} · sample_idx={int(row['sample_idx'])}"
    )
    metrics = (
        f"sim={row['similarity']:.3f}  ·  "
        f"|ΔH|={row['abs_dH']:.3f}  ·  "
        f"both_correct={row['both_correct']}"
    )
    fig.text(0.5, 1.06, zone_label, ha="center", va="bottom",
             fontsize=13, fontweight="bold", color=zone_color)
    fig.text(0.5, 1.01, meta, ha="center", va="bottom",
             fontsize=9, color="#333")
    fig.text(0.5, -0.02, metrics, ha="center", va="top",
             fontsize=9, color="#555")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------------

def render_trust_zone_exemplars(
    exemplars_csv: str | Path,
    experiments_dir: str | Path,
    output_dir: str | Path,
    image_loader_clean: Callable[[np.ndarray], torch.Tensor],
    image_loader_corr: Callable[[np.ndarray, str, int], torch.Tensor],
    zones: Iterable[str] = ("robust", "silent_drift",
                            "stubborn_failure", "expected_failure"),
    corruptions: list[str] | None = None,
    severities: list[int] | None = None,
    explainers: list[str] | None = None,
    seeds: list[int] | None = None,
) -> None:
    """Render one PNG per (condition × zone) exemplar from a trust-zone CSV.

    Parameters
    ----------
    exemplars_csv : path
        CSV written by ``auxiliary_analysis.export_trust_zone_exemplars``.
    experiments_dir : path
        Root containing ``experiment__n{N}__{EXPL}__seed{S}/`` dirs.
    output_dir : path
        PNGs go into ``{output_dir}/qualitative/``.
    image_loader_clean : callable
        ``(pair_idx: ndarray) -> Tensor[N, C, H, W]``. Called once per condition.
    image_loader_corr : callable
        ``(pair_idx: ndarray, corruption: str, severity: int) -> Tensor[N, C, H, W]``.
        Called once per condition.
    zones, corruptions, severities, explainers, seeds : optional filters
        Only rows matching all given filters are rendered. ``None`` = no filter.
    """
    exemplars_csv = Path(exemplars_csv)
    experiments_dir = Path(experiments_dir)
    output_dir = Path(output_dir)

    df = pd.read_csv(exemplars_csv)
    df = df.dropna(subset=["sample_idx"]).copy()
    df["sample_idx"] = df["sample_idx"].astype(int)

    # Apply filters
    df = df[df["zone"].isin(set(zones))]
    if corruptions is not None:
        df = df[df["corruption"].isin(corruptions)]
    if severities is not None:
        df = df[df["severity"].isin(severities)]
    if explainers is not None:
        df = df[df["explainer"].isin(explainers)]
    if seeds is not None:
        df = df[df["seed"].isin(seeds)]

    if df.empty:
        print("[warn] no exemplars to render after filtering")
        return

    n_rendered = 0
    n_errors = 0
    for (explainer, seed, corruption, severity), group in df.groupby(
        ["explainer", "seed", "corruption", "severity"]
    ):
        try:
            exp_dir = find_experiment_dir(experiments_dir, explainer, int(seed))
            if exp_dir is None:
                print(f"[warn] no experiment dir for {explainer} seed={seed}")
                n_errors += len(group)
                continue

            clean = _load_clean_arrays(exp_dir)
            artifact = _load_artifact_arrays(exp_dir, corruption, int(severity))

            pair_idx = clean["pair_idx"]
            pair_idx = np.asarray(_to_numpy(pair_idx))

            X_clean = image_loader_clean(pair_idx)
            X_corr = image_loader_corr(pair_idx, corruption, int(severity))

            sal_clean = clean["sal_clean"]
            sal_corr = artifact["sal_corr"]

            for _, row in group.iterrows():
                idx = int(row["sample_idx"])
                try:
                    img_clean_arr = _img_to_hwc01(X_clean[idx])
                    img_corr_arr = _img_to_hwc01(X_corr[idx])
                    sal_clean_2d = _sal_to_2d(sal_clean[idx])
                    sal_corr_2d = _sal_to_2d(sal_corr[idx])
                except Exception as e:
                    print(f"[warn] slicing failed for idx={idx} in "
                          f"{explainer}/seed{seed}/{corruption}/sev{severity}: {e}")
                    n_errors += 1
                    continue

                fname = (
                    f"qual__{row['zone']}__{explainer}__seed{int(seed)}__"
                    f"{corruption}__sev{int(severity)}__idx{idx}.png"
                )
                _save_exemplar_figure(
                    zone=row["zone"],
                    row=row,
                    img_clean_arr=img_clean_arr,
                    img_corr_arr=img_corr_arr,
                    sal_clean_2d=sal_clean_2d,
                    sal_corr_2d=sal_corr_2d,
                    out_path= PATHS.results / "qualitative_imgs" / fname,
                )
                n_rendered += 1

        except Exception as e:
            print(f"[error] condition {explainer}/seed{seed}/"
                  f"{corruption}/sev{severity}: {e}")
            n_errors += len(group)

    if n_errors:
        print(f"[info] {n_errors} exemplars were skipped due to errors")
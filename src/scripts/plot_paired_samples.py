from pathlib import Path
import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np

from configs.global_config import PATHS


def load_cifar10_test_image(cifar10_dir: Path, sample_idx: int) -> tuple[np.ndarray, int]:
    test_batch_path = cifar10_dir / "test_batch"

    with open(test_batch_path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")

    data = batch[b"data"]
    labels = batch[b"labels"]

    img = data[sample_idx].reshape(3, 32, 32).transpose(1, 2, 0)
    label = labels[sample_idx]
    return img, label


def load_cifar10_label_names(cifar10_dir: Path) -> list[str]:
    meta_path = cifar10_dir / "batches.meta"

    with open(meta_path, "rb") as f:
        meta = pickle.load(f, encoding="bytes")

    return [name.decode("utf-8") for name in meta[b"label_names"]]


def load_cifar10c_image(
    cifar10c_dir: Path,
    corruption: str,
    severity: int,
    sample_idx: int,
) -> np.ndarray:
    corruption_path = cifar10c_dir / f"{corruption}.npy"
    arr = np.load(corruption_path)

    if severity < 1 or severity > 5:
        raise ValueError(f"Severity must be in [1, 5], got {severity}.")
    if sample_idx < 0 or sample_idx >= 10000:
        raise ValueError(f"sample_idx must be in [0, 9999], got {sample_idx}.")

    offset = (severity - 1) * 10000 + sample_idx
    return arr[offset]


def plot_coupled_row(
    corruption: str,
    severities: list[int],
    sample_idx: int,
) -> None:
    cifar10_dir = Path(PATHS.data_clean)
    cifar10c_dir = Path(PATHS.data_corr)

    clean_img, label = load_cifar10_test_image(cifar10_dir, sample_idx)
    label_names = load_cifar10_label_names(cifar10_dir)

    n_cols = 1 + len(severities)
    fig, axes = plt.subplots(1, n_cols, figsize=(2.6 * n_cols, 2.8), dpi=300)

    if n_cols == 1:
        axes = [axes]

    axes[0].imshow(clean_img, interpolation="lanczos")
    axes[0].set_title(f"clean\n{label_names[label]}", fontsize=10)
    axes[0].axis("off")

    for ax, severity in zip(axes[1:], severities):
        corr_img = load_cifar10c_image(
            cifar10c_dir=cifar10c_dir,
            corruption=corruption,
            severity=severity,
            sample_idx=sample_idx,
        )
        ax.imshow(corr_img, interpolation="lanczos")
        ax.set_title(f"{corruption}\nseverity={severity}", fontsize=10)
        ax.axis("off")

    plt.tight_layout(pad=0.4)
    plt.savefig(f"results/qualitative_imgs/paired_samples_{corruption}_id{sample_idx}.png", dpi=300, bbox_inches="tight", pad_inches=0.2)


if __name__ == "__main__":
    # python src/scripts/plot_paired_samples.py --corruption fog --sample-idx 42
    parser = argparse.ArgumentParser()
    parser.add_argument("--corruption", type=str, required=True)
    parser.add_argument("--sample-idx", type=int, required=True)
    parser.add_argument(
        "--severities",
        type=int,
        nargs="+",
        default=[1, 2, 3, 5],
    )
    args = parser.parse_args()

    plot_coupled_row(
        corruption=args.corruption,
        severities=args.severities,
        sample_idx=args.sample_idx,
    )
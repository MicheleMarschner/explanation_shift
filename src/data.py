import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from src.configs.global_config import BATCH_SIZE, PATHS


def load_cifar10(cifar10_dir: str):
    path = os.path.join(cifar10_dir, "test_batch")
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")

    X = d[b"data"]          # (10000, 3072) uint8
    y = np.array(d[b"labels"], dtype=np.int64)  # (10000,)

    # reshape to (N, 32, 32, 3)
    X = X.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return X, y


def load_cifar10c_slice(cifarc_dir: str, corruption: str, severity: int):
    assert 1 <= severity <= 5
    X_c = np.load(os.path.join(cifarc_dir, f"{corruption}.npy"))   # (50000,32,32,3)
    y_c = np.load(os.path.join(cifarc_dir, "labels.npy"))          # (50000,)
    sl = slice((severity-1)*10000, severity*10000)
    return X_c[sl], y_c[sl]


def sample_cifar10_pair_indices(n_total: int = 10000, n_pairs: int = 500, seed: int = 51) -> torch.LongTensor:
    """
    Deterministically sample unique indices from the CIFAR-10 *test set*.
    Use these indices to select paired (clean, CIFAR-10-C) examples by position.

    Returns:
        idx: torch.LongTensor of shape [n_pairs] with values in [0, n_total).
    """
    rng = np.random.default_rng(seed)
    idx_np = rng.choice(n_total, size=n_pairs, replace=False)
    return torch.from_numpy(idx_np).long()
 

class CifarDataset(Dataset):
    def __init__(self, X_hwc_uint8, y, transform):
        self.X = X_hwc_uint8
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]                 # numpy HWC uint8
        # If transform uses PIL ops (Resize/Crop), it will fail on numpy.
        # Convert to PIL first.
        if not torch.is_tensor(x):          # x is numpy here
            x = Image.fromarray(x)
        x = self.transform(x)           # -> torch [3,H,W]
        y = int(self.y[idx])
        return x, y


def get_clean_data(path, idx, transform=None, batch_size=BATCH_SIZE):
    X_clean, y_clean = load_cifar10(path)
    X_clean_subset, y_clean_subset = X_clean[idx], y_clean[idx]

    clean_pairs_ds = CifarDataset(X_clean_subset, y_clean_subset, transform=transform)
    clean_dataloader = DataLoader(clean_pairs_ds, batch_size=batch_size, shuffle=False)  

    return clean_dataloader, X_clean_subset,  y_clean_subset


def get_corrupted_data(idx, path, transform=None, corruption=1, severity=1, batch_size=BATCH_SIZE):
    X_corr, y_corr = load_cifar10c_slice(path, corruption, severity)
    X_corr_subset,  y_corr_subset = X_corr[idx], y_corr[idx]

    corr_pairs_ds = CifarDataset(X_corr_subset, y_corr_subset, transform=transform)
    corr_dataloader = DataLoader(corr_pairs_ds, batch_size=batch_size, shuffle=False)  

    return corr_dataloader, X_corr_subset,  y_corr_subset
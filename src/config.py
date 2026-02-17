from pathlib import Path
import torch
from dataclasses import dataclass
from enum import Enum

@dataclass(frozen=True)
class Paths:
    """Holds the main filesystem paths used by the project"""
    data: Path
    results: Path
    runs: Path
    checkpoints: Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # repo root

PATHS = Paths(
    data=PROJECT_ROOT / "data",
    results=PROJECT_ROOT / "results",
    runs=PROJECT_ROOT / "experiments",
    checkpoints=PROJECT_ROOT / "checkpoints",
)

#--------------------------------------------#
#        CONSTANTS
#--------------------------------------------#

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 51

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_SD  = (0.2470, 0.2435, 0.2616)
N_PAIRS = 50  # later try 200
BATCH_SIZE = 64 

IG_STEPS = 32 # n_steps, How many points Captum samples along the path from baseline → input.


class TargetPolicy(str, Enum):
    PRED_CLEAN = "pred_clean"
    PRED_PER_DOMAIN = "pred_per_domain"



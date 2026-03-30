from pathlib import Path
import torch
from dataclasses import dataclass
from enum import Enum

@dataclass(frozen=True)
class Paths:
    """Holds the main filesystem paths used by the project"""
    data_clean: Path
    data_corr: Path
    results: Path
    runs: Path
    checkpoints: Path

@dataclass(frozen=True)
class ExperimentPaths:
    exp_dir: Path
    reference: Path
    artifacts: Path
    drift: Path
    quantus: Path

    @staticmethod
    def from_exp_dir(exp_dir: Path) -> "ExperimentPaths":
        return ExperimentPaths(
            exp_dir=exp_dir,
            reference=exp_dir / "00__reference",
            artifacts=exp_dir / "01__artifacts",
            drift=exp_dir / "02__drift",
            quantus=exp_dir / "03__quantus",
        )

PROJECT_ROOT = Path(__file__).resolve().parents[2] 

PATHS = Paths(
    data_clean=PROJECT_ROOT / "data/CIFAR-10",
    data_corr=PROJECT_ROOT / "data/CIFAR-10-C",
    results=PROJECT_ROOT / "results",
    runs=PROJECT_ROOT / "experiments",
    checkpoints=PROJECT_ROOT / "checkpoints"
)


class TargetPolicy(str, Enum):
    PRED_CLEAN = "pred_clean"
    PRED_PER_DOMAIN = "pred_per_domain"


#--------------------------------------------#
#        CONSTANTS
#--------------------------------------------#

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_SD  = (0.2470, 0.2435, 0.2616)
BATCH_SIZE = 64 

IG_STEPS = 64 # n_steps, How many points Captum samples along the path from baseline → input. 
BATCH_SIZE_EXPLAINER = 32 
TARGET_POLICY = TargetPolicy.PRED_CLEAN.value


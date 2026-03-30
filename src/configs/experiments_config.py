from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class ExperimentTemplate:
    """Defines the experiment grid used to generate runs."""
    N_PAIRS: int
    CORRUPTIONS: List[str]
    SEVERITIES: List[int]
    EXPLAINERS: List[str]
    SEEDS: List[int]


@dataclass(frozen=True)
class RunConfig:
    N_PAIRS: int
    seed: int
    explainer: str
    corruption: str
    severity: int

@dataclass(frozen=True)
class ReferenceJob:
    N_PAIRS: int
    seed: int
    explainer: str


## Experiment settings
EXP_CONFIGS = ExperimentTemplate(
    N_PAIRS = 1000,
    CORRUPTIONS = ["gaussian_noise", "defocus_blur", "brightness", "fog"], 
    SEVERITIES  = [1, 2, 3, 5],
    EXPLAINERS = ["GradCAM"],
    SEEDS = [7, 42, 52, 128, 1200]
)

#N_PAIRS = 1000 
#CORRUPTIONS = ["gaussian_noise", "defocus_blur", "brightness", "fog"] 
#SEVERITIES  = [1, 2, 3, 5],
#EXPLAINERS = ["IG"],   # GradCAM
#SEEDS = [7, 42, 52, 128, 1200]


# N_PAIRS = 250, 
# CORRUPTIONS = ["gaussian_noise"], 
#    SEVERITIES  = [5],
#    EXPLAINERS = ["IG"],
#    SEEDS = [51]
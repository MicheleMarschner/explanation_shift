from collections import defaultdict
from itertools import product
import torchvision.transforms as T

from typing import Dict, List, Tuple

from src.data import sample_cifar10_pair_indices
from src.models import load_model
from src.experiment_stages.stage_00_clean_ref import compute_clean_reference
from src.experiment_stages.stage_01_artifacts import run_experiment
from src.experiment_stages.stage_02_drift_metrics import compute_drift_metrics
from src.experiment_stages.stage_03_quantus import run_quantus_metrics
from src.experiment_stages.stage_04_metaquantus import run_metaquantus_stage
from src.configs.global_config import PATHS, CIFAR10_MEAN, CIFAR10_SD
from src.utils import create_file_path, to_np_idx, set_seeds
from src.configs.experiments_config import ReferenceJob, RunConfig, ExperimentTemplate


def expand_template(t: ExperimentTemplate) -> List[RunConfig]:
    runs: List[RunConfig] = []
    for seed, expl, corr, sev in product(t.SEEDS, t.EXPLAINERS, t.CORRUPTIONS, t.SEVERITIES):
        runs.append(RunConfig(
            N_PAIRS=int(t.N_PAIRS),
            seed=int(seed),
            explainer=str(expl),
            corruption=str(corr),
            severity=int(sev),
        ))
    return runs

"""
def group_by_seed_explainer(runs: List[RunConfig]) -> Dict[Tuple[int, str], List[RunConfig]]:
    groups = defaultdict(list)
    for r in runs:
        groups[(r.seed, r.explainer)].append(r)
    return groups

"""

def expand_reference_jobs(t: ExperimentTemplate) -> List[ReferenceJob]:
    jobs: List[ReferenceJob] = []
    for seed, expl in product(t.SEEDS, t.EXPLAINERS):
        jobs.append(
            ReferenceJob(
                N_PAIRS=int(t.N_PAIRS),
                seed=int(seed),
                explainer=str(expl),
            )
        )
    return jobs

def expand_metaquantus_jobs(t: ExperimentTemplate) -> List[dict]:
    jobs: List[dict] = []
    for seed in t.SEEDS:
        jobs.append(
            {
                "N_PAIRS": int(t.N_PAIRS),
                "seed": int(seed),
            }
        )
    return jobs


def run_reference_job(job: ReferenceJob, exp_config, overwrite: bool = False):
    # Set up basics:
    resnet_model = load_model()
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_SD),
    ])

    # deterministic + subset per seed
    set_seeds(job.seed, deterministic=True)
    pair_idx = sample_cifar10_pair_indices(n_pairs=int(job.N_PAIRS), seed=job.seed)
    pair_idx_np = to_np_idx(pair_idx)

    # per-seed run dir
    experiment_dir = PATHS.runs / f"experiment__n{int(job.N_PAIRS)}__{job.explainer}__seed{job.seed}"
    clean_path = create_file_path(experiment_dir, "00__reference", "00__clean_ref")


    if overwrite or not clean_path.exists():
        compute_clean_reference(
            pair_idx=pair_idx_np,
            exp_config=exp_config,
            save_path=clean_path,
            model=resnet_model,
            transform=transform,
            explainer_name=job.explainer,
            seed=job.seed,
        )

    print(f"reference stored under {clean_path}")

def run_metaquantus_job(job: dict, overwrite: bool = False):
    set_seeds(job["seed"], deterministic=True)
    pair_idx = sample_cifar10_pair_indices(n_pairs=int(job["N_PAIRS"]), seed=job["seed"])
    pair_idx_np = to_np_idx(pair_idx)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_SD),
    ])

    # eigener seed-basierter Ordner, nicht an einzelnen Explainer gebunden
    experiment_dir = PATHS.runs / f"metaquantus__n{int(job['N_PAIRS'])}__seed{job['seed']}"
    metaquantus_path = create_file_path(
        experiment_dir,
        "04__metaquantus",
        "04__metaquantus",
    )

    if overwrite or not metaquantus_path.exists():
        run_metaquantus_stage(
            pair_idx=pair_idx_np,
            save_path=metaquantus_path,
            transform=transform
        )

    print(f"metaquantus stored under {metaquantus_path}")



def run_condition_job(job: RunConfig, stage: str, overwrite: bool = False):
    # Set up basics:
    resnet_model = load_model()
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_SD),
    ])

    # deterministic + subset per seed
    set_seeds(job.seed, deterministic=True)
    pair_idx = sample_cifar10_pair_indices(n_pairs=int(job.N_PAIRS), seed=job.seed)
    pair_idx_np = to_np_idx(pair_idx)

    # per-seed run dir
    experiment_dir = PATHS.runs / f"experiment__n{int(job.N_PAIRS)}__{job.explainer}__seed{job.seed}"
    clean_path = create_file_path(experiment_dir, "00__reference", "00__clean_ref")

    if not clean_path.exists():
        raise FileNotFoundError(f"Missing clean reference: {clean_path} (run --stage reference first)")

    artifact_path = create_file_path(
        experiment_dir, "01__artifacts", "01__artifacts", job.corruption, job.severity
    )
    drift_path = create_file_path(
        experiment_dir, "02__drift", "02__drift", job.corruption, job.severity
    )
    quantus_path = create_file_path(
        experiment_dir, "03__quantus", "03__quantus", job.corruption, job.severity
    )

    if stage == "artifact":
        if overwrite or not artifact_path.exists():
            run_experiment(
                pair_idx=pair_idx_np,
                corruption=job.corruption,
                severity=job.severity,
                clean_path=clean_path,
                save_path=artifact_path,
                model=resnet_model,
                transform=transform,
                explainer_name=job.explainer,
            )
        print(f"artifacts stored under {artifact_path}")
        return

    if stage == "drift":
        if overwrite or not drift_path.exists():
            if not artifact_path.exists():
                raise FileNotFoundError(f"Missing artifact: {artifact_path} (run --stage artifact first)")
            compute_drift_metrics(
                clean_path=clean_path,
                artifact_path=artifact_path,
                save_path=drift_path,
            )
        print(f"drift stored under {drift_path}")
        return

    if stage == "quantus":
        quantus_clean_path = create_file_path(
            experiment_dir, "03__quantus", "03__quantus", "clean", 0
        )

        if overwrite or not quantus_clean_path.exists():
            run_quantus_metrics(
                pair_idx=pair_idx_np,
                corruption="clean",
                severity=0,
                clean_path=clean_path,
                artifact_path=None,
                save_path=quantus_clean_path,
                model=resnet_model,
                transform=transform,
                explainer_name=job.explainer,
                mode="clean",
            )
        print(f"quantus(clean) stored under {quantus_clean_path}")

        if overwrite or not quantus_path.exists():
            if not artifact_path.exists():
                raise FileNotFoundError(f"Missing artifact: {artifact_path} (run --stage artifact first)")
            run_quantus_metrics(
                pair_idx=pair_idx_np,
                corruption=job.corruption,
                severity=job.severity,
                clean_path=clean_path,
                artifact_path=artifact_path,
                save_path=quantus_path,
                model=resnet_model,
                transform=transform,
                explainer_name=job.explainer,
                mode="corr",
            )
        print(f"quantus stored under {quantus_path}")
        return

    raise ValueError(f"Unknown stage: {stage}")


def run_train_pipeline(exp_config, stage, overwrite=False):
    if stage == "reference":
        reference_jobs = expand_reference_jobs(exp_config)

        for job in reference_jobs:
            run_reference_job(job=job, exp_config=exp_config, overwrite=overwrite)
        return

    if stage == "metaquantus":
        meta_jobs = expand_metaquantus_jobs(exp_config)

        for job in meta_jobs:
            run_metaquantus_job(job=job, overwrite=overwrite)
        return

    condition_jobs = expand_template(exp_config)

    for job in condition_jobs:
        run_condition_job(job=job, stage=stage, overwrite=overwrite)
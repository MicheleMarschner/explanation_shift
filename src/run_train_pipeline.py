from collections import defaultdict
from itertools import product
import torchvision.transforms as T
from captum.attr import IntegratedGradients

from typing import Dict, List, Tuple

from src.data import sample_cifar10_pair_indices
from src.models import load_model
from src.experiment_stages.stage_00_clean_ref import compute_clean_reference
from src.experiment_stages.stage_01_artifacts import run_experiment
from src.experiment_stages.stage_02_drift_metrics import compute_drift_metrics
from src.experiment_stages.stage_03_quantus import run_quantus_metrics
from src.configs.global_config import PATHS, CIFAR10_MEAN, CIFAR10_SD
from src.utils import create_file_path, to_np_idx, set_seeds
from src.configs.experiments_config import RunConfig, ExperimentTemplate


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

def group_by_seed_explainer(runs: List[RunConfig]) -> Dict[Tuple[int, str], List[RunConfig]]:
    groups = defaultdict(list)
    for r in runs:
        groups[(r.seed, r.explainer)].append(r)
    return groups


def build_explainer(expl_name: str, model):
    if expl_name == "IG":
        return IntegratedGradients(model)
    raise ValueError(f"Unknown explainer: {expl_name}")


def run_train_pipeline(exp_config, stage, overwrite=False):
    experiments = expand_template(exp_config)
    groups = group_by_seed_explainer(experiments)

    # Set up basics:
    resnet_model = load_model()
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_SD),
    ])

    for (seed, expl_name), cond_runs in groups.items():

        # deterministic + subset per seed (consistent across stages)
        set_seeds(seed, deterministic=True)
        pair_idx = sample_cifar10_pair_indices(n_pairs=int(exp_config.N_PAIRS), seed=seed)
        pair_idx_np = to_np_idx(pair_idx)

        # per-seed run dir (prevents overwriting)
        experiment_dir = PATHS.runs / f"experiment__n{int(exp_config.N_PAIRS)}__{expl_name}__seed{seed}"
        clean_path = create_file_path(experiment_dir, "00__reference", "00__clean_ref")

        explainer = build_explainer(expl_name, resnet_model)

        # ---------- STAGE 00 ----------
        if stage == "reference":
            if overwrite or not clean_path.exists():
                compute_clean_reference(
                    pair_idx=pair_idx_np,
                    exp_config=exp_config,
                    save_path=clean_path,
                    model=resnet_model,
                    transform=transform,
                    explainer=explainer,
                    seed=seed
                )
            print(f"reference stored under {clean_path}")
            continue  # stage-only, don’t run other stages

        # from here on, we need clean_ref on disk
        if not clean_path.exists():
            raise FileNotFoundError(f"Missing clean reference: {clean_path} (run --stage reference first)")

        if stage == "quantus":
            quantus_clean_path  = create_file_path(experiment_dir, "03__quantus", "03__quantus", "clean", 0)
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
                    mode="clean",
                )
            print(f"quantus(clean) stored under {quantus_clean_path}")

        # ---------- STAGE 01/02/03 (loop conditions) ----------
        for r in cond_runs:
            sev = r.severity
            corr = r.corruption

            artifact_path = create_file_path(experiment_dir, "01__artifacts", "01__artifacts", corr, sev)
            drift_path  = create_file_path(experiment_dir, "02__drift", "02__drift", corr, sev)
            quantus_path  = create_file_path(experiment_dir, "03__quantus", "03__quantus", corr, sev)

            if stage == "artifact":
                if overwrite or not artifact_path.exists():
                    run_experiment(
                        pair_idx=pair_idx_np,
                        corruption=corr,
                        severity=sev,
                        clean_path=clean_path,
                        save_path=artifact_path,
                        model=resnet_model,
                        transform=transform,
                        explainer=explainer
                    )
                print(f"artifacts stored under {artifact_path}")
                continue

            if stage == "drift":
                if overwrite or not drift_path.exists():
                    if not artifact_path.exists():
                        raise FileNotFoundError(f"Missing artifact: {artifact_path} (run --stage artifact first)")
                    compute_drift_metrics(
                        clean_path=clean_path,
                        artifact_path=artifact_path,
                        save_path=drift_path
                    )
                print(f"drift stored under {drift_path}")
                continue

            if stage == "quantus":
                if overwrite or not quantus_path.exists():
                    if not artifact_path.exists():
                        raise FileNotFoundError(f"Missing artifact: {artifact_path} (run --stage artifact first)")
                    run_quantus_metrics(
                        pair_idx=pair_idx_np,
                        corruption=corr,
                        severity=sev,
                        clean_path=clean_path,
                        artifact_path=artifact_path,
                        save_path=quantus_path,
                        model=resnet_model,
                        transform=transform,
                        mode="corr",
                    )
                print(f"quantus stored under {quantus_path}")
                continue
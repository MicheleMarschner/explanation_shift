import argparse
from collections import defaultdict
import importlib.util
from itertools import product
from pathlib import Path
import torchvision.transforms as T
from captum.attr import IntegratedGradients

from typing import Any, Dict, List, Tuple

from src.utils import create_file_path, ensure_dirs, to_np_idx, set_seeds
from src.configs.global_config import PATHS, CIFAR10_MEAN, CIFAR10_SD
from src.configs.experiments_config import RunConfig, ExperimentTemplate
from src.data import sample_cifar10_pair_indices
from src.models import load_model
from src.experiment_stages.stage_00_clean_ref import compute_clean_reference
from src.experiment_stages.stage_01_artifacts import run_experiment
from src.experiment_stages.stage_02_drift_metrics import compute_drift_metrics
from src.experiment_stages.stage_03_quantus import run_quantus_metrics



def load_experiment_config(file: str) -> Dict[str, Any]:
    """Load experiment setup from a Python config file by importing and executing it"""
    path = Path(file)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    # Import an import spec from the file path given as argument
    spec = importlib.util.spec_from_file_location("exp_cfg", str(path))

    # Create a new empty module object from that spec (nothing executed yet).
    module = importlib.util.module_from_spec(spec)
    
    # Execute the module code (i.e., run the .py file) which makes the top-level 
    # variables available via `module.CONFIG
    spec.loader.exec_module(module)  # type: ignore
    
    return module.EXP_CONFIGS


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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["reference", "artifact", "drift", "quantus"], default="reference", help="Which stages to run.")
    p.add_argument("--config", type=str, required=True, help="Path to config python file, e.g. configs/experiment_config.py")
    p.add_argument("--overwrite", action="store_true")

    return p.parse_args()



def main() -> None:
    
    args = parse_args()
    ensure_dirs(PATHS)
    exp_config = load_experiment_config(args.config)
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
        experiments_dir = PATHS.runs / f"experiment__n{int(exp_config.N_PAIRS)}__{expl_name}__seed{seed}"
        clean_path = create_file_path(experiments_dir, "00__reference", "00__clean_ref")

        explainer = build_explainer(expl_name, resnet_model)

        # ---------- STAGE 00 ----------
        if args.stage == "reference":
            if args.overwrite or not clean_path.exists():
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

        if args.stage == "quantus":
            quantus_clean_path  = create_file_path(experiments_dir, "03__quantus", "03__quantus", "clean", 0)
            if args.overwrite or not quantus_clean_path.exists():
                run_quantus_metrics(
                    clean_path=clean_path,
                    artifact_path=None,
                    out_path=quantus_clean_path,
                    exp_config=exp_config,
                    model=resnet_model,
                    transform=transform,
                    mode="clean",
                )
            print(f"quantus(clean) stored under {quantus_clean_path}")

        # ---------- STAGE 01/02/03 (loop conditions) ----------
        for r in cond_runs:
            sev = r.severity
            corr = r.corruption

            artifact_path = create_file_path(experiments_dir, "01__artifacts", "01__artifacts", corr, sev)
            drift_path  = create_file_path(experiments_dir, "02__drift", "02__drift", corr, sev)
            quantus_path  = create_file_path(experiments_dir, "03__quantus", "03__quantus", corr, sev)

            if args.stage == "artifact":
                if args.overwrite or not artifact_path.exists():
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

            if args.stage == "drift":
                if args.overwrite or not drift_path.exists():
                    if not artifact_path.exists():
                        raise FileNotFoundError(f"Missing artifact: {artifact_path} (run --stage artifact first)")
                    compute_drift_metrics(
                        clean_path=clean_path,
                        artifact_path=artifact_path,
                        save_path=drift_path,
                    )
                print(f"drift stored under {drift_path}")
                continue

            if args.stage == "quantus":
                if args.overwrite or not quantus_path.exists():
                    if not artifact_path.exists():
                        raise FileNotFoundError(f"Missing artifact: {artifact_path} (run --stage artifact first)")
                    run_quantus_metrics(
                        clean_path=clean_path,
                        artifact_path=artifact_path,
                        out_path=quantus_path,
                        exp_config=exp_config,
                        model=resnet_model,
                        transform=transform,
                        mode="corr",
                    )
                print(f"quantus stored under {quantus_path}")
                continue


if __name__ == "__main__":
    main()
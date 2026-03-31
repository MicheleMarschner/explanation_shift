import argparse
import importlib.util
from pathlib import Path

from typing import Any, Dict

from src.run_train_pipeline import (
    run_train_pipeline,
    run_reference_job,
    expand_template,
    expand_reference_jobs,
    run_condition_job,
    expand_metaquantus_jobs,
    run_metaquantus_job,
)
from src.run_eval_pipeline import run_eval_pipeline
from src.utils import ensure_dirs
from src.configs.global_config import PATHS


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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "eval"], default="train", help="Which mode to run.")
    p.add_argument(
        "--stage",
        choices=["reference", "artifact", "drift", "quantus", "metaquantus"],
        default="reference",
        help="Which stages to run.",
    )
    p.add_argument("--config", type=str, required=True, help="Path to config python file, e.g. configs/experiment_config.py")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--job-type",
        choices=["reference", "condition", "metaquantus"],
        default=None,
        help="Optional: run exactly one job of this type.",
    )
    p.add_argument("--job-index", type=int, default=None,
                help="Optional: index of the single job to run.")
    p.add_argument("--job-mode", choices=["all", "single"], default="all",
                help="Run all jobs locally (default) or a single selected job.")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs(PATHS)
    exp_config = load_experiment_config(args.config)

    if args.mode == "train":
        if args.job_mode == "all":
            run_train_pipeline(
                exp_config=exp_config,
                stage=args.stage,
                overwrite=args.overwrite,
            )
            return

        if args.job_type is None or args.job_index is None:
            raise ValueError("For --job-mode single, both --job-type and --job-index are required.")

        if args.job_type == "reference":
            if args.stage != "reference":
                raise ValueError("Reference jobs can only be run with --stage reference.")

            jobs = expand_reference_jobs(exp_config)
            if not (0 <= args.job_index < len(jobs)):
                raise IndexError(
                    f"job-index {args.job_index} out of range for {len(jobs)} reference jobs."
                )

            run_reference_job(
                job=jobs[args.job_index],
                exp_config=exp_config,
                overwrite=args.overwrite,
            )
            return

        if args.job_type == "condition":
            if args.stage not in {"artifact", "drift", "quantus"}:
                raise ValueError(
                    "Condition jobs can only be run with --stage artifact, drift, or quantus."
                )

            jobs = expand_template(exp_config)
            if not (0 <= args.job_index < len(jobs)):
                raise IndexError(
                    f"job-index {args.job_index} out of range for {len(jobs)} condition jobs."
                )

            run_condition_job(
                job=jobs[args.job_index],
                stage=args.stage,
                overwrite=args.overwrite,
            )
            return

        if args.job_type == "metaquantus":
            if args.stage != "metaquantus":
                raise ValueError("MetaQuantus jobs can only be run with --stage metaquantus.")

            jobs = expand_metaquantus_jobs(exp_config)
            if not (0 <= args.job_index < len(jobs)):
                raise IndexError(
                    f"job-index {args.job_index} out of range for {len(jobs)} metaquantus jobs."
                )

            run_metaquantus_job(
                job=jobs[args.job_index],
                overwrite=args.overwrite,
            )
            return

    if args.mode == "eval":
        run_eval_pipeline(
            exp_config=exp_config,
            stage=args.stage,
            overwrite=args.overwrite,
        )

if __name__ == "__main__":
    main()
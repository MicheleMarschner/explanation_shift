import argparse
import importlib.util
from pathlib import Path

from typing import Any, Dict

from src import run_train_pipeline
from src import run_eval_pipeline
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
    p.add_argument("--stage", choices=["reference", "artifact", "drift", "quantus"], default="reference", help="Which stages to run.")
    p.add_argument("--config", type=str, required=True, help="Path to config python file, e.g. configs/experiment_config.py")
    p.add_argument("--overwrite", action="store_true")

    return p.parse_args()


def main() -> None:
    
    args = parse_args()
    ensure_dirs(PATHS)
    exp_config = load_experiment_config(args.config)

    # build identifier

    if args.mode == "train":
        run_train_pipeline(exp_config=exp_config, stage=args.stage, overwrite=args.overwrite)
    
    if args.mode == "eval":
        run_eval_pipeline(exp_config=exp_config, stage=args.stage, overwrite=args.overwrite)



if __name__ == "__main__":
    main()
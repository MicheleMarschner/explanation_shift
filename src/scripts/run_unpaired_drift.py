import argparse
from pathlib import Path

from src.experiment_stages.stage_02_drift_metrics import compute_drift_metrics_unpaired


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Path to one existing experiment directory, e.g. experiments/experiment__n1000__IG__seed42",
    )
    p.add_argument(
        "--corruption",
        type=str,
        required=True,
        help="Corruption name, e.g. gaussian_noise",
    )
    p.add_argument(
        "--severity",
        type=int,
        required=True,
        help="Severity level, e.g. 1",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    experiment_dir = args.experiment_dir
    corruption = args.corruption
    severity = args.severity

    clean_path = experiment_dir / "00__reference" / "00__clean_ref.pt"
    artifact_path = experiment_dir / "01__artifacts" / f"01__artifacts__{corruption}__sev{severity}.pt"
    save_path = experiment_dir / "02__drift_unpaired" / f"02__drift_unpaired__{corruption}__sev{severity}.pt"

    if not clean_path.exists():
        raise FileNotFoundError(f"Missing clean reference: {clean_path}")

    if not artifact_path.exists():
        raise FileNotFoundError(f"Missing artifact: {artifact_path}")

    row = compute_drift_metrics_unpaired(
        clean_path=clean_path,
        artifact_path=artifact_path,
        save_path=save_path,
    )

    print("Unpaired drift result:")
    for k, v in row.items():
        print(f"{k}: {v}")

    print(f"\nSaved to: {save_path}")


if __name__ == "__main__":
    main()

"""
python scripts/run_unpaired_drift.py \
  --experiment-dir /sc/home/michele.marschner/project/explanation_shift/experiments/experiment__n1000__IG__seed42 \
  --corruption gaussian_noise \
  --severity 1
"""
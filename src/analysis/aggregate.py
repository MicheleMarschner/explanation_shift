from pathlib import Path
from typing import Any, Callable, Iterable
import pandas as pd
import torch


def _load_pt(path: Path) -> dict[str, Any]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return obj


def _collect_rows(
    pt_dir: Path,
    pattern: str,
    row_extractor: Callable[[dict[str, Any]], dict[str, Any] | None],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    if not pt_dir.exists():
        return rows

    for pt_path in sorted(pt_dir.glob(pattern)):
        payload = _load_pt(pt_path)
        row = row_extractor(payload)
        if row is not None:
            rows.append(row)

    return rows


def _write_csv(rows: Iterable[dict[str, Any]], csv_path: Path) -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    if not df.empty:
        sort_cols = [c for c in ("corruption", "severity") if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols).reset_index(drop=True)
            df.to_csv(csv_path, index=False)

    return df


def _extract_stage01_row(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "corruption": payload["corruption"],
        "severity": int(payload["severity"]),
        "time_sec": float(payload["time_sec"]),
        "acc_corr": float(payload["corrupt_reference"]["acc_corr"]),
    }


def _extract_row_field(payload: dict[str, Any]) -> dict[str, Any]:
    return dict(payload["row"])


def aggregate_stage01(run_dir: Path) -> pd.DataFrame:
    pt_dir = run_dir / "01__artifacts"
    csv_path = pt_dir / "01__artifact_results.csv"
    rows = _collect_rows(pt_dir, "*.pt", _extract_stage01_row)
    return _write_csv(rows, csv_path)


def aggregate_stage02(run_dir: Path) -> pd.DataFrame:
    pt_dir = run_dir / "02__drift"
    csv_path = pt_dir / "02__drift_results.csv"
    rows = _collect_rows(pt_dir, "*.pt", _extract_row_field)
    return _write_csv(rows, csv_path)


def aggregate_stage03(run_dir: Path, mode: str = "corr") -> pd.DataFrame:
    pt_dir = run_dir / "03__quantus"
    csv_path = pt_dir / f"03__quantus_results__{mode}.csv"

    rows = []
    if pt_dir.exists():
        for pt_path in sorted(pt_dir.glob("*.pt")):
            payload = _load_pt(pt_path)

            payload_mode = payload.get("meta", {}).get("mode")
            if payload_mode != mode:
                continue

            rows.append(_extract_row_field(payload))

    return _write_csv(rows, csv_path)


def aggregate_stage04(run_dir: Path) -> pd.DataFrame:
    pt_dir = run_dir / "04__metaquantus"
    csv_path = pt_dir / "04__metaquantus_results.csv"

    def extract_row(payload: dict[str, Any]) -> dict[str, Any] | None:
        results = payload.get("results", {})
        row: dict[str, Any] = {}

        for category, category_dict in results.items():
            for metric_name, metric_payload in category_dict.items():
                prefix = f"{category}__{metric_name}"

                for group_name in (
                    "results_meta_consistency_scores",
                    "results_consistency_scores",
                    "results_intra_scores",
                    "results_inter_scores",
                ):
                    group = metric_payload.get(group_name, {})
                    if isinstance(group, dict):
                        for k, v in group.items():
                            row[f"{prefix}__{group_name}__{k}"] = v

        return row if row else None

    rows = _collect_rows(pt_dir, "*.pt", extract_row)
    return _write_csv(rows, csv_path)


def aggregate_experiment(run_dir: Path) -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}

    if (run_dir / "01__artifacts").exists():
        outputs["stage01"] = aggregate_stage01(run_dir)

    if (run_dir / "02__drift").exists():
        outputs["stage02"] = aggregate_stage02(run_dir)

    if (run_dir / "03__quantus").exists():
        outputs["stage03_corr"] = aggregate_stage03(run_dir, mode="corr")
        outputs["stage03_clean"] = aggregate_stage03(run_dir, mode="clean")

    if (run_dir / "04__metaquantus").exists():
        outputs["stage04"] = aggregate_stage04(run_dir)

    return outputs


def _is_experiment_dir(path: Path) -> bool:
    if not path.is_dir():
        return False

    stage_dirs = {
        "01__artifacts",
        "02__drift",
        "03__quantus",
        "04__metaquantus",
    }
    return any((path / stage_dir).exists() for stage_dir in stage_dirs)


def aggregate_all_experiments(experiments_root: Path) -> dict[str, dict[str, pd.DataFrame]]:
    experiments_root = Path(experiments_root)

    outputs: dict[str, dict[str, pd.DataFrame]] = {}

    for run_dir in sorted(experiments_root.iterdir()):
        if not _is_experiment_dir(run_dir):
            continue
        outputs[run_dir.name] = aggregate_experiment(run_dir)

    return outputs
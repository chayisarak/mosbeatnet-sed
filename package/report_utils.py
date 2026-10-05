from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

import config


FLOAT_FORMAT = getattr(config, "REPORT_FLOAT_FORMAT", "%.4f")


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _run_is_completed(run_dir: Path) -> bool:
    status_path = run_dir / "run_status.json"
    if not status_path.is_file():
        return False

    try:
        return str(_read_json(str(status_path)).get("status", "")) == "completed"
    except (OSError, json.JSONDecodeError):
        return False


def _latest_run(seed_dir: Path) -> Optional[Path]:
    runs = [
        path
        for path in seed_dir.iterdir()
        if path.is_dir()
        and path.name.startswith("run_")
        and (path / "run_config.json").is_file()
    ]
    if not runs:
        return None

    completed = [path for path in runs if _run_is_completed(path)]
    candidates = completed or runs
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))


def latest_runs_for_experiment(exp_name: str) -> list[dict]:
    exp_dir = Path(config.OUTPUT_BASE) / exp_name
    if not exp_dir.is_dir():
        return []

    rows = []
    for model_dir in sorted(exp_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        model_key = model_dir.name
        model_name = config.MODEL_DISPLAY_NAMES.get(model_key, model_key)

        for seed_dir in sorted(model_dir.glob("seed_*")):
            if not seed_dir.is_dir():
                continue

            try:
                seed = int(seed_dir.name.split("seed_", 1)[1])
            except (IndexError, ValueError):
                continue

            run_dir = _latest_run(seed_dir)
            if run_dir is None:
                continue

            rows.append({
                "experiment": exp_name,
                "model_key": model_key,
                "model": model_name,
                "seed": seed,
                "run_id": run_dir.name,
                "run_dir": str(run_dir),
            })

    return rows


def build_all_models_training_history(exp_name: str) -> Optional[pd.DataFrame]:
    frames = []

    for info in latest_runs_for_experiment(exp_name):
        history_path = Path(info["run_dir"]) / "metrics" / "history.csv"
        if not history_path.is_file():
            continue

        frame = pd.read_csv(history_path)
        if frame.empty:
            continue

        if "learning_rate" in frame.columns:
            lr = pd.to_numeric(frame["learning_rate"], errors="coerce")
            frame["learning_rate"] = lr.map(
                lambda value: "" if pd.isna(value) else f"{value:.2e}"
            )

        frame.insert(0, "Run ID", info["run_id"])
        frame.insert(0, "Environment", "mixed")
        frame.insert(0, "Seed", info["seed"])
        frame.insert(0, "Model Key", info["model_key"])
        frame.insert(0, "Model", info["model"])
        frame.insert(0, "Experiment", exp_name)
        frames.append(frame)

    if not frames:
        return None

    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged = merged.sort_values(
        ["Model", "Seed", "epoch"],
        kind="stable",
    ).reset_index(drop=True)
    merged.insert(0, "No", range(1, len(merged) + 1))
    return merged


def build_all_models_training_summary(exp_name: str) -> Optional[pd.DataFrame]:
    rows = []

    for info in latest_runs_for_experiment(exp_name):
        run_dir = Path(info["run_dir"])
        history_path = run_dir / "metrics" / "history.csv"
        best_path = run_dir / "metrics" / "best_metrics.json"

        if not history_path.is_file() or not best_path.is_file():
            continue

        history = pd.read_csv(history_path)
        if history.empty:
            continue

        best = _read_json(str(best_path))
        stopped_epoch = int(history["epoch"].iloc[-1])

        rows.append({
            "Experiment": exp_name,
            "Model": info["model"],
            "Model Key": info["model_key"],
            "Seed": info["seed"],
            "Environment": "mixed",
            "Run ID": info["run_id"],
            "Best Val Loss": best.get("best_val_loss"),
            "Best Val Loss Epoch": best.get("best_val_loss_epoch"),
            "Best Val Macro F1": best.get("best_val_f1_macro"),
            "Best Val Macro F1 Epoch": best.get("best_val_f1_macro_epoch"),
            "Stopped Epoch": stopped_epoch,
            "Epochs Run": int(len(history)),
            "Early Stop Metric": best.get("early_stop_metric"),
            "Early Stop Count": best.get("early_stop_no_improve_checks"),
            "Patience": best.get("patience"),
        })

    if not rows:
        return None

    frame = pd.DataFrame(rows)
    frame = frame.sort_values(["Model", "Seed"], kind="stable").reset_index(drop=True)
    frame.insert(0, "No", range(1, len(frame) + 1))
    return frame


def write_experiment_reports(exp_name: str) -> dict[str, str]:
    exp_dir = Path(config.OUTPUT_BASE) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}

    history = build_all_models_training_history(exp_name)
    if history is not None:
        path = exp_dir / "all_models_training_history.csv"
        history.to_csv(path, index=False, float_format=FLOAT_FORMAT)
        outputs["training_history"] = str(path)
        print(f"[report] all-model training history -> {path}")

    train_summary = build_all_models_training_summary(exp_name)
    if train_summary is not None:
        path = exp_dir / "all_models_training_summary.csv"
        train_summary.to_csv(path, index=False, float_format=FLOAT_FORMAT)
        outputs["training_summary"] = str(path)
        print(f"[report] all-model training summary -> {path}")

    return outputs
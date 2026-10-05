# run_utils.py
from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

import config


RUN_SCHEMA_VERSION = 2

_MODEL_ALIASES = {
    "mosbeatnet": "mosbeatnet_v1",
    "mosbeatnet_v1": "mosbeatnet_v1",
    "mosbeatnetv1": "mosbeatnet_v1",
    "mosbeatnet_v2": "mosbeatnet_v2",
    "mosbeatnetv2": "mosbeatnet_v2",
    "mosbeatnet_v2_se": "mosbeatnet_v2_se",
    "mosbeatnetv2_se": "mosbeatnet_v2_se",
    "mosbeatnet_v2_in": "mosbeatnet_v2_in",
    "mosbeatnetv2_in": "mosbeatnet_v2_in",
    "mosqplus": "mosqplus",
    "mosqplusmodel": "mosqplus",
    "sednet": "sednet",
    "sednetsegmentlevel": "sednet",
    "cfresnet": "cfresnet1d_small",
    "cfresnet_small": "cfresnet1d_small",
    "cfresnet1d": "cfresnet1d_small",
    "cfresnet1d_small": "cfresnet1d_small",
    "cfresnet_medium": "cfresnet1d_medium",
    "cfresnet1d_medium": "cfresnet1d_medium",
    "cfresnet_large": "cfresnet1d_large",
    "cfresnet1d_large": "cfresnet1d_large",
    "mtrcnn": "mtrcnn",
}


def canonical_model_key(model_key: str) -> str:
    key = str(model_key).strip().lower()
    try:
        return _MODEL_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"Unknown model key: {model_key}") from exc


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return os.fspath(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def stable_json_dumps(value: Any) -> str:
    return json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def stable_hash(value: Any, length: Optional[int] = None) -> str:
    digest = hashlib.sha256(stable_json_dumps(value).encode("utf-8")).hexdigest()
    return digest if length is None else digest[: int(length)]


def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _relative_or_absolute(path: str) -> str:
    absolute = os.path.abspath(os.fspath(path))
    project_root = os.path.abspath(getattr(config, "PROJ_DIR", os.getcwd()))
    try:
        common = os.path.commonpath([absolute, project_root])
    except ValueError:
        common = None
    if common == project_root:
        return os.path.relpath(absolute, project_root).replace("\\", "/")
    return absolute.replace("\\", "/")


def _normalise_config_paths(value: Any) -> Any:
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if str(key) == "output":
                continue
            result[str(key)] = _normalise_config_paths(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_normalise_config_paths(item) for item in value]
    if isinstance(value, str):
        looks_like_path = (
            os.path.sep in value
            or "/" in value
            or value.lower().endswith((".csv", ".wav", ".pt", ".pth"))
        )
        return _relative_or_absolute(value) if looks_like_path else value
    return _json_safe(value)


def get_git_info(project_dir: Optional[str] = None) -> Dict[str, Any]:
    root = os.path.abspath(project_dir or getattr(config, "PROJ_DIR", os.getcwd()))

    def run_git(*args: str) -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", "-C", root, *args],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            return result.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    commit = run_git("rev-parse", "HEAD")
    short_commit = run_git("rev-parse", "--short", "HEAD")
    branch = run_git("rev-parse", "--abbrev-ref", "HEAD")
    status = run_git("status", "--porcelain")
    return {
        "commit": commit,
        "short_commit": short_commit,
        "branch": branch,
        "dirty": bool(status) if status is not None else None,
    }


def get_environment_info() -> Dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    gpu_names = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            try:
                gpu_names.append(torch.cuda.get_device_name(index))
            except Exception:
                gpu_names.append(f"cuda:{index}")

    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "cuda_available": cuda_available,
        "torch_cuda": getattr(torch.version, "cuda", None),
        "cudnn": torch.backends.cudnn.version() if cuda_available else None,
        "gpu_names": gpu_names,
    }

    optional_packages = ["sklearn", "librosa", "soundfile", "matplotlib"]
    for package_name in optional_packages:
        try:
            module = __import__(package_name)
            versions[package_name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[package_name] = None
    return versions


def build_resolved_run_config(
    exp_cfg: Mapping[str, Any],
    model_key: str,
    model_tag: str,
    model_kwargs: Mapping[str, Any],
    training_seed: int,
    repeat: bool,
    mode: Optional[str],
) -> Dict[str, Any]:
    canonical_key = canonical_model_key(model_key)
    semantic_exp_cfg = _normalise_config_paths(copy.deepcopy(dict(exp_cfg)))
    payload = {
        "schema_version": RUN_SCHEMA_VERSION,
        "experiment": semantic_exp_cfg,
        "model": {
            "key": canonical_key,
            "tag": model_tag,
            "kwargs": _json_safe(dict(model_kwargs)),
        },
        "training": {
            "seed": int(training_seed),
            "deterministic": bool(repeat),
        },
        "invocation_mode": mode or "train_eval",
    }
    hash_payload = {
        "schema_version": payload["schema_version"],
        "experiment": payload["experiment"],
        "model": payload["model"],
        "training": payload["training"],
    }
    payload["config_hash"] = stable_hash(hash_payload)
    return payload


def _unique_run_dir(parent: str, base_run_id: str) -> Tuple[str, str]:
    run_id = base_run_id
    run_dir = os.path.join(parent, run_id)
    suffix = 2
    while os.path.exists(run_dir):
        run_id = f"{base_run_id}_{suffix}"
        run_dir = os.path.join(parent, run_id)
        suffix += 1
    return run_id, run_dir


def create_run_layout(
    exp_name: str,
    model_key: str,
    seed: int,
    config_hash: str,
    git_info: Optional[Mapping[str, Any]] = None,
    output_base: Optional[str] = None,
) -> Dict[str, str]:
    canonical_key = canonical_model_key(model_key)
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    short_git = (git_info or {}).get("short_commit")
    run_id = f"run_{timestamp}_cfg{config_hash[:8]}"
    if short_git:
        run_id += f"_git{short_git}"

    root = os.path.abspath(output_base or config.OUTPUT_BASE)
    parent = os.path.join(root, exp_name, canonical_key, f"seed_{int(seed)}")
    os.makedirs(parent, exist_ok=True)
    run_id, run_dir = _unique_run_dir(parent, run_id)

    layout = {
        "run_id": run_id,
        "base": run_dir,
        "checkpoints": os.path.join(run_dir, "checkpoints"),
        "metrics": os.path.join(run_dir, "metrics"),
        "evaluation": os.path.join(run_dir, "evaluation"),
        "tensorboard": os.path.join(run_dir, "tensorboard"),
        "csv": os.path.join(run_dir, "csv"),
        "segment_cache": os.path.join(run_dir, "segment_cache"),
        "code_snapshot": os.path.join(run_dir, "code_snapshot"),
    }
    for path in layout.values():
        if path == run_id:
            continue
        os.makedirs(path, exist_ok=True)
    return layout


def make_run_exp_cfg(exp_cfg: Mapping[str, Any], layout: Mapping[str, str]) -> Dict[str, Any]:
    result = copy.deepcopy(dict(exp_cfg))
    result["output"] = {
        "base": layout["base"],
        "model_dir": layout["checkpoints"],
        "eval_dir": layout["evaluation"],
        "run_dir": layout["tensorboard"],
        "csv_dir": layout["csv"],
        "seg_dir": layout["segment_cache"],
    }
    result["repro_run_dir"] = layout["base"]
    result["repro_run_id"] = layout["run_id"]
    return result


def _write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)


def write_initial_run_artifacts(
    layout: Mapping[str, str],
    resolved_config: Mapping[str, Any],
    command: Sequence[str],
    git_info: Mapping[str, Any],
) -> None:
    _write_json(os.path.join(layout["base"], "run_config.json"), resolved_config)
    _write_json(os.path.join(layout["base"], "environment.json"), get_environment_info())
    _write_json(os.path.join(layout["base"], "git.json"), git_info)
    with open(os.path.join(layout["base"], "command.txt"), "w", encoding="utf-8") as handle:
        handle.write(" ".join(map(str, command)).strip() + "\n")
    update_run_status(layout["base"], "created")


def update_run_status(run_dir: str, status: str, **details: Any) -> None:
    path = os.path.join(run_dir, "run_status.json")
    payload: Dict[str, Any] = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            payload = {}
    payload.update(
        {
            "schema_version": RUN_SCHEMA_VERSION,
            "status": str(status),
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    payload.update(_json_safe(details))
    _write_json(path, payload)


def snapshot_code(
    destination: str,
    project_dir: Optional[str] = None,
    file_names: Optional[Sequence[str]] = None,
) -> List[str]:
    root = os.path.abspath(project_dir or getattr(config, "PROJ_DIR", os.getcwd()))
    names = list(
        file_names
        or (
            "config.py",
            "data_paths.py",
            "experiments.py",
            "generate.py",
            "main.py",
            "mosq_dataloader.py",
            "run_utils.py",
            "simulation.py",
            "model.py",
            "evaluate.py",
            "CFResnet1D.py",
            "mrtcnn.py",
        )
    )
    os.makedirs(destination, exist_ok=True)
    copied = []
    for name in names:
        source = os.path.join(root, name)
        if not os.path.isfile(source):
            continue
        target = os.path.join(destination, os.path.basename(name))
        shutil.copy2(source, target)
        copied.append(target)
    return copied


def _csv_summary(path: str) -> Dict[str, Any]:
    frame = pd.read_csv(path)
    summary: Dict[str, Any] = {
        "path": _relative_or_absolute(path),
        "sha256": sha256_file(path),
        "size_bytes": os.path.getsize(path),
        "rows": int(len(frame)),
        "columns": list(map(str, frame.columns)),
    }
    if "file_name" in frame.columns:
        summary["unique_audio_files"] = int(frame["file_name"].nunique(dropna=True))
    if "source_name" in frame.columns:
        summary["source_names"] = sorted(
            frame["source_name"].dropna().astype(str).unique().tolist()
        )
    if "dataset_type" in frame.columns:
        summary["dataset_types"] = sorted(
            frame["dataset_type"].dropna().astype(str).unique().tolist()
        )
    if "seed" in frame.columns:
        seeds = []
        for value in frame["seed"].dropna().unique().tolist():
            try:
                seeds.append(int(value))
            except (TypeError, ValueError):
                seeds.append(str(value))
        summary["generation_seeds"] = sorted(seeds, key=str)
    return summary


def write_data_manifest(
    run_dir: str,
    source_csvs: Iterable[str],
    merged_csvs: Iterable[str],
    filename: str = "data_manifest.json",
) -> Dict[str, Any]:
    source_paths = list(dict.fromkeys(os.path.abspath(os.fspath(path)) for path in source_csvs))
    merged_paths = list(dict.fromkeys(os.path.abspath(os.fspath(path)) for path in merged_csvs))

    missing = [path for path in source_paths + merged_paths if not os.path.isfile(path)]
    if missing:
        details = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Cannot build data manifest; missing CSV files:\n{details}")

    source_entries = [_csv_summary(path) for path in source_paths]
    merged_entries = [_csv_summary(path) for path in merged_paths]
    generation_seeds = sorted(
        {
            seed
            for entry in source_entries + merged_entries
            for seed in entry.get("generation_seeds", [])
        },
        key=str,
    )
    payload = {
        "schema_version": RUN_SCHEMA_VERSION,
        "source_metadata": source_entries,
        "merged_metadata": merged_entries,
        "generation_seeds": generation_seeds,
    }
    payload["data_manifest_hash"] = stable_hash(payload)
    _write_json(os.path.join(run_dir, filename), payload)
    return payload


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_run_config(run_dir: str) -> Dict[str, Any]:
    path = os.path.join(os.path.abspath(run_dir), "run_config.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"run_config.json not found: {path}")
    return read_json(path)


def load_data_manifest(run_dir: str) -> Dict[str, Any]:
    path = os.path.join(os.path.abspath(run_dir), "data_manifest.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"data_manifest.json not found: {path}")
    return read_json(path)


def find_latest_run(
    exp_name: str,
    model_key: str,
    seed: int,
    output_base: Optional[str] = None,
) -> str:
    canonical_key = canonical_model_key(model_key)
    root = os.path.abspath(output_base or config.OUTPUT_BASE)
    parent = os.path.join(root, exp_name, canonical_key, f"seed_{int(seed)}")
    if not os.path.isdir(parent):
        raise FileNotFoundError(f"No run directory found: {parent}")

    candidates = []
    for name in os.listdir(parent):
        path = os.path.join(parent, name)
        if not os.path.isdir(path) or not name.startswith("run_"):
            continue
        config_path = os.path.join(path, "run_config.json")
        if os.path.isfile(config_path):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No reproducible runs found in: {parent}")
    return max(candidates, key=lambda path: (os.path.getmtime(path), os.path.basename(path)))


def checkpoint_path(run_dir: str, reason: str = "best_macro_f1") -> str:
    aliases = {
        "best_f1": "best_macro_f1.pt",
        "best_macro_f1": "best_macro_f1.pt",
        "best_loss": "best_val_loss.pt",
        "best_val_loss": "best_val_loss.pt",
        "last": "last.pt",
    }
    try:
        filename = aliases[str(reason)]
    except KeyError as exc:
        raise ValueError(
            "checkpoint reason must be one of: best_macro_f1, best_val_loss, last"
        ) from exc
    path = os.path.join(os.path.abspath(run_dir), "checkpoints", filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def capture_rng_state() -> Dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Optional[Mapping[str, Any]]) -> None:
    if not state:
        return
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch_cpu") is not None:
        torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])

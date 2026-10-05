#main.py

from __future__ import annotations

import argparse
import inspect
import multiprocessing as mp
import os
import random
from pathlib import Path
import sys
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

import config
from experiments import EXP_LIST
from evaluate import (
    cleanup_evaluation_outputs,
    combine_evaluation_summaries,
    get_eval_run_dir,
    get_eval_tag_dir,
    get_model_predictions,
    run_evaluation_from_predictions,
    write_all_model_reports,
    write_evaluation_summary,
)
from model import (
    Mosbeatnet,
    MosbeatnetV2,
    MosqPlusModel,
    SEDNetSegmentLevel,
)
from CFResnet1D import CFResNet1DSequence
from mrtcnn import MTRCNNSegmentLevel
from mosq_dataloader import (
    AudioAugmentor,
    AudioSequenceDataset,
    get_consistent_label_map,
    merge_metadata,
    train_model,
    set_label_mode,
    validate_metadata_labels,
)
from report_utils import write_experiment_reports
from run_utils import (
    build_resolved_run_config,
    canonical_model_key,
    checkpoint_path as run_checkpoint_path,
    create_run_layout,
    find_latest_run,
    get_git_info,
    load_data_manifest,
    load_run_config,
    make_run_exp_cfg,
    snapshot_code,
    update_run_status,
    write_data_manifest,
    write_initial_run_artifacts,
)

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

MODEL_CHOICES = [
    "mosbeatnet_v2",
    "mosqplus",
    "sednet",
    "cfresnet1d_small",
    "mtrcnn",
]
DEFAULT_MODELS = [
    "mosbeatnet_v2",
    "mosqplus",
    "sednet",
    "cfresnet1d_small",
    "mtrcnn",
]

# =============================================================================
# Basic setup
# =============================================================================


def set_seed(seed: int, deterministic: bool) -> None:
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_segment_samples(exp_cfg: dict) -> int:
    return int(round(config.SR * float(exp_cfg["seg_sec"])))


def make_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


# =============================================================================
# Metadata preparation
# =============================================================================


def merge_existing_metadata(files, output_path: str, tag: str) -> str:
    paths = [os.path.abspath(path) for path in make_list(files)]
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        details = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing metadata for evaluation tag={tag}:\n{details}")
    return merge_metadata(paths, output_path, strict=True)


def build_train_val_csvs(exp_cfg: dict) -> tuple[str, str]:
    csv_dir = exp_cfg["output"]["csv_dir"]
    os.makedirs(csv_dir, exist_ok=True)
    train_csv = merge_metadata(
        exp_cfg["train_csvs"],
        os.path.join(csv_dir, "simulated_all_train.csv"),
    )
    val_csv = merge_metadata(
        exp_cfg["val_csvs"],
        os.path.join(csv_dir, "simulated_all_val.csv"),
    )
    return train_csv, val_csv


def eval_label_space(exp_cfg: dict, tag: str, spec: dict) -> str:
    if spec.get("label_space"):
        return str(spec["label_space"]).strip().casefold()

    old_view = str(spec.get("view", "")).strip().casefold()
    if old_view in {"species", "species_sex"}:
        return old_view
    if tag.startswith("humbug") or tag.startswith("biodcase"):
        return "species"
    return str(exp_cfg["target_type"]).strip().casefold()


def build_eval_csv_map(exp_cfg: dict) -> dict[str, dict]:
    csv_dir = exp_cfg["output"]["csv_dir"]
    result = {}
    for tag, spec in exp_cfg["eval_sets"].items():
        result[tag] = {
            "csv": merge_existing_metadata(
                spec["csvs"],
                os.path.join(csv_dir, f"test_{tag}.csv"),
                tag=tag,
            ),
            "label_space": eval_label_space(exp_cfg, tag, spec),
        }
    return result


def all_source_csvs(exp_cfg: dict) -> list[str]:
    paths = list(exp_cfg.get("train_csvs", [])) + list(exp_cfg.get("val_csvs", []))
    for spec in exp_cfg.get("eval_sets", {}).values():
        paths.extend(make_list(spec.get("csvs")))
    return list(dict.fromkeys(os.path.abspath(path) for path in paths))


def all_eval_source_csvs(exp_cfg: dict) -> list[str]:
    paths = []
    for spec in exp_cfg.get("eval_sets", {}).values():
        paths.extend(make_list(spec.get("csvs")))
    return list(dict.fromkeys(os.path.abspath(path) for path in paths))


# =============================================================================
# Models and checkpoints
# =============================================================================


def build_cfresnet(num_classes: int, variant: str, device: torch.device):
    return CFResNet1DSequence(
        num_classes=num_classes,
        variant=variant,
        pool_size=5,
        kernel_size=11,
        modes=16,
    ).to(device)


def build_sednet_spec(num_classes: int, exp_cfg: dict, device: torch.device):
    full_kwargs = {
        "sr": config.SR,
        "segment_sec": exp_cfg["seg_sec"],
        "n_classes": num_classes,
        **config.SEDNET_CFG,
    }
    try:
        return SEDNetSegmentLevel(**full_kwargs).to(device), full_kwargs
    except TypeError:
        basic_kwargs = {
            "sr": config.SR,
            "segment_sec": exp_cfg["seg_sec"],
            "n_classes": num_classes,
        }
        try:
            return SEDNetSegmentLevel(**basic_kwargs).to(device), basic_kwargs
        except TypeError:
            fallback_kwargs = {"sr": config.SR, "n_classes": num_classes}
            return SEDNetSegmentLevel(**fallback_kwargs).to(device), fallback_kwargs


def build_model_spec(model_key: str, num_classes: int, exp_cfg: dict, device: torch.device):
    key = canonical_model_key(model_key)
    n_timesteps = get_segment_samples(exp_cfg)

    if key == "mosbeatnet_v1":
        kwargs = {"n_timesteps": n_timesteps, "n_outputs": num_classes}
        return "Mosbeatnet", Mosbeatnet(**kwargs).to(device), kwargs
    if key == "mosbeatnet_v2":
        kwargs = {"n_timesteps": n_timesteps, "n_outputs": num_classes}
        return "MosbeatnetV2", MosbeatnetV2(**kwargs).to(device), kwargs
    # if key == "mosbeatnet_v2_se":
    #     kwargs = {"n_timesteps": n_timesteps, "n_outputs": num_classes}
    #     return "MosbeatnetV2_SE", MosbeatnetV2_SE(**kwargs).to(device), kwargs
    # if key == "mosbeatnet_v2_in":
    #     kwargs = {"n_timesteps": n_timesteps, "n_outputs": num_classes}
    #     return "MosbeatnetV2_in", MosbeatnetV2_in(**kwargs).to(device), kwargs
    if key == "mosqplus":
        kwargs = {"n_timesteps": n_timesteps, "n_outputs": num_classes}
        return "MosqPlusModel", MosqPlusModel(**kwargs).to(device), kwargs
    if key == "sednet":
        model, kwargs = build_sednet_spec(num_classes, exp_cfg, device)
        return "SEDNetSegmentLevel", model, kwargs
    if key.startswith("cfresnet1d_"):
        variant = key.rsplit("_", 1)[-1]
        kwargs = {
            "num_classes": num_classes,
            "variant": variant,
            "pool_size": 5,
            "kernel_size": 11,
            "modes": 16,
        }
        return f"CFResNet1D_{variant}", build_cfresnet(num_classes, variant, device), kwargs
    if key == "mtrcnn":
        kwargs = {"n_outputs": num_classes}
        return "MTRCNN", MTRCNNSegmentLevel(**kwargs).to(device), kwargs
    raise ValueError(f"Unknown model: {model_key}")


def resolve_checkpoint(run_dir: str, ckpt_path=None, reason: str = "best_macro_f1") -> str:
    if ckpt_path is not None:
        path = os.path.abspath(ckpt_path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    return run_checkpoint_path(run_dir, reason)


def torch_load_checkpoint(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def checkpoint_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("model_state", "model_state_dict", "state_dict"):
        if key in checkpoint:
            return checkpoint[key]
    return checkpoint


def validate_checkpoint_metadata(
    checkpoint,
    model_key: str,
    model_tag: str,
    model_kwargs: dict,
    exp_cfg: dict,
    label_map: dict,
    config_hash: str,
    data_manifest_hash: str,
) -> None:
    if not isinstance(checkpoint, dict):
        raise TypeError("Reproducible loading requires a dictionary checkpoint")

    checks = {
        "model_key": (checkpoint.get("model_key"), canonical_model_key(model_key)),
        "model_name": (checkpoint.get("model_name"), model_tag),
        "exp_id": (checkpoint.get("exp_id"), exp_cfg["exp_name"]),
        "label_map": (checkpoint.get("label_map"), label_map),
        "model_kwargs": (checkpoint.get("model_kwargs"), model_kwargs),
        "config_hash": (checkpoint.get("config_hash"), config_hash),
        "data_manifest_hash": (
            checkpoint.get("data_manifest_hash"),
            data_manifest_hash,
        ),
    }
    mismatches = [
        f"{name}: saved={saved!r}, current={current!r}"
        for name, (saved, current) in checks.items()
        if saved is not None and saved != current
    ]
    if mismatches:
        raise ValueError(
            "Checkpoint does not match the selected run/configuration:\n  - "
            + "\n  - ".join(mismatches)
        )


def load_checkpoint_model(
    model_key: str,
    model_tag: str,
    model_kwargs: dict,
    exp_cfg: dict,
    label_map: dict,
    device: torch.device,
    run_dir: str,
    config_hash: str,
    data_manifest_hash: str,
    ckpt_path=None,
    checkpoint_reason: str = "best_macro_f1",
    allow_partial: bool = False,
):
    path = resolve_checkpoint(run_dir, ckpt_path, checkpoint_reason)
    rebuilt_tag, model, rebuilt_kwargs = build_model_spec(
        model_key,
        len(label_map),
        exp_cfg,
        device,
    )
    if rebuilt_tag != model_tag or rebuilt_kwargs != model_kwargs:
        raise ValueError(
            "Model specification changed before checkpoint loading: "
            f"saved={model_tag}/{model_kwargs}, rebuilt={rebuilt_tag}/{rebuilt_kwargs}"
        )

    checkpoint = torch_load_checkpoint(path, device)
    validate_checkpoint_metadata(
        checkpoint,
        model_key,
        model_tag,
        model_kwargs,
        exp_cfg,
        label_map,
        config_hash,
        data_manifest_hash,
    )

    result = model.load_state_dict(
        checkpoint_state_dict(checkpoint),
        strict=not bool(allow_partial),
    )
    if allow_partial:
        if getattr(result, "missing_keys", None):
            print(f"[load_model] missing keys: {result.missing_keys}")
        if getattr(result, "unexpected_keys", None):
            print(f"[load_model] unexpected keys: {result.unexpected_keys}")

    model.to(device).eval()
    print(f"[load_model] {model_tag} <- {path} (epoch={checkpoint.get('epoch', '?')})")
    return model, path


# =============================================================================
# Training
# =============================================================================


def supports_kwarg(func, name: str) -> bool:
    try:
        return name in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


def keep_supported_kwargs(func, kwargs: dict) -> dict:
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in params}


def train_one(
    model,
    model_key: str,
    model_tag: str,
    model_kwargs: dict,
    train_csv: str,
    val_csv: str,
    exp_cfg: dict,
    device: torch.device,
    label_map: dict,
    seed: int,
    deterministic: bool,
    run_dir: str,
    resolved_config: dict,
    data_manifest: dict,
    git_info: dict,
):
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(exp_cfg["learning_rate"]),
        weight_decay=float(exp_cfg.get("weight_decay", 0.0)),
    )
    loss_fn = nn.CrossEntropyLoss()

    train_kwargs = {
        "simulation_dir": config.SIM_DATA_DIR,
        "device": device,
        "label_map": label_map,
        "model_name": model_tag,
        "model_key": canonical_model_key(model_key),
        "model_kwargs": model_kwargs,
        "exp_id": exp_cfg["exp_name"],
        "mix_mode": exp_cfg["aug_mode"],
        "label_mode": exp_cfg["target_type"],
        "epochs": exp_cfg["epochs"],
        "batch_size": exp_cfg["batch"],
        "patience": exp_cfg["patience"],
        "regenerate_every": exp_cfg["refresh_aug_every"],
        "num_workers": exp_cfg["workers"],
        "prefetch_factor": exp_cfg["prefetch"],
        "persistent_workers": exp_cfg["persistent_workers"],
        "use_weighted_sampler": exp_cfg.get("use_weighted_sampler", True),
        "normalize_audio": exp_cfg["use_norm"],
        "target_rms": exp_cfg["rms_target"],
        "max_peak": exp_cfg["peak_limit"],
        "run_seed": seed,
        "seg_dir": exp_cfg["output"]["seg_dir"],
        "run_dir": run_dir,
        "resolved_config": resolved_config,
        "config_hash": resolved_config["config_hash"],
        "data_manifest_hash": data_manifest["data_manifest_hash"],
        "generation_seeds": data_manifest.get("generation_seeds", []),
        "git_info": git_info,
        "deterministic": deterministic,
    }
    if supports_kwarg(train_model, "segment_duration"):
        train_kwargs["segment_duration"] = exp_cfg["seg_sec"]

    return train_model(
        model,
        optimizer,
        loss_fn,
        train_csv,
        val_csv,
        **keep_supported_kwargs(train_model, train_kwargs),
    )


# =============================================================================
# Evaluation views
# =============================================================================


def label_map_from_order(label_order) -> dict:
    return {label: index for index, label in enumerate(label_order)}


def species_label(label: str) -> str:
    if label == "Noise":
        return "Noise"
    base, separator, suffix = str(label).rpartition("_")
    if separator and suffix in {"F", "M", "U"}:
        return base
    return str(label)


def get_eval_definition(exp_cfg: dict, label_space: str) -> tuple[str, dict]:
    label_space = str(label_space).strip().casefold()

    if label_space == "species":
        return "species", label_map_from_order(config.SPECIES_LABEL_ORDER)

    if label_space == "species_sex":
        if exp_cfg["target_type"] != "species_sex":
            raise ValueError(
                "Species+Sex evaluation requires a Species+Sex model, "
                f"got target_type={exp_cfg['target_type']}"
            )
        return "species_sex", label_map_from_order(exp_cfg["label_order"])

    raise ValueError(f"Unsupported evaluation label space: {label_space}")


def remap_predictions(
    y_pred,
    model_label_map: dict,
    eval_label_map: dict,
    label_space: str,
    model_label_space: str,
) -> np.ndarray:
    values = np.asarray(y_pred).reshape(-1)
    label_space = str(label_space).strip().casefold()
    model_label_space = str(model_label_space).strip().casefold()

    if label_space == model_label_space and model_label_map == eval_label_map:
        return values.astype(np.int64, copy=False)

    inverse_model_map = {index: label for label, index in model_label_map.items()}
    output = np.empty(values.shape[0], dtype=np.int64)

    for index, predicted_id in enumerate(values):
        model_label = inverse_model_map[int(predicted_id)]
        if label_space == "species":
            target_label = species_label(model_label)
        elif label_space == model_label_space:
            target_label = model_label
        else:
            raise ValueError(
                f"Cannot map {model_label_space} predictions to {label_space}"
            )
        output[index] = int(eval_label_map[target_label])

    return output



def make_loader(dataset, exp_cfg: dict, shuffle: bool = False) -> DataLoader:
    kwargs = {
        "dataset": dataset,
        "batch_size": exp_cfg["batch"],
        "shuffle": shuffle,
        "num_workers": exp_cfg["workers"],
        "pin_memory": torch.cuda.is_available(),
    }
    if exp_cfg["workers"] > 0:
        kwargs["persistent_workers"] = exp_cfg.get("persistent_workers", False)
        kwargs["prefetch_factor"] = exp_cfg.get("prefetch", 2)
    return DataLoader(**kwargs)


def make_eval_dataset(
    test_csv: str,
    exp_cfg: dict,
    eval_label_map: dict,
    label_mode: str,
    tag: str,
):
    validate_metadata_labels(test_csv, label_mode=label_mode, label_map=eval_label_map)
    return AudioSequenceDataset(
        metadata_path=test_csv,
        audio_dir=config.SIM_DATA_DIR,
        label_map=eval_label_map,
        segment_duration=exp_cfg["seg_sec"],
        label_mode=label_mode,
        dataset_type="test",
        env_type=tag,
        seg_dir=exp_cfg["output"]["seg_dir"],
        prefer_simfile_path=True,
    )


def print_prediction_scores(y_true, y_pred, label_map: dict, tag: str) -> None:
    all_labels = sorted(label_map.values())
    present_labels = sorted(np.unique(y_true).astype(int).tolist())

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=present_labels,
        average="macro",
        zero_division=0,
    )
    macro_f1_all = f1_score(
        y_true,
        y_pred,
        labels=all_labels,
        average="macro",
        zero_division=0,
    )
    weighted_f1 = f1_score(
        y_true,
        y_pred,
        labels=present_labels,
        average="weighted",
        zero_division=0,
    )

    print(
        f"[{tag}] acc={accuracy:.4f}  f1_macro={macro_f1:.4f}  "
        f"f1_macro_all={macro_f1_all:.4f}  "
        f"f1_weighted={weighted_f1:.4f}"
    )



def run_eval(
    model,
    checkpoint_path: str,
    test_csv: str,
    exp_cfg: dict,
    device: torch.device,
    model_label_map: dict,
    tag: str,
    label_space: str,
    model_key: str,
    seed: int,
):
    label_mode, eval_label_map = get_eval_definition(
        exp_cfg,
        label_space,
    )
    dataset = make_eval_dataset(
        test_csv,
        exp_cfg,
        eval_label_map,
        label_mode,
        tag,
    )
    loader = make_loader(dataset, exp_cfg, shuffle=False)

    raw_pred, y_true, file_names = get_model_predictions(
        model,
        loader,
        device,
    )
    y_pred = remap_predictions(
        raw_pred,
        model_label_map,
        eval_label_map,
        label_space,
        exp_cfg["target_type"],
    )
    y_true = np.asarray(y_true).reshape(-1).astype(
        np.int64,
        copy=False,
    )
    print_prediction_scores(y_true, y_pred, eval_label_map, tag)

    canonical_key = canonical_model_key(model_key)
    model_display_name = config.MODEL_DISPLAY_NAMES.get(
        canonical_key,
        canonical_key,
    )

    checkpoint_tag = os.path.splitext(
        os.path.basename(checkpoint_path)
    )[0]
    eval_run_dir = get_eval_run_dir(exp_cfg, checkpoint_tag)
    output_dir = get_eval_tag_dir(exp_cfg, checkpoint_tag, tag)
    os.makedirs(output_dir, exist_ok=True)

    # Eval-only may reuse the same run/checkpoint directory.
    # If an earlier evaluation stopped before cleanup, stale prediction files
    # can remain and their pred_id will not match the current predictions.
    # Clear those generated evaluation artifacts before writing a fresh result.
    cleanup_evaluation_outputs(output_dir)

    results = run_evaluation_from_predictions(
        y_pred=y_pred,
        y_true=y_true,
        file_names=file_names,
        label_dict=eval_label_map,
        output_dir=output_dir,
        model_name=model_display_name,
        environment=tag,
        segment_duration=exp_cfg["seg_sec"],
        overlap=0.0,
        metadata_csv_path=test_csv,
        iou_threshold=exp_cfg.get("iou_threshold", 0.3),
        include_noise_in_sequence=exp_cfg.get(
            "include_noise_in_sequence",
            False,
        ),
        evaluation_label_space=label_space,
        save=True,
    )

    summary_df, summary_path = write_evaluation_summary(
        results=results,
        out_dir=output_dir,
        model_name=model_display_name,
        tag=tag,
        n_segments=len(y_true),
        experiment=exp_cfg["exp_name"],
        seed=int(seed),
        checkpoint=checkpoint_tag,
        eval_set=tag,
    )
    cleanup_evaluation_outputs(output_dir)
    return summary_df, summary_path, eval_run_dir


# =============================================================================
# Alpha sweep
# =============================================================================


def get_alpha_values(mix_mode: str) -> list[float]:
    if mix_mode == "convex":
        return np.round(np.arange(0.0, 1.0 + 1e-8, 0.1), 2).tolist()
    return np.round(np.arange(0.7, 1.3 + 1e-8, 0.1), 2).tolist()


def score_alpha(model, loader, device: torch.device) -> dict:
    loss_fn = nn.CrossEntropyLoss()
    loss_sum = 0.0
    n_items = 0
    predictions = []
    targets = []

    with torch.inference_mode():
        for batch in loader:
            x, y = batch[:2]
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            output = model(x)
            logits = output.reshape(-1, output.shape[-1])
            y_flat = y.reshape(-1)
            loss_sum += loss_fn(logits, y_flat).item() * y_flat.numel()
            n_items += y_flat.numel()
            predictions.append(logits.argmax(dim=-1).cpu().numpy())
            targets.append(y_flat.cpu().numpy())

    y_true = np.concatenate(targets)
    y_pred = np.concatenate(predictions)
    return {
        "loss": loss_sum / max(n_items, 1),
        "accuracy": float((y_pred == y_true).mean()),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def sweep_alpha(
    model,
    checkpoint_path: str,
    test_csv: str,
    exp_cfg: dict,
    device: torch.device,
    label_map: dict,
) -> None:
    rows = []
    mix_mode = exp_cfg["aug_mode"]
    model.eval()

    for alpha in get_alpha_values(mix_mode):
        dataset = AudioSequenceDataset(
            metadata_path=test_csv,
            audio_dir=config.SIM_DATA_DIR,
            label_map=label_map,
            segment_duration=exp_cfg["seg_sec"],
            label_mode=exp_cfg["target_type"],
            dataset_type="test",
            env_type="alpha_sweep",
            seg_dir=exp_cfg["output"]["seg_dir"],
            prefer_simfile_path=True,
            augmentor=AudioAugmentor(
                fixed_alpha=alpha,
                mix_mode=mix_mode,
                noise_level_range=(0.001, 0.01),
            ),
        )
        scores = score_alpha(model, make_loader(dataset, exp_cfg), device)
        rows.append({"alpha": alpha, **scores})
        print(
            f"  a={alpha:.2f} loss={scores['loss']:.4f} "
            f"acc={scores['accuracy']:.4f} f1_m={scores['macro_f1']:.4f}"
        )

    checkpoint_tag = os.path.splitext(os.path.basename(checkpoint_path))[0]
    output_dir = os.path.join(exp_cfg["output"]["eval_dir"], checkpoint_tag, "alpha_sweep")
    os.makedirs(output_dir, exist_ok=True)
    frame = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, f"alpha_sweep_{mix_mode}.csv")
    png_path = os.path.join(output_dir, f"alpha_sweep_{mix_mode}.png")
    frame.to_csv(csv_path, index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(frame["alpha"], frame["accuracy"], marker="o", label="Accuracy")
    plt.plot(frame["alpha"], frame["weighted_f1"], marker="s", label="Weighted F1")
    plt.title(f"Alpha sweep ({mix_mode})")
    plt.xlabel("Alpha")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()
    print(f"[sweep] {csv_path}")
    print(f"[sweep] {png_path}")


# =============================================================================
# Run pipeline
# =============================================================================


def parse_run_mode(args) -> tuple[bool, bool, bool]:
    if args.mode == "train":
        return True, False, False
    if args.mode == "eval":
        return False, True, bool(args.with_sweep)
    return True, True, bool(args.with_sweep)


def layout_from_existing_run(run_dir: str) -> dict[str, str]:
    base = os.path.abspath(run_dir)
    layout = {
        "run_id": os.path.basename(base),
        "base": base,
        "checkpoints": os.path.join(base, "checkpoints"),
        "metrics": os.path.join(base, "metrics"),
        "evaluation": os.path.join(base, "evaluation"),
        "tensorboard": os.path.join(base, "tensorboard"),
        "csv": os.path.join(base, "csv"),
        "segment_cache": os.path.join(base, "segment_cache"),
        "code_snapshot": os.path.join(base, "code_snapshot"),
    }
    for key, path in layout.items():
        if key != "run_id":
            os.makedirs(path, exist_ok=True)
    return layout


def print_run_header(exp_cfg: dict, model_key: str, seed: int, deterministic: bool, mode, run_dir: str):
    print(f"\n{'=' * 72}")
    print(f"  exp={exp_cfg['exp_name']}")
    print(f"  model={canonical_model_key(model_key)}  mix={exp_cfg['aug_mode']}")
    print(f"  seed={seed}  deterministic={deterministic}  mode={mode or 'train+eval'}")
    print(f"  run_dir={run_dir}")
    print(f"{'=' * 72}")


def run_experiment(
    model_key: str,
    base_exp_cfg: dict,
    do_train: bool,
    do_eval: bool,
    do_sweep: bool,
    seed: int,
    deterministic: bool,
    ckpt_path=None,
    mode=None,
    requested_run_dir=None,
    checkpoint_reason: str = "best_macro_f1",
    allow_partial: bool = False,
):
    canonical_key = canonical_model_key(model_key)
    device = get_device()
    set_seed(seed, deterministic)

    num_classes = len(base_exp_cfg.get("label_order", []))
    if num_classes <= 0:
        raise ValueError(f"{base_exp_cfg['exp_name']} does not define label_order")

    model_tag, model, model_kwargs = build_model_spec(
        canonical_key,
        num_classes,
        base_exp_cfg,
        device,
    )
    git_info = get_git_info(config.PROJ_DIR)
    resolved_config = build_resolved_run_config(
        exp_cfg=base_exp_cfg,
        model_key=canonical_key,
        model_tag=model_tag,
        model_kwargs=model_kwargs,
        training_seed=seed,
        repeat=deterministic,
        mode=mode,
    )

    saved_run_config = None
    saved_manifest = None

    if do_train:
        if requested_run_dir is not None:
            raise ValueError("--run_dir is only valid with --mode eval")

        layout = create_run_layout(
            exp_name=base_exp_cfg["exp_name"],
            model_key=canonical_key,
            seed=seed,
            config_hash=resolved_config["config_hash"],
            git_info=git_info,
        )
        write_initial_run_artifacts(
            layout,
            resolved_config,
            [sys.executable, *sys.argv],
            git_info,
        )

        if config.REPRO_CFG.get("snapshot_code", True):
            copied = snapshot_code(
                layout["code_snapshot"],
                project_dir=config.PROJ_DIR,
            )
            print(f"[run] code snapshot: {len(copied)} files")

    else:
        run_dir = (
            os.path.abspath(requested_run_dir)
            if requested_run_dir is not None
            else find_latest_run(
                base_exp_cfg["exp_name"],
                canonical_key,
                seed,
            )
        )
        layout = layout_from_existing_run(run_dir)

        saved_run_config = load_run_config(run_dir)
        saved_config_hash = saved_run_config.get("config_hash")
        current_config_hash = resolved_config["config_hash"]

        if not saved_config_hash:
            raise ValueError(
                f"Selected run has no saved config_hash: {run_dir}"
            )

        if saved_config_hash != current_config_hash:
            print(
                "[eval warning] current code/config differs from training run"
            )
            print(f"  saved  : {saved_config_hash}")
            print(f"  current: {current_config_hash}")
            print(
                "  checkpoint validation will use the saved training config hash"
            )

        saved_manifest = load_data_manifest(run_dir)
        saved_manifest_hash = saved_manifest.get("data_manifest_hash")

        if not saved_manifest_hash:
            raise ValueError(
                f"Selected run has no saved data_manifest_hash: {run_dir}"
            )

    exp_cfg = make_run_exp_cfg(base_exp_cfg, layout)
    for path in exp_cfg["output"].values():
        os.makedirs(path, exist_ok=True)

    print_run_header(
        exp_cfg,
        canonical_key,
        seed,
        deterministic,
        mode,
        layout["base"],
    )

    try:
        if do_train:
            update_run_status(layout["base"], "preparing_data")

        label_map = label_map_from_order(exp_cfg["label_order"])

        train_csv = None
        val_csv = None

        if do_train:
            train_csv, val_csv = build_train_val_csvs(exp_cfg)

            validate_metadata_labels(train_csv,exp_cfg["target_type"],label_map)
            validate_metadata_labels(val_csv,exp_cfg["target_type"],label_map)

            train_df = pd.read_csv(train_csv)
            train_df = set_label_mode(train_df,label_mode=exp_cfg["target_type"])

            train_labels = set(train_df["label"].fillna("Noise").astype(str))

            expected = set(exp_cfg["label_order"])
            missing = expected - train_labels


            if exp_cfg.get("allow_missing_train_labels", False):
                allowed = set(exp_cfg.get("missing_train_labels", []))

                if missing - allowed:
                    raise ValueError(f"Unexpected missing train labels: {sorted(missing - allowed)}")

                print(f"[labels] allowed missing: {sorted(missing)}")

            elif missing:
                raise ValueError(f"Training data is missing required labels: {sorted(missing)}")

        # Eval CSVs are rebuilt from the current metadata.
        # In eval-only mode, train/val CSVs from the old run are not touched.
        eval_map = build_eval_csv_map(exp_cfg)

        for tag, eval_spec in eval_map.items():
            label_mode, eval_label_map = get_eval_definition(
                exp_cfg,
                eval_spec["label_space"],
            )
            validate_metadata_labels(
                eval_spec["csv"],
                label_mode,
                eval_label_map,
            )
            print(
                f"[eval set] {tag}: "
                f"label_space={eval_spec['label_space']} "
                f"csv={eval_spec['csv']}"
            )

        if do_train:
            data_manifest = write_data_manifest(
                run_dir=layout["base"],
                source_csvs=all_source_csvs(base_exp_cfg),
                merged_csvs=[
                    train_csv,
                    val_csv,
                    *[spec["csv"] for spec in eval_map.values()],
                ],
                filename="data_manifest.json",
            )

            checkpoint_config_hash = resolved_config["config_hash"]
            checkpoint_manifest_hash = data_manifest["data_manifest_hash"]

        else:
            evaluation_manifest = write_data_manifest(
                run_dir=layout["base"],
                source_csvs=all_eval_source_csvs(base_exp_cfg),
                merged_csvs=[
                    spec["csv"]
                    for spec in eval_map.values()
                ],
                filename="evaluation_manifest_current.json",
            )

            checkpoint_config_hash = saved_run_config["config_hash"]
            checkpoint_manifest_hash = saved_manifest["data_manifest_hash"]

            print(
                "[eval] current evaluation metadata is stored separately"
            )
            print(
                "  evaluation_manifest_current.json hash="
                f"{evaluation_manifest.get('data_manifest_hash')}"
            )
            print(
                "[eval] checkpoint validation uses hashes saved at training time"
            )
            print(f"  config_hash       : {checkpoint_config_hash}")
            print(f"  data_manifest_hash: {checkpoint_manifest_hash}")

        best_checkpoint = ckpt_path

        if do_train:
            update_run_status(layout["base"], "training")
            _, best_checkpoint = train_one(
                model,
                canonical_key,
                model_tag,
                model_kwargs,
                train_csv,
                val_csv,
                exp_cfg,
                device,
                label_map,
                seed,
                deterministic,
                layout["base"],
                resolved_config,
                data_manifest,
                git_info,
            )
            update_run_status(
                layout["base"],
                "trained",
                best_checkpoint=best_checkpoint,
            )

        evaluation_model = None
        selected_checkpoint = None

        if do_eval or do_sweep:
            evaluation_model, selected_checkpoint = load_checkpoint_model(
                model_key=canonical_key,
                model_tag=model_tag,
                model_kwargs=model_kwargs,
                exp_cfg=exp_cfg,
                label_map=label_map,
                device=device,
                run_dir=layout["base"],
                config_hash=checkpoint_config_hash,
                data_manifest_hash=checkpoint_manifest_hash,
                ckpt_path=best_checkpoint or ckpt_path,
                checkpoint_reason=checkpoint_reason,
                allow_partial=allow_partial,
            )

        if do_eval:
            if do_train:
                update_run_status(layout["base"], "evaluating")

            eval_summary_dir = None

            for tag, eval_spec in eval_map.items():
                _, _, eval_summary_dir = run_eval(
                    evaluation_model,
                    selected_checkpoint,
                    eval_spec["csv"],
                    exp_cfg,
                    device,
                    label_map,
                    tag,
                    eval_spec["label_space"],
                    canonical_key,
                    seed,
                )

            if eval_summary_dir is not None:
                # combine_evaluation_summaries() also builds the pooled
                # all_domains evaluation from the individual prediction CSVs.
                combine_evaluation_summaries(eval_summary_dir)

        if do_sweep:
            if do_train:
                update_run_status(layout["base"], "alpha_sweep")

            sweep_tag = exp_cfg["sweep_tag"]
            sweep_spec = eval_map[sweep_tag]

            if sweep_spec["label_space"] != exp_cfg["target_type"]:
                raise ValueError(
                    "Alpha sweep requires the model label space, "
                    f"got {sweep_tag}={sweep_spec['label_space']}"
                )

            sweep_alpha(
                evaluation_model,
                selected_checkpoint,
                sweep_spec["csv"],
                exp_cfg,
                device,
                label_map,
            )

        # Eval-only should not change the training run status.
        if do_train:
            update_run_status(
                layout["base"],
                "completed",
                trained=True,
                evaluated=bool(do_eval),
                swept=bool(do_sweep),
            )

        print(f"[run] completed -> {layout['base']}")
        return layout["base"]

    except Exception as exc:
        # Do not mark an old successful training run as failed
        # just because a later eval-only command failed.
        if do_train:
            update_run_status(
                layout["base"],
                "failed",
                error_type=type(exc).__name__,
                error=str(exc),
                traceback=traceback.format_exc(),
            )
        raise


# =============================================================================
# CLI
# =============================================================================


def parse_seed_list(seed: int, seeds_text: str | None) -> list[int]:
    if not seeds_text:
        return [int(seed)]
    values = [int(token.strip()) for token in seeds_text.split(",") if token.strip()]
    if not values:
        raise ValueError("--seeds cannot be empty")
    return list(dict.fromkeys(values))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate the six MosBeatNet cross-domain experiments"
    )
    parser.add_argument("--exp", choices=list(EXP_LIST), default=None)
    parser.add_argument("--model", choices=MODEL_CHOICES, default=None)
    parser.add_argument("--models", nargs="+", choices=MODEL_CHOICES, default=None)
    parser.add_argument("--mode", choices=["train", "eval"], default=None)
    parser.add_argument("--with-sweep", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default=None)

    deterministic_group = parser.add_mutually_exclusive_group()
    deterministic_group.add_argument(
        "--repeat",
        "--deterministic",
        dest="repeat",
        action="store_true",
    )
    deterministic_group.add_argument(
        "--fast_nondeterministic",
        dest="repeat",
        action="store_false",
    )
    parser.set_defaults(repeat=True)

    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument(
        "--checkpoint_reason",
        choices=["best_macro_f1", "best_val_loss", "last"],
        default="best_macro_f1",
    )
    parser.add_argument("--allow_partial_checkpoint", action="store_true")
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue to the next model/seed if one run fails.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    do_train, do_eval, do_sweep = parse_run_mode(args)
    exp_keys = [args.exp] if args.exp else list(EXP_LIST)
    if args.models:
        model_keys = args.models
    elif args.model:
        model_keys = [args.model]
    else:
        model_keys = DEFAULT_MODELS
    seeds = parse_seed_list(args.seed, args.seeds)

    if args.run_dir is not None:
        if args.mode != "eval":
            raise ValueError("--run_dir requires --mode eval")
        if len(exp_keys) != 1 or len(model_keys) != 1 or len(seeds) != 1:
            raise ValueError(
                "--run_dir requires one experiment, one model and one seed"
            )

    print("\n[run plan]")
    print(f"  experiments: {exp_keys}")
    print(
        f"  models: {[canonical_model_key(key) for key in model_keys]}"
    )
    print(f"  seeds: {seeds}")
    print(f"  mode: {args.mode or 'train+eval'}")
    print(f"  deterministic: {args.repeat}")
    print(f"  with_sweep: {args.with_sweep}\n")

    failures = []

    for exp_key in exp_keys:
        exp_cfg = EXP_LIST[exp_key]
        experiment_run_dirs = []

        for model_key in model_keys:
            for seed in seeds:
                print("=" * 78)
                print(
                    f"[start] exp={exp_key} | "
                    f"model={canonical_model_key(model_key)} | "
                    f"seed={seed} | mode={args.mode or 'train+eval'}"
                )
                print("=" * 78)

                try:
                    completed_run_dir = run_experiment(
                        model_key=model_key,
                        base_exp_cfg=exp_cfg,
                        do_train=do_train,
                        do_eval=do_eval,
                        do_sweep=do_sweep,
                        seed=seed,
                        deterministic=args.repeat,
                        ckpt_path=args.ckpt,
                        mode=args.mode,
                        requested_run_dir=args.run_dir,
                        checkpoint_reason=args.checkpoint_reason,
                        allow_partial=args.allow_partial_checkpoint,
                    )
                    experiment_run_dirs.append(completed_run_dir)
                except Exception as exc:
                    if not args.continue_on_error:
                        raise

                    item = {
                        "exp": exp_key,
                        "model": canonical_model_key(model_key),
                        "seed": seed,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    failures.append(item)

                    print(
                        "[run error] "
                        f"exp={item['exp']} "
                        f"model={item['model']} "
                        f"seed={item['seed']}"
                    )
                    print(f"  {item['error']}")
                    print("[run error] continuing to next run")

        # Training reports are handled by report_utils.py.
        try:
            write_experiment_reports(exp_cfg["exp_name"])
        except Exception as exc:
            print(
                "[report warning] training reports failed: "
                f"{type(exc).__name__}: {exc}"
            )

        # Evaluation reports are handled by evaluate.py.
        if experiment_run_dirs:
            experiment_dir = str(
                Path(experiment_run_dirs[-1]).parents[2]
            )

            write_all_model_reports(
                experiment_dir=experiment_dir,
                checkpoint_name=args.checkpoint_reason,
            )
        else:
            print(
                "[all models] skipped because no run completed "
                f"for experiment {exp_key}"
            )

    if failures:
        print("\n" + "=" * 78)
        print("[batch summary] failed runs")
        print("=" * 78)
        for item in failures:
            print(
                f"  exp={item['exp']} | "
                f"model={item['model']} | "
                f"seed={item['seed']} | "
                f"{item['error']}"
            )
        print(f"[batch summary] failures={len(failures)}")


if __name__ == "__main__":
    if os.name != "nt":
        mp.set_start_method("fork", force=True)
    main()
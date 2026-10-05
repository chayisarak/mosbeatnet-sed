# data_paths.py
from __future__ import annotations

import os
from collections.abc import Sequence

import config


def raw_noise_dir(split: str) -> str:
    if split == "train":
        return os.path.join(config.NOISE_RAW_DIR, "Env_Train_Norm")
    if split in {"val", "test"}:
        return os.path.join(config.NOISE_RAW_DIR, "Env_Val_test_Norm")
    raise ValueError(f"Unknown split: {split}")


def raw_source_dir(source: str, split: str) -> str:
    if source not in config.RAW_SOURCE_DIRS:
        raise ValueError(f"Unknown source: {source}")

    root = config.RAW_SOURCE_DIRS[source]

    # BioDCASE uses the original raw source directory directly.
    # Split membership is filtered from the official ID files in generate.py.
    if source == config.BIODCASE_SOURCE:
        return os.path.join(root, "raw_audio")

    if split == "train":
        return os.path.join(root, "train")
    if split in {"val", "test"}:
        return os.path.join(root, "val_test")

    raise ValueError(f"Unknown split: {split}")


def generated_audio_dir(source: str, background: str, split: str, target_type: str | None = None) -> str:
    if split == "test":
        return os.path.join(config.GENERATED_DATASET_DIR, "test", source, background)

    if target_type not in {"species", "species_sex"}:
        raise ValueError("target_type is required for train/val")

    return os.path.join(config.GENERATED_DATASET_DIR, target_type, source, split, background)


def noise_only_audio_dir(env: str, split: str = "test") -> str:
    return os.path.join(config.GENERATED_DATASET_DIR, split, "noise_only", env)


def metadata_filename(source: str, background: str, split: str, target_type: str | None = None) -> str:
    if split == "test":
        return f"metadata_test_{source}_{background}.csv"

    if target_type not in {"species", "species_sex"}:
        raise ValueError("target_type is required for train/val")

    return f"metadata_{target_type}_{source}_{background}_{split}.csv"


def metadata_paths(
    source: str,
    backgrounds: Sequence[str],
    split: str,
    target_type: str | None = None,
) -> list[str]:
    return [
        os.path.join(
            config.GENERATED_METADATA_DIR,
            metadata_filename(source, background, split, target_type=target_type),
        )
        for background in backgrounds
    ]


def noise_only_metadata_paths() -> list[str]:
    return [
        os.path.join(
            config.GENERATED_METADATA_DIR,
            f"metadata_test_noise_only_{env}.csv",
        )
        for env in config.ENVS
    ]
from __future__ import annotations
import argparse
import hashlib
import os
import random
import re
from pathlib import Path
import numpy as np
import config
import data_paths
from simulation import get_source_coverage_requirement, list_wavs, process_noise_only, process_simulation

SOURCE_TARGETS = {
    config.INDOOR_SOURCE: ("species_sex", "species"),
    config.OUTDOOR_SOURCE: ("species_sex", "species"),
    config.HUMBUG_SOURCE: ("species",),
    config.BIODCASE_SOURCE: ("species",),
}
SOURCE_BACKGROUND = {
    config.INDOOR_SOURCE: "real",
    config.OUTDOOR_SOURCE: "gaussian",
    config.HUMBUG_SOURCE: "real",
    config.BIODCASE_SOURCE: "real",
}
TEST_TARGET = {
    config.INDOOR_SOURCE: "species_sex",
    config.OUTDOOR_SOURCE: "species_sex",
    config.HUMBUG_SOURCE: "species",
    config.BIODCASE_SOURCE: "species",
}
BIODCASE_NAME = re.compile(r"^S_(?P<species_id>\d+)_D_(?P<domain_id>\d+)_(?P<clip_index>\d+)$", re.IGNORECASE)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def job_seed(seed, *parts):
    text = "|".join([str(seed), *map(str, parts)])
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def split_items(text, valid):
    if text == "all":
        return list(valid)
    items = [x.strip() for x in text.split(",") if x.strip()]
    bad = [x for x in items if x not in valid]
    if bad:
        raise ValueError(f"Unknown values: {bad}")
    return items


def hash_value(namespace, path, seed):
    text = f"{namespace}|{seed}|{path}".encode()
    return int(hashlib.sha256(text).hexdigest()[:16], 16)


def split_val_test(files, split, namespace):
    if split not in {"val", "test"}:
        return sorted(files)
    
    files = sorted(set(map(os.path.abspath, files)))
    if len(files) < 2:
        raise ValueError(f"Need at least 2 files for {namespace}")
    root = os.path.commonpath(files)
    files = sorted(files, key=lambda p: hash_value(namespace, os.path.relpath(p, root), config.RAW_SPLIT_SEED))
    n_val = int(round(len(files) * config.VAL_TEST_VAL_RATIO))
    n_val = min(max(n_val, 1), len(files) - 1)
    return files[:n_val] if split == "val" else files[n_val:]


def clean_id(value):
    return Path(str(value).strip().strip('"').strip("'")).stem


def find_biodcase_file(filename):
    root = Path(config.BIODCASE_RAW_DIR)
    search_dirs = [root, root / "metadata", root / "splits"]

    for folder in search_dirs:
        path = folder / filename
        if path.is_file():
            return path
        
    wanted = filename.casefold()
    for folder in search_dirs:
        if not folder.is_dir():
            continue
        for path in folder.rglob("*"):
            if path.is_file() and path.name.casefold() == wanted:
                return path
            
    raise FileNotFoundError(f"Could not find {filename} under {root}")


def read_biodcase_ids(split):
    filename = config.BIODCASE_SPLIT_FILES[split]
    path = find_biodcase_file(filename)
    ids = set()

    with path.open("r", encoding="utf-8-sig") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            first = re.split(r"[\t,;]", line, maxsplit=1)[0].strip()
            file_id = clean_id(first)
            if file_id.casefold() in {"id", "file_id", "filename", "file_name"}:
                continue
            if file_id:
                ids.add(file_id.casefold())

    if not ids:
        raise ValueError(f"No IDs found in {path}")
    
    return ids, path


def get_biodcase_files(split):
    root = data_paths.raw_source_dir(config.BIODCASE_SOURCE, split)
    all_files = list_wavs(root, recursive=True)
    split_ids, id_path = read_biodcase_ids(split)

    print(f"[BioDCASE] audio root: {root}")
    print(f"[BioDCASE] WAV files found: {len(all_files)}")
    print(f"[BioDCASE] {split} IDs found: {len(split_ids)} from {id_path}")

    if not all_files:
        raise ValueError(f"No WAV files found under BioDCASE audio root: {root}")
    
    files = []
    valid_name_count = 0
    split_match_count = 0
    species_count = {species: 0 for species in config.SPECIES_ORDER}
    for path in all_files:
        stem = Path(path).stem
        match = BIODCASE_NAME.fullmatch(stem)
        if match is None:
            continue
        valid_name_count += 1
        if stem.casefold() not in split_ids:
            continue
        split_match_count += 1
        species_id = int(match.group("species_id"))
        species = config.BIODCASE_SPECIES_IDS.get(species_id)
        if species is None:
            continue
        files.append(os.path.abspath(path))
        species_count[species] += 1

    files = sorted(set(files))
    if not files:
        sample_wavs = [Path(path).stem for path in all_files[:5]]
        sample_ids = sorted(split_ids)[:5]
        raise ValueError(
            f"No target BioDCASE WAV files for split={split}. "
            f"valid_filename={valid_name_count}, split_matches={split_match_count}. "
            f"sample_wav={sample_wavs}, sample_id={sample_ids}"
        )
    print(f"[BioDCASE] filename format matched: {valid_name_count}")
    print(f"[BioDCASE] split ID matched: {split_match_count}")
    print(f"[BioDCASE] target files kept: {len(files)}")
    print("[BioDCASE] " + ", ".join(f"{species}={count}" for species, count in species_count.items()))
    return files


def check_biodcase_lists():
    train_ids, _ = read_biodcase_ids("train")
    val_ids, _ = read_biodcase_ids("val")
    test_ids, _ = read_biodcase_ids("test")
    if train_ids & val_ids:
        raise ValueError("BioDCASE Training_ids and Validation_ids overlap")
    if train_ids & test_ids:
        raise ValueError("BioDCASE Training_ids and Test_ids overlap")
    if val_ids & test_ids:
        raise ValueError("BioDCASE Validation_ids and Test_ids overlap")
    try:
        trainval_path = find_biodcase_file(config.BIODCASE_TRAINVAL_FILE)
    except FileNotFoundError:
        return
    trainval_ids = set()
    with trainval_path.open("r", encoding="utf-8-sig") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            first = re.split(r"[\t,;]", line, maxsplit=1)[0].strip()
            file_id = clean_id(first)
            if file_id.casefold() not in {"id", "file_id", "filename", "file_name"} and file_id:
                trainval_ids.add(file_id.casefold())
    if train_ids | val_ids != trainval_ids:
        raise ValueError("BioDCASE Training_ids + Validation_ids do not match TrainVal_ids")
    if trainval_ids & test_ids:
        raise ValueError("BioDCASE TrainVal_ids and Test_ids overlap")


def get_source_files(source, split):
    if source == config.BIODCASE_SOURCE:
        return get_biodcase_files(split)
    root = data_paths.raw_source_dir(source, split)
    files = list_wavs(root, recursive=True)
    if split in {"val", "test"}:
        files = split_val_test(files, split, f"mosquito:{source}")
    if not files:
        raise ValueError(f"No WAV files: source={source}, split={split}")
    return files


def get_noise_files(split):
    root = data_paths.raw_noise_dir(split)
    result = {}
    for env, folder in config.NOISE_ENV_FOLDER.items():
        files = list_wavs(os.path.join(root, folder), recursive=True)
        if split in {"val", "test"}:
            files = split_val_test(files, split, f"noise:{env}")
        if not files:
            raise ValueError(f"No noise files: split={split}, env={env}")
        result[env] = {"file_paths": files, "file_names": [os.path.basename(p) for p in files]}
    return result


def get_num_simulations(source, split):
    if source == config.BIODCASE_SOURCE:
        return int(config.BIODCASE_N_SIMS[split])
    return int(config.N_SIMS_PER_SOURCE[split])


def get_job_num(source, split, target_type, env, source_files, requested_num, background_type="real"):
    info = get_source_coverage_requirement(
        mosquito_dirs=source_files, env_target=env, dataset_type=split,
        target_type=target_type, background_type=background_type,
    )
    requested_num = int(requested_num)
    minimum_num = int(info["minimum_simulations"])
    final_num = max(requested_num, minimum_num)
    env_label = "gaussian" if background_type == "gaussian" else env
    if final_num > requested_num:
        print(f"[coverage] {source}/{split}/{env_label}: n {requested_num} -> {final_num}")
    class_text = ", ".join(f"{label}={count}" for label, count in info["class_sizes"].items())
    print(
        f"[source pool] {source}/{split}/{env_label} | mode={info['sampling_mode']} | "
        f"raw={info['raw_file_count']} | classes: {class_text}"
    )
    return final_num


def generate_one(source, split, target_type, num, seed, dry_run=False):
    background_type = SOURCE_BACKGROUND[source]
    envs = (None,) if background_type == "gaussian" else tuple(config.ENVS)
    source_files = get_source_files(source, split)
    if background_type == "real":
        noise_dir = data_paths.raw_noise_dir(split)
        noise_list = get_noise_files(split)
    else:
        noise_dir = ""
        noise_list = None
    for env in envs:
        background = "gaussian" if background_type == "gaussian" else env
        job_num = get_job_num(
            source, split, target_type, env, source_files, num, background_type=background_type
        )
        output_dir = data_paths.generated_audio_dir(source, background, split, target_type=target_type)
        metadata_name = data_paths.metadata_filename(source, background, split, target_type=target_type)
        print(f"{split:5s} | {target_type:11s} | {source:13s} | {background:8s} | n={job_num}")
        if dry_run:
            continue
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(config.GENERATED_METADATA_DIR, exist_ok=True)
        process_simulation(
            mosquito_dirs=source_files, noise_dir=noise_dir, output_dir=output_dir,
            num_simulations=job_num, env_target=env, dataset_type=split, source_name=source,
            metadata_dir=config.GENERATED_METADATA_DIR, metadata_name=metadata_name,
            target_type=target_type, noise_list=noise_list, background_type=background_type,
            seed=job_seed(seed, target_type, source, split, background),
        )


def generate_noise_only(num, seed, dry_run=False):
    split = "test"
    noise_dir = data_paths.raw_noise_dir(split)
    noise_list = get_noise_files(split)
    for env in config.ENVS:
        output_dir = data_paths.noise_only_audio_dir(env)
        metadata_name = f"metadata_test_noise_only_{env}.csv"
        print(f"test  | noise_only  | {env:13s} | n={num}")
        if dry_run:
            continue
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(config.GENERATED_METADATA_DIR, exist_ok=True)
        process_noise_only(
            noise_dir=noise_dir,
            output_dir=output_dir,
            num_simulations=num,
            env_target=env,
            dataset_type="test",
            metadata_dir=config.GENERATED_METADATA_DIR,
            metadata_name=metadata_name,
            noise_list=noise_list,
            seed=job_seed(seed, "noise_only", env),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="all", help="all, train, val, test")
    parser.add_argument("--target", default="all", help="all, species_sex, species")
    parser.add_argument("--source", default="all", help="all or source name")
    parser.add_argument("--num", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--noise_num", type=int, default=None)
    args = parser.parse_args()
    set_seed(args.seed)
    splits = split_items(args.split, config.SPLITS)
    sources = split_items(args.source, config.MOSQUITO_SOURCES)
    targets = split_items(args.target, ("species_sex", "species"))
    if config.BIODCASE_SOURCE in sources:
        check_biodcase_lists()
    for split in splits:
        for source in sources:
            n = get_num_simulations(source, split) if args.num is None else int(args.num)
            if split == "test":
                generate_one(source, split, TEST_TARGET[source], n, args.seed, args.dry_run)
                continue
            for target in targets:
                if target in SOURCE_TARGETS[source]:
                    generate_one(source, split, target, n, args.seed, args.dry_run)
        if split == "test" and args.source == "all":
            noise_num = int(args.noise_num) if args.noise_num is not None else int(config.NOISE_ONLY_TEST)
            generate_noise_only(noise_num, args.seed, args.dry_run)


if __name__ == "__main__":
    main()

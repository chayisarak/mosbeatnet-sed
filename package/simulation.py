#simulation.py
from __future__ import annotations

import hashlib
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

import config


WAV_EXT = ".wav"
EPS = 1e-12
_SOURCE_DECODE_CACHE: dict[tuple[str, int], bool] = {}


# =============================================================================
# Basic audio and path helpers
# =============================================================================

def rms(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    if not np.isfinite(x).all():
        raise ValueError("audio contains NaN/Inf")
    return float(np.sqrt(np.mean(np.square(x))))


def normalize(audio, target_level=-20.0):
    """Compatibility helper; simulation uses explicit background RMS and SNR."""
    audio = np.asarray(audio, dtype=np.float32)
    current = rms(audio)
    if current > EPS:
        target_rms = 10.0 ** (float(target_level) / 20.0)
        audio = audio * (target_rms / current)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def set_random_seed(seed=None) -> None:
    if seed is None:
        return
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))


def stable_seed(*parts) -> int:
    text = "|".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(text).hexdigest()[:8], 16)


def load_audio(path: str, sr: int) -> np.ndarray:
    try:
        audio, _ = librosa.load(path, sr=int(sr), mono=True)
    except Exception as exc:
        raise RuntimeError(f"cannot load audio: {path} | {exc}") from exc

    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        raise ValueError(f"empty audio file: {path}")
    if not np.isfinite(audio).all():
        raise ValueError(f"audio has NaN/Inf: {path}")
    return audio


def list_wavs(folder: str, recursive: bool = True) -> list[str]:
    folder = os.path.abspath(os.fspath(folder))
    if not os.path.isdir(folder):
        return []

    paths: list[str] = []
    if recursive:
        for root, _, names in os.walk(folder):
            for name in names:
                if name.lower().endswith(WAV_EXT):
                    paths.append(os.path.join(root, name))
    else:
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if os.path.isfile(path) and name.lower().endswith(WAV_EXT):
                paths.append(path)
    return sorted(set(os.path.abspath(path) for path in paths))


def resolve_wav_inputs(inputs) -> list[str]:
    if isinstance(inputs, (str, bytes, os.PathLike)):
        inputs = [inputs]

    paths: list[str] = []
    for item in inputs:
        path = os.path.abspath(os.fspath(item))
        if os.path.isdir(path):
            paths.extend(list_wavs(path, recursive=True))
        elif os.path.isfile(path) and path.lower().endswith(WAV_EXT):
            paths.append(path)
    return sorted(set(paths))


def fit_peak(audio, peak_limit=None, return_info=False):
    if peak_limit is None:
        peak_limit = float(config.SIM_MIX_CFG.get("peak_limit", 0.98))

    audio = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    gain = min(1.0, float(peak_limit) / peak) if peak > 0 else 1.0
    result = (audio * gain).astype(np.float32)
    return (result, float(gain)) if return_info else result


# =============================================================================
# Species, sex and source metadata parsing
# =============================================================================

def _normalise_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def _species_lookup(allowed_species: Iterable[str]) -> dict[str, str]:
    allowed = [str(value).strip() for value in allowed_species if str(value).strip()]
    lookup = {_normalise_token(species): species for species in allowed}
    for alias, canonical in getattr(config, "SPECIES_ALIASES", {}).items():
        if canonical in allowed:
            lookup[_normalise_token(alias)] = canonical
    return lookup


def infer_species_from_path(path: str, allowed_species: Iterable[str]) -> str | None:
    lookup = _species_lookup(allowed_species)
    stem = Path(path).stem

    # Original BioDCASE filenames are flat, e.g. S_1_D_5_000001.wav.
    match = re.match(r"^S_(\d+)_D_(\d+)_", stem, re.IGNORECASE)
    if match:
        species = getattr(config, "BIODCASE_SPECIES_IDS", {}).get(int(match.group(1)))
        if species in allowed_species:
            return species

    candidates = list(Path(path).parts) + re.split(r"[_\-\s]+", stem)
    for token in reversed(candidates):
        canonical = lookup.get(_normalise_token(token))
        if canonical is not None:
            return canonical

    normalized_stem = _normalise_token(stem)
    matches = [(len(alias), canonical) for alias, canonical in lookup.items() if alias and alias in normalized_stem]
    return max(matches)[1] if matches else None


def infer_sex_from_filename(path: str) -> str:
    stem = Path(path).stem
    tokens = [token for token in re.split(r"[_\-\s]+", stem) if token]

    for token in tokens[1:5]:
        match = re.fullmatch(r"\d*([fFmM])", token)
        if match:
            return match.group(1).upper()

    for token in tokens:
        low = token.casefold()
        if low in {"f", "female"}:
            return "F"
        if low in {"m", "male"}:
            return "M"
    return "U"


def infer_source_environment(path: str) -> str:
    for token in Path(path).parts:
        low = token.casefold()
        if low in {"urban", "forest"}:
            return low
    return "unknown"


def infer_source_device(path: str) -> str:
    stem = Path(path).stem
    match = re.match(r"^S_\d+_D_(\d+)_", stem, re.IGNORECASE)
    if match:
        return f"D{match.group(1)}"

    for token in Path(path).parts:
        if re.fullmatch(r"[dD]\d+", token):
            return token.upper()
    return "unknown"


def raw_recording_id(path: str) -> str:
    normalized = os.path.normcase(os.path.abspath(path)).encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()[:16]


def parse_mos_name( path: str, allowed_species: Iterable[str] | None = None, target_type: str = "species_sex", ):
    allowed_species = list(allowed_species or config.SPECIES_ORDER)
    species = infer_species_from_path(path, allowed_species)
    if species is None:
        return None

    sex = infer_sex_from_filename(path)
    target_type = str(target_type).strip().lower()
    if target_type == "species_sex":
        if sex not in {"F", "M"}:
            return None
        balance_label = f"{species}_{sex}"
        label = balance_label
    elif target_type == "species":
        balance_label = species
        label = species
    else:
        raise ValueError("target_type must be 'species' or 'species_sex'")

    return {
        "path": os.path.abspath(path),
        "species": species,
        "sex": sex,
        "label": label,
        "balance_label": balance_label,
        "source_environment": infer_source_environment(path),
        "source_device": infer_source_device(path),
        "raw_recording_id": raw_recording_id(path),
    }


def build_mos_list( mosquito_inputs, allowed_species, target_type: str = "species_sex", ):
    """Build the valid source pool used by coverage-aware scheduling."""
    canonical_species = [str(value).strip() for value in allowed_species if str(value).strip()]
    if not canonical_species:
        raise ValueError("allowed_species is empty")

    min_source_sec = float(config.SIM_MIX_CFG.get("min_source_sec", 0.05))
    validate_decode = bool(config.SIM_MIX_CFG.get("validate_source_decode", False))
    source_sr = int(config.SAMPLING_RATE)

    pools: defaultdict[str, list[dict]] = defaultdict(list)
    skipped = Counter()

    for path in resolve_wav_inputs(mosquito_inputs):
        info = parse_mos_name(path, canonical_species, target_type=target_type)
        if info is None:
            if target_type == "species_sex" and infer_species_from_path(path, canonical_species):
                skipped["unknown_sex"] += 1
            else:
                skipped["unparsed"] += 1
            continue

        try:
            duration = float(sf.info(path).duration)
            if duration + 1e-9 < min_source_sec:
                skipped["too_short"] += 1
                continue
            if validate_decode:
                cache_key = (os.path.abspath(path), source_sr)
                is_valid = _SOURCE_DECODE_CACHE.get(cache_key)
                if is_valid is None:
                    try:
                        probe = load_audio(path, source_sr)
                        is_valid = rms(probe) >= EPS
                    except Exception:
                        is_valid = False
                    _SOURCE_DECODE_CACHE[cache_key] = bool(is_valid)
                if not is_valid:
                    skipped["decode_or_silent"] += 1
                    continue
        except Exception:
            skipped["unreadable"] += 1
            continue

        info["source_duration"] = duration
        pools[info["balance_label"]].append(info)

    pools = { label: sorted(rows, key=lambda item: item["path"]) for label, rows in sorted(pools.items()) if rows }
    if not pools:
        raise ValueError(
            "No usable mosquito WAV files. "
            f"target_type={target_type}, allowed_species={canonical_species}"
        )

    expected = (
        {f"{species}_{sex}" for species in canonical_species for sex in ("F", "M")}
        if target_type == "species_sex"
        else set(canonical_species)
    )
    missing_classes = sorted(expected - set(pools))
    if missing_classes:
        print(f"[warning] missing valid source classes: {missing_classes}")

    print(f"[mosquito pool] target_type={target_type}")
    for label, files in pools.items():
        print(f"  {label}: {len(files)} valid files")
    print("  skipped: " + ", ".join(f"{key}={value}" for key, value in sorted(skipped.items())))
    return pools


# =============================================================================
# Source and noise file selectors
# =============================================================================

class select_mos_source():
    """Cover raw sources first, then reuse them according to the sampling mode."""

    def __init__(self, pools: Mapping[str, Sequence[dict]], seed: int, sampling_mode: str = "balanced"):
        self.pools = {str(label): [dict(item) for item in rows] for label, rows in sorted(pools.items()) if rows}
        if not self.pools:
            raise ValueError("source scheduler received an empty pool")

        sampling_mode = str(sampling_mode).strip().lower()
        if sampling_mode not in {"balanced", "raw"}:
            raise ValueError("sampling_mode must be 'balanced' or 'raw'")

        self.sampling_mode = sampling_mode
        self.rng = random.Random(int(seed))
        self.class_use = Counter({label: 0 for label in self.pools})
        self.file_use = Counter({item["path"]: 0 for rows in self.pools.values() for item in rows})
        self.used_file_count = 0
        self._history = []

        self._coverage_by_class = {}
        self._coverage_pos = {}
        self._reuse_by_class = {}
        self._reuse_pos = {}

        for label, rows in self.pools.items():
            coverage_rows = list(rows)
            reuse_rows = list(rows)
            self.rng.shuffle(coverage_rows)
            self.rng.shuffle(reuse_rows)
            self._coverage_by_class[label] = coverage_rows
            self._coverage_pos[label] = 0
            self._reuse_by_class[label] = reuse_rows
            self._reuse_pos[label] = 0

        self._coverage_all = [item for rows in self.pools.values() for item in rows]
        self._reuse_all = list(self._coverage_all)
        self.rng.shuffle(self._coverage_all)
        self.rng.shuffle(self._reuse_all)
        self._coverage_all_pos = 0
        self._reuse_all_pos = 0

    def snapshot(self):
        return {
            "history_len": len(self._history),
            "class_use": dict(self.class_use),
            "used_file_count": int(self.used_file_count),
            "coverage_pos": dict(self._coverage_pos),
            "coverage_all_pos": int(self._coverage_all_pos),
            "reuse_pos": dict(self._reuse_pos),
            "reuse_all_pos": int(self._reuse_all_pos),
            "rng_state": self.rng.getstate(),
        }

    def restore(self, state) -> None:
        while len(self._history) > int(state["history_len"]):
            path, _ = self._history.pop()
            self.file_use[path] -= 1

        self.class_use = Counter(state["class_use"])
        self.used_file_count = int(state["used_file_count"])
        self._coverage_pos = dict(state["coverage_pos"])
        self._coverage_all_pos = int(state["coverage_all_pos"])
        self._reuse_pos = dict(state["reuse_pos"])
        self._reuse_all_pos = int(state["reuse_all_pos"])
        self.rng.setstate(state["rng_state"])

    def _coverage_done(self) -> bool:
        return self.used_file_count >= self.total_files

    def _pick_unseen_balanced(self) -> dict:
        labels = [
            label for label, rows in self._coverage_by_class.items()
            if self._coverage_pos[label] < len(rows)
        ]
        if not labels:
            raise RuntimeError("coverage queue is empty before coverage completed")

        minimum_class_use = min(self.class_use[label] for label in labels)
        choices = [label for label in labels if self.class_use[label] == minimum_class_use]
        label = self.rng.choice(choices)

        pos = self._coverage_pos[label]
        self._coverage_pos[label] += 1
        return dict(self._coverage_by_class[label][pos])

    def _pick_unseen_raw(self) -> dict:
        if self._coverage_all_pos >= len(self._coverage_all):
            raise RuntimeError("raw coverage queue is empty before coverage completed")

        item = self._coverage_all[self._coverage_all_pos]
        self._coverage_all_pos += 1
        return dict(item)

    def _pick_balanced_reuse(self) -> dict:
        minimum_class_use = min(self.class_use.values())
        labels = [label for label in self.pools if self.class_use[label] == minimum_class_use]
        label = self.rng.choice(labels)

        rows = self._reuse_by_class[label]
        pos = self._reuse_pos[label] % len(rows)
        self._reuse_pos[label] += 1
        return dict(rows[pos])

    def _pick_raw_reuse(self) -> dict:
        pos = self._reuse_all_pos % len(self._reuse_all)
        self._reuse_all_pos += 1
        return dict(self._reuse_all[pos])

    def pick(self) -> dict:
        if not self._coverage_done():
            chosen = self._pick_unseen_balanced() if self.sampling_mode == "balanced" else self._pick_unseen_raw()
        else:
            chosen = self._pick_balanced_reuse() if self.sampling_mode == "balanced" else self._pick_raw_reuse()

        path = chosen["path"]
        label = chosen["balance_label"]
        next_file_use = int(self.file_use[path] + 1)
        next_class_use = int(self.class_use[label] + 1)

        chosen["source_use_index"] = next_file_use
        chosen["class_use_index"] = next_class_use
        chosen["source_use_cycle"] = next_file_use

        if self.file_use[path] == 0:
            self.used_file_count += 1

        self.file_use[path] += 1
        self.class_use[label] += 1
        self._history.append((path, label))
        return chosen

    @property
    def total_files(self) -> int:
        return len(self.file_use)

    @property
    def missing_count(self) -> int:
        return self.total_files - self.used_file_count

    @property
    def max_class_size(self) -> int:
        return max(len(rows) for rows in self.pools.values())

    @property
    def n_classes(self) -> int:
        return len(self.pools)

    @property
    def coverage_slots_needed(self) -> int:
        return self.total_files

    def missing_paths(self) -> list[str]:
        return sorted(path for path, count in self.file_use.items() if count == 0)

    def usage_frame(self) -> pd.DataFrame:
        rows = []
        for label, items in self.pools.items():
            for item in items:
                rows.append({
                    "sampling_mode": self.sampling_mode,
                    "balance_label": label,
                    "species": item["species"],
                    "sex": item["sex"],
                    "raw_recording_id": item["raw_recording_id"],
                    "mos_path": item["path"],
                    "source_environment": item["source_environment"],
                    "source_device": item["source_device"],
                    "use_count": int(self.file_use[item["path"]]),
                })
        return pd.DataFrame(rows).sort_values(
            ["balance_label", "use_count", "mos_path"],
            ascending=[True, True, True],
        )


class select_noise_file():
    """Cycle through all valid noise files in each environment before reuse."""

    def __init__(self, noise_list: Mapping, seed: int):
        self.files = { str(env): sorted(set(data.get("file_paths", []))) for env, data in noise_list.items() }
        self.rng = random.Random(int(seed))
        self.use = Counter(path for paths in self.files.values() for path in [])
        for paths in self.files.values():
            for path in paths:
                self.use[path] = 0

    def snapshot(self):
        return dict(self.use), self.rng.getstate()

    def restore(self, state) -> None:
        use, rng_state = state
        self.use = Counter(use)
        self.rng.setstate(rng_state)

    def pick(self, env: str) -> tuple[str, int]:
        paths = self.files.get(env, [])
        if not paths:
            raise ValueError(f"no valid background files for {env}")
        minimum = min(self.use[path] for path in paths)
        choices = [path for path in paths if self.use[path] == minimum]
        path = self.rng.choice(choices)
        use_index = int(self.use[path] + 1)
        self.use[path] += 1
        return path, use_index

    def missing_paths(self, env: str) -> list[str]:
        return sorted(path for path in self.files.get(env, []) if self.use[path] == 0)

    def usage_frame(self, env: str) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "simulation_environment": env,
                "noise_file": path,
                "use_count": int(self.use[path]),
            }
            for path in self.files.get(env, [])
        ]).sort_values(["use_count", "noise_file"])


# =============================================================================
# Noise loading and new mixing logic
# =============================================================================

def get_noise_files_by_env(noise_base_dir: str):
    folder_map = getattr( config, "NOISE_ENV_FOLDER", {"urban": "Env_Urban", "forest": "Env_Forest"}, )
    if not os.path.isdir(noise_base_dir):
        raise FileNotFoundError(f"noise folder not found: {noise_base_dir}")

    minimum_duration = float(config.AUDIO_DURATION)
    result = { str(env).lower(): {"file_paths": [], "file_names": []} for env in folder_map }

    for raw_env, folder_name in folder_map.items():
        env = str(raw_env).lower()
        result[env]["file_paths"].extend( list_wavs(os.path.join(noise_base_dir, folder_name), recursive=True) )

    for path in list_wavs(noise_base_dir, recursive=False):
        name = os.path.basename(path).casefold()
        for env in result:
            if env in name:
                result[env]["file_paths"].append(path)
                break

    for env, data in result.items():
        valid = []
        for path in sorted(set(data["file_paths"])):
            try:
                if float(sf.info(path).duration) + 1e-9 >= minimum_duration:
                    valid.append(os.path.abspath(path))
            except Exception:
                continue
        data["file_paths"] = valid
        data["file_names"] = [os.path.basename(path) for path in valid]
        print(f"[noise pool] {env}: {len(valid)} valid files")

    if not any(data["file_paths"] for data in result.values()):
        raise ValueError( f"No background WAV files at least {minimum_duration:.3f}s long " f"in {noise_base_dir}" )
    return result


def validate_noise_list(noise_list: Mapping, required_envs: Iterable[str] = config.ENVS):
    for env in required_envs:
        files = noise_list.get(env, {}).get("file_paths", [])
        if not files:
            raise ValueError(f"noise list has no files for environment={env}")

def add_gaussian_noise(duration_sec: float, sr: int):
    """Generate Gaussian background independent of environment."""
    total_samples = int(round(float(duration_sec) * int(sr)))
    target_dbfs = float(config.GAUSSIAN_BACKGROUND["noise_target_rms_dbfs"])
    target_rms = 10.0 ** (target_dbfs / 20.0)

    background = np.random.normal(0, 1, total_samples).astype(np.float32)
    current_rms = rms(background)
    if current_rms < EPS:
        raise ValueError("generated silent Gaussian noise")

    background *= target_rms / current_rms
    return background.astype(np.float32)

def add_background_noise(
    noise_base_dir,
    env_name,
    dataset_type,
    source_name=None,
    use_hb_env=False,
    noise_list=None,
    noise_scheduler: select_noise_file | None = None,
):
    """Add the same controlled real Urban/Forest background logic to all sources."""
    del dataset_type, source_name
    env_map = config.HB_ENVIRONMENTS if use_hb_env else config.ENVIRONMENTS
    if env_name not in env_map:
        raise ValueError(f"unknown environment: {env_name}")

    duration = float(config.AUDIO_DURATION)
    sr = int(config.SAMPLING_RATE)
    total_samples = int(round(duration * sr))

    noise_list = noise_list if noise_list is not None else get_noise_files_by_env(noise_base_dir)
    if noise_scheduler is None:
        noise_scheduler = select_noise_file(noise_list, stable_seed(env_name, "noise"))

    noise_file, noise_use_index = noise_scheduler.pick(env_name)
    source = load_audio(noise_file, sr)
    if len(source) < total_samples:
        raise ValueError(f"background shorter than output duration: {noise_file}")

    start_sample = random.randint(0, len(source) - total_samples)
    background = source[start_sample:start_sample + total_samples].copy()
    background = background - np.mean(background, dtype=np.float64)

    current_rms = rms(background)
    if current_rms < EPS:
        raise ValueError(f"silent background: {noise_file}")

    target_dbfs = float(env_map[env_name]["noise_target_rms_dbfs"])
    target_rms = 10.0 ** (target_dbfs / 20.0)
    background = background * (target_rms / current_rms)

    stem = os.path.splitext(os.path.basename(noise_file))[0]
    subenv = stem.split("_")[-1] if "_" in stem else stem
    return (
        np.asarray(background, dtype=np.float32),
        noise_file,
        subenv,
        round(start_sample / sr, 6),
        noise_use_index,
    )


# =============================================================================
# Event positions, source cutting and SNR scaling
# =============================================================================

def choose_mos_event_count(source_scheduler: select_mos_source, remaining_clips: int) -> int:
    mix_cfg = config.SIM_MIX_CFG
    min_events = max(1, int(mix_cfg["min_mos_events"]))
    max_events = max(min_events, int(mix_cfg["max_mos_events"]))
    remaining_clips = max(1, int(remaining_clips))

    if get_source_mode() == "off" or source_scheduler.missing_count <= 0:
        return random.randint(min_events, max_events)

    needed = int(np.ceil(source_scheduler.missing_count / remaining_clips))
    if needed > max_events and get_source_mode() == "strict":
        raise ValueError(
            f"not enough event capacity to cover remaining sources: "
            f"missing={source_scheduler.missing_count}, remaining_clips={remaining_clips}, "
            f"max_events={max_events}"
        )

    lower = max(min_events, min(needed, max_events))
    return random.randint(lower, max_events)


def gen_mos_pos(duration_sec, n_events=None):
    mix_cfg = config.SIM_MIX_CFG
    duration = float(duration_sec)

    if n_events is None:
        n_events = random.randint(int(mix_cfg["min_mos_events"]), int(mix_cfg["max_mos_events"]))
    n_events = int(n_events)

    min_events = int(mix_cfg["min_mos_events"])
    max_events = int(mix_cfg["max_mos_events"])
    if not min_events <= n_events <= max_events:
        raise ValueError(f"n_events must be between {min_events} and {max_events}, got {n_events}")

    lengths = np.random.uniform(
        float(mix_cfg["min_mos_event_sec"]),
        float(mix_cfg["max_mos_event_sec"]),
        n_events,
    )
    margin = float(mix_cfg["event_margin_sec"])

    required = float(np.sum(lengths)) + margin * (n_events - 1)
    if required > duration:
        raise ValueError(
            "event configuration cannot fit in one clip: "
            f"required={required:.3f}s, duration={duration:.3f}s"
        )

    gaps = np.random.dirichlet(np.ones(n_events + 1)) * (duration - required)
    positions = []
    cursor = float(gaps[0])

    for index, length in enumerate(lengths):
        start = cursor
        end = start + float(length)
        positions.append((start, end))
        cursor = end + float(gaps[index + 1])
        if index < n_events - 1:
            cursor += margin

    return positions


def apply_smooth_volume(audio, sr, fade_in_time=None, fade_out_time=None):
    default_fade = float(config.SIM_MIX_CFG.get("fade_sec", 0.020))
    fade_in_time = default_fade if fade_in_time is None else float(fade_in_time)
    fade_out_time = default_fade if fade_out_time is None else float(fade_out_time)

    audio = np.asarray(audio, dtype=np.float32).copy()
    fade_in = min(int(round(fade_in_time * int(sr))), len(audio) // 2)
    fade_out = min(int(round(fade_out_time * int(sr))), len(audio) // 2)

    if fade_in > 1:
        phase = np.linspace(0.0, np.pi / 2.0, fade_in)
        audio[:fade_in] *= np.sin(phase) ** 2
    if fade_out > 1:
        phase = np.linspace(0.0, np.pi / 2.0, fade_out)
        audio[-fade_out:] *= np.cos(phase) ** 2
    return audio.astype(np.float32)


def cut_mosquito_source(path: str, requested_samples: int, sr: int):
    """Seeded random crop for long files; loop short files to exact length."""
    source = load_audio(path, sr)
    if rms(source) < EPS:
        raise ValueError(f"silent mosquito recording: {path}")

    random_crop = bool(config.SIM_MIX_CFG.get("random_source_crop", True))
    if len(source) >= requested_samples:
        maximum_start = len(source) - requested_samples
        start_sample = random.randint(0, maximum_start) if random_crop else maximum_start // 2
        event = source[start_sample:start_sample + requested_samples]
        source_looped = False
        source_end_sample = start_sample + requested_samples
    else:
        repeats = int(np.ceil(requested_samples / len(source)))
        event = np.tile(source, repeats)[:requested_samples]
        start_sample = 0
        source_end_sample = len(source)
        source_looped = True

    event = event - np.mean(event, dtype=np.float64)
    event = apply_smooth_volume(event, sr)
    if len(event) != requested_samples or rms(event) < EPS:
        raise ValueError(f"invalid cut from mosquito recording: {path}")

    start = float(start_sample) / sr
    end = float(source_end_sample) / sr


    return (event.astype(np.float32), start, end, source_looped)


def prepare_mos(
    mos_env,
    mos_list,
    background_noise,
    mos_positions,
    source_name="unknown",
    noise_env=None,
    background_type="real",
    use_hb_env=False,
    source_scheduler: select_mos_source | None = None,
):
    background_type = str(background_type).strip().lower()
    if background_type not in {"real", "gaussian"}:
        raise ValueError(f"unknown background_type: {background_type}")

    env_map = config.HB_ENVIRONMENTS if use_hb_env else config.ENVIRONMENTS
    if background_type == "real":
        noise_env = mos_env if noise_env is None else noise_env
        if mos_env not in env_map:
            raise ValueError(f"unknown mosquito environment: {mos_env}")
        if noise_env not in env_map:
            raise ValueError(f"unknown noise environment: {noise_env}")
        snr_range = env_map[noise_env]["snr_range"]
    else:
        snr_range = config.GAUSSIAN_BACKGROUND["snr_range"]

    sr = int(config.SAMPLING_RATE)
    background_noise = np.asarray(background_noise, dtype=np.float32)
    mosquito_audio = np.zeros(len(background_noise), dtype=np.float32)
    rows = []

    if source_scheduler is None:
        source_scheduler = select_mos_source(mos_list, stable_seed(source_name, noise_env), sampling_mode="balanced")

    snr_low, snr_high = sorted(map(float, snr_range))

    for event_index, (start, end) in enumerate(mos_positions):
        chosen = source_scheduler.pick()
        start_sample = int(round(float(start) * sr))
        requested = max(1, int(round((float(end) - float(start)) * sr)))

        event, source_start, source_end, source_looped = cut_mosquito_source( chosen["path"], requested, sr, )

        end_sample = min(start_sample + len(event), len(background_noise))
        event = event[:end_sample - start_sample]
        if len(event) == 0:
            raise ValueError("mosquito event falls outside clip")

        local_background = np.asarray(background_noise[start_sample:end_sample], dtype=np.float64)
        event64 = np.asarray(event, dtype=np.float64)
        noise_power = float(np.mean(np.square(local_background)))
        mosquito_power = float(np.mean(np.square(event64)))
        if noise_power < EPS or mosquito_power < EPS:
            raise RuntimeError("cannot calculate SNR from silent signal")

        target_snr = random.uniform(snr_low, snr_high)
        scale = np.sqrt( noise_power * (10.0 ** (target_snr / 10.0)) / mosquito_power )
        event = (event64 * scale).astype(np.float32)
        real_snr = 10.0 * np.log10( float(np.mean(np.square(event.astype(np.float64)))) / noise_power )

        mosquito_audio[start_sample:end_sample] += event
        rows.append({
            "event_type": "Mosquito",
            "event_index": int(event_index),
            "start_time": round(start_sample / sr, 3),
            "end_time": round(end_sample / sr, 3),
            "species": chosen["species"],
            "sex": chosen["sex"],
            "label": chosen["label"],
            "balance_label": chosen["balance_label"],
            "snr": round(float(real_snr), 3),
            "target_snr": round(float(target_snr), 3),
            "real_snr": round(float(real_snr), 3),
            "snr_kind": "source_event_to_added_background",
            "mosquito_rms": round(rms(event), 8),
            "mos_path": chosen["path"],
            "raw_recording_id": chosen["raw_recording_id"],
            "source_environment": chosen["source_environment"],
            "source_device": chosen["source_device"],
            "mos_source_start": round(float(source_start), 6),
            "mos_source_end": round(float(source_end), 6),
            "mos_source_duration": round(float(chosen["source_duration"]), 6),
            "mos_is_looped": bool(source_looped),
            "source_use_index": int(chosen["source_use_index"]),
            "class_use_index": int(chosen["class_use_index"]),
            "source_use_cycle": int(chosen["source_use_cycle"]),
            "source_name": source_name,
        })

    return mosquito_audio.astype(np.float32), {"audio_labels": rows}

# =============================================================================
# Metadata rows
# =============================================================================

def make_noise_row(start_time, end_time, env_name, source_name):
    return {
        "event_type": "Noise",
        "event_index": None,
        "start_time": round(float(start_time), 3),
        "end_time": round(float(end_time), 3),
        "species": "-",
        "sex": "-",
        "label": "Noise",
        "balance_label": "Noise",
        "snr": None,
        "target_snr": None,
        "real_snr": None,
        "snr_kind": "not_applicable",
        "mosquito_rms": None,
        "mos_path": None,
        "raw_recording_id": None,
        "source_environment": "not_applicable",
        "source_device": "not_applicable",
        "mos_source_start": None,
        "mos_source_end": None,
        "mos_source_duration": None,
        "mos_is_looped": False,
        "source_use_index": None,
        "class_use_index": None,
        "source_use_cycle": None,
        "environment": env_name,
        "source_name": source_name,
    }


def annotate_noise_events(mos_rows, duration_sec, env_name, source_name):
    rows = []
    cursor = 0.0
    for row in sorted(mos_rows, key=lambda item: item["start_time"]):
        if float(row["start_time"]) > cursor:
            rows.append(make_noise_row(cursor, row["start_time"], env_name, source_name))
        rows.append(row)
        cursor = max(cursor, float(row["end_time"]))
    if cursor < float(duration_sec):
        rows.append(make_noise_row(cursor, duration_sec, env_name, source_name))
    return rows


# =============================================================================
# Source usage settings and reports
# =============================================================================

def get_source_mode() -> str:
    mode = str(config.SIM_MIX_CFG.get("source_coverage_mode", "auto")).lower()
    if mode not in {"auto", "strict", "off"}:
        raise ValueError("source_coverage_mode must be auto, strict or off")
    return mode


def get_source_sampling_mode(dataset_type: str) -> str:
    split = str(dataset_type).strip().lower()
    if split in {"train", "training"} or split.startswith("train_"):
        return "balanced"
    return "raw"

def get_allowed_species(env_target=None, background_type="real", use_hb_env=False):
    background_type = str(background_type).strip().lower()
    if background_type not in {"real", "gaussian"}:
        raise ValueError(f"unknown background_type: {background_type}")
    if background_type == "gaussian":
        return list(config.SPECIES_ORDER)

    env_map = config.HB_ENVIRONMENTS if use_hb_env else config.ENVIRONMENTS
    if env_target not in env_map:
        raise ValueError(f"unknown environment: {env_target}")
    return list(env_map[env_target]["mosquito_species"])


def get_source_coverage_requirement( mosquito_dirs, env_target=None, dataset_type: str = "train", target_type: str = "species", use_hb_env: bool = False, background_type: str = "real", ) -> dict:
    """Return source coverage capacity for one simulation job."""
    allowed_species = get_allowed_species(env_target=env_target, background_type=background_type, use_hb_env=use_hb_env)
    mos_list = build_mos_list(mosquito_dirs, allowed_species, target_type=target_type)

    sampling_mode = get_source_sampling_mode(dataset_type)
    scheduler = select_mos_source(mos_list, seed=0, sampling_mode=sampling_mode)
    max_events = max(1, int(config.SIM_MIX_CFG["max_mos_events"]))
    minimum_simulations = int(np.ceil(scheduler.total_files / max_events))
    class_sizes = {label: len(rows) for label, rows in scheduler.pools.items()}

    return {
        "sampling_mode": sampling_mode,
        "raw_file_count": scheduler.total_files,
        "class_count": scheduler.n_classes,
        "class_sizes": dict(sorted(class_sizes.items())),
        "required_event_slots": scheduler.total_files,
        "minimum_simulations": minimum_simulations,
    }


def check_source_usage(scheduler: select_mos_source, num_simulations: int) -> bool:
    min_events = max(1, int(config.SIM_MIX_CFG["min_mos_events"]))
    max_events = max(min_events, int(config.SIM_MIX_CFG["max_mos_events"]))
    min_slots = int(num_simulations) * min_events
    max_slots = int(num_simulations) * max_events
    slots_needed = scheduler.total_files
    enough_capacity = max_slots >= slots_needed
    mode = get_source_mode()

    print(f"[source usage] coverage mode: {mode}")
    print(f"[source usage] sampling mode: {scheduler.sampling_mode}")
    print(f"[source usage] valid files: {scheduler.total_files}")
    print(f"[source usage] classes: {scheduler.n_classes}")
    print(f"[source usage] minimum event slots: {min_slots}")
    print(f"[source usage] maximum event slots: {max_slots}")
    print(f"[source usage] coverage slots needed: {slots_needed}")

    if mode == "strict" and not enough_capacity:
        minimum_simulations = int(np.ceil(slots_needed / max_events))
        raise ValueError(
            f"not enough capacity to cover all valid source files. "
            f"Need at least {minimum_simulations} simulations with max_mos_events={max_events}."
        )

    if mode == "auto" and not enough_capacity:
        print("[source usage] Some source files may not be used.")

    return enough_capacity and mode != "off"


def save_usage_reports(
    scheduler: select_mos_source,
    noise_scheduler: select_noise_file | None,
    metadata_dir: str,
    source_name: str,
    env_name: str,
    dataset_type: str,
    target_type: str,
) -> tuple[str, str | None]:
    if dataset_type == "test":
        file_tag = f"test_{source_name}_{env_name}"
    else:
        file_tag = f"{target_type}_{source_name}_{env_name}_{dataset_type}"
    source_path = os.path.join( metadata_dir, f"usage_sources_{file_tag}.csv", )
    scheduler.usage_frame().to_csv(source_path, index=False)

    noise_path = None
    if noise_scheduler is not None:
        noise_path = os.path.join( metadata_dir, f"usage_noise_{file_tag}.csv", )
        noise_scheduler.usage_frame(env_name).to_csv( noise_path, index=False, )

    return source_path, noise_path


# =============================================================================
# Main generation functions
# =============================================================================

def create_simulation_data(
    mosquito_dirs,
    noise_dir,
    output_dir,
    num_simulations,
    mos_env,
    noise_env,
    dataset_type,
    source_name,
    metadata_dir,
    metadata_name,
    file_prefix,
    seed,
    use_hb_env=False,
    target_type="species_sex",
    noise_list=None,
    background_type="real",
):

    seed = 0 if seed is None else int(seed)
    num_simulations = int(num_simulations)
    set_random_seed(seed)

    mix_cfg = config.SIM_MIX_CFG
    env_map = config.HB_ENVIRONMENTS if use_hb_env else config.ENVIRONMENTS
    # Environmental validation is required only for real backgrounds.
    background_type = str(background_type).strip().lower()
    if background_type not in {"real", "gaussian"}:
        raise ValueError(f"unknown background_type: {background_type}")

    if background_type == "real":
        if mos_env not in env_map:
            raise ValueError(f"unknown mosquito environment: {mos_env}")

        if noise_env not in env_map:
            raise ValueError(f"unknown noise environment: {noise_env}")

    duration = float(config.AUDIO_DURATION)
    sr = int(config.SAMPLING_RATE)

    allowed_species = get_allowed_species(env_target=mos_env, background_type=background_type, use_hb_env=use_hb_env)
    mos_list = build_mos_list(mosquito_dirs, allowed_species, target_type=target_type)

 

    mos_seed = stable_seed(seed, source_name, dataset_type, noise_env, "source")
    noise_seed = stable_seed(seed, source_name, dataset_type, noise_env, "noise")
    
    source_sampling_mode = get_source_sampling_mode(dataset_type)
    source_scheduler = select_mos_source(mos_list, mos_seed, sampling_mode=source_sampling_mode)
    check_all_source_files = check_source_usage(source_scheduler, num_simulations)


    if background_type == "gaussian":
        noise_scheduler = None
        check_all_noise_files = False
    else:
        if noise_list is None:
            noise_list = get_noise_files_by_env(noise_dir)

        validate_noise_list( noise_list, required_envs=(noise_env,), )
        noise_scheduler = select_noise_file(noise_list, noise_seed)
        noise_files = noise_scheduler.files.get(noise_env, [])
        check_all_noise_files = num_simulations >= len(noise_files)




    output_dir = os.path.abspath(os.fspath(output_dir))
    metadata_dir = ( os.path.join(os.path.dirname(output_dir), "metadata")
        if metadata_dir is None
        else os.path.abspath(os.fspath(metadata_dir))
    )
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    metadata_rows = []
    max_retry = int(mix_cfg.get("max_retry_per_sim", 10))

    for index in range(int(num_simulations)):
        source_snapshot = source_scheduler.snapshot()
        noise_snapshot = ( noise_scheduler.snapshot() if noise_scheduler is not None else None )
        last_error = None

        for attempt in range(max_retry):
            if attempt > 0:
                source_scheduler.restore(source_snapshot)
                if noise_scheduler is not None:
                    noise_scheduler.restore(noise_snapshot)

            try:
                if background_type == "gaussian":
                    background = add_gaussian_noise(duration,sr)
                    noise_file = None
                    subenv = "gaussian"
                    noise_source_start = None
                    noise_use_index = None
                else:
                    (
                        background,
                        noise_file,
                        subenv,
                        noise_source_start,
                        noise_use_index,
                    ) = add_background_noise(
                        noise_dir,
                        noise_env,
                        dataset_type,
                        source_name=source_name,
                        use_hb_env=use_hb_env,
                        noise_list=noise_list,
                        noise_scheduler=noise_scheduler,
                    )

                remaining_clips = num_simulations - index
                n_events = choose_mos_event_count(source_scheduler, remaining_clips)
                positions = gen_mos_pos(duration, n_events=n_events)
                mosquito_audio, metadata = prepare_mos(
                    mos_env,
                    mos_list,
                    background,
                    positions,
                    source_name=source_name,
                    noise_env=noise_env,
                    background_type=background_type,
                    use_hb_env=use_hb_env,
                    source_scheduler=source_scheduler,
                )
                final_audio, output_gain = fit_peak( background + mosquito_audio, return_info=True, )
                break
            except (ValueError, RuntimeError) as exc:
                last_error = exc
        else:
            source_scheduler.restore(source_snapshot)
            if noise_scheduler is not None:
                noise_scheduler.restore(noise_snapshot)
            raise RuntimeError( f"failed to generate clip after {max_retry} attempts: {last_error}" ) from last_error

        file_name = f"{file_prefix}_{index + 1:06d}.wav"
        audio_path = os.path.abspath(os.path.join(output_dir, file_name))
        sf.write(audio_path, final_audio, sr, subtype=str(mix_cfg.get("wav_subtype", "PCM_16")))

        metadata_env = "gaussian" if background_type == "gaussian" else noise_env

        common = {
            "environment": metadata_env,
            "simulation_environment": metadata_env,
            "mos_env": mos_env,
            "noise_env": noise_env,
            "source_name": source_name,
            "target_type": target_type,
            "source_sampling_mode": source_sampling_mode,
            "background_type": background_type,
            "noise_file": os.path.abspath(noise_file) if noise_file is not None else None,
            "noise_source_start": noise_source_start,
            "noise_use_index": int(noise_use_index) if noise_use_index is not None else None,
            "subenv": subenv,
            "noise_rms": round(rms(background) * output_gain, 8),
            "output_gain": round(float(output_gain), 8),
            "file_name": file_name,
            "simfile_path": audio_path,
            "duration": duration,
            "dataset_type": dataset_type,
            "seed": seed,
        }
        final_rows = annotate_noise_events(metadata["audio_labels"], duration, metadata_env, source_name)
        metadata_rows.extend({**row, **common} for row in final_rows)

    metadata_df = pd.DataFrame(metadata_rows)
    metadata_path = os.path.join(metadata_dir, metadata_name)
    metadata_df.to_csv(metadata_path, index=False)

    source_usage_path, noise_usage_path = save_usage_reports(
        source_scheduler,
        noise_scheduler,
        metadata_dir,
        source_name,
        noise_env,
        dataset_type,
        target_type,
    )

    missing_sources = source_scheduler.missing_paths()
    missing_noise = ( noise_scheduler.missing_paths(noise_env) if noise_scheduler is not None else [] )

    if check_all_source_files and missing_sources:
        raise RuntimeError(
            f"failed: {len(missing_sources)} valid source files "
            f"were not used. See {source_usage_path}"
        )

    if check_all_noise_files and missing_noise:
        raise RuntimeError( f"failed: {len(missing_noise)} noise files were not used. " f"See {noise_usage_path}" )

    mos_counts = (
        metadata_df.loc[
            metadata_df["event_type"] == "Mosquito",
            "label",
        ]
        .value_counts()
        .sort_index()
    )

    print(f"saved: {metadata_path}")
    print(f"saved: {source_usage_path}")
    if noise_usage_path is not None:
        print(f"saved: {noise_usage_path}")
    print(f"[usage] missing_sources={len(missing_sources)} missing_noise={len(missing_noise)}")
    print(f"[saved mosquito events] target_type={target_type}")
    print(mos_counts.to_string())
    return metadata_df


def process_simulation(
    mosquito_dirs,
    noise_dir,
    output_dir,
    num_simulations,
    env_target,
    dataset_type,
    source_name=None,
    metadata_dir=None,
    metadata_name=None,
    use_hb_env=False,
    seed=None,
    target_type="species_sex",
    noise_list=None,
    background_type="real",
):
    source_name = source_name or detect_source_name(mosquito_dirs)
    source_token = str(source_name).replace(" ", "_")
    
    if background_type == "gaussian":
        env_token = "gaussian"
    else:
        env_token = str(env_target).replace(" ", "_")

    split_token = str(dataset_type).replace(" ", "_")
    metadata_name = metadata_name or (f"metadata_{source_token}_{env_token}_{split_token}.csv")

    audio_file = create_simulation_data(
        mosquito_dirs=mosquito_dirs,
        noise_dir=noise_dir,
        output_dir=output_dir,
        num_simulations=num_simulations,
        mos_env=env_target,
        noise_env=env_target,
        dataset_type=dataset_type,
        source_name=source_name,
        metadata_dir=metadata_dir,
        metadata_name=metadata_name,
        file_prefix=f"{source_token}_{env_token}_{split_token}",
        seed=seed,
        use_hb_env=use_hb_env,
        target_type=target_type,
        noise_list=noise_list,
        background_type=background_type,
    )

    return audio_file


def process_noise_only(
    noise_dir,
    output_dir,
    num_simulations,
    env_target,
    dataset_type="test",
    metadata_dir=None,
    metadata_name=None,
    use_hb_env=False,
    seed=None,
    noise_list=None,
):
    seed = 0 if seed is None else int(seed)
    set_random_seed(seed)

    env_map = config.HB_ENVIRONMENTS if use_hb_env else config.ENVIRONMENTS
    if env_target not in env_map:
        raise ValueError(f"unknown environment: {env_target}")

    duration = float(config.AUDIO_DURATION)
    sr = int(config.SAMPLING_RATE)
    if noise_list is None:
        noise_list = get_noise_files_by_env(noise_dir)
    validate_noise_list(noise_list, required_envs=(env_target,))

    scheduler = select_noise_file(noise_list,stable_seed(seed, "noiseonly", dataset_type, env_target))

    output_dir = os.path.abspath(os.fspath(output_dir))
    metadata_dir = (
        os.path.join(os.path.dirname(output_dir), "metadata")
        if metadata_dir is None
        else os.path.abspath(os.fspath(metadata_dir))
    )
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    env_token = str(env_target).replace(" ", "_")
    split_token = str(dataset_type).replace(" ", "_")
    metadata_name = metadata_name or f"metadata_noiseonly_{env_token}_{split_token}.csv"
    rows = []

    for index in range(int(num_simulations)):
        (
            background,
            noise_file,
            subenv,
            noise_source_start,
            noise_use_index,
        ) = add_background_noise(
            noise_dir,
            env_target,
            dataset_type,
            source_name="noiseonly",
            use_hb_env=use_hb_env,
            noise_list=noise_list,
            noise_scheduler=scheduler,
        )
        audio, output_gain = fit_peak(background, return_info=True)

        file_name = f"noiseonly_{env_token}_{split_token}_{index + 1:06d}.wav"
        audio_path = os.path.abspath(os.path.join(output_dir, file_name))
        sf.write(audio_path, audio, sr, subtype=str(config.SIM_MIX_CFG.get("wav_subtype", "PCM_16")))

        row = make_noise_row(0.0, duration, env_target, "noiseonly")
        row.update({
            "simulation_environment": env_target,
            "mos_env": "-",
            "noise_env": env_target,
            "target_type": "noise_only",
            "background_type": "noise_only",
            "noise_file": os.path.abspath(noise_file),
            "noise_source_start": noise_source_start,
            "noise_use_index": int(noise_use_index),
            "subenv": subenv,
            "noise_rms": round(rms(background) * output_gain, 8),
            "output_gain": round(float(output_gain), 8),
            "file_name": file_name,
            "simfile_path": audio_path,
            "duration": duration,
            "dataset_type": dataset_type,
            "seed": seed,
        })
        rows.append(row)

    metadata_df = pd.DataFrame(rows)
    metadata_path = os.path.join(metadata_dir, metadata_name)
    metadata_df.to_csv(metadata_path, index=False)

    usage_path = os.path.join(metadata_dir,f"usage_noise_noiseonly_{env_token}_{split_token}.csv")
    scheduler.usage_frame(env_target).to_csv(usage_path, index=False)

    missing = scheduler.missing_paths(env_target)
    if int(num_simulations) >= len(scheduler.files.get(env_target, [])) and missing:
        raise RuntimeError(f"noise-only coverage failed; see {usage_path}")

    print(f"saved: {metadata_path}")
    print(f"saved: {usage_path}")
    return metadata_df


def process_cross_simulation(
    mosquito_dirs,
    noise_dir,
    output_dir,
    num_simulations,
    mos_env,
    noise_env,
    pair_type,
    dataset_type="test",
    source_name="miru_cross",
    metadata_dir=None,
    metadata_name=None,
    seed=None,
    target_type="species_sex",
    noise_list=None,
    background_type="real",
):
    source_token = str(source_name).replace(" ", "_")
    pair_token = str(pair_type).replace(" ", "_")
    noise_token = str(noise_env).replace(" ", "_")
    split_token = str(dataset_type).replace(" ", "_")
    metadata_name = metadata_name or ( f"metadata_{source_token}_{noise_token}_{split_token}_cross_{pair_token}.csv" )

    return create_simulation_data(
        mosquito_dirs=mosquito_dirs,
        noise_dir=noise_dir,
        output_dir=output_dir,
        num_simulations=num_simulations,
        mos_env=mos_env,
        noise_env=noise_env,
        dataset_type=dataset_type,
        source_name=source_name,
        metadata_dir=metadata_dir,
        metadata_name=metadata_name,
        file_prefix=f"{source_token}_{pair_token}_{split_token}",
        seed=seed,
        target_type=target_type,
        noise_list=noise_list,
        background_type=background_type,
    )


def detect_source_name(mosquito_inputs) -> str:
    paths = resolve_wav_inputs(mosquito_inputs)
    source_dirs = getattr(config, "SRC_MIRU_SOURCE_DIRS", {})
    biodcase_root = os.path.abspath( os.fspath(config.BIODCASE_RAW_DIR) )

    for path in paths[:20]:
        path = os.path.abspath(path)

        for source_name, root in source_dirs.items():
            source_root = os.path.abspath(os.fspath(root))
            try:
                if os.path.commonpath([path, source_root]) == source_root:
                    return str(source_name)
            except ValueError:
                continue

        try:
            if os.path.commonpath([path, biodcase_root]) == biodcase_root:
                return "biodcase"
        except ValueError:
            continue

    return "unknown"

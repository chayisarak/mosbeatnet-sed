

# dataloader.py
import gc
import hashlib
import json
import os
import random
import time
import warnings
from collections import Counter, OrderedDict, defaultdict
from contextlib import nullcontext
from typing import List, Optional, Tuple, Union

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
try:
    from torch.utils.tensorboard import SummaryWriter
except (ImportError, ModuleNotFoundError):
    class SummaryWriter:
        """No-op fallback when the optional tensorboard package is absent."""

        def __init__(self, *args, **kwargs):
            print("[TensorBoard] package not installed; scalar logging disabled")

        def add_scalar(self, *args, **kwargs):
            return None

        def close(self):
            return None

from tqdm.auto import tqdm

import config
from run_utils import (
    RUN_SCHEMA_VERSION,
    canonical_model_key,
    capture_rng_state,
    find_latest_run,
    checkpoint_path as run_checkpoint_path,
)


# -----------------------------------------------------------------------------
# Metadata
# -----------------------------------------------------------------------------


def merge_metadata( files: List[str], output_path: str, strict: bool = True, ) -> str:
    files = [os.path.abspath(os.fspath(path)) for path in files]
    missing = [path for path in files if not os.path.exists(path)]
    if missing and strict:
        details = "\n".join("  - {}".format(path) for path in missing)
        raise FileNotFoundError( "Required metadata files are missing:\n{}".format(details) )

    existing = [path for path in files if os.path.exists(path)]
    if not existing:
        raise ValueError("No metadata files found to merge.")

    frames = []
    required_columns = { "file_name", "start_time", "end_time", "label", "simfile_path" }
    for path in existing:
        frame = pd.read_csv(path)
        missing_columns = sorted(required_columns - set(frame.columns))
        if missing_columns:
            raise KeyError( "Metadata {} is missing columns: {}".format( path, missing_columns ) )
        frames.append(frame)

    merged = pd.concat(frames, ignore_index=True)
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    merged.to_csv(output_path, index=False)
    print( "[merge] {} files, {} rows -> {}".format( len(existing), len(merged), output_path ) )
    return output_path


# -----------------------------------------------------------------------------
# Audio helpers
# -----------------------------------------------------------------------------


def load_audio_sf(path: str, target_sr: int) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1, dtype=np.float32)

    audio = np.asarray(audio, dtype=np.float32)

    if not np.isfinite(audio).all():
        raise FloatingPointError("Audio contains NaN or Inf: {}".format(path))

    if int(sr) != int(target_sr):
        audio = librosa.resample(
            audio,
            orig_sr=int(sr),
            target_sr=int(target_sr),
            res_type="kaiser_fast",
        ).astype(np.float32)

    if not np.isfinite(audio).all():
        raise FloatingPointError("Resampled audio contains NaN or Inf: {}".format(path))

    return audio.astype(np.float32, copy=False)


def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def rms(x: np.ndarray) -> float:
    x64 = np.asarray(x, dtype=np.float64)
    if x64.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x64)) + 1e-12))


def loudness_normalize( x: np.ndarray, target_rms: float = 0.10, max_peak: float = 0.99, ) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)

    if not np.isfinite(x).all():
        raise FloatingPointError("Audio contains NaN or Inf before normalization")

    current_rms = rms(x)
    if current_rms > 1e-8:
        x = x * (float(target_rms) / current_rms)

    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > float(max_peak) and peak > 1e-8:
        x = x * (float(max_peak) / peak)

    if not np.isfinite(x).all():
        raise FloatingPointError("Audio contains NaN or Inf after normalization")

    return x.astype(np.float32, copy=False)


# -----------------------------------------------------------------------------
# Labels
# -----------------------------------------------------------------------------


def set_label_mode(df: pd.DataFrame, label_mode: str = "label") -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame, got {}".format(type(df)))

    df = df.copy()
    mode = (label_mode or "label").strip().lower()

    def is_missing_token(value):
        text = str(value).strip()
        return text in {"-", "", "nan", "NaN", "None", "none", "<NA>", "NA", "N/A"}

    def clean_sex(value):
        text = str(value).strip()
        if is_missing_token(text):
            return "U"

        low = text.lower()
        if low in {"f", "female"}:
            return "F"
        if low in {"m", "male"}:
            return "M"
        if low in {"u", "unknown", "unk"}:
            return "U"
        return text.upper()

    def clean_species(value):
        text = str(value).strip()
        if is_missing_token(text) or text.lower() == "noise":
            return "Noise"
        return text

    def clean_species_sex(value):
        text = str(value).strip()
        if is_missing_token(text) or text.lower() == "noise":
            return "Noise"

        bad_suffixes = ["_nan", "_NaN", "_None", "_none", "_<NA>", "_NA", "_N/A", "_-", "_"]
        for suffix in bad_suffixes:
            if text.endswith(suffix):
                base = text[: -len(suffix)]
                return "{}_U".format(base) if base else "Noise"

        if "_" in text:
            base, sex = text.rsplit("_", 1)
            if base and sex:
                return "{}_{}".format(base, clean_sex(sex))

        return text

    is_noise = pd.Series(False, index=df.index)

    if "event_type" in df.columns:
        is_noise = is_noise | df["event_type"].astype(str).str.strip().str.lower().eq("noise")

    if "label" in df.columns:
        is_noise = is_noise | df["label"].astype(str).str.strip().str.lower().eq("noise")

    if "species" in df.columns:
        species_text = df["species"].astype(str).str.strip()
        is_noise = is_noise | species_text.apply(is_missing_token)
        is_noise = is_noise | species_text.str.lower().eq("noise")

    noise_tokens = {
        "-": "Noise",
        "": "Noise",
        "nan": "Noise",
        "NaN": "Noise",
        "None": "Noise",
        "none": "Noise",
        "<NA>": "Noise",
        "NA": "Noise",
        "N/A": "Noise",
    }

    if mode == "label":
        if "label" not in df.columns:
            raise KeyError("label_mode='label' requires column: 'label'")
        target = df["label"].astype(str).str.strip()
        target = target.where(~is_noise, "Noise").replace(noise_tokens)

    elif mode == "species":
        if "species" not in df.columns:
            raise KeyError("label_mode='species' requires column: 'species'")
        target = df["species"].apply(clean_species)
        target = target.where(~is_noise, "Noise")

    elif mode == "sex":
        if "sex" not in df.columns:
            raise KeyError("label_mode='sex' requires column: 'sex'")
        target = df["sex"].apply(clean_sex)
        target = target.where(~is_noise, "Noise")

    elif mode == "species_sex":
        if "species_sex" in df.columns:
            target = df["species_sex"].apply(clean_species_sex)
            target = target.where(~is_noise, "Noise")

        elif {"species", "sex"}.issubset(df.columns):
            species = df["species"].apply(clean_species)
            sex = df["sex"].apply(clean_sex)
            target = pd.Series(
                np.where(~is_noise, species.astype(str) + "_" + sex.astype(str), "Noise"),
                index=df.index,
            ).apply(clean_species_sex)

        elif "label" in df.columns:
            target = df["label"].astype(str).where(~is_noise, "Noise")
            target = target.apply(clean_species_sex)

        else:
            raise KeyError(
                "label_mode='species_sex' requires 'species_sex', "
                "both 'species' and 'sex', or 'label'"
            )

    else:
        raise ValueError( "Unsupported label_mode='{}'. Use label, species, species_sex, or sex".format(label_mode) )

    target = pd.Series(target, index=df.index)
    target = target.replace(r"^\s*$", np.nan, regex=True)
    df["label"] = target.fillna("Noise").astype(str)
    return df


# -----------------------------------------------------------------------------
# Augmentation
# -----------------------------------------------------------------------------


class AudioAugmentor:
    def __init__(
        self,
        alpha_range=(0.7, 1.3),
        alpha_step=None,
        noise_level_range=(0.001, 0.01),
        fixed_alpha=None,
        mix_mode="weighted",
        pitch_shift_range=(-3, 3),
        stretch_range=(0.8, 1.2),
        p_pitch=0.0,
        p_stretch=0.0,
    ):
        self.alpha_range = alpha_range
        self.alpha_step = alpha_step
        self.noise_level_range = noise_level_range
        self.fixed_alpha = fixed_alpha
        self.mix_mode = (mix_mode or "weighted").lower()
        self.pitch_shift_range = pitch_shift_range
        self.stretch_range = stretch_range
        self.p_pitch = float(p_pitch)
        self.p_stretch = float(p_stretch)

        if alpha_step is not None:
            self.alpha_choices = np.round( np.arange(alpha_range[0], alpha_range[1] + 1e-9, alpha_step), 3 ).tolist()
        else:
            self.alpha_choices = None

        self.last_alpha = None
        self.last_noise_ratio = None

    def _sample_alpha(self) -> Tuple[float, float]:
        if self.fixed_alpha is not None:
            alpha = float(self.fixed_alpha)
        elif self.alpha_choices is not None:
            alpha = float(random.choice(self.alpha_choices))
        else:
            alpha = float(np.random.uniform(self.alpha_range[0], self.alpha_range[1]))

        noise_ratio = float(np.random.uniform(*self.noise_level_range))
        self.last_alpha = alpha
        self.last_noise_ratio = noise_ratio
        return alpha, noise_ratio

    def _alpha_to_lambda01(self, alpha: float) -> float:
        low, high = float(self.alpha_range[0]), float(self.alpha_range[1])
        if abs(high - low) < 1e-12:
            return 0.5
        return float(np.clip((alpha - low) / (high - low), 0.0, 1.0))

    def sample_params(self) -> Tuple[float, float]:
        return self._sample_alpha()

    def augment(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().squeeze().cpu().numpy()
        if not isinstance(audio, np.ndarray):
            raise TypeError("Audio must be np.ndarray or torch.Tensor, got {}".format(type(audio)))

        audio = audio.astype(np.float32, copy=False)
        original_length = int(audio.shape[0])
        audio_rms = rms(audio)
        alpha, noise_ratio = self._sample_alpha()

        noise = np.random.normal(0.0, 1.0, size=audio.shape).astype(np.float32)
        noise = noise / max(rms(noise), 1e-8)
        noise_scaled = noise * (audio_rms * noise_ratio)

        if self.mix_mode == "both":
            augmented = alpha * (audio + noise_scaled)
        elif self.mix_mode == "convex":
            lam = self._alpha_to_lambda01(alpha)
            augmented = lam * audio + (1.0 - lam) * noise_scaled
        else:
            augmented = alpha * audio + noise_scaled

        if self.p_pitch > 0 and random.random() < self.p_pitch:
            n_steps = float(random.uniform(*self.pitch_shift_range))
            augmented = librosa.effects.pitch_shift( augmented, sr=sr, n_steps=n_steps, ).astype(np.float32)

        if self.p_stretch > 0 and random.random() < self.p_stretch:
            rate = float(random.uniform(*self.stretch_range))
            augmented = librosa.effects.time_stretch(augmented, rate=rate).astype(np.float32)

        if augmented.shape[0] < original_length:
            augmented = np.pad(augmented, (0, original_length - augmented.shape[0]))
        else:
            augmented = augmented[:original_length]

        augmented = np.clip(augmented.astype(np.float32, copy=False), -1.0, 1.0)

        if not np.isfinite(augmented).all():
            raise FloatingPointError("Augmentation produced NaN or Inf")

        return augmented


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


class AudioSequenceDataset(Dataset):
    def __init__(
        self,
        metadata_path: Optional[str] = None,
        metadata_df: Optional[pd.DataFrame] = None,
        audio_dir: Optional[str] = None,
        segment_duration: float = 0.5,
        label_map: Optional[dict] = None,
        augmentor: Optional[AudioAugmentor] = None,
        save_metadata: bool = False,
        prefer_simfile_path: bool = True,
        root_dir: Optional[str] = None,
        augment_noise: bool = False,
        anchor: Optional[str] = None,
        path_anchors: Optional[Union[str, List[str]]] = None,
        normalize_audio: bool = True,
        target_rms: float = 0.10,
        max_peak: float = 0.99,
        label_mode: str = "label",
        dataset_type: Optional[str] = None,
        env_type: Optional[str] = None,
        seg_dir: Optional[str] = None,
        return_file_name: bool = False,
        audio_cache_max: int = 32,
        cache_segment_tensors: bool = False,
        use_disk_segment_cache: bool = False,
        rebuild_segment_cache: bool = False,
        validate_audio: bool = True,
    ):
        self.metadata_path = None

        if metadata_df is not None:
            if not isinstance(metadata_df, pd.DataFrame):
                raise TypeError("metadata_df must be DataFrame, got {}".format(type(metadata_df)))
            self.df = metadata_df.copy()
            lower_path = ""
        elif metadata_path is not None:
            if not os.path.exists(metadata_path):
                raise FileNotFoundError("Metadata CSV not found: {}".format(metadata_path))
            self.metadata_path = os.path.abspath(metadata_path)
            self.df = pd.read_csv(self.metadata_path)
            lower_path = self.metadata_path.lower()
        else:
            raise ValueError("Provide either metadata_path or metadata_df")

        self.label_mode = str(label_mode)
        self.df = set_label_mode(self.df, label_mode=self.label_mode)

        for column in ["file_name", "start_time", "end_time", "label"]:
            if column not in self.df.columns:
                raise KeyError("Metadata missing required column: '{}'".format(column))

        self.audio_dir = audio_dir
        self.segment_duration = float(segment_duration)
        self.sr = int(getattr(config, "SAMPLING_RATE", getattr(config, "SR", 8000)))
        self.max_segments = int( float(getattr(config, "AUDIO_DURATION", 10.0)) / self.segment_duration )

        self.augmentor = augmentor
        self.augment_noise = bool(augment_noise)
        self.prefer_simfile_path = bool(prefer_simfile_path)
        self.root_dir = os.path.normpath(root_dir) if root_dir else None
        self.return_file_name = bool(return_file_name)
        self.normalize_audio = bool(normalize_audio)
        self.target_rms = float(target_rms)
        self.max_peak = float(max_peak)
        self.validate_audio = bool(validate_audio)

        if path_anchors is None:
            path_anchors = anchor if anchor is not None else getattr(config, "PATH_ANCHORS", None)
        self.path_anchors = self._normalize_anchors(path_anchors)

        self.df["label"] = self.df["label"].fillna("Noise").astype(str)
        self.df["start_time"] = pd.to_numeric(self.df["start_time"], errors="coerce")
        self.df["end_time"] = pd.to_numeric(self.df["end_time"], errors="coerce")

        if label_map is None:
            warnings.warn( "label_map not provided. Building from this split only.", UserWarning, stacklevel=2, )
            self.label_map = self._create_label_map()
        else:
            self.label_map = label_map

        if "Noise" not in self.label_map:
            raise KeyError("label_map must contain the Noise class")

        self.file_names = self.df["file_name"].dropna().astype(str).unique().tolist()
        self.file_index = {name: index for index, name in enumerate(self.file_names)}

        self.meta_per_file = {}
        for file_name, frame in self.df.groupby("file_name", sort=False):
            clean_frame = frame.dropna(subset=["start_time", "end_time"]).reset_index(drop=True)
            self.meta_per_file[str(file_name)] = clean_frame

        if self.prefer_simfile_path and "simfile_path" in self.df.columns:
            paths = ( self.df[["file_name", "simfile_path"]] .dropna() .drop_duplicates(subset=["file_name"]) )
            self.path_map = dict(zip(paths["file_name"].astype(str), paths["simfile_path"]))
        else:
            self.path_map = {}

        self.audio_cache_max = max(0, int(audio_cache_max))
        self.audio_cache = OrderedDict()
        self.path_cache = {}
        self.label_cache = {}
        self.segment_tensor_map = {}
        self.cache_segment_tensors = bool(cache_segment_tensors)
        self.use_disk_segment_cache = bool(use_disk_segment_cache)
        self.rebuild_segment_cache = bool(rebuild_segment_cache)
        self.disk_segment_cache_path = None
        self.disk_segment_manifest_path = None
        self._disk_segment_array = None

        self.save_metadata = bool(save_metadata)
        self.segment_metadata = []

        if dataset_type is not None:
            self.dataset_type = str(dataset_type)
        else:
            self.dataset_type = ( "train" if "train" in lower_path else "val" if "val" in lower_path else "test" )

        if env_type is not None:
            self.env_type = str(env_type)
        else:
            self.env_type = (
                "gaussian" if "gaussian" in lower_path else
                "urban" if "urban" in lower_path else
                "forest" if "forest" in lower_path else
                "all" if "all" in lower_path else
                "general"
            )

        if seg_dir is None:
            seg_dir = getattr( config, "SEGMENT_DIR", os.path.join(getattr(config, "OUTPUT_BASE", "."), "segments"), )

        self.segment_metadata_path = self._get_segment_metadata_path(seg_dir)
        self.segment_cache_root = os.path.join(seg_dir, "tensor_cache")

        self._prepare_label_cache()

        if self.use_disk_segment_cache:
            self._prepare_disk_segment_cache()

        if self.cache_segment_tensors:
            self._precompute_segment_tensors()

        if self.save_metadata:
            self.process_all_segments()

    def _normalize_anchors(self, anchors):
        if anchors is None:
            return []
        if isinstance(anchors, str):
            anchors = [anchors]
        return [
            str(value).replace("\\", "/").strip()
            for value in anchors
            if value is not None and str(value).replace("\\", "/").strip()
        ]

    def _create_label_map(self) -> dict:
        labels = set(self.df["label"].dropna().unique().tolist())
        labels.add("Noise")
        return {label: index for index, label in enumerate(sorted(labels))}

    def _get_segment_metadata_path(self, seg_dir: str) -> str:
        filename = "{}_segment_{}_metadata.csv".format(self.dataset_type, self.env_type)
        return os.path.join(seg_dir, self.dataset_type, filename)

    def _segment_length(self) -> int:
        return int(round(self.sr * self.segment_duration))

    def _build_segment_cache_signature(self) -> str:
        hasher = hashlib.sha256()

        settings = {
            "cache_version": 1,
            "sampling_rate": self.sr,
            "segment_duration": self.segment_duration,
            "max_segments": self.max_segments,
            "normalize_audio": self.normalize_audio,
            "target_rms": self.target_rms,
            "max_peak": self.max_peak,
            "dataset_type": self.dataset_type,
            "env_type": self.env_type,
        }
        hasher.update(json.dumps(settings, sort_keys=True).encode("utf-8"))

        if self.metadata_path is not None:
            metadata_stat = os.stat(self.metadata_path)
            metadata_info = ( self.metadata_path, metadata_stat.st_size, metadata_stat.st_mtime_ns, )
            hasher.update(repr(metadata_info).encode("utf-8"))
        else:
            columns = [
                column
                for column in ["file_name", "start_time", "end_time", "label", "simfile_path"]
                if column in self.df.columns
            ]
            frame_hash = pd.util.hash_pandas_object( self.df[columns], index=True, ).values.tobytes()
            hasher.update(frame_hash)

        for file_name in self.file_names:
            audio_path = os.path.abspath(self._resolve_audio_path(file_name))
            if not os.path.exists(audio_path):
                raise FileNotFoundError("Audio file not found: {}".format(audio_path))
            audio_stat = os.stat(audio_path)
            audio_info = ( file_name, audio_path, audio_stat.st_size, audio_stat.st_mtime_ns, )
            hasher.update(repr(audio_info).encode("utf-8"))

        return hasher.hexdigest()[:20]

    def _cache_shape(self):
        return ( len(self.file_names), self.max_segments, self._segment_length(), )

    def _cache_is_valid(self) -> bool:
        if not self.disk_segment_cache_path or not self.disk_segment_manifest_path:
            return False
        if not os.path.exists(self.disk_segment_cache_path):
            return False
        if not os.path.exists(self.disk_segment_manifest_path):
            return False

        try:
            with open(self.disk_segment_manifest_path, "r", encoding="utf-8") as file:
                manifest = json.load(file)

            if tuple(manifest.get("shape", [])) != self._cache_shape():
                return False
            if manifest.get("dtype") != "float32":
                return False

            array = np.load(self.disk_segment_cache_path, mmap_mode="r")
            valid = tuple(array.shape) == self._cache_shape() and array.dtype == np.float32
            del array
            return bool(valid)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return False

    def _prepare_disk_segment_cache(self):
        signature = self._build_segment_cache_signature()
        cache_name = "{}_{}_{}".format( self.dataset_type, self.env_type, signature, )
        cache_dir = os.path.join(self.segment_cache_root, cache_name)
        os.makedirs(cache_dir, exist_ok=True)

        self.disk_segment_cache_path = os.path.join(cache_dir, "segments.npy")
        self.disk_segment_manifest_path = os.path.join(cache_dir, "manifest.json")

        if self.rebuild_segment_cache:
            for path in [self.disk_segment_cache_path, self.disk_segment_manifest_path]:
                if os.path.exists(path):
                    os.remove(path)

        if self._cache_is_valid():
            with open(self.disk_segment_manifest_path, "r", encoding="utf-8") as file:
                manifest = json.load(file)
            size_gb = os.path.getsize(self.disk_segment_cache_path) / (1024 ** 3)
            print(
                "[Segment cache] Reusing {} ({:.2f} GB, first build {:.1f} s)".format(
                    self.disk_segment_cache_path,
                    size_gb,
                    float(manifest.get("build_seconds", 0.0)),
                )
            )
            return

        shape = self._cache_shape()
        estimated_gb = int(np.prod(shape)) * np.dtype(np.float32).itemsize / (1024 ** 3)
        print(
            "[Segment cache] Building {} files, estimated size {:.2f} GB".format(
                len(self.file_names),
                estimated_gb,
            )
        )

        start_time = time.perf_counter()
        process_id = os.getpid()
        temp_cache_path = self.disk_segment_cache_path + ".{}.tmp.npy".format(process_id)
        temp_manifest_path = self.disk_segment_manifest_path + ".{}.tmp".format(process_id)

        try:
            cache_array = np.lib.format.open_memmap( temp_cache_path, mode="w+", dtype=np.float32, shape=shape, )

            for index, file_name in enumerate(
                tqdm(self.file_names, desc="segment cache", unit="file", leave=False)
            ):
                segments = self._compute_segments(file_name)
                cache_array[index] = segments.numpy()

            cache_array.flush()
            del cache_array
            os.replace(temp_cache_path, self.disk_segment_cache_path)

            build_seconds = time.perf_counter() - start_time
            manifest = {
                "cache_version": 1,
                "shape": list(shape),
                "dtype": "float32",
                "build_seconds": build_seconds,
                "file_count": len(self.file_names),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open(temp_manifest_path, "w", encoding="utf-8") as file:
                json.dump(manifest, file, indent=2)
            os.replace(temp_manifest_path, self.disk_segment_manifest_path)

            size_gb = os.path.getsize(self.disk_segment_cache_path) / (1024 ** 3)
            print(
                "[Segment cache] Saved {} ({:.2f} GB) in {:.1f} s".format(
                    self.disk_segment_cache_path,
                    size_gb,
                    build_seconds,
                )
            )
        except Exception:
            for path in [temp_cache_path, temp_manifest_path]:
                if os.path.exists(path):
                    os.remove(path)
            raise
        finally:
            self.audio_cache.clear()
            self._disk_segment_array = None

    def _get_disk_segment_array(self):
        if self._disk_segment_array is None:
            self._disk_segment_array = np.load( self.disk_segment_cache_path, mmap_mode="r", )
        return self._disk_segment_array

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_disk_segment_array"] = None
        return state

    def _strip_to_anchor(self, path: str):
        if not isinstance(path, str):
            return None

        normalized = path.replace("\\", "/")
        for anchor in self.path_anchors:
            index = normalized.lower().find(anchor.lower())
            if index != -1:
                tail = normalized[index + len(anchor):]
                return tail.lstrip("/").replace("/", os.sep)
        return None

    def _resolve_audio_path(self, file_name: str) -> str:
        cached = self.path_cache.get(file_name)
        if cached is not None:
            return cached

        candidate = self.path_map.get(file_name)
        resolved = None

        if isinstance(candidate, str) and candidate.strip():
            candidate = os.path.normpath(candidate)

            if os.path.isabs(candidate) and os.path.exists(candidate):
                resolved = candidate

            elif self.root_dir:
                tail = self._strip_to_anchor(candidate)
                if tail:
                    alt = os.path.normpath(os.path.join(self.root_dir, tail))
                    if os.path.exists(alt):
                        resolved = alt

                if resolved is None:
                    alt = os.path.normpath( os.path.join(self.root_dir, os.path.basename(candidate)) )
                    if os.path.exists(alt):
                        resolved = alt

            if resolved is None and os.path.exists(candidate):
                resolved = candidate

        if resolved is None:
            if self.audio_dir is None:
                raise ValueError("audio_dir is None and simfile_path is missing or unusable")
            resolved = os.path.normpath(os.path.join(self.audio_dir, file_name))

        self.path_cache[file_name] = resolved
        return resolved

    def _cache_get_audio(self, file_name):
        if file_name not in self.audio_cache:
            return None
        value = self.audio_cache.pop(file_name)
        self.audio_cache[file_name] = value
        return value.copy()

    def _cache_put_audio(self, file_name, audio):
        if self.audio_cache_max <= 0:
            return

        if file_name in self.audio_cache:
            self.audio_cache.pop(file_name)

        self.audio_cache[file_name] = audio

        while len(self.audio_cache) > self.audio_cache_max:
            self.audio_cache.popitem(last=False)

    def _get_audio_np(self, file_name: str) -> np.ndarray:
        cached = self._cache_get_audio(file_name)
        if cached is not None:
            return cached

        audio_path = self._resolve_audio_path(file_name)
        audio = load_audio_sf(audio_path, target_sr=self.sr)

        if self.normalize_audio:
            audio = loudness_normalize(audio, self.target_rms, self.max_peak)

        if self.validate_audio and not np.isfinite(audio).all():
            raise FloatingPointError("Invalid audio values: {}".format(audio_path))

        self._cache_put_audio(file_name, audio)
        return audio.copy()

    def _segment_audio_cpu(self, audio_1d: torch.Tensor) -> torch.Tensor:
        segment_length = self._segment_length()
        total_needed = self.max_segments * segment_length
        current_length = int(audio_1d.numel())

        if current_length < total_needed:
            pad = torch.zeros(total_needed - current_length, dtype=torch.float32)
            audio_1d = torch.cat([audio_1d, pad], dim=0)
        elif current_length > total_needed:
            audio_1d = audio_1d[:total_needed]

        segments = audio_1d.unfold(0, segment_length, segment_length)
        segments = segments[:self.max_segments]
        return segments.contiguous()

    def _compute_segments(self, file_name: str) -> torch.Tensor:
        audio = torch.from_numpy(self._get_audio_np(file_name)).float()
        segments = self._segment_audio_cpu(audio)

        if self.validate_audio and not torch.isfinite(segments).all():
            raise FloatingPointError("Segment tensor contains NaN or Inf: {}".format(file_name))

        return segments

    def _get_segments(self, file_name: str) -> torch.Tensor:
        cached = self.segment_tensor_map.get(file_name)
        if cached is not None:
            return cached.clone()

        if self.use_disk_segment_cache and self.disk_segment_cache_path:
            index = self.file_index[file_name]
            array = self._get_disk_segment_array()[index]
            segments = torch.from_numpy(np.array(array, dtype=np.float32, copy=True))
        else:
            segments = self._compute_segments(file_name)

        if self.cache_segment_tensors:
            self.segment_tensor_map[file_name] = segments.clone()

        return segments

    def _make_labels_for_file(self, file_name: str):
        file_meta = self.meta_per_file.get(file_name)

        labels = torch.full( (self.max_segments,), int(self.label_map["Noise"]), dtype=torch.long, )
        label_text = ["Noise"] * self.max_segments
        snrs = [None] * self.max_segments

        if file_meta is None or file_meta.empty:
            return labels, label_text, snrs

        has_snr = "snr" in file_meta.columns

        starts = file_meta["start_time"].to_numpy(dtype=np.float64)
        ends = file_meta["end_time"].to_numpy(dtype=np.float64)
        names = file_meta["label"].astype(str).to_numpy()
        snr_values = file_meta["snr"].to_numpy() if has_snr else None

        for segment_index in range(self.max_segments):
            segment_start = segment_index * self.segment_duration
            segment_end = segment_start + self.segment_duration
            segment_mid = (segment_start + segment_end) / 2.0

            overlap = np.maximum( 0.0, np.minimum(ends, segment_end) - np.maximum(starts, segment_start), )
            active = np.flatnonzero(overlap > 0)

            if active.size == 0:
                continue

            overlap_by_label = defaultdict(float)
            for row_index in active:
                overlap_by_label[str(names[row_index])] += float(overlap[row_index])

            noise_time = overlap_by_label.get("Noise", 0.0)
            mosquito = { label: duration for label, duration in overlap_by_label.items() if label != "Noise" }

            mosquito_time = sum(mosquito.values())
            if not mosquito or mosquito_time <= noise_time:
                continue

            best_time = max(mosquito.values())
            candidates = [label for label, duration in mosquito.items() if duration == best_time]

            if len(candidates) == 1:
                best_label = candidates[0]
            else:
                best_label = None
                best_distance = None
                best_start = None

                for row_index in active:
                    label = str(names[row_index])
                    if label not in candidates:
                        continue

                    clipped_start = max(starts[row_index], segment_start)
                    clipped_end = min(ends[row_index], segment_end)
                    clipped_mid = (clipped_start + clipped_end) / 2.0
                    distance = abs(clipped_mid - segment_mid)

                    if (
                        best_distance is None
                        or distance < best_distance
                        or (distance == best_distance and starts[row_index] < best_start)
                    ):
                        best_label = label
                        best_distance = distance
                        best_start = starts[row_index]

            if best_label is None:
                best_label = candidates[0]

            if best_label not in self.label_map:
                raise KeyError(
                    "Label {!r} in file {!r} is not present in the training "
                    "label map. Refusing to convert it silently to Noise.".format(
                        best_label, file_name
                    )
                )

            label_text[segment_index] = best_label
            labels[segment_index] = int(self.label_map[best_label])

            if has_snr:
                best_rows = [ row_index for row_index in active if str(names[row_index]) == best_label ]
                if best_rows:
                    best_row = max(best_rows, key=lambda row_index: overlap[row_index])
                    value = snr_values[best_row]
                    snrs[segment_index] = None if pd.isna(value) else float(value)

        return labels, label_text, snrs

    def _prepare_label_cache(self):
        print("[Dataset] Preparing labels for {} files...".format(len(self.file_names)))
        for file_name in self.file_names:
            labels, label_text, snrs = self._make_labels_for_file(file_name)
            self.label_cache[file_name] = ( labels, tuple(label_text), tuple(snrs), )

    def _label_segments(self, file_name: str):
        labels, label_text, snrs = self.label_cache[file_name]
        return labels.clone(), list(label_text), list(snrs)

    def _precompute_segment_tensors(self):
        print("[Dataset] Precomputing segments for {} files...".format(len(self.file_names)))
        for file_name in tqdm(self.file_names, desc="precompute", leave=False):
            self.segment_tensor_map[file_name] = self._get_segments(file_name)
        self.audio_cache.clear()

    def refresh_augmentor(self, new_augmentor: Optional[AudioAugmentor]):
        self.augmentor = new_augmentor
        if self.save_metadata:
            self.process_all_segments()

    def _augment_segments(self, segments: torch.Tensor, labels: torch.Tensor):
        if self.augmentor is None:
            return segments, [None] * self.max_segments, [None] * self.max_segments

        output = segments.clone()
        augmentor = self.augmentor
        noise_index = self.label_map.get("Noise")

        if noise_index is not None and not self.augment_noise:
            active_mask = labels != int(noise_index)
        else:
            active_mask = torch.ones(labels.shape[0], dtype=torch.bool)

        active_index = torch.nonzero(active_mask, as_tuple=False).flatten()
        alphas = [None] * self.max_segments
        noise_ratios = [None] * self.max_segments

        if active_index.numel() == 0:
            return output, alphas, noise_ratios

        if augmentor.p_pitch > 0 or augmentor.p_stretch > 0:
            for segment_index in active_index.cpu().tolist():
                segment_aug = augmentor.augment(output[segment_index].numpy(), sr=self.sr)
                alphas[segment_index] = augmentor.last_alpha
                noise_ratios[segment_index] = augmentor.last_noise_ratio
                output[segment_index] = torch.from_numpy(segment_aug).float()
            return output, alphas, noise_ratios

        work = output[active_index]
        count = int(active_index.numel())
        dtype = work.dtype

        if augmentor.fixed_alpha is not None:
            alpha = torch.full((count,), float(augmentor.fixed_alpha), dtype=dtype)
        elif augmentor.alpha_choices is not None:
            choices = torch.tensor(augmentor.alpha_choices, dtype=dtype)
            alpha = choices[torch.randint(0, len(choices), (count,))]
        else:
            alpha = torch.empty(count, dtype=dtype).uniform_(
                float(augmentor.alpha_range[0]),
                float(augmentor.alpha_range[1]),
            )

        noise_ratio = torch.empty(count, dtype=dtype).uniform_(
            float(augmentor.noise_level_range[0]),
            float(augmentor.noise_level_range[1]),
        )

        audio_rms = torch.sqrt(work.pow(2).mean(dim=1) + 1e-12)
        noise = torch.randn_like(work)
        noise = noise / torch.sqrt(noise.pow(2).mean(dim=1, keepdim=True) + 1e-12)
        noise_scaled = noise * (audio_rms[:, None] * noise_ratio[:, None])

        if augmentor.mix_mode == "both":
            work = alpha[:, None] * (work + noise_scaled)
        elif augmentor.mix_mode == "convex":
            low, high = float(augmentor.alpha_range[0]), float(augmentor.alpha_range[1])
            if abs(high - low) < 1e-12:
                lam = torch.full((count, 1), 0.5, dtype=dtype)
            else:
                lam = ((alpha - low) / (high - low)).clamp(0.0, 1.0).unsqueeze(1)
            work = lam * work + (1.0 - lam) * noise_scaled
        else:
            work = alpha[:, None] * work + noise_scaled

        work = work.clamp(-1.0, 1.0)

        if not torch.isfinite(work).all():
            raise FloatingPointError("Augmentation produced NaN or Inf")

        output[active_index] = work

        active_list = active_index.cpu().tolist()
        alpha_list = alpha.cpu().tolist()
        ratio_list = noise_ratio.cpu().tolist()

        for offset, segment_index in enumerate(active_list):
            alphas[segment_index] = float(alpha_list[offset])
            noise_ratios[segment_index] = float(ratio_list[offset])

        augmentor.last_alpha = float(alpha_list[-1])
        augmentor.last_noise_ratio = float(ratio_list[-1])

        return output, alphas, noise_ratios

    def process_all_segments(self):
        os.makedirs(os.path.dirname(self.segment_metadata_path), exist_ok=True)
        rows = []

        for file_name in self.file_names:
            segments = self._get_segments(file_name)
            labels, label_text, snrs = self._label_segments(file_name)
            _, alphas, noise_ratios = self._augment_segments(segments, labels)

            for segment_index in range(self.max_segments):
                start = segment_index * self.segment_duration
                end = start + self.segment_duration
                rows.append({
                    "file_name": file_name,
                    "segment_index": int(segment_index),
                    "start_time": round(float(start), 6),
                    "end_time": round(float(end), 6),
                    "label": label_text[segment_index],
                    "snr": snrs[segment_index],
                    "alpha": alphas[segment_index],
                    "noise_ratio": noise_ratios[segment_index],
                    "label_mode": self.label_mode,
                })

        self.segment_metadata = rows
        pd.DataFrame(rows).to_csv(self.segment_metadata_path, index=False)
        print("[Dataset] segment metadata saved -> {}".format(self.segment_metadata_path))

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        file_name = self.file_names[index]
        segments = self._get_segments(file_name)
        labels, _, _ = self._label_segments(file_name)
        segments, _, _ = self._augment_segments(segments, labels)

        if self.validate_audio and not torch.isfinite(segments).all():
            raise FloatingPointError("Invalid segment values: {}".format(file_name))

        item = (segments.unsqueeze(1).contiguous(), labels)
        if self.return_file_name:
            return item + (file_name,)
        return item


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------


def evaluate_model(
    model,
    criterion,
    data_loader,
    device,
    label_map,
    ignore_index=None,
    print_report=True,
    debug_checks=False,
):
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    use_amp = device.type == "cuda"
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    item_count = 0
    all_targets = []
    all_preds = []
    inverse_map = {value: key for key, value in label_map.items()}

    with torch.inference_mode():
        for batch in data_loader:
            inputs, targets = batch[:2]
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            amp_context = ( torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext() )
            with amp_context:
                outputs = model(inputs)
                logits = outputs.reshape(-1, outputs.shape[-1])
                target_flat = targets.reshape(-1)
                loss = criterion(logits, target_flat)

            if debug_checks and not torch.isfinite(loss).item():
                raise FloatingPointError("Validation loss contains NaN or Inf")

            batch_items = target_flat.numel()
            loss_sum += loss.detach().double() * batch_items
            item_count += batch_items
            all_targets.append(target_flat.detach())
            all_preds.append(logits.detach().argmax(dim=-1))

    if all_targets:
        all_targets = torch.cat(all_targets).cpu().numpy()
        all_preds = torch.cat(all_preds).cpu().numpy()
    else:
        all_targets = np.array([])
        all_preds = np.array([])

    if ignore_index is not None and all_targets.size > 0:
        mask = all_targets != ignore_index
        all_targets = all_targets[mask]
        all_preds = all_preds[mask]

    average_loss = (loss_sum / max(item_count, 1)).item()
    all_labels = sorted(inverse_map.keys())
    present_labels = ( sorted(np.unique(all_targets).astype(int).tolist()) if all_targets.size else [] )

    accuracy = ( accuracy_score(all_targets, all_preds) if all_targets.size else 0.0 )

    # Main validation Macro F1 uses only classes that are actually present in
    # the ground truth. This is important for Exp2, where Cx.Quin_F is part of
    # the canonical 9-class output but absent from Outdoor train/validation.
    macro_f1 = (
        f1_score(
            all_targets,
            all_preds,
            average="macro",
            labels=present_labels,
            zero_division=0,
        )
        if present_labels
        else 0.0
    )

    # Keep an all-configured-class version as a diagnostic only.
    macro_f1_all = (
        f1_score(
            all_targets,
            all_preds,
            average="macro",
            labels=all_labels,
            zero_division=0,
        )
        if all_labels
        else 0.0
    )

    weighted_f1 = (
        f1_score(
            all_targets,
            all_preds,
            average="weighted",
            labels=present_labels,
            zero_division=0,
        )
        if present_labels
        else 0.0
    )

    precision, recall, class_f1, support = precision_recall_fscore_support(
        all_targets,
        all_preds,
        labels=all_labels,
        zero_division=0,
    )

    if print_report:
        print("\n===== Evaluation =====")
        print(
            "Loss: {:.4f} | Acc: {:.4f} | F1-macro: {:.4f} | "
            "F1-macro-all: {:.4f} | F1-weighted: {:.4f}".format(
                average_loss,
                accuracy,
                macro_f1,
                macro_f1_all,
                weighted_f1,
            )
        )
        print("\n--- Per-Class (P / R / F1 / Support) ---")
        for class_id, p, r, f1_value, count in zip( all_labels, precision, recall, class_f1, support, ):
            class_name = inverse_map.get(class_id, str(class_id))
            print( "{:>20s} | P:{:.4f} R:{:.4f} F1:{:.4f} | n={}".format( class_name, p, r, f1_value, int(count), ) )

    return {
        "loss": average_loss,
        "acc": accuracy,
        "f1_macro": macro_f1,
        "f1_macro_all": macro_f1_all,
        "f1_weighted": weighted_f1,
        "f1_per_class": class_f1,
        "precision_per_class": precision,
        "recall_per_class": recall,
        "support_per_class": support,
        "labels": all_labels,
        "present_labels": present_labels,
    }


# -----------------------------------------------------------------------------
# Checkpoints
# -----------------------------------------------------------------------------


def _legacy_run_dir(exp_id: str, model_name: str, run_seed: Optional[int]) -> str:
    """Fallback for callers that have not yet adopted the run manager."""
    seed = 0 if run_seed is None else int(run_seed)
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = os.path.join(
        config.OUTPUT_BASE,
        exp_id,
        canonical_model_key(model_name),
        "seed_{}".format(seed),
        "run_{}_legacy".format(timestamp),
    )
    os.makedirs(os.path.join(path, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(path, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(path, "tensorboard"), exist_ok=True)
    warnings.warn(
        "train_model was called without run_dir; using a legacy-compatible "
        "reproducible folder: {}".format(path),
        RuntimeWarning,
        stacklevel=2,
    )
    return path


def _checkpoint_filename(save_reason: str) -> str:
    names = {
        "best_f1": "best_macro_f1.pt",
        "best_macro_f1": "best_macro_f1.pt",
        "best_loss": "best_val_loss.pt",
        "best_val_loss": "best_val_loss.pt",
        "last": "last.pt",
        "final": "last.pt",
    }
    try:
        return names[str(save_reason)]
    except KeyError as exc:
        raise ValueError( "save_reason must be best_macro_f1, best_val_loss, or last" ) from exc


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    val_loss: Optional[float],
    val_f1_macro: Optional[float],
    model_name: str,
    exp_id: str,
    mix_mode: str,
    label_mode: str,
    label_map: dict,
    train_csv: str,
    save_reason: str,
    round_num: int = None,
    timestamp: str = None,
    run_seed: Optional[int] = None,
    run_dir: Optional[str] = None,
    model_key: Optional[str] = None,
    model_kwargs: Optional[dict] = None,
    resolved_config: Optional[dict] = None,
    config_hash: Optional[str] = None,
    data_manifest_hash: Optional[str] = None,
    generation_seeds: Optional[list] = None,
    git_info: Optional[dict] = None,
    scaler=None,
    scheduler=None,
    best_state: Optional[dict] = None,
) -> Tuple[str, int, str, str]:
    del round_num
    if run_dir is None:
        run_dir = _legacy_run_dir(exp_id, model_name, run_seed)
    run_dir = os.path.abspath(run_dir)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    timestamp = timestamp or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    filename = _checkpoint_filename(save_reason)
    path = os.path.join(checkpoint_dir, filename)
    canonical_key = canonical_model_key(model_key or model_name)

    payload = {
        "schema_version": RUN_SCHEMA_VERSION,
        "model_key": canonical_key,
        "model_name": model_name,
        "model_kwargs": dict(model_kwargs or {}),
        "exp_id": exp_id,
        "mix_mode": mix_mode,
        "label_mode": label_mode,
        "label_map": dict(label_map),
        "save_reason": save_reason,
        "epoch": int(epoch),
        "val_loss": None if val_loss is None else float(val_loss),
        "val_f1_macro": None if val_f1_macro is None else float(val_f1_macro),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "rng_state": capture_rng_state(),
        "train_csv": os.path.abspath(str(train_csv)),
        "training_seed": run_seed,
        "generation_seeds": list(generation_seeds or []),
        "resolved_config": resolved_config,
        "config_hash": config_hash,
        "data_manifest_hash": data_manifest_hash,
        "git": dict(git_info or {}),
        "best_state": dict(best_state or {}),
        "saved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    torch.save(payload, path)

    print("[ckpt] {:16s} -> {}".format(save_reason, path))
    stem = os.path.splitext(filename)[0]
    return path, 0, timestamp, stem


def get_latest_checkpoint(
    model_name: str,
    exp_id: str,
    save_reason: str = "best_f1",
    run_dir: Optional[str] = None,
    seed: int = 42,
    model_key: Optional[str] = None,
) -> Optional[str]:
    try:
        if run_dir is None:
            run_dir = find_latest_run( exp_name=exp_id, model_key=model_key or model_name, seed=seed, )
        path = run_checkpoint_path(run_dir, save_reason)
    except (FileNotFoundError, ValueError) as exc:
        print("[ckpt] {}".format(exc))
        return None
    print("[ckpt] loading -> {}".format(path))
    return path


def get_consistent_label_map(
    train_csv: str,
    label_mode: str = "label",
    expected_labels: Optional[List[str]] = None,
) -> dict:
    frame = pd.read_csv(train_csv)
    frame = set_label_mode(frame, label_mode=label_mode)
    observed = set(frame["label"].fillna("Noise").astype(str).unique())
    observed.add("Noise")

    if expected_labels is None:
        ordered = ["Noise"] + sorted(label for label in observed if label != "Noise")
    else:
        ordered = list(dict.fromkeys(map(str, expected_labels)))
        if "Noise" not in ordered:
            ordered.insert(0, "Noise")

        expected = set(ordered)
        unexpected = sorted(observed - expected)
        missing = sorted(expected - observed)
        if unexpected:
            raise ValueError(
                "Training metadata contains unexpected labels for mode={}: {}".format(
                    label_mode, unexpected
                )
            )
        if missing:
            raise ValueError(
                "Training metadata is missing required labels for mode={}: {}".format(
                    label_mode, missing
                )
            )

    label_map = {label: index for index, label in enumerate(ordered)}
    print("[label_map] mode={} map={}".format(label_mode, label_map))
    return label_map


def validate_metadata_labels(metadata_csv: str,label_mode: str,label_map: dict) -> None:
    frame = pd.read_csv(metadata_csv)
    frame = set_label_mode(frame, label_mode=label_mode)
    labels = set(frame["label"].fillna("Noise").astype(str).unique())
    unknown = sorted(labels - set(label_map))
    if unknown:
        raise ValueError(
            "Metadata {} contains labels absent from the training map: {}".format(
                metadata_csv, unknown
            )
        )


def get_tb_log_dir( model_name: str, exp_id: str, ckpt_stem: str = None, run_dir: Optional[str] = None, ) -> str:
    if run_dir is not None:
        path = os.path.join(os.path.abspath(run_dir), "tensorboard")
    else:
        sub_dir = ckpt_stem or time.strftime("%Y%m%d_%H%M")
        path = os.path.join(config.OUTPUT_BASE, exp_id, "runs", model_name, sub_dir)
    os.makedirs(path, exist_ok=True)
    return path


def load_model(
    model_path: str,
    n_timesteps: int,
    n_outputs: int,
    device,
    model_name: str = None,
    strict: bool = True,
):
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not found: {}".format(model_path))

    from model import Mosbeatnet, MosqPlusModel, SEDNetSegmentLevel, SEDNet_segment

    if model_name and "Mosbeatnet" in model_name:
        model = Mosbeatnet(n_timesteps=n_timesteps, n_outputs=n_outputs).to(device)
    elif model_name and "MosqPlusModel" in model_name:
        model = MosqPlusModel(n_timesteps=n_timesteps, n_outputs=n_outputs).to(device)
    elif model_name == "SEDNet":
        model = SEDNet_segment(
            sr=getattr(config, "SR", 8000),
            segment_sec=float(n_timesteps) / float(getattr(config, "SR", 8000)),
            n_classes=n_outputs,
        ).to(device)
    elif model_name and "SEDNetSegmentLevel" in model_name:
        model = SEDNetSegmentLevel(
            sr=getattr(config, "SR", 8000),
            hop_len=getattr(config, "HOP", 512),
            n_classes=n_outputs,
        ).to(device)
    else:
        raise ValueError("Unknown model name: {}".format(model_name))

    checkpoint = torch.load(model_path, map_location=device)
    result = model.load_state_dict(checkpoint["model_state"], strict=bool(strict))

    if not strict:
        if result.missing_keys:
            print("[load_model] Missing keys: {}".format(result.missing_keys))
        if result.unexpected_keys:
            print("[load_model] Unexpected keys: {}".format(result.unexpected_keys))

    model.to(device)
    print(
        "[load_model] '{}' loaded from {} (epoch={} reason={})".format(
            model_name,
            model_path,
            checkpoint.get("epoch", "?"),
            checkpoint.get("save_reason", "?"),
        )
    )
    return model


# -----------------------------------------------------------------------------
# DataLoader helpers
# -----------------------------------------------------------------------------


def _make_worker_init(run_seed):
    def seed_worker(worker_id):
        # Avoid every worker starting its own CPU thread pool.
        torch.set_num_threads(1)

        base_seed = int(run_seed) if run_seed is not None else int(torch.initial_seed())
        seed = (base_seed + int(worker_id)) % (2 ** 32)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    return seed_worker


def _build_weighted_file_sampler(dataset, generator=None):
    class_counts = Counter()

    for file_name in dataset.file_names:
        labels, _, _ = dataset.label_cache[file_name]
        class_counts.update(labels.tolist())

    inverse_map = {value: key for key, value in dataset.label_map.items()}
    total = sum(class_counts.values())

    print("[Sampler] Segment-level class distribution:")
    for class_index in sorted(class_counts):
        class_name = inverse_map.get(class_index, str(class_index))
        count = class_counts[class_index]
        percent = 100.0 * count / total if total else 0.0
        print("  {:>22s} : {:7d} ({:.1f}%)".format(class_name, count, percent))

    class_weight = { class_index: 1.0 / max(count, 1) for class_index, count in class_counts.items() }

    file_weights = []
    for file_name in dataset.file_names:
        labels, _, _ = dataset.label_cache[file_name]
        weights = [class_weight[int(value)] for value in labels.tolist()]
        file_weights.append(float(np.mean(weights)))

    sample_weight = torch.tensor(file_weights, dtype=torch.double)

    return WeightedRandomSampler(
        sample_weight,
        num_samples=len(sample_weight),
        replacement=True,
        generator=generator,
    )


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------


def train_model(
    model,
    optimizer,
    criterion,
    train_csv,
    val_csv,
    simulation_dir,
    device,
    label_map,
    epochs: int = 100,
    batch_size: int = 32,
    regenerate_every: int = 20,
    patience: Optional[int] = 20,
    min_delta: float = 1e-4,
    early_stop_metric: str = "val_f1_macro",
    early_stop_start_epoch: int = 6,
    model_name: str = None,
    normalize_audio: bool = True,
    target_rms: float = 0.10,
    max_peak: float = 0.99,
    label_mode: str = "label",
    segment_duration: float = 0.5,
    exp_id: str = "exp",
    mix_mode: str = "weighted",
    num_workers: int = 4,
    prefetch_factor: int = 4,
    run_seed: Optional[int] = None,
    cache_segment_tensors: bool = False,
    use_disk_segment_cache: bool = False,
    rebuild_segment_cache: bool = False,
    eval_every: int = 1,
    audio_cache_max: int = 32,
    use_weighted_sampler: bool = False,
    gradient_clip: float = 5.0,
    persistent_workers: bool = True,
    log_interval: int = 20,
    debug_checks: bool = False,
    seg_dir: Optional[str] = None,
    run_dir: Optional[str] = None,
    model_key: Optional[str] = None,
    model_kwargs: Optional[dict] = None,
    resolved_config: Optional[dict] = None,
    config_hash: Optional[str] = None,
    data_manifest_hash: Optional[str] = None,
    generation_seeds: Optional[list] = None,
    git_info: Optional[dict] = None,
    deterministic: bool = False,
    scheduler=None,
):
    device = torch.device(device)

    if early_stop_metric not in {"val_loss", "val_f1_macro"}:
        raise ValueError("early_stop_metric must be 'val_loss' or 'val_f1_macro'")

    if patience is not None:
        patience = int(patience)
        if patience < 0:
            raise ValueError("patience must be >= 0 or None")

    min_delta = max(0.0, float(min_delta))
    early_stop_start_epoch = max(1, int(early_stop_start_epoch))
    early_stop_enabled = patience is not None and patience > 0

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    if cache_segment_tensors and num_workers > 0:
        print("[DataLoader] RAM segment cache enabled, changing num_workers to 0")
        num_workers = 0

    num_workers = max(0, int(num_workers))
    prefetch_factor = max(1, int(prefetch_factor))
    log_interval = max(1, int(log_interval))
    eval_every = max(1, int(eval_every))

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = not bool(deterministic)
        torch.backends.cudnn.deterministic = bool(deterministic)
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    model = model.to(device)
    model_class = model_name or model.__class__.__name__
    canonical_key = canonical_model_key(model_key or model_class)

    if run_dir is None:
        run_dir = _legacy_run_dir(exp_id, canonical_key, run_seed)

    run_dir = os.path.abspath(run_dir)

    metrics_dir = os.path.join(run_dir, "metrics")
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(
        "\n[Training] exp={} model={} key={} seed={} deterministic={}".format(
            exp_id,
            model_class,
            canonical_key,
            run_seed,
            bool(deterministic),
        )
    )
    print("[Training] run_dir={}".format(run_dir))
    print("[device] requested: {}".format(device))
    print(
        "[DataLoader] workers={} prefetch={} audio_cache={} disk_cache={} weighted_sampler={}".format(
            num_workers,
            prefetch_factor if num_workers > 0 else 0,
            audio_cache_max,
            use_disk_segment_cache,
            use_weighted_sampler,
        )
    )

    if early_stop_enabled:
        print(
            "[EarlyStopping] metric={} patience={} min_delta={} start_epoch={}".format(
                early_stop_metric,
                patience,
                min_delta,
                early_stop_start_epoch,
            )
        )
    else:
        print("[EarlyStopping] disabled")

    log_dir = get_tb_log_dir(model_class, exp_id, run_dir=run_dir)
    writer = SummaryWriter(log_dir=log_dir)
    print("[TensorBoard] {}".format(log_dir))

    dataset_kwargs = dict(
        audio_dir=simulation_dir,
        root_dir=getattr(config, "SIM_DATA_DIR", None),
        segment_duration=segment_duration,
        label_map=label_map,
        save_metadata=False,
        prefer_simfile_path=True,
        normalize_audio=normalize_audio,
        target_rms=target_rms,
        max_peak=max_peak,
        label_mode=label_mode,
        cache_segment_tensors=cache_segment_tensors,
        use_disk_segment_cache=use_disk_segment_cache,
        rebuild_segment_cache=rebuild_segment_cache,
        audio_cache_max=audio_cache_max,
        validate_audio=True,
    )

    train_dataset = AudioSequenceDataset(
        metadata_path=train_csv,
        augmentor=None,
        dataset_type="train",
        env_type=getattr(config, "ENV_SCOPE", None),
        seg_dir=seg_dir or os.path.join(run_dir, "segment_cache"),
        **dataset_kwargs
    )

    val_dataset = AudioSequenceDataset(
        metadata_path=val_csv,
        augmentor=None,
        dataset_type="val",
        env_type=getattr(config, "ENV_SCOPE", None),
        seg_dir=seg_dir or os.path.join(run_dir, "segment_cache"),
        **dataset_kwargs
    )

    sampler_generator = torch.Generator()
    train_loader_generator = torch.Generator()
    val_loader_generator = torch.Generator()

    base_seed = int(run_seed or 0)
    sampler_generator.manual_seed(base_seed)
    train_loader_generator.manual_seed(base_seed + 1000)
    val_loader_generator.manual_seed(base_seed + 2000)

    worker_init = _make_worker_init(run_seed)

    sampler = ( _build_weighted_file_sampler(train_dataset, sampler_generator) if use_weighted_sampler else None )

    def make_loader(dataset, sampler_obj=None, shuffle=False, generator=None):
        kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "sampler": sampler_obj,
            "shuffle": False if sampler_obj is not None else shuffle,
            "num_workers": num_workers,
            "pin_memory": device.type == "cuda",
            "worker_init_fn": worker_init,
            "generator": generator,
        }

        if num_workers > 0:
            kwargs["persistent_workers"] = bool(persistent_workers)
            kwargs["prefetch_factor"] = prefetch_factor

        return DataLoader(**kwargs)

    train_loader = make_loader(train_dataset, sampler_obj=sampler, shuffle=sampler is None, generator=train_loader_generator)

    val_loader = make_loader(val_dataset,shuffle=False,generator=val_loader_generator)

    use_amp = device.type == "cuda"

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    checkpoint_kwargs = dict(
        model_name=model_class,
        model_key=canonical_key,
        model_kwargs=dict(model_kwargs or {}),
        exp_id=exp_id,
        mix_mode=mix_mode,
        label_mode=label_mode,
        label_map=label_map,
        train_csv=str(train_csv),
        run_seed=run_seed,
        run_dir=run_dir,
        resolved_config=resolved_config,
        config_hash=config_hash,
        data_manifest_hash=data_manifest_hash,
        generation_seeds=list(generation_seeds or []),
        git_info=dict(git_info or {}),
        scaler=scaler,
        scheduler=scheduler,
    )

    best_val_loss = float("inf")
    best_val_f1 = -float("inf")
    best_loss_epoch = None
    best_f1_epoch = None
    best_f1_path = None

    best_stop_value = ( float("inf") if early_stop_metric == "val_loss" else -float("inf") )

    checks_no_improve = 0
    history = []

    history_path = os.path.join(metrics_dir, "history.csv")
    best_metrics_path = os.path.join(metrics_dir, "best_metrics.json")

    try:
        for epoch in range(int(epochs)):
            epoch_number = epoch + 1

            print("\nEpoch {}/{}".format(epoch_number, epochs))
            epoch_start = time.perf_counter()

            past_warmup = epoch >= 5

            should_refresh = (
                past_warmup
                and int(regenerate_every) > 0
                and ((epoch - 5) % int(regenerate_every) == 0)
            )

            if should_refresh:
                action = "Enabling" if epoch == 5 else "Refreshing"
                print("[AUG] {} augmentor mix_mode={}".format(action, mix_mode))

                train_dataset.refresh_augmentor(
                    AudioAugmentor(
                        alpha_range=(0.7, 1.3),
                        noise_level_range=(0.001, 0.01),
                        mix_mode=mix_mode,
                        p_pitch=0.0,
                        p_stretch=0.0,
                    )
                )

                train_loader = make_loader(
                    train_dataset,
                    sampler_obj=sampler,
                    shuffle=sampler is None,
                    generator=train_loader_generator,
                )

            model.train()

            train_loss_sum = torch.zeros((),device=device,dtype=torch.float64)

            correct = torch.zeros((),device=device,dtype=torch.long)

            item_count = 0

            progress = tqdm(
                train_loader,
                desc="Epoch {}/{} [train]".format(epoch_number, epochs),
                unit="batch",
                dynamic_ncols=True,
                leave=False,
            )

            for batch_index, batch in enumerate(progress):
                inputs, targets = batch[:2]

                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                amp_context = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if use_amp
                    else nullcontext()
                )

                with amp_context:
                    outputs = model(inputs)
                    logits = outputs.reshape(-1, outputs.shape[-1])
                    target_flat = targets.reshape(-1)
                    loss = criterion(logits, target_flat)

                if debug_checks and not torch.isfinite(loss).item():
                    raise FloatingPointError("Training loss contains NaN or Inf")

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                grad_norm = torch.nn.utils.clip_grad_norm_( model.parameters(), max_norm=float(gradient_clip), )

                if debug_checks and not torch.isfinite(grad_norm).item():
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError("Gradient norm contains NaN or Inf")

                scaler.step(optimizer)
                scaler.update()

                if scheduler is not None:
                    scheduler.step()

                batch_items = target_flat.numel()
                train_loss_sum += loss.detach().double() * batch_items
                correct += (logits.detach().argmax(dim=-1) == target_flat).sum()
                item_count += batch_items

                if ( (batch_index + 1) % log_interval == 0 or (batch_index + 1) == len(train_loader) ):
                    running_loss = (train_loss_sum / max(item_count, 1)).item()
                    running_acc = (correct.float() / max(item_count, 1)).item()

                    progress.set_postfix( loss="{:.4f}".format(running_loss), acc="{:.4f}".format(running_acc), )

            train_loss = (train_loss_sum / max(item_count, 1)).item()
            train_acc = (correct.float() / max(item_count, 1)).item()
            epoch_seconds = time.perf_counter() - epoch_start
            files_per_second = len(train_dataset) / max(epoch_seconds, 1e-9)

            writer.add_scalar("Loss/Train", train_loss, epoch_number)
            writer.add_scalar("Accuracy/Train", train_acc, epoch_number)
            writer.add_scalar( "Performance/EpochSeconds", epoch_seconds, epoch_number, )

            val_loss = None
            val_acc = None
            val_f1_macro = None
            val_f1_macro_all = None
            val_f1_weighted = None

            should_eval = (epoch_number % eval_every == 0 or epoch_number == int(epochs))

            if should_eval:
                val_metrics = evaluate_model(
                    model,
                    criterion,
                    val_loader,
                    device,
                    label_map=label_map,
                    print_report=True,
                    debug_checks=debug_checks,
                )

                val_loss = float(val_metrics["loss"])
                val_acc = float(val_metrics["acc"])
                val_f1_macro = float(val_metrics["f1_macro"])
                val_f1_macro_all = float(val_metrics["f1_macro_all"])
                val_f1_weighted = float(val_metrics["f1_weighted"])

                writer.add_scalar("Loss/Validation", val_loss, epoch_number)
                writer.add_scalar("Accuracy/Validation", val_acc, epoch_number)
                writer.add_scalar("F1/Macro", val_f1_macro, epoch_number)
                writer.add_scalar("F1/MacroAll", val_f1_macro_all, epoch_number)
                writer.add_scalar("F1/Weighted", val_f1_weighted, epoch_number)

                # Early stopping tracks the best value across all validation
                # checks, including the warm-up period. Counting no-improvement
                # starts only at early_stop_start_epoch.
                if early_stop_enabled:
                    current_stop_value = ( val_loss if early_stop_metric == "val_loss" else val_f1_macro )

                    if early_stop_metric == "val_loss":
                        stop_improved = ( current_stop_value < best_stop_value - min_delta )
                    else:
                        stop_improved = ( current_stop_value > best_stop_value + min_delta )

                    if stop_improved:
                        best_stop_value = current_stop_value
                        if epoch_number >= early_stop_start_epoch:
                            checks_no_improve = 0
                        print( "[EarlyStopping] {} improved to {:.6f}".format( early_stop_metric, best_stop_value, ) )
                    elif epoch_number >= early_stop_start_epoch:
                        checks_no_improve += 1
                        print(
                            "[EarlyStopping] no {} improvement: {}/{} | "
                            "current={:.6f} best={:.6f}".format(
                                early_stop_metric,
                                checks_no_improve,
                                patience,
                                current_stop_value,
                                best_stop_value,
                            )
                        )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_loss_epoch = epoch_number

                    save_checkpoint(
                        model,
                        optimizer,
                        epoch=epoch_number,
                        val_loss=val_loss,
                        val_f1_macro=val_f1_macro,
                        save_reason="best_val_loss",
                        best_state={
                            "best_val_loss": best_val_loss,
                            "best_val_f1_macro": best_val_f1,
                            "best_loss_epoch": best_loss_epoch,
                            "best_f1_epoch": best_f1_epoch,
                        },
                        **checkpoint_kwargs
                    )

                if val_f1_macro > best_val_f1:
                    best_val_f1 = val_f1_macro
                    best_f1_epoch = epoch_number

                    best_f1_path, _, _, _ = save_checkpoint(
                        model,
                        optimizer,
                        epoch=epoch_number,
                        val_loss=val_loss,
                        val_f1_macro=val_f1_macro,
                        save_reason="best_macro_f1",
                        best_state={
                            "best_val_loss": best_val_loss,
                            "best_val_f1_macro": best_val_f1,
                            "best_loss_epoch": best_loss_epoch,
                            "best_f1_epoch": best_f1_epoch,
                        },
                        **checkpoint_kwargs
                    )

            history.append({
                "epoch": epoch_number,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_f1_macro": val_f1_macro,
                "val_f1_macro_all": val_f1_macro_all,
                "val_f1_weighted": val_f1_weighted,
                "epoch_seconds": epoch_seconds,
                "files_per_second": files_per_second,
                "learning_rate": "{:.2e}".format(
                    float(optimizer.param_groups[0].get("lr", 0.0))
                ),
                "early_stop_no_improve": checks_no_improve,
            })

            pd.DataFrame(history).to_csv(
                history_path,
                index=False,
                float_format=getattr(config, "REPORT_FLOAT_FORMAT", "%.4f"),
            )

            saved_stop_value = ( None if best_stop_value in {float("inf"), -float("inf")} else best_stop_value )

            best_payload = {
                "best_val_loss": (
                    None
                    if best_val_loss == float("inf")
                    else best_val_loss
                ),
                "best_val_loss_epoch": best_loss_epoch,
                "best_val_f1_macro": (
                    None
                    if best_val_f1 == -float("inf")
                    else best_val_f1
                ),
                "best_val_f1_macro_epoch": best_f1_epoch,
                "last_val_f1_macro_all": val_f1_macro_all,
                "early_stop_metric": early_stop_metric,
                "early_stop_best_value": saved_stop_value,
                "early_stop_no_improve_checks": checks_no_improve,
                "early_stop_start_epoch": early_stop_start_epoch,
                "patience": patience,
                "min_delta": min_delta,
                "best_macro_f1_checkpoint": best_f1_path,
                "best_val_loss_checkpoint": os.path.join(
                    checkpoint_dir,
                    "best_val_loss.pt",
                ),
                "last_checkpoint": os.path.join(
                    checkpoint_dir,
                    "last.pt",
                ),
            }

            with open(best_metrics_path,"w",encoding="utf-8") as handle:

                json.dump(best_payload,handle,indent=2,sort_keys=True)

            save_checkpoint(
                model,
                optimizer,
                epoch=epoch_number,
                val_loss=val_loss,
                val_f1_macro=val_f1_macro,
                save_reason="last",
                best_state={
                    "best_val_loss": best_payload["best_val_loss"],
                    "best_val_f1_macro": best_payload["best_val_f1_macro"],
                    "best_loss_epoch": best_loss_epoch,
                    "best_f1_epoch": best_f1_epoch,
                },
                **checkpoint_kwargs
            )

            if should_eval:
                print(
                    "Duration: {:.1f}s | Train Loss: {:.4f} | Train Acc: {:.4f} | "
                    "Val Loss: {:.4f} | Val Acc: {:.4f} | Val F1-macro: {:.4f} | "
                    "Val F1-macro-all: {:.4f} | Val F1-weighted: {:.4f}".format(
                        epoch_seconds,
                        train_loss,
                        train_acc,
                        val_loss,
                        val_acc,
                        val_f1_macro,
                        val_f1_macro_all,
                        val_f1_weighted,
                    )
                )
            else:
                print(
                    "Duration: {:.1f}s | Train Loss: {:.4f} | Train Acc: {:.4f} | "
                    "Val: skipped (eval_every={})".format(
                        epoch_seconds,
                        train_loss,
                        train_acc,
                        eval_every,
                    )
                )

            should_stop = (
                early_stop_enabled
                and should_eval
                and epoch_number >= early_stop_start_epoch
                and checks_no_improve >= patience
            )

            if should_stop:
                print(
                    "[EarlyStopping] stopped at epoch {} | no {} improvement "
                    "for {} validation checks".format(
                        epoch_number,
                        early_stop_metric,
                        patience,
                    )
                )
                break

    finally:
        writer.close()

    if best_f1_path is None:
        last_path = os.path.join(checkpoint_dir, "last.pt")

        if os.path.isfile(last_path):
            best_f1_path = last_path

    return model, best_f1_path
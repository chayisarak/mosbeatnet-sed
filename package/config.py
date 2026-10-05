from __future__ import annotations

import os


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

CURR_DIR = os.path.abspath(os.path.dirname(__file__))
PROJ_DIR = (
    CURR_DIR
    if os.path.isdir(os.path.join(CURR_DIR, "data"))
    else os.path.abspath(os.path.join(CURR_DIR, ".."))
)

DATA_DIR = os.path.abspath(
    os.environ.get("MOSQUI_DATA_DIR", os.path.join(PROJ_DIR, "data"))
)
OUTPUT_BASE = os.path.abspath(
    os.environ.get("MOSQUI_OUTPUT_DIR", os.path.join(PROJ_DIR, "output"))
)
SIM_DATA_DIR = os.path.join(DATA_DIR, "simulated_data")

NOISE_RAW_DIR = os.path.abspath(
    os.environ.get(
        "MOSQUI_NOISE_DIR",
        os.path.join(DATA_DIR, "Environmental_Noise"),
    )
)

RAW_MIRU_DIR = os.path.abspath(
    os.environ.get("MOSQUI_RAW_MIRU_DIR", DATA_DIR)
)

# Original BioDCASE development dataset.
# No BioDCASE raw audio is copied.
BIODCASE_RAW_DIR = os.path.abspath(
    os.environ.get(
        "MOSQUI_BIODCASE_DIR",
        os.path.join(DATA_DIR, "BioDCASE"),
    )
)


# -----------------------------------------------------------------------------
# Sources
# -----------------------------------------------------------------------------

INDOOR_SOURCE = "indoor_miru"
OUTDOOR_SOURCE = "outdoor_miru"
HUMBUG_SOURCE = "humbug_miru"
BIODCASE_SOURCE = "biodcase"

MOSQUITO_SOURCES = (
    INDOOR_SOURCE,
    OUTDOOR_SOURCE,
    HUMBUG_SOURCE,
    BIODCASE_SOURCE,
)

MIRU_SOURCES = (
    INDOOR_SOURCE,
    OUTDOOR_SOURCE,
    HUMBUG_SOURCE,
)

SRC_MIRU_SOURCES = MIRU_SOURCES

RAW_SOURCE_DIRS = {
    INDOOR_SOURCE: os.path.join(RAW_MIRU_DIR, INDOOR_SOURCE),
    OUTDOOR_SOURCE: os.path.join(RAW_MIRU_DIR, OUTDOOR_SOURCE),
    HUMBUG_SOURCE: os.path.join(RAW_MIRU_DIR, HUMBUG_SOURCE),
    BIODCASE_SOURCE: BIODCASE_RAW_DIR,
}

SRC_MIRU_SOURCE_DIRS = {
    source: RAW_SOURCE_DIRS[source]
    for source in MIRU_SOURCES
}


# -----------------------------------------------------------------------------
# Generated dataset
# -----------------------------------------------------------------------------

GENERATED_DATASET_DIR = os.path.join(SIM_DATA_DIR, "gen_xdomain")
GENERATED_METADATA_DIR = os.path.join(GENERATED_DATASET_DIR, "metadata")


# -----------------------------------------------------------------------------
# Audio
# -----------------------------------------------------------------------------

SR = 8000
SAMPLING_RATE = SR
HOP = 128
SEG_LENGTH = 0.5
AUDIO_DURATION = 10.0

SEDNET_CFG = {
    "n_fft": 500,
    "hop_len": 500,
    "n_mels": 40,
    "dropout_rate": 0.5,
}


# -----------------------------------------------------------------------------
# Dataset settings
# -----------------------------------------------------------------------------

SPLITS = ("train", "val", "test")
ENVS = ("urban", "forest")

SPECIES_ORDER = [
    "Ae.Aegypti",
    "Ae.Albopictus",
    "An.Dirus",
    "Cx.Quin",
]

SPECIES_LABEL_ORDER = ["Noise", *SPECIES_ORDER]

SPECIES_SEX_LABEL_ORDER = [
    "Noise",
    *[
        f"{species}_{sex}"
        for species in SPECIES_ORDER
        for sex in ("F", "M")
    ],
]

OUTDOOR_MISSING_LABELS = ("Cx.Quin_F",)

SPECIES_ALIASES = {
    "ae.aegypti": "Ae.Aegypti",
    "aedes.aegypti": "Ae.Aegypti",
    "aedes_aegypti": "Ae.Aegypti",
    "aeaegypti": "Ae.Aegypti",

    "ae.albopictus": "Ae.Albopictus",
    "aedes.albopictus": "Ae.Albopictus",
    "aedes_albopictus": "Ae.Albopictus",
    "aealbopictus": "Ae.Albopictus",

    "an.dirus": "An.Dirus",
    "anopheles.dirus": "An.Dirus",
    "anopheles_dirus": "An.Dirus",
    "andirus": "An.Dirus",

    "cx.quin": "Cx.Quin",
    "cx.quinquefasciatus": "Cx.Quin",
    "culex.quinquefasciatus": "Cx.Quin",
    "culex_quinquefasciatus": "Cx.Quin",
    "cxquin": "Cx.Quin",
}


# -----------------------------------------------------------------------------
# BioDCASE
# -----------------------------------------------------------------------------

# BioDCASE filename species IDs:
# S_1 = Ae.Aegypti
# S_2 = Ae.Albopictus
# S_3 = Cx.Quin
# S_6 = An.Dirus
BIODCASE_SPECIES_IDS = {
    1: "Ae.Aegypti",
    2: "Ae.Albopictus",
    3: "Cx.Quin",
    6: "An.Dirus",
}

BIODCASE_SPLIT_FILES = {
    "train": "Training_ids.txt",
    "val": "Validation_ids.txt",
    "test": "Test_ids.txt",
}

BIODCASE_TRAINVAL_FILE = "TrainVal_ids.txt"


# -----------------------------------------------------------------------------
# Split settings
# -----------------------------------------------------------------------------

VAL_TEST_VAL_RATIO = 0.50
RAW_SPLIT_SEED = 42

N_SIMS_PER_SOURCE = {
    "train": 10500,
    "val": 2250,
    "test": 2250,
}

BIODCASE_N_SIMS = {
    "train": 35000,
    "val": 6000,
    "test": 6000,
}

NOISE_ONLY_TEST = 500


# -----------------------------------------------------------------------------
# Background settings
# -----------------------------------------------------------------------------

NOISE_ENV_FOLDER = {
    "urban": "Env_Urban",
    "forest": "Env_Forest",
}

# Every target species can be simulated in both real backgrounds.
ENVIRONMENTS = {
    env: {
        "noise_subdir": folder,
        "mosquito_species": SPECIES_ORDER.copy(),
        "noise_target_rms_dbfs": -30.0,
        "snr_range": (-5.0, 10.0),
    }
    for env, folder in NOISE_ENV_FOLDER.items()
}

HB_ENVIRONMENTS = ENVIRONMENTS

# Gaussian is a background type, not an environment.
GAUSSIAN_BACKGROUND = {
    "noise_target_rms_dbfs": -30.0,
    "snr_range": (-5.0, 10.0),
}


# -----------------------------------------------------------------------------
# Simulation
# -----------------------------------------------------------------------------

SIM_MIX_CFG = {
    "peak_limit": 0.98,
    "fade_sec": 0.020,
    "min_mos_events": 2,
    "max_mos_events": 5,
    "min_mos_event_sec": 0.55,
    "max_mos_event_sec": 1.50,
    "event_margin_sec": 0.05,
    "min_source_sec": 0.05,
    "embedded_background_sources": (),
    "embedded_background_mode": "gaussian",
    "max_retry_per_sim": 10,
    "wav_subtype": "PCM_16",
    "source_coverage_mode": "strict",
}


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

TRAIN_CFG = {
    "epochs": 100,
    "batch": 64,
    "patience": 10,
    "refresh_aug_every": 10,

    "use_norm": True,
    "rms_target": 0.10,
    "peak_limit": 0.99,

    "workers": 4,
    "prefetch": 2,
    "seg_sec": SEG_LENGTH,
    "persistent_workers": True,

    "learning_rate": 1e-4,
    "weight_decay": 0.0,

    "use_weighted_sampler": True,
    "use_disk_segment_cache": False,
}


# -----------------------------------------------------------------------------
# Evaluation / reporting
# -----------------------------------------------------------------------------

EVAL_CFG = {
    "iou_threshold": 0.3,
    "include_noise_in_sequence": False,
}

AUG_MODE = "weighted"
THESIS_TRAINING_SEEDS = (42, 123, 2026)

PATH_ANCHORS = ("simulated_data",)
ENV_SCOPE = "mixed"

REPORT_FLOAT_DIGITS = 4
REPORT_FLOAT_FORMAT = "%.4f"


# -----------------------------------------------------------------------------
# Model display names
# -----------------------------------------------------------------------------

MODEL_DISPLAY_NAMES = {
    "mosbeatnet_v1": "MosBeatNet-v1",
    "mosbeatnet_v2": "MosBeatNet-v2",
    "mosbeatnet_v2_se": "MosBeatNet-v2-SE",
    "mosbeatnet_v2_in": "MosBeatNet-v2-IN",

    "mosqplus": "MosqPlus",
    "sednet": "SEDNet",

    "cfresnet1d_small": "CFResNet1D-Small",
    "cfresnet1d_medium": "CFResNet1D-Medium",
    "cfresnet1d_large": "CFResNet1D-Large",

    "mtrcnn": "MTRCNN",
}


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------

REPRO_CFG = {
    "schema_version": 2,
    "strict_checkpoint_loading": True,
    "snapshot_code": True,
    "checkpoint_names": {
        "best_macro_f1": "best_macro_f1.pt",
        "best_val_loss": "best_val_loss.pt",
        "last": "last.pt",
    },
}


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------

viz_dir = os.path.join(PROJ_DIR, "visualization")
os.makedirs(viz_dir, exist_ok=True)
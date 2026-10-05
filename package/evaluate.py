
# evaluate.py
# =============================================================================
# IMPORTS AND CONSTANTS
# =============================================================================
from __future__ import annotations

import glob
import json
import os
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - sklearn normally installs scipy
    linear_sum_assignment = None


EPS = 1e-8
DEFAULT_IOU_THRESHOLD = 0.3
Event = Tuple[str, float, float, str]  # label, start, end, file_name

EVENT_METRIC_COLUMNS = [ "class", "Precision", "Recall", "F1-score", "Event Jaccard", "Support", "TP", "FP", "FN", ]

EVENT_MATCH_COLUMNS = [
    "file_name",
    "class",
    "true_start",
    "true_end",
    "pred_start",
    "pred_end",
    "iou",
    "overlap",
    "onset_error",
    "offset_error",
]

NOISE_LABELS = { "noise", "background", "bg", "-", "", "nan", "none", "na", "n/a", "<na>", }

ALL_DOMAIN_SPECIES = [ "Noise", "Ae.Aegypti", "Ae.Albopictus", "An.Dirus", "Cx.Quin", ]


# =============================================================================
# COMMON HELPERS AND VALIDATION
# =============================================================================


def ensure_dir(path: Optional[str]) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def is_noise_label(label: object) -> bool:
    return str(label).strip().casefold() in NOISE_LABELS


def to_detection_label(label: object) -> str:
    return "background" if is_noise_label(label) else "wingbeat"


def label_names_and_ids( label_dict: Dict[str, int], ) -> Tuple[List[str], List[int]]:
    pairs = sorted( ((str(name), int(class_id)) for name, class_id in label_dict.items()), key=lambda pair: pair[1], )
    return [name for name, _ in pairs], [class_id for _, class_id in pairs]


def id_to_name(label_dict: Dict[str, int]) -> Dict[int, str]:
    return { int(class_id): str(name) for name, class_id in label_dict.items() }


def validate_label_dict(label_dict: Dict[str, int]) -> None:
    if not label_dict:
        raise ValueError("label_dict is empty")

    names, ids = label_names_and_ids(label_dict)
    if len(set(names)) != len(names):
        raise ValueError("duplicate class name")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate class id")

    expected = list(range(len(ids)))
    if ids != expected:
        raise ValueError("class id is wrong")


def _as_int_vector(values, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        array = array.reshape(-1)
    if array.size == 0:
        return np.asarray([], dtype=np.int64)

    try:
        array = array.astype(np.int64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not int") from exc

    return array


def validate_prediction_arrays(
    y_pred,
    y_true,
    file_names: Sequence[str],
    label_dict: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    validate_label_dict(label_dict)

    y_pred = _as_int_vector(y_pred, "y_pred")
    y_true = _as_int_vector(y_true, "y_true")
    file_names = [str(value) for value in file_names]

    if not (len(y_pred) == len(y_true) == len(file_names)):
        raise ValueError("length not match")
    if len(y_true) == 0:
        raise ValueError("prediction arrays are empty")

    valid_ids = set(map(int, label_dict.values()))
    unknown_true = sorted(set(map(int, np.unique(y_true))) - valid_ids)
    unknown_pred = sorted(set(map(int, np.unique(y_pred))) - valid_ids)

    if unknown_true:
        raise ValueError(f"true id not found: {unknown_true}")
    if unknown_pred:
        raise ValueError(f"pred id not found: {unknown_pred}")

    return y_pred, y_true, file_names


def present_ordered_names( values, ordered_names: Sequence[str], ) -> List[str]:
    present = {str(value) for value in np.unique(values)}
    return [ str(name) for name in ordered_names if str(name) in present ]


def present_label_names_and_ids(y_true, label_dict: Dict[str, int]) -> Tuple[List[str], List[int]]:
    names, ids = label_names_and_ids(label_dict)
    present = {int(value) for value in np.unique(y_true)}
    pairs = [(name, class_id) for name, class_id in zip(names, ids) if class_id in present]
    if not pairs:
        return [], []
    return [name for name, _ in pairs], [class_id for _, class_id in pairs]


def value_from_metric_df( df: pd.DataFrame, metric_name: str, default: float = np.nan, ) -> float:
    if df is None or df.empty or "Metric" not in df.columns:
        return default

    rows = df.loc[df["Metric"] == metric_name, "Value"]
    if rows.empty:
        return default

    value = rows.iloc[0]
    return float(value) if pd.notna(value) else default


def safe_divide( numerator: float, denominator: float, default: float = np.nan, ) -> float:
    if denominator <= 0:
        return default
    return float(numerator) / float(denominator)


def _normalise_prediction_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    aliases = {
        "pred_label": "predicted_label",
        "prediction": "predicted_label",
        "target_label": "true_label",
        "filename": "file_name",
        "predicted_id": "pred_id",
        "prediction_id": "pred_id",
        "target_id": "true_id",
    }
    for old_name, new_name in aliases.items():
        if old_name in out.columns and new_name not in out.columns:
            out[new_name] = out[old_name]

    return out


def _normalise_file_key(value: object) -> str:
    return str(value).replace("\\", "/")


def _run_id_from_path(path: str) -> Optional[str]:
    for part in reversed(Path(path).parts):
        if part.startswith("run_"):
            return part
    return None


def save_class_support( prediction_df: pd.DataFrame, output_dir: str, label_name: str, ) -> str:
    df = _normalise_prediction_columns(prediction_df)
    if "true_label" not in df.columns:
        raise ValueError("true_label column not found")

    counts = df["true_label"].astype(str).value_counts(dropna=False)
    total = int(counts.sum())
    rows = []

    for class_name, count in counts.sort_index().items():
        rows.append({
            "Label": label_name,
            "Class": str(class_name),
            "N(segment)": int(count),
            "Proportion": safe_divide(int(count), total),
        })

    path = os.path.join(output_dir, "class_support.tsv")
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    return path


def save_noise_prediction_distribution(
    prediction_df: pd.DataFrame,
    output_dir: str,
    label_name: str,
) -> Optional[str]:
    df = _normalise_prediction_columns(prediction_df)
    required = {"true_label", "predicted_label"}
    if not required.issubset(df.columns):
        return None
    if df.empty or not df["true_label"].apply(is_noise_label).all():
        return None

    counts = df["predicted_label"].astype(str).value_counts(dropna=False)
    total = int(counts.sum())
    rows = []

    for class_name, count in counts.sort_index().items():
        rows.append({
            "Label": label_name,
            "Predicted Class": str(class_name),
            "N(segment)": int(count),
            "Proportion": safe_divide(int(count), total),
        })

    path = os.path.join(output_dir, "noise_prediction_distribution.tsv")
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    return path


# =============================================================================
# MODEL PREDICTIONS
# =============================================================================


def get_model_predictions(model, data_loader, device):
    """Return flattened predictions, targets and file names."""
    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_file_names: List[str] = []

    dataset = getattr(data_loader, "dataset", None)
    dataset_file_names = getattr(dataset, "file_names", None)
    offset = 0

    model.eval()
    device = torch.device(device)

    with torch.inference_mode():
        for batch in data_loader:
            if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                raise ValueError("batch format is wrong")

            inputs = batch[0].to(device, non_blocking=True)
            targets = batch[1]
            batch_size = int(inputs.shape[0])

            if len(batch) >= 3:
                batch_file_names = batch[2]
                if isinstance(batch_file_names, torch.Tensor):
                    batch_file_names = batch_file_names.detach().cpu().tolist()
                batch_file_names = [str(value) for value in batch_file_names]
            else:
                if dataset_file_names is None:
                    raise ValueError("file_names not found")
                batch_file_names = [ str(dataset_file_names[offset + index]) for index in range(batch_size) ]
                offset += batch_size

            if len(batch_file_names) != batch_size:
                raise ValueError("file_names length not match")

            outputs = model(inputs)
            if outputs.ndim == 2:
                outputs = outputs.unsqueeze(1)
            if outputs.ndim != 3:
                raise ValueError("model output shape is wrong")

            predictions = outputs.argmax(dim=-1).detach().cpu().numpy()
            targets_np = targets.detach().cpu().numpy()

            if targets_np.ndim == 1:
                targets_np = targets_np[:, None]

            if predictions.shape != targets_np.shape:
                raise ValueError("pred and true shape not match")

            all_preds.append(predictions.reshape(-1))
            all_labels.append(targets_np.reshape(-1))

            steps_per_file = int(targets_np.shape[1])
            for file_name in batch_file_names:
                all_file_names.extend([file_name] * steps_per_file)

    y_pred = (
        np.concatenate(all_preds).astype(np.int64, copy=False)
        if all_preds
        else np.asarray([], dtype=np.int64)
    )
    y_true = (
        np.concatenate(all_labels).astype(np.int64, copy=False)
        if all_labels
        else np.asarray([], dtype=np.int64)
    )

    if not (len(y_pred) == len(y_true) == len(all_file_names)):
        raise ValueError("output length not match")

    return y_pred, y_true, all_file_names


# =============================================================================
# SEGMENT CLASSIFICATION
# =============================================================================


def get_classwise_metrics( y_true, y_pred, label_dict: Dict[str, int], ) -> pd.DataFrame:
    """Return one-vs-rest metrics for every configured class."""
    y_pred, y_true, _ = validate_prediction_arrays(
        y_pred,
        y_true,
        ["_"] * len(np.asarray(y_true).reshape(-1)),
        label_dict,
    )

    label_names, labels = label_names_and_ids(label_dict)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = int(cm.sum())

    rows = []
    for index, class_id in enumerate(labels):
        class_name = label_names[index]
        tp = int(cm[index, index])
        fp = int(cm[:, index].sum() - tp)
        fn = int(cm[index, :].sum() - tp)
        tn = int(total - tp - fp - fn)
        support = int(cm[index, :].sum())
        predicted_support = int(cm[:, index].sum())

        precision_default = 0.0 if support > 0 else np.nan
        precision = safe_divide(tp, tp + fp, default=precision_default)
        recall = safe_divide(tp, tp + fn, default=np.nan)
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if pd.notna(recall) and precision + recall > 0
            else (0.0 if support > 0 else np.nan)
        )

        rows.append({
            "class": class_name,
            "class_id": class_id,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "Support": support,
            "Predicted Support": predicted_support,
            "Specificity": safe_divide(tn, tn + fp),
            "Class Accuracy": safe_divide(tp + tn, total),
            "Class IoU": safe_divide(tp, tp + fp + fn),
            "FPR": safe_divide(fp, fp + tn),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
        })

    return pd.DataFrame(rows)


def _segment_summary_from_classwise(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classwise_df: pd.DataFrame,
) -> pd.DataFrame:
    present = classwise_df["Support"] > 0
    present_df = classwise_df.loc[present].copy()

    if present_df.empty:
        raise ValueError("class not found in y_true")

    total_support = float(present_df["Support"].sum())

    weighted_precision = float(
        (
            present_df["Precision"].fillna(0.0)
            * present_df["Support"]
        ).sum()
        / max(total_support, EPS)
    )
    weighted_recall = float(
        (
            present_df["Recall"].fillna(0.0)
            * present_df["Support"]
        ).sum()
        / max(total_support, EPS)
    )
    weighted_f1 = float(
        (
            present_df["F1-score"].fillna(0.0)
            * present_df["Support"]
        ).sum()
        / max(total_support, EPS)
    )

    all_macro_precision = float(classwise_df["Precision"].fillna(0.0).mean())
    all_macro_recall = float(classwise_df["Recall"].fillna(0.0).mean())
    all_macro_f1 = float(classwise_df["F1-score"].fillna(0.0).mean())

    return pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Macro Precision",
            "Macro Recall",
            "Macro F1-score",
            "Balanced Accuracy",
            "Weighted Precision",
            "Weighted Recall",
            "Weighted F1-score",
            "Macro Precision (All Classes)",
            "Macro Recall (All Classes)",
            "Macro F1-score (All Classes)",
            "Present GT Classes",
            "Configured Classes",
        ],
        "Value": [
            accuracy_score(y_true, y_pred),
            float(present_df["Precision"].fillna(0.0).mean()),
            float(present_df["Recall"].fillna(0.0).mean()),
            float(present_df["F1-score"].fillna(0.0).mean()),
            float(present_df["Recall"].fillna(0.0).mean()),
            weighted_precision,
            weighted_recall,
            weighted_f1,
            all_macro_precision,
            all_macro_recall,
            all_macro_f1,
            int(present.sum()),
            int(len(classwise_df)),
        ],
    })


def save_confusion_matrix_png(
    matrix: np.ndarray,
    class_names: Sequence[str],
    output_path: str,
    normalized: bool = False,
) -> None:
    class_names = [str(name) for name in class_names]
    size = max(6.5, 0.75 * len(class_names) + 2.5)

    fig, ax = plt.subplots(figsize=(size, size))
    shown = np.nan_to_num(matrix.astype(float), nan=0.0)
    image = ax.imshow(shown, aspect="auto")
    fig.colorbar(image, ax=ax)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalized Confusion Matrix" if normalized else "Confusion Matrix")

    if len(class_names) <= 10:
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = matrix[row, col]
                if normalized and np.isnan(value):
                    label = ""
                elif normalized:
                    label = f"{value:.2f}"
                else:
                    label = str(int(value))
                if label:
                    ax.text(col, row, label, ha="center", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def evaluate_segment_classification(
    y_pred,
    y_true,
    label_dict: Dict[str, int],
    output_dir: Optional[str] = None,
    model_name: Optional[str] = None,
    environment: Optional[str] = None,
    save: bool = False,
):
    y_pred, y_true, _ = validate_prediction_arrays(
        y_pred,
        y_true,
        ["_"] * len(np.asarray(y_true).reshape(-1)),
        label_dict,
    )

    class_report_df = get_classwise_metrics( y_true=y_true, y_pred=y_pred, label_dict=label_dict, )
    results_df = _segment_summary_from_classwise( y_true=y_true, y_pred=y_pred, classwise_df=class_report_df, )

    label_names, labels = label_names_and_ids(label_dict)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true:{name}" for name in label_names],
        columns=[f"pred:{name}" for name in label_names],
    )

    row_sum = cm.sum(axis=1, keepdims=True)
    cm_normalized = np.divide(
        cm.astype(np.float64),
        row_sum,
        out=np.full_like(cm, np.nan, dtype=np.float64),
        where=row_sum > 0,
    )
    cm_normalized_df = pd.DataFrame( cm_normalized, index=cm_df.index, columns=cm_df.columns, )

    print("\n=== Segment-Level Classification ===")
    print(results_df.to_string(index=False))

    if save:
        ensure_dir(output_dir)
        prefix = f"{model_name or 'model'}_{environment or 'env'}"
        results_df.to_csv( os.path.join(output_dir, f"{prefix}_segment_results.csv"), index=False, )
        class_report_df.to_csv( os.path.join(output_dir, f"{prefix}_segment_class_report.csv"), index=False, )
        cm_df.to_csv( os.path.join(output_dir, f"{prefix}_confusion_matrix.csv") )
        cm_normalized_df.to_csv( os.path.join( output_dir, f"{prefix}_confusion_matrix_normalized.csv", ) )
        save_confusion_matrix_png(
            cm,
            label_names,
            os.path.join(output_dir, f"{prefix}_confusion_matrix.png"),
            normalized=False,
        )
        save_confusion_matrix_png(
            cm_normalized,
            label_names,
            os.path.join(output_dir, f"{prefix}_confusion_matrix_normalized.png"),
            normalized=True,
        )

    return results_df, class_report_df


# =============================================================================
# SEGMENT DETECTION
# =============================================================================


def evaluate_segment_detection(
    y_true,
    y_pred,
    label_dict: Dict[str, int],
    output_dir: Optional[str] = None,
    model_name: Optional[str] = None,
    environment: Optional[str] = None,
    save: bool = False,
):
    y_pred, y_true, _ = validate_prediction_arrays(
        y_pred,
        y_true,
        ["_"] * len(np.asarray(y_true).reshape(-1)),
        label_dict,
    )

    inverse = id_to_name(label_dict)
    y_true_binary = np.asarray([ to_detection_label(inverse[int(value)]) for value in y_true ])
    y_pred_binary = np.asarray([ to_detection_label(inverse[int(value)]) for value in y_pred ])

    binary_label_dict = { "background": 0, "wingbeat": 1, }
    name_to_id = binary_label_dict
    y_true_ids = np.asarray( [name_to_id[value] for value in y_true_binary], dtype=np.int64, )
    y_pred_ids = np.asarray( [name_to_id[value] for value in y_pred_binary], dtype=np.int64, )

    class_report_df = get_classwise_metrics( y_true=y_true_ids, y_pred=y_pred_ids, label_dict=binary_label_dict, )
    results_df = _segment_summary_from_classwise(
        y_true=y_true_ids,
        y_pred=y_pred_ids,
        classwise_df=class_report_df,
    )

    def class_value(class_name: str, metric: str) -> float:
        rows = class_report_df.loc[ class_report_df["class"] == class_name, metric, ]
        return ( float(rows.iloc[0]) if len(rows) and pd.notna(rows.iloc[0]) else np.nan )

    detail_df = pd.DataFrame({
        "Metric": [
            "Wingbeat Precision",
            "Wingbeat Recall",
            "Wingbeat F1-score",
            "Background Precision",
            "Background Recall",
            "Background F1-score",
        ],
        "Value": [
            class_value("wingbeat", "Precision"),
            class_value("wingbeat", "Recall"),
            class_value("wingbeat", "F1-score"),
            class_value("background", "Precision"),
            class_value("background", "Recall"),
            class_value("background", "F1-score"),
        ],
    })
    results_df = pd.concat( [results_df, detail_df], ignore_index=True, )

    report_df = class_report_df.rename(columns={
        "Precision": "precision",
        "Recall": "recall",
        "F1-score": "f1-score",
        "Support": "support",
    })

    print("\n=== Segment-Level Detection: Wingbeat vs Background ===")
    print(results_df.to_string(index=False))

    if save:
        ensure_dir(output_dir)
        prefix = f"{model_name or 'model'}_{environment or 'env'}_binary"
        results_df.to_csv( os.path.join(output_dir, f"{prefix}_segment_results.csv"), index=False, )
        report_df.to_csv( os.path.join( output_dir, f"{prefix}_segment_classification_report.csv", ), index=False, )

    return results_df, report_df


# =============================================================================
# PREDICTION METADATA AND CSV
# =============================================================================


def _metadata_by_file( metadata_csv_path: Optional[str], ) -> Dict[str, pd.DataFrame]:
    if metadata_csv_path is None:
        return {}

    metadata_csv_path = os.path.abspath(os.fspath(metadata_csv_path))
    if not os.path.exists(metadata_csv_path):
        warnings.warn( f"metadata file not found: {metadata_csv_path}", RuntimeWarning, )
        return {}

    metadata_df = pd.read_csv(metadata_csv_path)
    if "file_name" not in metadata_df.columns:
        warnings.warn( "file_name column not found", RuntimeWarning, )
        return {}

    metadata_df = metadata_df.copy()
    metadata_df["_file_key"] = ( metadata_df["file_name"] .astype(str) .map(_normalise_file_key) )

    return {
        str(file_key): group.reset_index(drop=True)
        for file_key, group in metadata_df.groupby("_file_key", sort=False)
    }


def _best_metadata_row_for_segment(
    file_metadata: pd.DataFrame,
    start_time: float,
    end_time: float,
) -> Optional[pd.Series]:
    if file_metadata is None or file_metadata.empty:
        return None

    if not {"start_time", "end_time"}.issubset(file_metadata.columns):
        return file_metadata.iloc[0]

    starts = pd.to_numeric( file_metadata["start_time"], errors="coerce", ).to_numpy(dtype=np.float64)
    ends = pd.to_numeric( file_metadata["end_time"], errors="coerce", ).to_numpy(dtype=np.float64)

    overlaps = np.maximum( 0.0, np.minimum(ends, end_time) - np.maximum(starts, start_time), )
    active = np.flatnonzero(overlaps > 0)
    if active.size == 0:
        return None

    labels = (
        file_metadata["label"].astype(str).to_numpy()
        if "label" in file_metadata.columns
        else np.asarray([""] * len(file_metadata))
    )

    overlap_by_label: Dict[str, float] = {}
    for row_index in active:
        label = str(labels[row_index])
        overlap_by_label[label] = ( overlap_by_label.get(label, 0.0) + float(overlaps[row_index]) )

    mosquito_overlap = {
        label: duration
        for label, duration in overlap_by_label.items()
        if not is_noise_label(label)
    }
    noise_overlap = sum( duration for label, duration in overlap_by_label.items() if is_noise_label(label) )

    # Mirror AudioSequenceDataset: mosquito wins only when its total overlap is
    # greater than Noise overlap. Otherwise select the best Noise row.
    if mosquito_overlap and sum(mosquito_overlap.values()) > noise_overlap:
        best_label_duration = max(mosquito_overlap.values())
        candidate_labels = {
            label
            for label, duration in mosquito_overlap.items()
            if np.isclose(duration, best_label_duration)
        }
        candidates = [ row_index for row_index in active if str(labels[row_index]) in candidate_labels ]
    else:
        candidates = [ row_index for row_index in active if is_noise_label(labels[row_index]) ]
        if not candidates:
            candidates = active.tolist()

    segment_mid = (float(start_time) + float(end_time)) / 2.0

    def rank(row_index: int):
        clipped_start = max(starts[row_index], start_time)
        clipped_end = min(ends[row_index], end_time)
        clipped_mid = (clipped_start + clipped_end) / 2.0
        return ( float(overlaps[row_index]), -abs(clipped_mid - segment_mid), -float(starts[row_index]), )

    best_index = max(candidates, key=rank)
    return file_metadata.iloc[int(best_index)]


def save_segment_predictions_csv(
    y_pred,
    y_true,
    file_names: Sequence[str],
    label_dict: Dict[str, int],
    output_dir: str,
    model_name: Optional[str] = None,
    environment: Optional[str] = None,
    segment_duration: float = 0.5,
    overlap: float = 0.0,
    metadata_csv_path: Optional[str] = None,
    save: bool = True,
):
    """Build fresh segment predictions and optionally overwrite the CSV."""
    y_pred, y_true, file_names = validate_prediction_arrays(
        y_pred=y_pred,
        y_true=y_true,
        file_names=file_names,
        label_dict=label_dict,
    )

    if save:
        ensure_dir(output_dir)

    segment_duration = float(segment_duration)
    overlap = float(overlap)
    step_size = segment_duration - overlap
    if segment_duration <= 0:
        raise ValueError("segment_duration must be more than 0")
    if step_size <= 0:
        raise ValueError("overlap is too large")

    inverse = id_to_name(label_dict)
    file_segment_counters: Dict[str, int] = {}
    metadata_map = _metadata_by_file(metadata_csv_path)

    context_columns = [
        "event_type",
        "species",
        "sex",
        "label",
        "snr",
        "target_snr",
        "real_snr",
        "source_name",
        "source_environment",
        "source_device",
        "simulation_environment",
        "mos_dataset",
        "mos_path",
        "raw_recording_id",
        "noise_file",
        "subenv",
    ]

    rows = []
    for pred_id, true_id, file_name in zip( y_pred, y_true, file_names, ):
        file_name = str(file_name)
        file_key = _normalise_file_key(file_name)
        segment_index = file_segment_counters.get(file_key, 0)
        file_segment_counters[file_key] = segment_index + 1

        start_time = float(segment_index) * step_size
        end_time = start_time + segment_duration

        row = {
            "model_name": model_name,
            "environment": environment,
            "file_name": file_name,
            "segment_index": int(segment_index),
            "start_time": round(start_time, 6),
            "end_time": round(end_time, 6),
            "true_id": int(true_id),
            "true_label": inverse[int(true_id)],
            "pred_id": int(pred_id),
            "predicted_label": inverse[int(pred_id)],
            "confidence": np.nan,
            "correct": bool(int(true_id) == int(pred_id)),
        }

        metadata_row = _best_metadata_row_for_segment(
            metadata_map.get(file_key),
            start_time=start_time,
            end_time=end_time,
        )
        for column in context_columns:
            value = ( metadata_row.get(column, np.nan) if metadata_row is not None else np.nan )
            row[f"metadata_{column}"] = value

        rows.append(row)

    prediction_df = pd.DataFrame(rows)
    output_path = os.path.join( os.path.abspath(output_dir), "segment_predictions.csv", )
    if save:
        prediction_df.to_csv(output_path, index=False)
        print(f"[eval] segment predictions -> {output_path}")

    return output_path if save else None, prediction_df


# =============================================================================
# LABEL MAPPING
# =============================================================================


SEX_SUFFIXES = { "f", "m", "female", "male", }


def split_species_sex( label: object, ) -> Tuple[str, Optional[str]]:
    text = str(label).strip()
    if is_noise_label(text):
        return "Noise", None
    if "_" not in text:
        return text, None

    species, suffix = text.rsplit("_", 1)
    suffix_key = suffix.strip().casefold()
    if suffix_key in SEX_SUFFIXES:
        sex = "F" if suffix_key.startswith("f") else "M"
        return species, sex

    return text, None


def map_label_to_space( label: object, label_space: str, ) -> str:
    label_space = str(label_space).strip().casefold()

    if label_space in {"binary", "wingbeat", "detection"}:
        return to_detection_label(label)

    if is_noise_label(label):
        return "Noise"

    species, sex = split_species_sex(label)

    if label_space == "species":
        return species
    if label_space == "sex":
        return sex if sex is not None else "UnknownSex"
    if label_space == "species_sex":
        return str(label)

    raise ValueError(f"label space not found: {label_space}")


def label_dict_from_names( names: Sequence[str], ) -> Dict[str, int]:
    unique = sorted(set(map(str, names)))
    ordered = []

    for preferred in [ "Noise", "background", "wingbeat", "F", "M", "UnknownSex", ]:
        if preferred in unique and preferred not in ordered:
            ordered.append(preferred)

    ordered.extend( value for value in unique if value not in ordered )
    return { name: index for index, name in enumerate(ordered) }


def map_prediction_dataframe( df: pd.DataFrame, label_space: str, ) -> Tuple[pd.DataFrame, Dict[str, int]]:
    out = _normalise_prediction_columns(df)

    required = {"true_label", "predicted_label"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    true_labels = out["true_label"].map( lambda label: map_label_to_space(label, label_space) )
    pred_labels = out["predicted_label"].map( lambda label: map_label_to_space(label, label_space) )

    # Build the ID map from both ground truth and predictions.
    all_labels = pd.concat( [true_labels, pred_labels], ignore_index=True, )
    mapped_label_dict = label_dict_from_names(all_labels.tolist())

    out = out.copy()
    out["true_label"] = true_labels
    out["predicted_label"] = pred_labels
    out["true_id"] = true_labels.map(mapped_label_dict).astype(np.int64)
    out["pred_id"] = pred_labels.map(mapped_label_dict).astype(np.int64)
    out["correct"] = out["true_id"] == out["pred_id"]

    return out, mapped_label_dict


def arrays_from_prediction_dataframe(
    df: pd.DataFrame,
    label_dict: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(y_true, y_pred)``."""
    out = _normalise_prediction_columns(df)

    required = {"true_label", "predicted_label"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    true_ids = out["true_label"].map(label_dict)
    pred_ids = out["predicted_label"].map(label_dict)

    unknown_true = sorted( out.loc[true_ids.isna(), "true_label"].astype(str).unique() )
    unknown_pred = sorted( out.loc[pred_ids.isna(), "predicted_label"].astype(str).unique() )

    if unknown_true:
        raise ValueError(f"true labels not found: {unknown_true}")
    if unknown_pred:
        raise ValueError(f"predicted labels not found: {unknown_pred}")

    return ( true_ids.astype(np.int64).to_numpy(), pred_ids.astype(np.int64).to_numpy(), )


def infer_label_space( label_dict: Dict[str, int], ) -> str:
    non_noise = [ name for name in label_dict if not is_noise_label(name) ]
    if non_noise and all( split_species_sex(name)[1] is not None for name in non_noise ):
        return "species_sex"
    return "species"


# =============================================================================
# SEQUENCE EVENT UTILITIES
# =============================================================================


def extract_events_from_segments(
    df: pd.DataFrame,
    label_col: str,
    time_cols: Tuple[str, str] = ("start_time", "end_time"),
    merge_tolerance: float = 1e-6,
    exclude_noise: bool = False,
) -> List[Event]:
    """Merge adjacent segments with the same label within each file.

    Equal labels separated by a real time gap are not merged.
    """
    out = _normalise_prediction_columns(df)
    required = { "file_name", label_col, time_cols[0], time_cols[1], }
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    events: List[Event] = []

    for file_name, group in out.groupby("file_name", sort=True):
        group = group.sort_values( [time_cols[0], time_cols[1]], kind="stable", )

        current_label: Optional[str] = None
        current_start: Optional[float] = None
        current_end: Optional[float] = None

        for _, row in group.iterrows():
            label = str(row[label_col])
            start = float(row[time_cols[0]])
            end = float(row[time_cols[1]])

            if not np.isfinite(start) or not np.isfinite(end) or end <= start:
                raise ValueError(f"bad segment time: {file_name}")

            contiguous = ( current_end is not None and start <= current_end + float(merge_tolerance) )

            if ( current_label is None or label != current_label or not contiguous ):
                if current_label is not None:
                    if not (exclude_noise and is_noise_label(current_label)):
                        events.append(( current_label, float(current_start), float(current_end), str(file_name), ))

                current_label = label
                current_start = start
                current_end = end
            else:
                current_end = max(float(current_end), end)

        if current_label is not None:
            if not (exclude_noise and is_noise_label(current_label)):
                events.append(( current_label, float(current_start), float(current_end), str(file_name), ))

    return events


def extract_events_from_metadata(
    metadata_csv_path: str,
    label_space: str = "species",
    include_noise: bool = False,
) -> List[Event]:
    """Load exact ground-truth event boundaries from simulation metadata."""
    metadata_csv_path = os.path.abspath(os.fspath(metadata_csv_path))
    if not os.path.exists(metadata_csv_path):
        raise FileNotFoundError(f"metadata file not found: {metadata_csv_path}")

    df = pd.read_csv(metadata_csv_path)
    required = {"file_name", "start_time", "end_time", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"metadata missing columns: {sorted(missing)}")

    if "event_type" in df.columns:
        event_type = df["event_type"].astype(str).str.casefold()
        if include_noise:
            keep = event_type.isin({"mosquito", "noise"})
        else:
            keep = event_type.eq("mosquito")
        df = df.loc[keep].copy()
    elif not include_noise:
        df = df.loc[ ~df["label"].apply(is_noise_label) ].copy()

    events: List[Event] = []
    for _, row in df.iterrows():
        label = map_label_to_space(row["label"], label_space)
        if not include_noise and is_noise_label(label):
            continue

        start = float(row["start_time"])
        end = float(row["end_time"])
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            raise ValueError(f"bad event time: {row['file_name']}")

        events.append(( label, start, end, str(row["file_name"]), ))

    return sorted( events, key=lambda event: ( _normalise_file_key(event[3]), event[1], event[2], event[0], ), )


def temporal_iou( interval1: Tuple[float, float], interval2: Tuple[float, float], ) -> float:
    intersection_start = max(interval1[0], interval2[0])
    intersection_end = min(interval1[1], interval2[1])
    intersection = max( 0.0, intersection_end - intersection_start, )
    union = ( max(interval1[1], interval2[1]) - min(interval1[0], interval2[0]) )
    return intersection / union if union > 0 else 0.0


def temporal_overlap( interval1: Tuple[float, float], interval2: Tuple[float, float], ) -> float:
    return max( 0.0, min(interval1[1], interval2[1]) - max(interval1[0], interval2[0]), )


def total_interval_duration( intervals: Iterable[Tuple[float, float]], ) -> float:
    valid = sorted(
        (float(start), float(end))
        for start, end in intervals
        if np.isfinite(start) and np.isfinite(end) and end > start
    )
    if not valid:
        return 0.0

    total = 0.0
    current_start, current_end = valid[0]

    for start, end in valid[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue

        total += current_end - current_start
        current_start, current_end = start, end

    return total + current_end - current_start


def _match_group_hungarian(
    gt_group: Sequence[Tuple[int, Event]],
    pred_group: Sequence[Tuple[int, Event]],
    iou_threshold: float,
) -> List[Tuple[int, int, float, float]]:
    if not gt_group or not pred_group:
        return []

    iou_matrix = np.zeros( (len(gt_group), len(pred_group)), dtype=np.float64, )
    overlap_matrix = np.zeros_like(iou_matrix)

    for gt_pos, (_, gt_event) in enumerate(gt_group):
        for pred_pos, (_, pred_event) in enumerate(pred_group):
            iou_matrix[gt_pos, pred_pos] = temporal_iou( (gt_event[1], gt_event[2]), (pred_event[1], pred_event[2]), )
            overlap_matrix[gt_pos, pred_pos] = temporal_overlap(
                (gt_event[1], gt_event[2]),
                (pred_event[1], pred_event[2]),
            )

    valid = iou_matrix >= float(iou_threshold)
    if not valid.any():
        return []

    if linear_sum_assignment is None:
        candidates = []
        for gt_pos, pred_pos in zip(*np.nonzero(valid)):
            candidates.append((
                float(iou_matrix[gt_pos, pred_pos]),
                float(overlap_matrix[gt_pos, pred_pos]),
                gt_pos,
                pred_pos,
            ))
        candidates.sort( key=lambda item: (item[0], item[1]), reverse=True, )

        used_gt = set()
        used_pred = set()
        assignments = []
        for iou_value, overlap_value, gt_pos, pred_pos in candidates:
            if gt_pos in used_gt or pred_pos in used_pred:
                continue
            used_gt.add(gt_pos)
            used_pred.add(pred_pos)
            assignments.append(( gt_group[gt_pos][0], pred_group[pred_pos][0], iou_value, overlap_value, ))
        return assignments

    benefit = np.where( valid, 1.0 + iou_matrix, 0.0, )
    row_index, col_index = linear_sum_assignment(-benefit)

    assignments = []
    for gt_pos, pred_pos in zip(row_index, col_index):
        if benefit[gt_pos, pred_pos] <= 0:
            continue
        assignments.append((
            gt_group[gt_pos][0],
            pred_group[pred_pos][0],
            float(iou_matrix[gt_pos, pred_pos]),
            float(overlap_matrix[gt_pos, pred_pos]),
        ))

    return assignments


def match_events_by_iou(
    gt_events: Sequence[Event],
    pred_events: Sequence[Event],
    class_names: Sequence[str],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
):
    """Match events one-to-one by file, class and temporal IoU."""
    iou_threshold = float(iou_threshold)
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be 0 to 1")

    class_names = [str(name) for name in class_names]
    class_set = set(class_names)

    counts = OrderedDict(
        (
            class_name,
            {
                "TP": 0,
                "FP": 0,
                "FN": 0,
                "Support": 0,
            },
        )
        for class_name in class_names
    )

    gt_groups: Dict[Tuple[str, str], List[Tuple[int, Event]]] = {}
    pred_groups: Dict[Tuple[str, str], List[Tuple[int, Event]]] = {}

    for index, event in enumerate(gt_events):
        label, _, _, file_name = event
        label = str(label)
        if label not in class_set:
            continue
        counts[label]["Support"] += 1
        key = (_normalise_file_key(file_name), label)
        gt_groups.setdefault(key, []).append((index, event))

    for index, event in enumerate(pred_events):
        label, _, _, file_name = event
        label = str(label)
        if label not in class_set:
            continue
        key = (_normalise_file_key(file_name), label)
        pred_groups.setdefault(key, []).append((index, event))

    gt_matched = set()
    pred_matched = set()
    match_rows = []

    for key in sorted(set(gt_groups) | set(pred_groups)):
        gt_group = gt_groups.get(key, [])
        pred_group = pred_groups.get(key, [])

        assignments = _match_group_hungarian( gt_group=gt_group, pred_group=pred_group, iou_threshold=iou_threshold, )

        for gt_index, pred_index, iou_value, overlap_value in assignments:
            gt_event = gt_events[gt_index]
            pred_event = pred_events[pred_index]
            label = str(gt_event[0])

            gt_matched.add(gt_index)
            pred_matched.add(pred_index)
            counts[label]["TP"] += 1

            match_rows.append({
                "file_name": str(gt_event[3]),
                "class": label,
                "true_start": float(gt_event[1]),
                "true_end": float(gt_event[2]),
                "pred_start": float(pred_event[1]),
                "pred_end": float(pred_event[2]),
                "iou": float(iou_value),
                "overlap": float(overlap_value),
                "onset_error": float(pred_event[1] - gt_event[1]),
                "offset_error": float(pred_event[2] - gt_event[2]),
            })

    for index, event in enumerate(gt_events):
        label = str(event[0])
        if label in class_set and index not in gt_matched:
            counts[label]["FN"] += 1

    for index, event in enumerate(pred_events):
        label = str(event[0])
        if label in class_set and index not in pred_matched:
            counts[label]["FP"] += 1

    return counts, pd.DataFrame(match_rows, columns=EVENT_MATCH_COLUMNS)


def counts_to_metrics_df( counts: Dict[str, Dict[str, int]], ) -> pd.DataFrame:
    rows = []

    for class_name, values in counts.items():
        tp = int(values["TP"])
        fp = int(values["FP"])
        fn = int(values["FN"])
        support = int(values["Support"])

        precision = safe_divide(tp, tp + fp, default=np.nan)
        recall = safe_divide(tp, tp + fn)
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if pd.notna(recall) and precision + recall > 0
            else (0.0 if support > 0 else np.nan)
        )

        rows.append({
            "class": class_name,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "Event Jaccard": safe_divide(tp, tp + fp + fn),
            "Support": support,
            "TP": tp,
            "FP": fp,
            "FN": fn,
        })

    return pd.DataFrame(rows, columns=EVENT_METRIC_COLUMNS)


def summarize_event_metrics( per_class_df: pd.DataFrame, prefix: str, ) -> pd.DataFrame:
    total_tp = int(per_class_df["TP"].sum())
    total_fp = int(per_class_df["FP"].sum())
    total_fn = int(per_class_df["FN"].sum())
    total_support = int(per_class_df["Support"].sum())

    micro_precision = safe_divide( total_tp, total_tp + total_fp, default=np.nan, )
    micro_recall = safe_divide( total_tp, total_tp + total_fn, )
    micro_f1 = (
        2.0 * micro_precision * micro_recall
        / (micro_precision + micro_recall)
        if pd.notna(micro_recall)
        and micro_precision + micro_recall > 0
        else (0.0 if total_support > 0 else np.nan)
    )
    event_jaccard = safe_divide( total_tp, total_tp + total_fp + total_fn, )

    present_df = per_class_df.loc[ per_class_df["Support"] > 0 ].copy()

    if present_df.empty:
        macro_precision = np.nan
        macro_recall = np.nan
        macro_f1 = np.nan
        weighted_f1 = np.nan
    else:
        macro_precision = float( present_df["Precision"].fillna(0.0).mean() )
        macro_recall = float( present_df["Recall"].fillna(0.0).mean() )
        macro_f1 = float( present_df["F1-score"].fillna(0.0).mean() )
        weighted_f1 = float(
            (
                present_df["F1-score"].fillna(0.0)
                * present_df["Support"]
            ).sum()
            / max(total_support, EPS)
        )

    return pd.DataFrame({
        "Metric": [
            f"{prefix} Jaccard",
            "Micro Precision",
            "Micro Recall",
            "Micro F1-score",
            "Macro Precision",
            "Macro Recall",
            "Macro F1-score",
            "Weighted F1-score",
            "TP",
            "FP",
            "FN",
            "Support",
        ],
        "Value": [
            event_jaccard,
            micro_precision,
            micro_recall,
            micro_f1,
            macro_precision,
            macro_recall,
            macro_f1,
            weighted_f1,
            total_tp,
            total_fp,
            total_fn,
            total_support,
        ],
    })


def _ground_truth_events(
    segment_df: pd.DataFrame,
    metadata_csv_path: Optional[str],
    label_space: str,
    include_noise: bool,
) -> Tuple[List[Event], str]:
    if metadata_csv_path is not None and os.path.exists(metadata_csv_path):
        events = extract_events_from_metadata(
            metadata_csv_path=metadata_csv_path,
            label_space=label_space,
            include_noise=include_noise,
        )
        return events, "exact_metadata"

    warnings.warn( "metadata not found, use segment time", RuntimeWarning, )
    events = extract_events_from_segments( segment_df, label_col="true_label", exclude_noise=not include_noise, )
    return events, "segment_derived"


# =============================================================================
# SEQUENCE CLASSIFICATION
# =============================================================================


def evaluate_sequence_metrics_from_df(
    df: pd.DataFrame,
    label_dict: Dict[str, int],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    include_noise: bool = False,
    output_dir: Optional[str] = None,
    model_name: Optional[str] = None,
    environment: Optional[str] = None,
    save: bool = False,
    metadata_csv_path: Optional[str] = None,
    label_space: str = "species",
):
    """Event-level class evaluation from exact GT metadata when available."""
    out = _normalise_prediction_columns(df)

    gt_events, gt_boundary_source = _ground_truth_events(
        segment_df=out,
        metadata_csv_path=metadata_csv_path,
        label_space=label_space,
        include_noise=include_noise,
    )
    pred_events = extract_events_from_segments( out, label_col="predicted_label", exclude_noise=not include_noise, )

    class_names, _ = label_names_and_ids(label_dict)
    if not include_noise:
        class_names = [ name for name in class_names if not is_noise_label(name) ]

    counts, matches = match_events_by_iou(
        gt_events=gt_events,
        pred_events=pred_events,
        class_names=class_names,
        iou_threshold=iou_threshold,
    )
    per_class_df = counts_to_metrics_df(counts)
    overall_df = summarize_event_metrics( per_class_df, prefix="Sequence", )
    overall_df = pd.concat([
        overall_df,
        pd.DataFrame({
            "Metric": [
                "Ground Truth Boundary Source",
                "IoU Threshold",
            ],
            "Value": [
                gt_boundary_source,
                float(iou_threshold),
            ],
        }),
    ], ignore_index=True)

    print( f"\n=== Event Classification, IoU >= {iou_threshold} " f"({gt_boundary_source}) ===" )
    print(overall_df.to_string(index=False))

    if save:
        ensure_dir(output_dir)
        tag = str(iou_threshold).replace(".", "p")
        noise_suffix = "_incl_noise" if include_noise else ""
        prefix = ( f"{model_name or 'model'}_" f"{environment or 'env'}_sequence_iou{tag}{noise_suffix}" )
        overall_df.to_csv( os.path.join(output_dir, f"{prefix}_overall.csv"), index=False, )
        per_class_df.to_csv( os.path.join(output_dir, f"{prefix}_per_class.csv"), index=False, )
        matches.to_csv( os.path.join(output_dir, f"{prefix}_matches.csv"), index=False, )

    return overall_df, per_class_df, matches


# =============================================================================
# EVENT DETECTION
# =============================================================================


def make_detection_dataframe( df: pd.DataFrame, ) -> pd.DataFrame:
    out = _normalise_prediction_columns(df)
    out = out.copy()
    out["true_label"] = out["true_label"].apply( to_detection_label )
    out["predicted_label"] = out["predicted_label"].apply( to_detection_label )
    return out


def evaluate_event_detection_from_df(
    raw_df: pd.DataFrame,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    output_dir: Optional[str] = None,
    model_name: Optional[str] = None,
    environment: Optional[str] = None,
    save: bool = False,
    metadata_csv_path: Optional[str] = None,
):
    """Positive-event wingbeat detection with exact GT boundaries."""
    df = make_detection_dataframe(raw_df)

    if metadata_csv_path is not None and os.path.exists(metadata_csv_path):
        gt_events = extract_events_from_metadata(
            metadata_csv_path=metadata_csv_path,
            label_space="binary",
            include_noise=False,
        )
        gt_boundary_source = "exact_metadata"
    else:
        gt_events = [
            event
            for event in extract_events_from_segments(
                df,
                label_col="true_label",
                exclude_noise=True,
            )
            if event[0] == "wingbeat"
        ]
        gt_boundary_source = "segment_derived"

    pred_events = [
        event
        for event in extract_events_from_segments(
            df,
            label_col="predicted_label",
            exclude_noise=True,
        )
        if event[0] == "wingbeat"
    ]

    counts, matches = match_events_by_iou(
        gt_events=gt_events,
        pred_events=pred_events,
        class_names=["wingbeat"],
        iou_threshold=iou_threshold,
    )
    per_class_df = counts_to_metrics_df(counts)
    overall_df = summarize_event_metrics( per_class_df, prefix="Detection", )
    overall_df = pd.concat([
        overall_df,
        pd.DataFrame({
            "Metric": [
                "Ground Truth Boundary Source",
                "IoU Threshold",
            ],
            "Value": [
                gt_boundary_source,
                float(iou_threshold),
            ],
        }),
    ], ignore_index=True)

    print( f"\n=== Event Detection, IoU >= {iou_threshold} " f"({gt_boundary_source}) ===" )
    print(overall_df.to_string(index=False))

    if save:
        ensure_dir(output_dir)
        tag = str(iou_threshold).replace(".", "p")
        prefix = ( f"{model_name or 'model'}_" f"{environment or 'env'}_detection_iou{tag}" )
        overall_df.to_csv( os.path.join(output_dir, f"{prefix}_overall.csv"), index=False, )
        per_class_df.to_csv( os.path.join(output_dir, f"{prefix}_per_class.csv"), index=False, )
        matches.to_csv( os.path.join(output_dir, f"{prefix}_matches.csv"), index=False, )

    return overall_df, per_class_df, matches


# =============================================================================
# NOISE-ONLY EVALUATION
# =============================================================================


def is_noise_only_ground_truth( y_true, label_dict: Dict[str, int], ) -> bool:
    y_true = np.asarray(y_true).reshape(-1)
    if y_true.size == 0:
        return False

    inverse = id_to_name(label_dict)
    return all( is_noise_label(inverse[int(class_id)]) for class_id in y_true )


def evaluate_noise_only_from_df( raw_df: pd.DataFrame, ) -> pd.DataFrame:
    df = make_detection_dataframe(raw_df)

    required = { "file_name", "start_time", "end_time", "true_label", "predicted_label", }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("noise-only dataframe is empty")
    if not df["true_label"].eq("background").all():
        raise ValueError("ground truth is not noise-only")

    false_positive = df["predicted_label"].eq("wingbeat")
    n_segments = int(len(df))
    n_false_positive_segments = int(false_positive.sum())

    file_names = df["file_name"].astype(str)
    n_files = int(file_names.nunique())
    n_false_alarm_files = int(file_names[false_positive].nunique())

    false_alarm_events = [
        event
        for event in extract_events_from_segments(
            df,
            label_col="predicted_label",
            exclude_noise=True,
        )
        if event[0] == "wingbeat"
    ]
    false_alarm_seconds = sum( event[2] - event[1] for event in false_alarm_events )

    audio_seconds = 0.0
    for _, group in df.groupby("file_name", sort=False):
        intervals = zip(group["start_time"], group["end_time"])
        audio_seconds += total_interval_duration(intervals)

    audio_hours = audio_seconds / 3600.0

    return pd.DataFrame({
        "Metric": [
            "Noise Segments",
            "False Positive Segments",
            "Segment False Positive Rate",
            "Background Accuracy",
            "N Files",
            "Files With False Alarm",
            "File False Alarm Rate",
            "False Alarm Events",
            "False Alarm Duration Seconds",
            "Evaluated Audio Hours",
            "False Alarms Per Hour",
        ],
        "Value": [
            n_segments,
            n_false_positive_segments,
            safe_divide(n_false_positive_segments, n_segments),
            safe_divide(
                n_segments - n_false_positive_segments,
                n_segments,
            ),
            n_files,
            n_false_alarm_files,
            safe_divide(n_false_alarm_files, n_files),
            len(false_alarm_events),
            float(false_alarm_seconds),
            audio_hours,
            safe_divide(len(false_alarm_events), audio_hours),
        ],
    })


# =============================================================================
# FILE-LEVEL DETECTION
# =============================================================================


def evaluate_file_presence( y_pred, y_true, file_names: Sequence[str], label_dict: Dict[str, int], ) -> pd.DataFrame:
    y_pred, y_true, file_names = validate_prediction_arrays(
        y_pred=y_pred,
        y_true=y_true,
        file_names=file_names,
        label_dict=label_dict,
    )

    inverse = id_to_name(label_dict)
    file_names_array = np.asarray( [_normalise_file_key(value) for value in file_names] )

    true_presence = []
    pred_presence = []

    for file_name in sorted(set(file_names_array)):
        indices = np.flatnonzero(file_names_array == file_name)
        true_has = any( to_detection_label(inverse[int(y_true[index])]) == "wingbeat" for index in indices )
        pred_has = any( to_detection_label(inverse[int(y_pred[index])]) == "wingbeat" for index in indices )

        true_presence.append( "wingbeat" if true_has else "background" )
        pred_presence.append( "wingbeat" if pred_has else "background" )

    all_labels = ["background", "wingbeat"]
    present_labels = [ label for label in all_labels if label in set(true_presence) ]
    report = classification_report(
        true_presence,
        pred_presence,
        labels=all_labels,
        output_dict=True,
        zero_division=0,
    )
    macro_f1 = float(np.mean([ report[label]["f1-score"] for label in present_labels ]))

    false_alarm_files = sum(
        true_label == "background" and pred_label == "wingbeat"
        for true_label, pred_label in zip(true_presence, pred_presence)
    )

    return pd.DataFrame({
        "Metric": [
            "File Presence Accuracy",
            "File Presence Macro F1-score",
            "Files With False Alarm",
            "File False Alarm Rate",
            "N Files",
        ],
        "Value": [
            accuracy_score(true_presence, pred_presence),
            macro_f1,
            false_alarm_files,
            safe_divide(false_alarm_files, len(true_presence)),
            len(true_presence),
        ],
    })


# =============================================================================
# EVALUATION PIPELINE
# =============================================================================


def evaluate_label_space(
    segment_df: pd.DataFrame,
    label_dict: Dict[str, int],
    label_space: str,
    output_dir: str,
    model_name: str,
    environment: str,
    iou_threshold: float,
    include_noise_in_sequence: bool,
    save: bool,
    metadata_csv_path: Optional[str] = None,
):
    y_true_view, y_pred_view = arrays_from_prediction_dataframe( segment_df, label_dict, )

    segment_results_df, segment_class_df = (
        evaluate_segment_classification(
            y_pred=y_pred_view,
            y_true=y_true_view,
            label_dict=label_dict,
            output_dir=output_dir,
            model_name=model_name,
            environment=environment,
            save=save,
        )
    )

    binary_segment_results_df, binary_segment_report_df = (
        evaluate_segment_detection(
            y_true=y_true_view,
            y_pred=y_pred_view,
            label_dict=label_dict,
            output_dir=output_dir,
            model_name=model_name,
            environment=environment,
            save=save,
        )
    )

    # Keep the main event-class metric mosquito-only so its definition stays
    # consistent across experiments. Also compute a full sequence view where
    # Noise is treated as one of the event classes.
    sequence_overall_df, sequence_class_df, sequence_matches_df = (
        evaluate_sequence_metrics_from_df(
            df=segment_df,
            label_dict=label_dict,
            iou_threshold=iou_threshold,
            include_noise=False,
            output_dir=output_dir,
            model_name=model_name,
            environment=environment,
            save=save,
            metadata_csv_path=metadata_csv_path,
            label_space=label_space,
        )
    )
    sequence_with_noise_overall_df, sequence_with_noise_class_df, sequence_with_noise_matches_df = (
        evaluate_sequence_metrics_from_df(
            df=segment_df,
            label_dict=label_dict,
            iou_threshold=iou_threshold,
            include_noise=True,
            output_dir=output_dir,
            model_name=model_name,
            environment=environment,
            save=save,
            metadata_csv_path=metadata_csv_path,
            label_space=label_space,
        )
    )

    detection_overall_df, detection_class_df, detection_matches_df = (
        evaluate_event_detection_from_df(
            raw_df=segment_df,
            iou_threshold=iou_threshold,
            output_dir=output_dir,
            model_name=model_name,
            environment=environment,
            save=save,
            metadata_csv_path=metadata_csv_path,
        )
    )

    return {
        "label_space": label_space,
        "label_dict": label_dict,
        "segment_predictions": segment_df,
        "segment_results": segment_results_df,
        "segment_class_report": segment_class_df,
        "binary_segment_results": binary_segment_results_df,
        "binary_segment_report": binary_segment_report_df,
        "sequence_overall": sequence_overall_df,
        "sequence_class_report": sequence_class_df,
        "sequence_matches": sequence_matches_df,
        "sequence_with_noise_overall": sequence_with_noise_overall_df,
        "sequence_with_noise_class_report": sequence_with_noise_class_df,
        "sequence_with_noise_matches": sequence_with_noise_matches_df,
        "detection_overall": detection_overall_df,
        "detection_class_report": detection_class_df,
        "detection_matches": detection_matches_df,
    }


def run_evaluation_from_predictions(
    y_pred,
    y_true,
    file_names: Sequence[str],
    label_dict: Dict[str, int],
    output_dir: str,
    model_name: str,
    environment: str = "test",
    segment_duration: float = 0.5,
    overlap: float = 0.0,
    metadata_csv_path: Optional[str] = None,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    include_noise_in_sequence: bool = False,
    evaluation_label_space: Optional[str] = None,
    save: bool = True,
    no: Optional[Union[int, str]] = None,
):
    """Run one evaluation label space from segment predictions."""
    if save:
        ensure_dir(output_dir)

    y_pred, y_true, file_names = validate_prediction_arrays(
        y_pred=y_pred,
        y_true=y_true,
        file_names=file_names,
        label_dict=label_dict,
    )

    label_space = str( evaluation_label_space or infer_label_space(label_dict) ).strip().casefold()
    if label_space not in {"species", "species_sex"}:
        raise ValueError(f"Unsupported evaluation label space: {label_space}")

    prediction_path, prediction_df = save_segment_predictions_csv(
        y_pred=y_pred,
        y_true=y_true,
        file_names=file_names,
        label_dict=label_dict,
        output_dir=output_dir,
        model_name=model_name,
        environment=environment,
        segment_duration=segment_duration,
        overlap=overlap,
        metadata_csv_path=metadata_csv_path,
        save=save,
    )

    noise_only = is_noise_only_ground_truth(y_true, label_dict)
    evaluation_regime = "Noise-only" if noise_only else "Mosquito+Background"
    noise_only_results_df = None

    if noise_only:
        noise_only_results_df = evaluate_noise_only_from_df(prediction_df)
        print("\n=== Noise-Only False-Alarm Evaluation ===")
        print(noise_only_results_df.to_string(index=False))
        if save:
            noise_only_results_df.to_csv(
                os.path.join(
                    output_dir,
                    f"{model_name}_{environment}_noise_only_results.csv",
                ),
                index=False,
                float_format="%.4f",
            )

    if save:
        label_name = "Species+Sex" if label_space == "species_sex" else "Species"
        save_class_support(prediction_df, output_dir, label_name)
        if noise_only:
            save_noise_prediction_distribution( prediction_df, output_dir, label_name, )

    result = evaluate_label_space(
        segment_df=prediction_df,
        label_dict=label_dict,
        label_space=label_space,
        output_dir=output_dir,
        model_name=model_name,
        environment=environment,
        iou_threshold=iou_threshold,
        include_noise_in_sequence=include_noise_in_sequence,
        save=save,
        metadata_csv_path=metadata_csv_path,
    )
    result["segment_predictions_path"] = prediction_path
    result["evaluation_regime"] = evaluation_regime
    result["noise_only_results"] = noise_only_results_df
    result["label_space"] = label_space

    file_presence_df = evaluate_file_presence(
        y_pred=y_pred,
        y_true=y_true,
        file_names=file_names,
        label_dict=label_dict,
    )
    if save:
        file_presence_df.to_csv(
            os.path.join(
                output_dir,
                f"{model_name}_{environment}_file_presence_results.csv",
            ),
            index=False,
        )

    result["file_presence"] = file_presence_df
    result["n_segments"] = int(len(y_true))
    result["n_files"] = int(len(set(map(str, file_names))))
    result["iou_threshold"] = float(iou_threshold)
    result["model_name"] = model_name
    result["environment"] = environment
    result["metadata_csv_path"] = metadata_csv_path
    result["no"] = no

    if save:
        manifest = {
            "model_name": model_name,
            "environment": environment,
            "label_space": label_space,
            "evaluation_regime": evaluation_regime,
            "n_segments": int(len(y_true)),
            "n_files": int(len(set(map(str, file_names)))),
            "iou_threshold": float(iou_threshold),
            "include_noise_in_sequence": bool(include_noise_in_sequence),
            "sequence_class_views": ["mosquito_only", "including_noise"],
            "metadata_csv_path": (
                os.path.abspath(metadata_csv_path)
                if metadata_csv_path is not None
                else None
            ),
            "prediction_path": os.path.abspath(prediction_path),
        }
        with open( os.path.join(output_dir, "evaluation_manifest.json"), "w", encoding="utf-8", ) as file:
            json.dump(manifest, file, indent=2)

    return result


# =============================================================================
# ALL-DOMAINS POOLED EVALUATION
# =============================================================================


def _all_domains_label_dict() -> Dict[str, int]:
    return { class_name: class_id for class_id, class_name in enumerate(ALL_DOMAIN_SPECIES) }


def _load_all_domains_inputs(run_dir: str):
    run_dir = os.path.abspath(run_dir)
    prediction_frames = []
    metadata_frames = []
    source_sets = []

    for eval_name in sorted(os.listdir(run_dir)):
        eval_dir = os.path.join(run_dir, eval_name)
        if not os.path.isdir(eval_dir) or eval_name == "all_domains":
            continue

        prediction_path = os.path.join(eval_dir, "segment_predictions.csv")
        summary_path = os.path.join(eval_dir, "evaluation_summary.csv")
        if not os.path.exists(prediction_path) or not os.path.exists(summary_path):
            continue

        prediction_df = _normalise_prediction_columns(pd.read_csv(prediction_path))
        required = {"file_name", "true_label", "predicted_label"}
        if prediction_df.empty or not required.issubset(prediction_df.columns):
            continue

        prediction_df = prediction_df.copy()
        prediction_df["file_name"] = prediction_df["file_name"].astype(str).map(
            lambda value: f"{eval_name}::{_normalise_file_key(value)}"
        )
        prediction_df["true_label"] = prediction_df["true_label"].map(
            lambda value: map_label_to_space(value, "species")
        )
        prediction_df["predicted_label"] = prediction_df["predicted_label"].map(
            lambda value: map_label_to_space(value, "species")
        )

        label_dict = _all_domains_label_dict()
        unknown_true = sorted(set(prediction_df["true_label"]) - set(label_dict))
        unknown_pred = sorted(set(prediction_df["predicted_label"]) - set(label_dict))
        if unknown_true or unknown_pred:
            raise ValueError( f"all_domains labels are not supported: true={unknown_true}, pred={unknown_pred}" )

        prediction_df["true_id"] = prediction_df["true_label"].map(label_dict).astype(np.int64)
        prediction_df["pred_id"] = prediction_df["predicted_label"].map(label_dict).astype(np.int64)
        prediction_df["correct"] = prediction_df["true_id"] == prediction_df["pred_id"]
        prediction_frames.append(prediction_df)
        source_sets.append(eval_name)

        manifest_path = os.path.join(eval_dir, "evaluation_manifest.json")
        if not os.path.exists(manifest_path):
            continue

        with open(manifest_path, "r", encoding="utf-8") as file:
            manifest = json.load(file)

        metadata_path = manifest.get("metadata_csv_path")
        if not metadata_path or not os.path.exists(metadata_path):
            continue

        metadata_df = pd.read_csv(metadata_path, low_memory=False)
        if "file_name" not in metadata_df.columns:
            continue

        metadata_df = metadata_df.copy()
        metadata_df["file_name"] = metadata_df["file_name"].astype(str).map(
            lambda value: f"{eval_name}::{_normalise_file_key(value)}"
        )
        if "label" in metadata_df.columns:
            metadata_df["label"] = metadata_df["label"].map( lambda value: map_label_to_space(value, "species") )
        metadata_frames.append(metadata_df)

    if not prediction_frames:
        return None

    return {
        "predictions": pd.concat(prediction_frames, ignore_index=True, sort=False),
        "metadata": (
            pd.concat(metadata_frames, ignore_index=True, sort=False)
            if metadata_frames
            else None
        ),
        "source_sets": source_sets,
    }


def build_all_domains_evaluation(run_dir: str) -> Optional[pd.DataFrame]:
    pooled = _load_all_domains_inputs(run_dir)
    if pooled is None:
        return None

    output_dir = os.path.join(run_dir, "all_domains")
    ensure_dir(output_dir)

    prediction_df = pooled["predictions"]
    label_dict = _all_domains_label_dict()
    y_true, y_pred = arrays_from_prediction_dataframe(prediction_df, label_dict)
    file_names = prediction_df["file_name"].astype(str).tolist()

    metadata_path = None
    if pooled["metadata"] is not None:
        metadata_path = os.path.join(output_dir, "all_domains_metadata.csv")
        pooled["metadata"].to_csv(metadata_path, index=False)

    first_summary = None
    for eval_name in pooled["source_sets"]:
        path = os.path.join(run_dir, eval_name, "evaluation_summary.csv")
        if os.path.exists(path):
            frame = pd.read_csv(path)
            if not frame.empty:
                first_summary = frame.iloc[0]
                break

    model_name = (
        str(first_summary.get("Model"))
        if first_summary is not None
        else str(prediction_df.get("model_name", pd.Series(["model"])).iloc[0])
    )
    experiment = first_summary.get("Experiment") if first_summary is not None else None
    seed = first_summary.get("Seed") if first_summary is not None else None
    checkpoint = first_summary.get("Checkpoint") if first_summary is not None else os.path.basename(run_dir)

    segment_duration = 0.5
    if {"start_time", "end_time"}.issubset(prediction_df.columns):
        duration = pd.to_numeric( prediction_df["end_time"] - prediction_df["start_time"], errors="coerce", ).dropna()
        if not duration.empty and float(duration.median()) > 0:
            segment_duration = float(duration.median())

    results = run_evaluation_from_predictions(
        y_pred=y_pred,
        y_true=y_true,
        file_names=file_names,
        label_dict=label_dict,
        output_dir=output_dir,
        model_name=model_name,
        environment="all_domains",
        segment_duration=segment_duration,
        overlap=0.0,
        metadata_csv_path=metadata_path,
        iou_threshold=DEFAULT_IOU_THRESHOLD,
        include_noise_in_sequence=False,
        evaluation_label_space="species",
        save=True,
    )

    summary_df, _ = write_evaluation_summary(
        results=results,
        out_dir=output_dir,
        model_name=model_name,
        tag="all_domains",
        n_segments=len(y_true),
        experiment=experiment,
        seed=seed,
        checkpoint=checkpoint,
        eval_set="all_domains",
    )

    print("[eval] all_domains <- " + ", ".join(pooled["source_sets"]))
    return summary_df


# =============================================================================
# SUMMARY TABLES
# =============================================================================


def parse_eval_tag_for_summary( tag: str, ) -> Tuple[str, str]:
    tag = str(tag).strip().casefold()

    source_names = {
        "indoor": "INDOOR_MIRU",
        "indoor_miru": "INDOOR_MIRU",
        "outdoor": "OUTDOOR_MIRU",
        "outdoor_miru": "OUTDOOR_MIRU",
        "humbug": "HUMBUGDB",
        "humbug_miru": "HUMBUGDB",
        "biodcase": "BIODCASE",
        "mosquito": "MOSQUITO_ALL",
        "all": "ALL",
    }

    if tag in {"noise_only", "noiseonly"}:
        return "NOISE_ONLY", "MIXED"
    if tag in {"all_domains", "alldomains"}:
        return "ALL", "ALL"

    for suffix, environment in [
        ("_urban", "URBAN"),
        ("_forest", "FOREST"),
        ("_gaussian", "GAUSSIAN"),
        ("_mixed", "MIXED"),
    ]:
        if tag.endswith(suffix):
            source = tag[:-len(suffix)]
            return source_names.get(source, source.upper()), environment

    if tag == "cross":
        return "CROSS", "CROSS"

    return source_names.get(tag, tag.upper()), tag.upper()


def summary_label_from_result( result: Dict[str, object], ) -> Optional[str]:
    label_space = str( result.get("label_space", "species") ).strip().casefold()

    if label_space == "species_sex":
        return "Species+Sex"
    if label_space == "species":
        return "Species"
    return None


def report_value( report_df: pd.DataFrame, class_name: str, metric_name: str, ) -> float:
    if report_df is None or report_df.empty:
        return np.nan
    if "class" not in report_df.columns or metric_name not in report_df.columns:
        return np.nan

    rows = report_df.loc[ report_df["class"] == class_name, metric_name, ]
    if rows.empty or pd.isna(rows.iloc[0]):
        return np.nan
    return float(rows.iloc[0])


def build_summary_row(
    result: Dict[str, object],
    model_name: str,
    tag: str,
    n_segments: int,
    experiment: Optional[str] = None,
    seed: Optional[int] = None,
    checkpoint: Optional[str] = None,
    eval_set: Optional[str] = None,
) -> Optional[OrderedDict]:
    label_name = summary_label_from_result(result)
    if label_name is None:
        return None

    wingbeat_source, background_env = parse_eval_tag_for_summary(tag)
    segment_results = result["segment_results"]
    segment_class = result["segment_class_report"]
    binary_report = result["binary_segment_report"]
    sequence_overall = result["sequence_overall"]
    sequence_class = result["sequence_class_report"]
    sequence_with_noise_overall = result.get("sequence_with_noise_overall")
    sequence_with_noise_class = result.get("sequence_with_noise_class_report")
    detection_overall = result["detection_overall"]
    label_dict = result["label_dict"]

    n_files = int( result["segment_predictions"]["file_name"] .astype(str) .nunique() )

    row = OrderedDict()
    row["Experiment"] = experiment
    row["Model"] = model_name
    row["Seed"] = seed
    row["Checkpoint"] = checkpoint
    row["Eval Set"] = eval_set or tag
    row["Wingbeat Source"] = wingbeat_source
    row["Background Env"] = background_env
    row["Test Type"] = (
        "Pooled"
        if str(tag).strip().casefold() == "all_domains"
        else result.get("evaluation_regime", "Mosquito+Background")
    )
    row["Label"] = label_name
    row["N(files)"] = n_files
    row["N(segment)"] = int(n_segments)

    # Segment-level wingbeat/background detection.
    row["Seg Wingbeat PR"] = report_value( binary_report, "wingbeat", "precision", )
    row["Seg Wingbeat RE"] = report_value( binary_report, "wingbeat", "recall", )
    row["Seg Wingbeat F1"] = report_value( binary_report, "wingbeat", "f1-score", )
    row["Seg Background RE"] = report_value( binary_report, "background", "recall", )

    # Segment-level classification in the selected label space.
    row["Seg ACC"] = value_from_metric_df( segment_results, "Accuracy", )
    row["Seg Macro F1"] = value_from_metric_df( segment_results, "Macro F1-score", )
    row["Seg Weighted F1"] = value_from_metric_df( segment_results, "Weighted F1-score", )

    for class_name in label_dict.keys():
        row[f"SegF1_{class_name}"] = report_value( segment_class, class_name, "F1-score", )

    # Sequence = event-level classification: same class + temporal IoU match.
    row["Event Class Jaccard"] = value_from_metric_df( sequence_overall, "Sequence Jaccard", )
    row["Event Class Macro F1"] = value_from_metric_df( sequence_overall, "Macro F1-score", )
    row["Event Class Jaccard incl. Noise"] = value_from_metric_df( sequence_with_noise_overall, "Sequence Jaccard", )
    row["Event Class Macro F1 incl. Noise"] = value_from_metric_df( sequence_with_noise_overall, "Macro F1-score", )
    row["EventF1_Noise"] = report_value( sequence_with_noise_class, "Noise", "F1-score", )

    for class_name in label_dict.keys():
        if is_noise_label(class_name):
            continue
        row[f"EventF1_{class_name}"] = report_value( sequence_class, class_name, "F1-score", )

    # Event detection stays class-agnostic: any mosquito = wingbeat.
    row["Event Det Jaccard"] = value_from_metric_df( detection_overall, "Detection Jaccard", )
    row["Event Det PR"] = value_from_metric_df( detection_overall, "Micro Precision", )
    row["Event Det RE"] = value_from_metric_df( detection_overall, "Micro Recall", )
    row["Event Det F1"] = value_from_metric_df( detection_overall, "Micro F1-score", )

    noise_only_results = result.get("noise_only_results")
    row["Noise Seg FPR"] = value_from_metric_df( noise_only_results, "Segment False Positive Rate", )
    row["Noise File FAR"] = value_from_metric_df( noise_only_results, "File False Alarm Rate", )
    row["False Alarm Events"] = value_from_metric_df( noise_only_results, "False Alarm Events", )
    row["False Alarms/hour"] = value_from_metric_df( noise_only_results, "False Alarms Per Hour", )

    if row["Test Type"] == "Noise-only":
        # For noise-only data, mosquito recall/F1 and mosquito classification
        # are undefined. Keep the false-alarm metrics as the main result.
        row["Seg Wingbeat RE"] = np.nan
        row["Seg Wingbeat F1"] = np.nan
        row["Seg Macro F1"] = np.nan
        row["Seg Weighted F1"] = np.nan
        row["Event Class Jaccard"] = np.nan
        row["Event Class Macro F1"] = np.nan
        row["Event Det Jaccard"] = np.nan
        row["Event Det RE"] = np.nan
        row["Event Det F1"] = np.nan

        for column in list(row):
            if column.startswith("EventF1_") and column != "EventF1_Noise":
                row[column] = np.nan

    return row


def build_evaluation_summary(
    results: Dict[str, object],
    model_name: str,
    tag: str,
    n_segments: int,
    experiment: Optional[str] = None,
    seed: Optional[int] = None,
    checkpoint: Optional[str] = None,
    eval_set: Optional[str] = None,
) -> pd.DataFrame:
    row = build_summary_row(
        results,
        model_name=model_name,
        tag=tag,
        n_segments=n_segments,
        experiment=experiment,
        seed=seed,
        checkpoint=checkpoint,
        eval_set=eval_set,
    )
    output = pd.DataFrame([row] if row is not None else [])
    if not output.empty:
        output.insert(0, "No", range(1, len(output) + 1))
    return output


def write_evaluation_summary(
    results: Dict[str, object],
    out_dir: str,
    model_name: str,
    tag: str,
    n_segments: int,
    experiment: Optional[str] = None,
    seed: Optional[int] = None,
    checkpoint: Optional[str] = None,
    eval_set: Optional[str] = None,
    filename: str = "evaluation_summary.csv",
) -> Tuple[pd.DataFrame, str]:
    ensure_dir(out_dir)

    summary_df = build_evaluation_summary(
        results=results,
        model_name=model_name,
        tag=tag,
        n_segments=n_segments,
        experiment=experiment,
        seed=seed,
        checkpoint=checkpoint,
        eval_set=eval_set,
    )
    run_id = _run_id_from_path(out_dir)
    if "Run ID" not in summary_df.columns:
        insert_at = summary_df.columns.get_loc("Seed") + 1 if "Seed" in summary_df.columns else 1
        summary_df.insert(insert_at, "Run ID", run_id)

    output_path = os.path.join(out_dir, filename)
    summary_df.to_csv(output_path, index=False, float_format="%.4f")

    tsv_path = os.path.join(out_dir, "evaluation_summary.tsv")
    summary_df.to_csv(tsv_path, sep="	", index=False, float_format="%.4f")

    print(f"[eval] summary CSV -> {output_path}")
    print(f"[eval] summary TSV -> {tsv_path}")
    return summary_df, output_path


# =============================================================================
# OUTPUT MANAGEMENT
# =============================================================================


def cleanup_evaluation_outputs( out_dir: str, ) -> int:
    """Remove only obsolete summary clutter."""
    patterns = [
        "*_wide_summary.csv",
        "*_compact_summary.csv",
        "*_all_views_wide_summary.csv",
        "*_all_views_compact_summary.csv",
        "view_*/*_wide_summary.csv",
        "view_*/*_compact_summary.csv",
    ]

    removed = 0
    for pattern in patterns:
        for path in glob.glob(os.path.join(out_dir, pattern)):
            if os.path.isfile(path):
                os.remove(path)
                removed += 1

    if removed:
        print(f"[eval] removed {removed} obsolete summary files")
    return removed


def _latest_checkpoint_dirs( experiment_dir: str, checkpoint_name: str, ) -> List[Path]:
    experiment_path = Path(experiment_dir)
    selected = []

    for model_dir in sorted(path for path in experiment_path.iterdir() if path.is_dir()):
        for seed_dir in sorted(path for path in model_dir.glob("seed_*") if path.is_dir()):
            candidates = []
            for run_dir in seed_dir.glob("run_*"):
                checkpoint_dir = run_dir / "evaluation" / checkpoint_name
                if checkpoint_dir.is_dir():
                    candidates.append((run_dir.name, checkpoint_dir))

            if candidates:
                candidates.sort(key=lambda item: item[0], reverse=True)
                selected.append(candidates[0][1])

    return selected


def write_all_model_reports( experiment_dir: str, checkpoint_name: str, ) -> Dict[str, str]:
    experiment_dir = os.path.abspath(experiment_dir)
    summary_frames = []
    support_frames = []
    noise_frames = []

    experiment_name = Path(experiment_dir).name

    for checkpoint_dir in _latest_checkpoint_dirs(experiment_dir, checkpoint_name):
        run_dir = checkpoint_dir.parent.parent
        seed_dir = run_dir.parent
        model_dir = seed_dir.parent
        run_id = run_dir.name
        model_name = model_dir.name

        try:
            seed = int(seed_dir.name.replace("seed_", "", 1))
        except ValueError:
            seed = np.nan

        combined_path = checkpoint_dir / "all_environment_evaluation_summary.tsv"
        if combined_path.exists():
            frame = pd.read_csv(combined_path, sep="\t")
            if not frame.empty:
                frame["Model"] = model_name
                frame["Seed"] = seed
                frame["Run ID"] = run_id
                frame["Checkpoint"] = checkpoint_name
                summary_frames.append(frame)

        for class_support_path in checkpoint_dir.glob("*/class_support.tsv"):
            frame = pd.read_csv(class_support_path, sep="\t")
            if frame.empty:
                continue

            frame.insert(0, "Eval Set", class_support_path.parent.name)
            frame.insert(0, "Checkpoint", checkpoint_name)
            frame.insert(0, "Run ID", run_id)
            frame.insert(0, "Seed", seed)
            frame.insert(0, "Model", model_name)
            frame.insert(0, "Experiment", experiment_name)
            support_frames.append(frame)

        noise_distribution_path = checkpoint_dir / "noise_only" / "noise_prediction_distribution.tsv"
        if noise_distribution_path.exists():
            frame = pd.read_csv(noise_distribution_path, sep="\t")
            if not frame.empty:
                frame.insert(0, "Eval Set", "noise_only")
                frame.insert(0, "Checkpoint", checkpoint_name)
                frame.insert(0, "Run ID", run_id)
                frame.insert(0, "Seed", seed)
                frame.insert(0, "Model", model_name)
                frame.insert(0, "Experiment", experiment_name)
                noise_frames.append(frame)

    summary_path = os.path.join(experiment_dir, "all_environment_evaluation_summary.tsv")
    support_path = os.path.join(experiment_dir, "all_models_class_support.tsv")
    noise_path = os.path.join(experiment_dir, "all_models_noise_prediction_distribution.tsv")

    if summary_frames:
        summary = pd.concat(summary_frames, ignore_index=True, sort=False)
        if "No" in summary.columns:
            summary = summary.drop(columns=["No"])
        summary.insert(0, "No", range(1, len(summary) + 1))
        summary.to_csv(summary_path, sep="\t", index=False, float_format="%.4f")
    else:
        pd.DataFrame().to_csv(summary_path, sep="\t", index=False)

    if support_frames:
        pd.concat(support_frames, ignore_index=True, sort=False).to_csv(
            support_path,
            sep="\t",
            index=False,
            float_format="%.4f",
        )
    else:
        pd.DataFrame().to_csv(support_path, sep="\t", index=False)

    if noise_frames:
        pd.concat(noise_frames, ignore_index=True, sort=False).to_csv(
            noise_path,
            sep="\t",
            index=False,
            float_format="%.4f",
        )
    else:
        pd.DataFrame().to_csv(noise_path, sep="\t", index=False)

    print(f"[eval] all-model summary -> {summary_path}")
    print(f"[eval] all-model class support -> {support_path}")
    print(f"[eval] all-model noise distribution -> {noise_path}")

    return { "summary": summary_path, "class_support": support_path, "noise_prediction_distribution": noise_path, }


def combine_evaluation_summaries(
    run_dir: str,
    output_name: str = "all_environment_evaluation_summary.csv",
    add_eval_folder: bool = False,
) -> Optional[pd.DataFrame]:
    build_all_domains_evaluation(run_dir)
    summary_paths = []

    for root, _, files in os.walk(run_dir):
        if "evaluation_summary.csv" not in files:
            continue

        path = os.path.join(root, "evaluation_summary.csv")
        if os.path.abspath(os.path.dirname(path)) == os.path.abspath(run_dir):
            continue
        summary_paths.append(path)

    summary_paths = sorted(summary_paths)
    if not summary_paths:
        print( "[eval] no evaluation_summary.csv files found " f"under {run_dir}" )
        return None

    frames = []
    for path in summary_paths:
        frame = pd.read_csv(path)
        if frame.empty:
            continue

        if "No" in frame.columns:
            frame = frame.drop(columns=["No"])

        if add_eval_folder and "Eval Folder" not in frame.columns:
            frame.insert( 0, "Eval Folder", os.path.relpath(os.path.dirname(path), run_dir), )
        frames.append(frame)

    if not frames:
        return None

    merged = pd.concat(frames, ignore_index=True, sort=False)

    sort_columns = [ column for column in [ "Model", "Seed", "Eval Set", "Label", ] if column in merged.columns ]
    if sort_columns:
        merged = merged.sort_values( sort_columns, kind="stable", ).reset_index(drop=True)

    merged.insert(0, "No", range(1, len(merged) + 1))

    csv_name = output_name
    if not csv_name.lower().endswith(".csv"):
        csv_name = os.path.splitext(csv_name)[0] + ".csv"

    csv_path = os.path.join(run_dir, csv_name)
    tsv_path = os.path.splitext(csv_path)[0] + ".tsv"

    merged.to_csv( csv_path, index=False, float_format="%.4f", )
    merged.to_csv( tsv_path, sep="\t", index=False, float_format="%.4f", )

    print(f"[eval] combined summary CSV -> {csv_path}")
    print(f"[eval] combined summary TSV -> {tsv_path}")

    checkpoint_name = os.path.basename(os.path.abspath(run_dir))
    run_path = Path(run_dir)
    if len(run_path.parents) >= 5:
        experiment_dir = str(run_path.parents[4])
        write_all_model_reports( experiment_dir=experiment_dir, checkpoint_name=checkpoint_name, )

    return merged


# =============================================================================
# COMPATIBILITY HELPERS
# =============================================================================


def get_eval_run_dir( exp_cfg, ckpt_stem: str, ) -> str:
    return os.path.join( exp_cfg["output"]["eval_dir"], ckpt_stem, )


def get_eval_tag_dir( exp_cfg, ckpt_stem: str, tag: str, ) -> str:
    return os.path.join( get_eval_run_dir(exp_cfg, ckpt_stem), tag, )


def match_events_counts( gt_events, pred_events, class_names, iou_threshold=DEFAULT_IOU_THRESHOLD, ):
    counts, _ = match_events_by_iou( gt_events, pred_events, class_names, iou_threshold, )
    return counts


def match_events_old_iou(gt_events,pred_events,iou_threshold=DEFAULT_IOU_THRESHOLD):
    classes = sorted({ str(event[0]) for event in list(gt_events) + list(pred_events) })
    counts, _ = match_events_by_iou( gt_events, pred_events, classes, iou_threshold, )
    tp = sum(value["TP"] for value in counts.values())
    fp = sum(value["FP"] for value in counts.values())
    fn = sum(value["FN"] for value in counts.values())
    return tp, fp, fn


def compute_sequence_consistent_metrics_from_csv(
    csv_path: str,
    label_dict: Dict[str, int],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    include_noise: bool = False,
    output_dir: Optional[str] = None,
    model_name: Optional[str] = None,
    environment: Optional[str] = None,
    save: bool = False,
    metadata_csv_path: Optional[str] = None,
    label_space: str = "species",
):
    df = pd.read_csv(csv_path)
    overall, per_class, _ = (
        evaluate_sequence_metrics_from_df(
            df=df,
            label_dict=label_dict,
            iou_threshold=iou_threshold,
            include_noise=include_noise,
            output_dir=output_dir,
            model_name=model_name,
            environment=environment,
            save=save,
            metadata_csv_path=metadata_csv_path,
            label_space=label_space,
        )
    )
    return overall, per_class


def compute_detection_temporal_metrics_from_csv(
    csv_path: str,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    output_dir: Optional[str] = None,
    model_name: Optional[str] = None,
    environment: Optional[str] = None,
    save: bool = False,
    metadata_csv_path: Optional[str] = None,
):
    df = pd.read_csv(csv_path)
    overall, per_class, _ = (
        evaluate_event_detection_from_df(
            raw_df=df,
            iou_threshold=iou_threshold,
            output_dir=output_dir,
            model_name=model_name,
            environment=environment,
            save=save,
            metadata_csv_path=metadata_csv_path,
        )
    )
    return overall, per_class


def convert_csv_to_binary(input_csv_path,output_csv_path,mosquito_labels=None):
    df = _normalise_prediction_columns(pd.read_csv(input_csv_path))

    if mosquito_labels is None:
        mapper = to_detection_label
    else:
        mosquito_labels = set(map(str, mosquito_labels))

        def mapper(value):
            return ( "wingbeat" if str(value) in mosquito_labels else "background" )

    for column in ["true_label","predicted_label"]:
        df[column] = df[column].apply(mapper)

    ensure_dir(os.path.dirname(output_csv_path) or ".")
    df.to_csv(output_csv_path,index=False)
    print(f"Saved binary prediction CSV to: {output_csv_path}")

    return output_csv_path


def save_summary_tsv(rows,output_path,essential_only=False):
    del essential_only

    ensure_dir(os.path.dirname(output_path) or ".")
    df = pd.DataFrame(rows).copy()

    if len(df) and "No" in df.columns:
        df = df.drop(columns=["No"])
    if len(df):
        df.insert(0, "No", range(1, len(df) + 1), )

    df.to_csv(output_path,sep="\t",index=False)
    print(f"Saved summary TSV to: {output_path}")
    return df
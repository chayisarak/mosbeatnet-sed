from pathlib import Path
from collections import Counter
import csv

metadata_dir = Path("/home/mosbnet/mosdet/mosbeatnet_main/data/simulated_data/gen_xdomain/metadata")
segment_duration = 0.5
species_output = metadata_dir / "class_distribution_species.csv"
species_sex_output = metadata_dir / "class_distribution_species_sex.csv"

def get_one_value(rows, column):
    values = {row.get(column, "").strip() for row in rows if row.get(column, "").strip()}
    if len(values) == 1:
        return next(iter(values))
    if len(values) == 0:
        return ""
    return "mixed"

def get_env(rows):
    for column in ["simulation_environment", "environment", "background_type"]:
        value = get_one_value(rows, column)
        if value and value != "mixed":
            return value
    return ""

def is_noise(row):
    noise_values = {"", "-", "noise", "nan", "none", "na", "n/a", "<na>"}
    return (
        row.get("event_type", "").strip().lower() == "noise"
        or row.get("label", "").strip().lower() in noise_values
        or row.get("species", "").strip().lower() in noise_values
    )

def clean_sex(value):
    low = str(value).strip().lower()
    if low in {"f", "female"}:
        return "F"
    if low in {"m", "male"}:
        return "M"
    if low in {"", "-", "u", "unknown", "unk", "nan", "none", "na", "n/a", "<na>"}:
        return "U"
    return str(value).strip().upper()

def get_species_label(row):
    if is_noise(row):
        return "Noise"
    species = row.get("species", "").strip()
    if species:
        return species
    label = row.get("label", "").strip()
    base, sep, suffix = label.rpartition("_")
    return base if sep and suffix in {"F", "M", "U"} else label

def get_species_sex_label(row):
    if is_noise(row):
        return "Noise"
    species_sex = row.get("species_sex", "").strip()
    if species_sex:
        return species_sex
    species = row.get("species", "").strip()
    if species:
        return f"{species}_{clean_sex(row.get('sex', ''))}"
    return row.get("label", "").strip()

def count_labels(rows, view):
    counts = Counter()
    for row in rows:
        label = get_species_label(row) if view == "species" else get_species_sex_label(row)
        if label:
            counts[label] += 1
    return counts

def save_distribution(output_path, data, labels):
    labels = sorted(labels)
    fields = ["set_name", "dataset_type", "source_name", "env", "background_type", "metadata_target_type", "evaluation_view"] + labels + ["total_segments", "duration_sec", "n_file"]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in data:
            row = {
                "set_name": item["set_name"],
                "dataset_type": item["dataset_type"],
                "source_name": item["source_name"],
                "env": item["env"],
                "background_type": item["background_type"],
                "metadata_target_type": item["metadata_target_type"],
                "evaluation_view": item["evaluation_view"],
            }
            for label in labels:
                row[label] = item["counts"].get(label, 0)
            row["total_segments"] = item["total_segments"]
            row["duration_sec"] = item["duration_sec"]
            row["n_file"] = item["n_file"]
            writer.writerow(row)

csv_files = sorted(metadata_dir.glob("metadata_*.csv"))
species_sets, species_sex_sets = [], []
species_labels, species_sex_labels = set(), set()

print("Metadata folder :", metadata_dir.resolve())
print("Metadata files  :", len(csv_files))

for csv_path in csv_files:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "label" not in rows[0]:
        continue

    dataset_type = get_one_value(rows, "dataset_type").lower()
    target_type = get_one_value(rows, "target_type").lower()
    source_name = get_one_value(rows, "source_name")
    background_type = get_one_value(rows, "background_type")
    env = get_env(rows)
    n_file = len({row.get("file_name", "").strip() for row in rows if row.get("file_name", "").strip()})

    if dataset_type == "test":
        views = ["species"]
        name_lower = csv_path.name.lower()
        source_lower = source_name.lower()
        is_humbug = "humbug" in name_lower or "humbug" in source_lower
        is_biodcase = "biodcase" in name_lower or "biodcase" in source_lower
        if not is_humbug and not is_biodcase:
            views.append("species_sex")
    elif target_type in {"species", "species_sex"}:
        views = [target_type]
    else:
        print("Skip unknown target_type:", csv_path.name, "->", target_type)
        continue

    for view in views:
        counts = count_labels(rows, view)
        total_segments = sum(counts.values())
        duration_sec = total_segments * segment_duration

        info = {
            "set_name": csv_path.stem,
            "dataset_type": dataset_type,
            "source_name": source_name,
            "env": env,
            "background_type": background_type,
            "metadata_target_type": target_type,
            "evaluation_view": view,
            "counts": counts,
            "total_segments": total_segments,
            "duration_sec": duration_sec,
            "n_file": n_file,
        }

        if view == "species":
            species_sets.append(info)
            species_labels.update(counts.keys())
        else:
            species_sex_sets.append(info)
            species_sex_labels.update(counts.keys())

        print("=" * 100)
        print(csv_path.name)
        print(f"dataset_type={dataset_type} | target_type={target_type} | view={view} | source={source_name} | env={env} | background={background_type}")
        print(f"n_segment={total_segments:,} | duration={duration_sec:,.1f} sec | n_file={n_file:,}")
        print(f"{'class':35s}{'n_segment':>15s}{'duration_sec':>17s}")
        print("-" * 67)
        for label in sorted(counts):
            n = counts[label]
            print(f"{label:35s}{n:15,d}{n * segment_duration:17,.1f}")

save_distribution(species_output, species_sets, species_labels)
save_distribution(species_sex_output, species_sex_sets, species_sex_labels)

print("=" * 100)
print("Saved:")
print(species_output.resolve())
print(species_sex_output.resolve())
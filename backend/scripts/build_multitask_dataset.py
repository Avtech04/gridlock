from __future__ import annotations

import argparse
import re
import random
import shutil
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"PyYAML is required. Use the project venv. Details: {exc}")


CLASS_NAMES = [
    "helmet",
    "no_helmet",
    "license_plate",
    "triple_riding",
    "seatbelt",
    "no_seatbelt",
    "red_light",
    "green_light",
    "yellow_light",
    "stop_line",
    "no_stopping_sign",
    "no_entry_sign",
    "stop_sign",
    "illegal_parking",
    "wrong_side",
    "right_side",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "auto_rickshaw",
    "person",
]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}

NAME_MAP = {
    "with helmet": "helmet",
    "withhelmet": "helmet",
    "helmet": "helmet",
    "without helmet": "no_helmet",
    "withouthelmet": "no_helmet",
    "no helmet": "no_helmet",
    "no_helmet": "no_helmet",
    "plate": "license_plate",
    "licence": "license_plate",
    "license": "license_plate",
    "license plate": "license_plate",
    "licence plate": "license_plate",
    "number plate": "license_plate",
    "number_plate": "license_plate",
    "registration plate": "license_plate",
    "tripleriding": "triple_riding",
    "triple riding": "triple_riding",
    "triple_riding": "triple_riding",
    "tripple ridding": "triple_riding",
    "tripple riding": "triple_riding",
    "tripple-ridding": "triple_riding",
    "seatbelt": "seatbelt",
    "seat belt": "seatbelt",
    "person seatbelt": "seatbelt",
    "person-seatbelt": "seatbelt",
    "no seatbelt": "no_seatbelt",
    "no_seatbelt": "no_seatbelt",
    "without seatbelt": "no_seatbelt",
    "person noseatbelt": "no_seatbelt",
    "person no seatbelt": "no_seatbelt",
    "person-noseatbelt": "no_seatbelt",
    "red light": "red_light",
    "red_light": "red_light",
    "green light": "green_light",
    "green_light": "green_light",
    "yellow light": "yellow_light",
    "yellow_light": "yellow_light",
    "stop line": "stop_line",
    "stop_line": "stop_line",
    "zebra crossing": "stop_line",
    "crosswalk": "stop_line",
    "no stopping": "no_stopping_sign",
    "no_stopping": "no_stopping_sign",
    "no parking": "no_stopping_sign",
    "no parking sign": "no_stopping_sign",
    "no entry": "no_entry_sign",
    "no_entry": "no_entry_sign",
    "stop sign": "stop_sign",
    "stop_sign": "stop_sign",
    "illegally parked": "illegal_parking",
    "illegally_parked": "illegal_parking",
    "illegal parking": "illegal_parking",
    "parked vehicle": "illegal_parking",
    "wrong side": "wrong_side",
    "wrong-side": "wrong_side",
    "wrong_side": "wrong_side",
    "wrong way": "wrong_side",
    "right side": "right_side",
    "right-side": "right_side",
    "right_side": "right_side",
    "correct side": "right_side",
    "car": "car",
    "bus": "bus",
    "truck": "truck",
    "person": "person",
    "pedestrian": "person",
    "bike": "motorcycle",
    "motobike": "motorcycle",
    "motorbike": "motorcycle",
    "motorcycle": "motorcycle",
    "two wheeler": "motorcycle",
    "two-wheeler": "motorcycle",
    "auto": "auto_rickshaw",
    "auto rickshaw": "auto_rickshaw",
    "autorickshaw": "auto_rickshaw",
    "rikshaw": "auto_rickshaw",
    "rickshaw": "auto_rickshaw",
}

TRAIN_LIMITS = {
    "helmet": 1800,
    "no_helmet": 3200,
    "license_plate": 3200,
    "triple_riding": 2600,
    "seatbelt": 1400,
    "no_seatbelt": 1400,
    "red_light": 1800,
    "green_light": 1400,
    "yellow_light": 900,
    "stop_line": 2200,
    "no_stopping_sign": 300,
    "no_entry_sign": 220,
    "stop_sign": 360,
    "illegal_parking": 700,
    "wrong_side": 900,
    "right_side": 1200,
    "car": 1800,
    "motorcycle": 1800,
    "bus": 800,
    "truck": 800,
    "auto_rickshaw": 600,
    "person": 1000,
}
VAL_LIMITS = {
    "helmet": 350,
    "no_helmet": 600,
    "license_plate": 600,
    "triple_riding": 450,
    "seatbelt": 350,
    "no_seatbelt": 350,
    "red_light": 450,
    "green_light": 350,
    "yellow_light": 220,
    "stop_line": 450,
    "no_stopping_sign": 100,
    "no_entry_sign": 80,
    "stop_sign": 100,
    "illegal_parking": 180,
    "wrong_side": 220,
    "right_side": 350,
    "car": 450,
    "motorcycle": 450,
    "bus": 220,
    "truck": 220,
    "auto_rickshaw": 180,
    "person": 280,
}

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


@dataclass(frozen=True)
class Record:
    source: str
    split: str
    image: Path
    rows: tuple[str, ...]
    class_counts: Counter[str]


def norm_name(value: str) -> str:
    return value.strip().lower().replace("-", " ").replace("_", " ")


def target_for_name(value: str) -> str | None:
    return NAME_MAP.get(norm_name(value))


def yolo_geometry_to_bbox(parts: list[str]) -> list[str] | None:
    try:
        values = [float(value) for value in parts[1:]]
    except ValueError:
        return None
    if len(values) == 4:
        xc, yc, width, height = values
    elif len(values) >= 6 and len(values) % 2 == 0:
        xs = values[0::2]
        ys = values[1::2]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        xc = (x1 + x2) / 2
        yc = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
    else:
        return None
    xc = max(0.0, min(1.0, xc))
    yc = max(0.0, min(1.0, yc))
    width = max(0.0001, min(1.0, width))
    height = max(0.0001, min(1.0, height))
    return [f"{xc:.6f}", f"{yc:.6f}", f"{width:.6f}", f"{height:.6f}"]


def yolo_row_target(row: str, names: dict[int, str]) -> tuple[str, str] | None:
    parts = row.strip().split()
    if len(parts) < 5:
        return None
    try:
        source_id = int(float(parts[0]))
    except ValueError:
        return None
    target_name = target_for_name(names.get(source_id, str(source_id)))
    if target_name is None:
        return None
    bbox = yolo_geometry_to_bbox(parts)
    if bbox is None:
        return None
    return target_name, " ".join([str(CLASS_TO_ID[target_name]), *bbox])


def names_from_yaml(path: Path) -> dict[int, str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace")) or {}
    names = data.get("names", {})
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    return {int(idx): str(name) for idx, name in names.items()}


def split_from_path(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    return "train" if "train" in parts else "val"


def find_yolo_image(label_path: Path) -> Path | None:
    image_dir = Path(str(label_path.parent).replace("/labels", "/images"))
    for ext in IMAGE_EXTS:
        candidate = image_dir / f"{label_path.stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def collect_yolo_dataset(source: str, dataset_yaml: Path) -> list[Record]:
    names = names_from_yaml(dataset_yaml)
    records: list[Record] = []
    root = dataset_yaml.parent
    for label_path in sorted(root.rglob("*.txt")):
        if "labels" not in {part.lower() for part in label_path.parts}:
            continue
        image_path = find_yolo_image(label_path)
        if image_path is None:
            continue
        rows: list[str] = []
        counts: Counter[str] = Counter()
        for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            mapped = yolo_row_target(line, names)
            if mapped is None:
                continue
            target_name, out_row = mapped
            rows.append(out_row)
            counts[target_name] += 1
        if rows:
            records.append(Record(source, split_from_path(label_path), image_path, tuple(rows), counts))
    return records


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")[:80]


def safe_extract_zip(zip_path: Path, output_dir: Path) -> Path:
    target = output_dir / slug(zip_path.stem)
    marker = target / ".extracted"
    if marker.exists():
        return target
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                continue
            archive.extract(member, target)
    marker.write_text(zip_path.name + "\n", encoding="utf-8")
    return target


def collect_roboflow_zips(zip_roots: list[Path], extract_root: Path) -> list[Record]:
    records: list[Record] = []
    zip_paths: list[Path] = []
    for root in zip_roots:
        if root.is_file() and root.suffix == ".zip":
            zip_paths.append(root)
        elif root.exists():
            zip_paths.extend(sorted(path for path in root.glob("*.zip") if ".yolo" in path.name.lower()))
    for zip_path in sorted(set(zip_paths)):
        extracted = safe_extract_zip(zip_path, extract_root)
        for dataset_yaml in sorted(extracted.rglob("data.yaml")):
            source = slug(zip_path.stem)
            records.extend(collect_yolo_dataset(source, dataset_yaml))
    return records


def find_voc_image(xml_path: Path, root: Path, filename: str) -> Path | None:
    candidates = [
        xml_path.with_name(filename),
        root / filename,
        root / "images" / filename,
        root / "Images" / filename,
        root / "JPEGImages" / filename,
        root / "Indian_Number_Plates" / "Sample_Images" / filename,
        root / "number_plate_images_ocr" / "number_plate_images_ocr" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(root.rglob(filename))
    return matches[0] if matches else None


def voc_box_to_yolo(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    width: float,
    height: float,
) -> tuple[float, float, float, float] | None:
    xmin = max(0.0, min(xmin, width))
    xmax = max(0.0, min(xmax, width))
    ymin = max(0.0, min(ymin, height))
    ymax = max(0.0, min(ymax, height))
    if xmax <= xmin or ymax <= ymin or width <= 0 or height <= 0:
        return None
    return (
        ((xmin + xmax) / 2) / width,
        ((ymin + ymax) / 2) / height,
        (xmax - xmin) / width,
        (ymax - ymin) / height,
    )


def collect_voc_dataset(source: str, root: Path, seed: int, train_split: float = 0.85) -> list[Record]:
    records: list[Record] = []
    xml_files = sorted(root.rglob("*.xml"))
    random.Random(seed).shuffle(xml_files)
    train_cutoff = int(len(xml_files) * train_split)
    for index, xml_path in enumerate(xml_files):
        try:
            ann = ET.parse(xml_path).getroot()
        except ET.ParseError:
            continue
        filename = ann.findtext("filename")
        size = ann.find("size")
        if not filename or size is None:
            continue
        image_path = find_voc_image(xml_path, root, filename)
        if image_path is None:
            continue
        width = float(size.findtext("width", "0") or 0)
        height = float(size.findtext("height", "0") or 0)
        rows: list[str] = []
        counts: Counter[str] = Counter()
        for obj in ann.findall("object"):
            target_name = target_for_name(obj.findtext("name", ""))
            if target_name is None:
                continue
            bbox = obj.find("bndbox")
            if bbox is None:
                continue
            box = voc_box_to_yolo(
                float(bbox.findtext("xmin", "0") or 0),
                float(bbox.findtext("ymin", "0") or 0),
                float(bbox.findtext("xmax", "0") or 0),
                float(bbox.findtext("ymax", "0") or 0),
                width,
                height,
            )
            if box is None:
                continue
            rows.append(f"{CLASS_TO_ID[target_name]} " + " ".join(f"{value:.6f}" for value in box))
            counts[target_name] += 1
        if rows:
            split = "train" if index < train_cutoff else "val"
            records.append(Record(source, split, image_path, tuple(rows), counts))
    return records


def select_balanced(records: list[Record], limits: dict[str, int], seed: int) -> list[Record]:
    shuffled = records[:]
    random.Random(seed).shuffle(shuffled)
    selected: list[Record] = []
    counts: Counter[str] = Counter()
    for record in shuffled:
        if any(counts[name] < limits.get(name, 0) for name in record.class_counts):
            selected.append(record)
            counts.update(record.class_counts)
    return selected


def write_dataset(records: list[Record], output: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    seen_names: Counter[str] = Counter()
    for split in ["train", "val"]:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    for record in records:
        split = record.split
        stem = f"{record.source}_{record.image.stem}"
        seen_names[stem] += 1
        if seen_names[stem] > 1:
            stem = f"{stem}_{seen_names[stem]}"
        out_image = output / "images" / split / f"{stem}{record.image.suffix.lower()}"
        out_label = output / "labels" / split / f"{stem}.txt"
        shutil.copy2(record.image, out_image)
        out_label.write_text("\n".join(record.rows) + "\n", encoding="utf-8")
        counts.update(record.class_counts)
    return counts


def write_yaml(output: Path) -> None:
    lines = [
        f"path: {output.resolve()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    lines.extend(f"  {idx}: {name}" for idx, name in enumerate(CLASS_NAMES))
    (output / "dataset.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_all(raw_from_user: Path, raw_extra: Path, seed: int) -> list[Record]:
    records: list[Record] = []
    if raw_from_user.exists():
        records.extend(collect_voc_dataset("old_helmet", raw_from_user / "helmet_detection", seed))
        records.extend(collect_voc_dataset("old_carplate", raw_from_user / "car_plate_detection", seed))
        records.extend(collect_voc_dataset("old_indianplate", raw_from_user / "indian_number_plates", seed))

    yaml_paths = [
        raw_extra / "traffic_two_wheeler" / "master_traffic_violation_dataset" / "data.yaml",
        raw_extra / "helmet_violations" / "HelmetViolations" / "data.yaml",
        raw_extra / "helmet_violations" / "HelmetViolationsV2" / "data.yaml",
        raw_extra / "seatbelt_dms" / "data.yaml",
        raw_extra / "traffic_violation_23class" / "data.yaml",
    ]
    for yaml_path in yaml_paths:
        if yaml_path.exists():
            records.extend(collect_yolo_dataset(yaml_path.parent.name, yaml_path))
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build balanced YOLO dataset for traffic violation evidence.")
    parser.add_argument("--raw-from-user", type=Path, default=Path("data/raw_from_user/traffic_datasets"))
    parser.add_argument("--raw-extra", type=Path, default=Path("data/raw_extra/traffic_extra_datasets"))
    parser.add_argument("--roboflow-zip-root", action="append", type=Path, default=[Path(".")])
    parser.add_argument("--roboflow-extract-root", type=Path, default=Path("data/raw_extra/roboflow_extracted"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/traffic_multitask_detector"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = collect_all(args.raw_from_user, args.raw_extra, args.seed)
    records.extend(collect_roboflow_zips(args.roboflow_zip_root, args.roboflow_extract_root))
    train_records = [record for record in records if record.split == "train"]
    val_records = [record for record in records if record.split != "train"]
    selected_train = select_balanced(train_records, TRAIN_LIMITS, args.seed)
    selected_val = select_balanced(val_records, VAL_LIMITS, args.seed + 1)
    args.output.mkdir(parents=True, exist_ok=True)
    train_counts = write_dataset(selected_train, args.output)
    val_counts = write_dataset(selected_val, args.output)
    write_yaml(args.output)
    print(f"source_records={len(records)} train_candidates={len(train_records)} val_candidates={len(val_records)}")
    print(f"selected_train={len(selected_train)} selected_val={len(selected_val)}")
    print("train_counts=" + ", ".join(f"{name}:{train_counts[name]}" for name in CLASS_NAMES))
    print("val_counts=" + ", ".join(f"{name}:{val_counts[name]}" for name in CLASS_NAMES))
    print(f"dataset_yaml={args.output / 'dataset.yaml'}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


CLASS_NAMES = ["helmet", "no_helmet", "license_plate"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}

SOURCE_CLASS_MAP = {
    "with helmet": "helmet",
    "helmet": "helmet",
    "without helmet": "no_helmet",
    "no helmet": "no_helmet",
    "no_helmet": "no_helmet",
    "licence": "license_plate",
    "license": "license_plate",
    "license plate": "license_plate",
    "licence plate": "license_plate",
    "number plate": "license_plate",
    "number_plate": "license_plate",
    "plate": "license_plate",
}


@dataclass(frozen=True)
class SourceSpec:
    name: str
    root: Path


def norm_name(value: str) -> str:
    return value.strip().lower().replace("-", " ").replace("_", " ")


def find_image(xml_path: Path, root: Path, filename: str) -> Path | None:
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


def voc_to_yolo_box(
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


def parse_annotation(xml_path: Path, root: Path) -> tuple[Path, list[str]] | None:
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return None

    ann = tree.getroot()
    filename = ann.findtext("filename")
    size = ann.find("size")
    if not filename or size is None:
        return None

    width = float(size.findtext("width", "0") or 0)
    height = float(size.findtext("height", "0") or 0)
    image_path = find_image(xml_path, root, filename)
    if image_path is None:
        return None

    rows: list[str] = []
    for obj in ann.findall("object"):
        source_name = norm_name(obj.findtext("name", ""))
        target_name = SOURCE_CLASS_MAP.get(source_name)
        if target_name is None:
            continue
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        box = voc_to_yolo_box(
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

    return (image_path, rows) if rows else None


def unique_output_name(source_name: str, image_path: Path) -> str:
    return f"{source_name}_{image_path.stem}{image_path.suffix.lower()}"


def build_dataset(sources: list[SourceSpec], output_dir: Path, train_split: float, seed: int) -> None:
    records: list[tuple[str, Path, list[str]]] = []
    skipped = 0
    class_counts: Counter[str] = Counter()

    for source in sources:
        xml_files = sorted(source.root.rglob("*.xml"))
        for xml_path in xml_files:
            parsed = parse_annotation(xml_path, source.root)
            if parsed is None:
                skipped += 1
                continue
            image_path, rows = parsed
            records.append((source.name, image_path, rows))
            for row in rows:
                class_counts[CLASS_NAMES[int(row.split()[0])]] += 1

    if not records:
        raise SystemExit("No usable VOC annotations found for specialist classes.")

    random.seed(seed)
    random.shuffle(records)
    split_index = int(len(records) * train_split)
    partitions = {
        "train": records[:split_index],
        "val": records[split_index:],
    }

    for partition in partitions:
        (output_dir / "images" / partition).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / partition).mkdir(parents=True, exist_ok=True)

    for partition, partition_records in partitions.items():
        for source_name, image_path, rows in partition_records:
            out_name = unique_output_name(source_name, image_path)
            out_image = output_dir / "images" / partition / out_name
            out_label = output_dir / "labels" / partition / f"{Path(out_name).stem}.txt"
            shutil.copy2(image_path, out_image)
            out_label.write_text("\n".join(rows) + "\n", encoding="utf-8")

    yaml_lines = [
        f"path: {output_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    yaml_lines.extend(f"  {idx}: {name}" for idx, name in enumerate(CLASS_NAMES))
    (output_dir / "dataset.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    print(f"records={len(records)} skipped={skipped}")
    print(f"train={len(partitions['train'])} val={len(partitions['val'])}")
    print("class_counts=" + ", ".join(f"{name}:{class_counts[name]}" for name in CLASS_NAMES))
    print(f"dataset_yaml={output_dir / 'dataset.yaml'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build merged YOLO dataset for helmet and plate specialist detection.")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw_from_user/traffic_datasets"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/specialist_detector"))
    parser.add_argument("--train-split", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = [
        SourceSpec("helmet", args.raw_root / "helmet_detection"),
        SourceSpec("carplate", args.raw_root / "car_plate_detection"),
        SourceSpec("indianplate", args.raw_root / "indian_number_plates"),
    ]
    missing = [str(source.root) for source in sources if not source.root.exists()]
    if missing:
        raise SystemExit("Missing source folders: " + ", ".join(missing))
    build_dataset(sources, args.output, args.train_split, args.seed)


if __name__ == "__main__":
    main()

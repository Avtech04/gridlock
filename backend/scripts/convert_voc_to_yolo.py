from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_class_map(items: list[str]) -> dict[str, str]:
    mapping = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Invalid class map item: {item}. Use source=target.")
        source, target = item.split("=", 1)
        mapping[source.strip().lower()] = target.strip()
    return mapping


def find_image(xml_path: Path, root: Path, filename: str) -> Path | None:
    candidates = [
        xml_path.with_name(filename),
        root / filename,
        root / "images" / filename,
        root / "Images" / filename,
        root / "JPEGImages" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(root.rglob(filename))
    return matches[0] if matches else None


def voc_to_yolo_box(box: tuple[float, float, float, float], width: float, height: float) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = box
    x_center = ((xmin + xmax) / 2) / width
    y_center = ((ymin + ymax) / 2) / height
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height
    return x_center, y_center, box_width, box_height


def convert(input_dir: Path, output_dir: Path, class_map: dict[str, str], split: float, seed: int) -> None:
    xml_files = sorted(input_dir.rglob("*.xml"))
    if not xml_files:
        raise SystemExit(f"No Pascal VOC XML files found under {input_dir}")

    target_classes = sorted(set(class_map.values()))
    class_to_id = {name: idx for idx, name in enumerate(target_classes)}

    random.seed(seed)
    random.shuffle(xml_files)
    train_count = int(len(xml_files) * split)
    partitions = {
        "train": xml_files[:train_count],
        "val": xml_files[train_count:],
    }

    for partition in partitions:
        (output_dir / "images" / partition).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / partition).mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped = 0
    for partition, files in partitions.items():
        for xml_path in files:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            filename_node = root.find("filename")
            size_node = root.find("size")
            if filename_node is None or size_node is None:
                skipped += 1
                continue

            width = float(size_node.findtext("width", "0"))
            height = float(size_node.findtext("height", "0"))
            if width <= 0 or height <= 0:
                skipped += 1
                continue

            image_path = find_image(xml_path, input_dir, filename_node.text or "")
            if image_path is None:
                skipped += 1
                continue

            rows = []
            for obj in root.findall("object"):
                source_name = (obj.findtext("name") or "").strip().lower()
                if source_name not in class_map:
                    continue
                bbox = obj.find("bndbox")
                if bbox is None:
                    continue
                xmin = float(bbox.findtext("xmin", "0"))
                ymin = float(bbox.findtext("ymin", "0"))
                xmax = float(bbox.findtext("xmax", "0"))
                ymax = float(bbox.findtext("ymax", "0"))
                if xmax <= xmin or ymax <= ymin:
                    continue
                target_name = class_map[source_name]
                yolo = voc_to_yolo_box((xmin, ymin, xmax, ymax), width, height)
                rows.append(f"{class_to_id[target_name]} " + " ".join(f"{v:.6f}" for v in yolo))

            if not rows:
                skipped += 1
                continue

            out_image = output_dir / "images" / partition / image_path.name
            out_label = output_dir / "labels" / partition / f"{image_path.stem}.txt"
            shutil.copy2(image_path, out_image)
            out_label.write_text("\n".join(rows) + "\n", encoding="utf-8")
            kept += 1

    yaml_text = [
        f"path: {output_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for name, idx in class_to_id.items():
        yaml_text.append(f"  {idx}: {name}")
    (output_dir / "dataset.yaml").write_text("\n".join(yaml_text) + "\n", encoding="utf-8")

    print(f"Converted {kept} images. Skipped {skipped}.")
    print(f"Dataset YAML: {output_dir / 'dataset.yaml'}")
    print("Classes:", class_to_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Pascal VOC XML annotations to YOLO format.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--class-map", nargs="+", required=True, help="source=target pairs")
    parser.add_argument("--train-split", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    convert(args.input, args.output, parse_class_map(args.class_map), args.train_split, args.seed)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Pascal VOC class names.")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()

    counter = Counter()
    for xml_path in args.path.rglob("*.xml"):
        root = ET.parse(xml_path).getroot()
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip()
            if name:
                counter[name] += 1

    for name, count in counter.most_common():
        print(f"{name}\t{count}")


if __name__ == "__main__":
    main()

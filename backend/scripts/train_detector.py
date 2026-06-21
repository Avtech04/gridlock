from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Gridlock traffic violation detector.")
    parser.add_argument("--data", default="backend/datasets/traffic_violations.yaml", help="YOLO dataset YAML path")
    parser.add_argument("--model", default="yolo26s.pt", help="Base model or checkpoint")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", default=-1, help="Batch size. Use -1 for Ultralytics auto batch.")
    parser.add_argument("--device", default=None, help="cuda device id, cpu, or mps")
    parser.add_argument("--project", default="backend/runs/train")
    parser.add_argument("--name", default="gridlock_detector")
    parser.add_argument("--cache", default=False, help="Ultralytics cache mode: False, ram, or disk")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of training data to use")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise SystemExit(f"Ultralytics is not installed. Run pip install -r backend/requirements.txt. Details: {exc}")

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Dataset YAML not found: {data_path}")

    model = YOLO(args.model)
    train_args = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": int(args.batch),
        "project": args.project,
        "name": args.name,
        "patience": 25,
        "cos_lr": True,
        "close_mosaic": 15,
        "multi_scale": 0.15,
        "plots": True,
        "cache": args.cache,
        "fraction": args.fraction,
    }
    if args.device:
        train_args["device"] = args.device
    model.train(**train_args)


if __name__ == "__main__":
    main()

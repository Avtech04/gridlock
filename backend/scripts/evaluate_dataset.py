from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
import uuid
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from evaluator import evaluate_predictions, load_ground_truth  # noqa: E402
from main import DB_PATH, run_analysis  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Gridlock predictions against labelled ground truth.")
    parser.add_argument("--images", required=True, help="Directory containing evaluation images")
    parser.add_argument("--ground-truth", required=True, help="Ground-truth JSON path")
    parser.add_argument("--dataset-name", default="local-eval")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--write-db", action="store_true", help="Write metrics into violations.db for the UI")
    parser.add_argument("--output", default=None, help="Optional metrics JSON output path")
    return parser.parse_args()


async def predict_images(images_dir: Path, ground_truth: dict[str, list[dict]]) -> dict[str, list[dict]]:
    predictions = {}
    for image_name in sorted(ground_truth):
        image_path = images_dir / image_name
        if not image_path.exists():
            print(f"[warn] missing image: {image_path}")
            predictions[image_name] = []
            continue
        session_id = f"eval_{uuid.uuid4().hex[:8]}"
        analysis = await run_analysis(str(image_path), session_id)
        predictions[image_name] = analysis.get("violations", [])
        print(f"[eval] {image_name}: {len(predictions[image_name])} predictions")
    return predictions


def write_db(dataset_name: str, metrics: dict) -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluation_runs (
                id TEXT PRIMARY KEY,
                dataset_name TEXT,
                metrics_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO evaluation_runs (id, dataset_name, metrics_json, created_at) VALUES (?, ?, ?, ?)",
            (uuid.uuid4().hex[:12], dataset_name, json.dumps(metrics), datetime.now().isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


async def main_async() -> None:
    args = parse_args()
    images_dir = Path(args.images)
    ground_truth = load_ground_truth(args.ground_truth)
    predictions = await predict_images(images_dir, ground_truth)
    metrics = evaluate_predictions(ground_truth, predictions, iou_threshold=args.iou)
    metrics["dataset_name"] = args.dataset_name

    print(json.dumps(metrics, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if args.write_db:
        write_db(args.dataset_name, metrics)


if __name__ == "__main__":
    asyncio.run(main_async())

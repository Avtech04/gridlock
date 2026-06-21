"""
Gridlock AI Enforcer API.

Pipeline:
1. Image quality analysis and conservative preprocessing.
2. Vehicle, road-user, plate, and safety-equipment detection.
3. Optional OCR for detected plates.
4. Rule-based violation mapping with optional vision-model review.
5. Evidence rendering, storage, analytics, and real ground-truth evaluation hooks.
"""

from __future__ import annotations

import base64
import json
import shutil
import sqlite3
import time
import uuid
import zipfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np
import yaml
from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from detector import detect, get_detector_status
from ocr import ocr_status, recognize_plates
from video_analyzer import analyze_video_file
from violation_mapper import VALID_VIOLATION_TYPES, analyze_violations

try:
    from google import genai
except Exception:  # pragma: no cover - optional dependency
    genai = None

try:
    from groq import Groq
except Exception:  # pragma: no cover - optional dependency
    Groq = None


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.yaml"
UPLOAD_DIR = BASE_DIR / "uploads"
ANNOTATED_DIR = BASE_DIR / "annotated"
WEB_DIR = BASE_DIR.parent / "web"
DB_PATH = BASE_DIR / "violations.db"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_BATCH_IMAGES = 60
MAX_ZIP_BYTES = 250 * 1024 * 1024
MAX_ZIP_UNCOMPRESSED_BYTES = 500 * 1024 * 1024
MAX_BATCH_IMAGE_BYTES = 30 * 1024 * 1024

UPLOAD_DIR.mkdir(exist_ok=True)
ANNOTATED_DIR.mkdir(exist_ok=True)

with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
    CFG = yaml.safe_load(handle) or {}


app = FastAPI(title="Gridlock AI Enforcer API", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/annotated", StaticFiles(directory=str(ANNOTATED_DIR)), name="annotated")
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
if WEB_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(WEB_DIR), html=True), name="ui")


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    with get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS violations (
                id TEXT PRIMARY KEY,
                image_id TEXT NOT NULL,
                original_image TEXT,
                annotated_image TEXT,
                violation_type TEXT NOT NULL,
                confidence REAL,
                license_plate TEXT,
                vehicle_type TEXT,
                bbox_x REAL,
                bbox_y REAL,
                bbox_w REAL,
                bbox_h REAL,
                timestamp TEXT NOT NULL,
                preprocessing_applied TEXT,
                description TEXT,
                evidence TEXT,
                review_status TEXT DEFAULT 'pending',
                review_comment TEXT,
                reviewed_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_sessions (
                id TEXT PRIMARY KEY,
                original_image TEXT,
                annotated_image TEXT,
                total_vehicles INTEGER,
                total_violations INTEGER,
                total_pedestrians INTEGER,
                processing_time_ms REAL,
                preprocessing_applied TEXT,
                timestamp TEXT NOT NULL,
                raw_ai_response TEXT,
                review_status TEXT DEFAULT 'pending',
                approved_at TEXT,
                review_note TEXT
            )
            """
        )
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
        # Older local databases may not have review/evidence columns.
        existing = {row[1] for row in conn.execute("PRAGMA table_info(violations)").fetchall()}
        violation_columns = {
            "evidence": "ALTER TABLE violations ADD COLUMN evidence TEXT",
            "review_status": "ALTER TABLE violations ADD COLUMN review_status TEXT DEFAULT 'approved'",
            "review_comment": "ALTER TABLE violations ADD COLUMN review_comment TEXT",
            "reviewed_at": "ALTER TABLE violations ADD COLUMN reviewed_at TEXT",
        }
        for column, statement in violation_columns.items():
            if column not in existing:
                conn.execute(statement)
        conn.execute("UPDATE violations SET review_status = 'approved' WHERE review_status IS NULL OR review_status = ''")

        session_existing = {row[1] for row in conn.execute("PRAGMA table_info(analysis_sessions)").fetchall()}
        session_columns = {
            "review_status": "ALTER TABLE analysis_sessions ADD COLUMN review_status TEXT DEFAULT 'approved'",
            "approved_at": "ALTER TABLE analysis_sessions ADD COLUMN approved_at TEXT",
            "review_note": "ALTER TABLE analysis_sessions ADD COLUMN review_note TEXT",
        }
        for column, statement in session_columns.items():
            if column not in session_existing:
                conn.execute(statement)
        conn.execute("UPDATE analysis_sessions SET review_status = 'approved' WHERE review_status IS NULL OR review_status = ''")
        conn.commit()


init_db()


def load_config_file() -> dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def save_config_file(config: dict[str, Any]) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def current_calibration() -> dict[str, Any]:
    return load_config_file().get("calibration", {})


AI_API_KEY = ""
AI_PROVIDER = "none"
groq_client = None
gemini_client = None


def configure_ai(api_key: str) -> dict[str, str]:
    global AI_API_KEY, AI_PROVIDER, groq_client, gemini_client
    if api_key.startswith("gsk_"):
        if Groq is None:
            raise HTTPException(status_code=503, detail="groq package is not installed. Install backend requirements.")
        groq_client = Groq(api_key=api_key)
        gemini_client = None
        AI_PROVIDER = "groq"
        AI_API_KEY = api_key
        return {"provider": "groq", "message": "Groq vision review configured"}

    if genai is None:
        raise HTTPException(status_code=503, detail="google-genai package is not installed. Install backend requirements.")
    gemini_client = genai.Client(api_key=api_key)
    groq_client = None
    AI_PROVIDER = "gemini"
    AI_API_KEY = api_key
    return {"provider": "gemini", "message": "Gemini vision review configured"}


def safe_json_parse(text: str) -> dict[str, Any]:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


ANALYSIS_PROMPT = """You are a traffic evidence reviewer. Return only JSON.

Analyze the image for these violation types:
- Helmet Non-compliance
- Seatbelt Non-compliance
- Triple Riding
- Wrong-side Driving
- Stop-line Violation
- Red-light Violation
- Illegal Parking

Use this exact schema:
{"scene_description":"","weather_conditions":"clear/rainy/foggy/night/overcast/unknown","total_vehicles":0,"total_pedestrians":0,"traffic_light_state":"red/green/yellow/not_visible/unknown","violations":[{"type":"Helmet Non-compliance","confidence":0.0,"vehicle_type":"motorcycle","license_plate":null,"bbox_percent":{"x":0,"y":0,"w":0,"h":0},"description":""}]}

Rules:
- Do not report compliance as a violation.
- Use tight percent bounding boxes around the violating vehicle or road user.
- If uncertain, lower confidence below 0.75.
- Prefer empty violations over guessing.
"""


async def analyze_with_ai(image_path: str) -> dict[str, Any]:
    if AI_PROVIDER == "groq":
        return await _analyze_groq(image_path)
    if AI_PROVIDER == "gemini":
        return await _analyze_gemini(image_path)
    return {}


async def _analyze_groq(image_path: str) -> dict[str, Any]:
    if groq_client is None:
        return {}
    image_data = Path(image_path).read_bytes()
    img_base64 = base64.b64encode(image_data).decode("utf-8")
    mime_type = "image/png" if Path(image_path).suffix.lower() == ".png" else "image/jpeg"
    response = groq_client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": ANALYSIS_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{img_base64}"}},
                ],
            }
        ],
        temperature=0.05,
        max_tokens=2048,
    )
    return safe_json_parse(response.choices[0].message.content)


async def _analyze_gemini(image_path: str) -> dict[str, Any]:
    if gemini_client is None or genai is None:
        return {}
    image_data = Path(image_path).read_bytes()
    mime_type = "image/png" if Path(image_path).suffix.lower() == ".png" else "image/jpeg"
    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            genai.types.Content(
                parts=[
                    genai.types.Part(text=ANALYSIS_PROMPT),
                    genai.types.Part(inline_data=genai.types.Blob(mime_type=mime_type, data=image_data)),
                ]
            )
        ],
    )
    return safe_json_parse(response.text)


ANALYTICS_BRIEF_PROMPT = """You are the analytics commander for Gridlock AI, a traffic violation evidence platform.
Return only compact JSON using this exact schema:
{"headline":"","executive_summary":"","priority_actions":[""],"risk_signals":[""],"operations":[""],"presentation_notes":[""]}

Rules:
- Use the supplied analytics JSON only.
- Do not claim formal model accuracy unless evaluation metrics are present.
- Write concise, judge-friendly, practical insights.
- Focus on enforcement value, review quality, OCR/plate usefulness, throughput, and next best actions.
"""


def _normalize_brief_payload(payload: dict[str, Any]) -> dict[str, Any]:
    def text(value: Any, fallback: str = "") -> str:
        return str(value or fallback).strip()

    def text_list(value: Any, fallback: list[str]) -> list[str]:
        if isinstance(value, list):
            cleaned = [str(item).strip() for item in value if str(item).strip()]
            return cleaned[:5] or fallback
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return fallback

    return {
        "headline": text(payload.get("headline"), "Gridlock AI enforcement brief"),
        "executive_summary": text(payload.get("executive_summary"), "Traffic evidence is being processed into searchable, reviewable violation records."),
        "priority_actions": text_list(payload.get("priority_actions"), ["Review pending detections before issuing final enforcement records."]),
        "risk_signals": text_list(payload.get("risk_signals"), ["Use labelled ground truth before claiming formal model accuracy."]),
        "operations": text_list(payload.get("operations"), ["Continue processing representative traffic images and approve/reject detections."]),
        "presentation_notes": text_list(payload.get("presentation_notes"), ["Show annotated evidence, plate extraction, review approval, analytics, and performance readiness."]),
    }


def _local_analytics_brief(analytics: dict[str, Any]) -> dict[str, Any]:
    summary = analytics.get("summary", {})
    performance = analytics.get("performance", {})
    by_type = analytics.get("violations_by_type", [])
    avg_conf = analytics.get("avg_confidence_by_type", [])
    top_plates = analytics.get("top_offending_plates", [])

    total_images = int(summary.get("total_images_analyzed") or 0)
    total_violations = int(summary.get("total_violations") or 0)
    pending = int(summary.get("pending_violations") or 0)
    rejected = int(summary.get("rejected_violations") or 0)
    avg_ms = float(performance.get("avg_processing_ms") or 0)
    top_type = by_type[0]["type"] if by_type else "No dominant class yet"
    top_count = by_type[0]["count"] if by_type else 0
    plate_signal = ", ".join(item["plate"] for item in top_plates[:3]) if top_plates else "No recurring plate yet"
    confidence_sorted = sorted(avg_conf, key=lambda item: item.get("confidence", 0), reverse=True)
    strongest_class = confidence_sorted[0]["type"] if confidence_sorted else "No confidence trend yet"

    density = round(total_violations / max(total_images, 1), 2)
    return {
        "headline": f"{top_type} is the current enforcement hotspot" if total_violations else "Gridlock AI is ready for evidence intake",
        "executive_summary": (
            f"{total_images} image/session(s) have produced {total_violations} approved violation record(s), "
            f"with an average processing time of {round(avg_ms / 1000, 2)}s per case. "
            f"The current violation density is {density} approved finding(s) per processed image."
        ),
        "priority_actions": [
            f"Use the top class '{top_type}' ({top_count} record(s)) as the first demo story.",
            "Open annotated evidence from Records to show explainable bounding boxes and plate metadata.",
            "Run a labelled evaluation set before presenting Precision, Recall, F1, or mAP as final accuracy.",
        ],
        "risk_signals": [
            f"{pending} pending and {rejected} rejected finding(s) show the human-review loop is active.",
            f"Strongest average-confidence class: {strongest_class}. Treat lower-confidence classes as review-first.",
            f"Recurring plate signal: {plate_signal}.",
        ],
        "operations": [
            "Approve clean detections and remove false IDs so Analytics reflects reviewed enforcement records.",
            "Use ZIP batch mode for stress testing multiple scenes and environmental conditions.",
            "Keep calibration disabled unless the camera angle is fixed and the stop-line or parking zone is known.",
        ],
        "presentation_notes": [
            "Start with Analyze, then show Records search, Analytics brief, Performance readiness, and Config calibration.",
            "Phrase current metrics as operational throughput and review quality until labelled ground truth is evaluated.",
        ],
    }


async def _ai_analytics_brief(analytics: dict[str, Any]) -> dict[str, Any] | None:
    payload = json.dumps(analytics, ensure_ascii=True)[:14000]
    prompt = f"{ANALYTICS_BRIEF_PROMPT}\n\nAnalytics JSON:\n{payload}"
    if AI_PROVIDER == "groq" and groq_client is not None:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.25,
            max_tokens=900,
        )
        return _normalize_brief_payload(safe_json_parse(response.choices[0].message.content))

    if AI_PROVIDER == "gemini" and gemini_client is not None:
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        return _normalize_brief_payload(safe_json_parse(response.text))

    return None


def preprocess_image(image_path: str) -> dict[str, Any]:
    image = cv2.imread(image_path)
    if image is None:
        return {"path": image_path, "original_path": image_path, "steps": [], "quality_metrics": {}}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    processed = image.copy()
    steps: list[str] = []

    if brightness < 85 or contrast < 38:
        lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l_channel)
        processed = cv2.cvtColor(cv2.merge([enhanced_l, a_channel, b_channel]), cv2.COLOR_LAB2BGR)
        steps.append("CLAHE illumination normalization")

    if sharpness < 95:
        blur = cv2.GaussianBlur(processed, (0, 0), 1.2)
        processed = cv2.addWeighted(processed, 1.55, blur, -0.55, 0)
        steps.append("Unsharp masking")

    processed = cv2.fastNlMeansDenoisingColored(processed, None, 5, 5, 7, 21)
    steps.append("Color denoising")

    output_path = str(Path(image_path).with_name(f"{Path(image_path).stem}_preprocessed{Path(image_path).suffix}"))
    cv2.imwrite(output_path, processed)
    return {
        "path": output_path,
        "original_path": image_path,
        "steps": steps,
        "quality_metrics": {
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "sharpness": round(sharpness, 2),
        },
    }


def attach_plates_to_detections(detections: dict[str, Any], recognized_plates: list[dict[str, Any]]) -> None:
    detections["recognized_plates"] = recognized_plates
    for vehicle in detections.get("vehicles", []):
        vehicle_bbox = vehicle.get("bbox_percent", {})
        best = None
        for plate in recognized_plates:
            bbox = plate.get("bbox_percent", {})
            px = bbox.get("x", 0) + bbox.get("w", 0) / 2
            py = bbox.get("y", 0) + bbox.get("h", 0) / 2
            vx1 = vehicle_bbox.get("x", 0)
            vy1 = vehicle_bbox.get("y", 0)
            vx2 = vx1 + vehicle_bbox.get("w", 0)
            vy2 = vy1 + vehicle_bbox.get("h", 0)
            if vx1 - 2 <= px <= vx2 + 2 and vy1 - 2 <= py <= vy2 + 2:
                if best is None or plate.get("confidence", 0) > best.get("confidence", 0):
                    best = plate
        if best:
            vehicle["license_plate"] = best.get("text")
            vehicle["plate_confidence"] = best.get("confidence")


def validate_violations(violations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for violation in violations:
        vtype = violation.get("type", "")
        if vtype not in VALID_VIOLATION_TYPES:
            continue
        if "compliance" in vtype.lower() and "non-compliance" not in vtype.lower():
            continue
        if float(violation.get("confidence", 0.0)) < 0.45:
            continue
        bbox = violation.get("bbox_percent", {})
        if bbox.get("w", 0) <= 0 or bbox.get("h", 0) <= 0:
            continue
        violation["bbox_percent"] = {
            "x": max(0, min(100, bbox.get("x", 0))),
            "y": max(0, min(100, bbox.get("y", 0))),
            "w": max(0.1, min(100, bbox.get("w", 0))),
            "h": max(0.1, min(100, bbox.get("h", 0))),
        }
        cleaned.append(violation)
    return cleaned


def _safe_zip_image_path(name: str) -> PurePosixPath | None:
    normalized = name.replace("\\", "/").strip()
    if not normalized:
        return None

    path = PurePosixPath(normalized)
    if path.is_absolute():
        return None

    parts = path.parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        return None
    if parts[0] == "__MACOSX" or parts[-1].startswith("._"):
        return None
    if Path(parts[-1]).suffix.lower() not in IMAGE_EXTENSIONS:
        return None
    return path


def _safe_batch_image_filename(batch_id: str, index: int, source_name: str) -> str:
    source_path = PurePosixPath(source_name.replace("\\", "/"))
    ext = Path(source_path.name).suffix.lower()
    if ext not in IMAGE_EXTENSIONS:
        ext = ".jpg"
    stem = Path(source_path.name).stem
    safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in stem)
    safe_stem = safe_stem.strip("._")[:54] or "traffic_image"
    return f"{batch_id}_{index:03d}_{safe_stem}{ext}"


def _compact_batch_analysis(analysis: dict[str, Any], source_filename: str) -> dict[str, Any]:
    detection = analysis.get("detection", {})
    recognized_plates = detection.get("recognized_plates", []) or []
    return {
        "status": analysis.get("status", "success"),
        "source_filename": source_filename,
        "session_id": analysis.get("session_id"),
        "timestamp": analysis.get("timestamp"),
        "processing_time_ms": analysis.get("processing_time_ms", 0),
        "scene": analysis.get("scene", {}),
        "preprocessing": analysis.get("preprocessing", {}),
        "detection": {
            "total_vehicles": detection.get("total_vehicles", 0),
            "total_pedestrians": detection.get("total_pedestrians", 0),
            "recognized_plates": recognized_plates,
            "traffic_light": analysis.get("scene", {}).get("traffic_light", "not_visible"),
        },
        "violation_count": len(analysis.get("violations", [])),
        "violations": analysis.get("violations", []),
        "evidence": analysis.get("evidence", {}),
    }


def _violation_from_db_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "record_id": row["id"],
        "type": row["violation_type"],
        "confidence": row["confidence"] or 0,
        "license_plate": row["license_plate"],
        "vehicle_type": row["vehicle_type"],
        "bbox_percent": {
            "x": row["bbox_x"] or 0,
            "y": row["bbox_y"] or 0,
            "w": row["bbox_w"] or 0,
            "h": row["bbox_h"] or 0,
        },
        "description": row["description"] or "",
        "evidence": row["evidence"] or "",
        "review_status": row["review_status"] or "pending",
    }


def _public_annotated_url(path: str | None) -> str:
    if not path:
        return ""
    return f"/annotated/{Path(path).name}"


def _public_upload_url(path: str | None) -> str:
    if not path:
        return ""
    candidate = Path(path)
    if candidate.suffix.lower() not in IMAGE_EXTENSIONS:
        return ""
    return f"/uploads/{candidate.name}"


def _record_row_for_api(row: sqlite3.Row) -> dict[str, Any]:
    record = dict(row)
    annotated_url = _public_annotated_url(record.get("annotated_image"))
    original_url = _public_upload_url(record.get("original_image"))
    record["annotated_image_url"] = annotated_url
    record["original_image_url"] = original_url
    record["evidence_image_url"] = annotated_url or original_url
    record["bbox_percent"] = {
        "x": record.get("bbox_x") or 0,
        "y": record.get("bbox_y") or 0,
        "w": record.get("bbox_w") or 0,
        "h": record.get("bbox_h") or 0,
    }
    return record


def _draw_label(image: Any, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    y = max(th + 8, y)
    cv2.rectangle(image, (x, y - th - 7), (x + tw + 6, y + 2), color, -1)
    cv2.putText(image, text, (x + 3, y - 4), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _draw_percent_box(image: Any, bbox: dict[str, float], color: tuple[int, int, int], thickness: int, label: str) -> None:
    h, w = image.shape[:2]
    x = int(bbox.get("x", 0) * w / 100)
    y = int(bbox.get("y", 0) * h / 100)
    bw = int(bbox.get("w", 0) * w / 100)
    bh = int(bbox.get("h", 0) * h / 100)
    if bw <= 1 or bh <= 1:
        return
    cv2.rectangle(image, (x, y), (x + bw, y + bh), color, thickness)
    _draw_label(image, label, x, y, color)


def draw_calibration_overlay(image: Any) -> None:
    calibration = current_calibration()
    height, width = image.shape[:2]

    if calibration.get("stop_line_rule_enabled"):
        y = int(float(calibration.get("stop_line_y_percent", 60)) / 100 * height)
        cv2.line(image, (0, y), (width, y), (0, 165, 255), 2)
        _draw_label(image, "Stop line", 8, max(y - 8, 18), (0, 165, 255))

    if calibration.get("red_light_rule_enabled"):
        y = int(float(calibration.get("red_light_vehicle_bottom_y_percent", 52)) / 100 * height)
        cv2.line(image, (0, y), (width, y), (0, 0, 255), 2)
        _draw_label(image, "Red-light gate", 8, max(y - 8, 18), (0, 0, 255))

    if calibration.get("illegal_parking_rule_enabled"):
        for index, roi in enumerate(calibration.get("illegal_parking_rois", []) or [], start=1):
            x1 = int(float(roi.get("x", 0)) / 100 * width)
            y1 = int(float(roi.get("y", 0)) / 100 * height)
            x2 = int((float(roi.get("x", 0)) + float(roi.get("w", 0))) / 100 * width)
            y2 = int((float(roi.get("y", 0)) + float(roi.get("h", 0))) / 100 * height)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 80, 0), 2)
            _draw_label(image, f"No parking zone {index}", x1, max(y1 - 8, 18), (255, 80, 0))


def generate_annotated_image(image_path: str, detections: dict[str, Any], violations: list[dict[str, Any]], session_id: str) -> str:
    image = cv2.imread(image_path)
    if image is None:
        return ""

    draw_calibration_overlay(image)

    for vehicle in detections.get("vehicles", []):
        label = f"{vehicle.get('type', 'vehicle')} {vehicle.get('confidence', 0) * 100:.0f}%"
        if vehicle.get("license_plate"):
            label = f"{label} {vehicle['license_plate']}"
        _draw_percent_box(image, vehicle.get("bbox_percent", {}), (36, 160, 90), 1, label)

    for plate in detections.get("recognized_plates", []):
        _draw_percent_box(image, plate.get("bbox_percent", {}), (0, 190, 255), 2, plate.get("text", "plate"))

    for sign in detections.get("no_parking_signs", []):
        _draw_percent_box(image, sign.get("bbox_percent", {}), (255, 80, 0), 2, "No parking sign")

    for stop_line in detections.get("stop_lines", []):
        _draw_percent_box(image, stop_line.get("bbox_percent", {}), (0, 165, 255), 2, "Stop line")

    colors = {
        "Helmet Non-compliance": (0, 0, 255),
        "Seatbelt Non-compliance": (0, 100, 255),
        "Triple Riding": (0, 0, 200),
        "Wrong-side Driving": (255, 0, 255),
        "Stop-line Violation": (0, 165, 255),
        "Red-light Violation": (0, 0, 255),
        "Illegal Parking": (255, 80, 0),
        "Mobile Phone Use": (200, 0, 255),
        "Overloading": (0, 120, 255),
    }
    for violation in violations:
        vtype = violation.get("type", "Violation")
        label = f"{vtype} {violation.get('confidence', 0) * 100:.0f}%"
        _draw_percent_box(image, violation.get("bbox_percent", {}), colors.get(vtype, (0, 0, 255)), 3, label)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.rectangle(image, (8, image.shape[0] - 30), (340, image.shape[0] - 6), (0, 0, 0), -1)
    cv2.putText(image, f"Gridlock evidence {timestamp}", (14, image.shape[0] - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    output_path = ANNOTATED_DIR / f"{session_id}_annotated.jpg"
    cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return str(output_path)


async def run_analysis(original_path: str, session_id: str) -> dict[str, Any]:
    started = time.time()
    preprocess = preprocess_image(original_path)
    processed_path = preprocess.get("path", original_path)

    detections = detect(processed_path)
    analysis_image_path = processed_path
    original_detections = None
    if str(processed_path) != str(original_path) and (
        not detections.get("vehicles") or not detections.get("traffic_lights")
    ):
        original_detections = detect(str(original_path))
        if len(original_detections.get("vehicles", [])) > len(detections.get("vehicles", [])):
            detections = original_detections
            analysis_image_path = str(original_path)
            preprocess.setdefault("steps", []).append("Detector fallback to original image")
        elif not detections.get("traffic_lights") and original_detections.get("traffic_lights"):
            detections["traffic_lights"] = original_detections.get("traffic_lights", [])
            detections["raw_boxes"].extend(
                entry
                for entry in original_detections.get("raw_boxes", [])
                if entry.get("type") == "traffic_light"
            )
            preprocess.setdefault("steps", []).append("Traffic-light fallback to original image")

    recognized_plates = recognize_plates(analysis_image_path, detections.get("license_plates", []), detections.get("vehicles", []))
    attach_plates_to_detections(detections, recognized_plates)

    ai_analysis = {}
    if AI_PROVIDER != "none":
        try:
            ai_analysis = await analyze_with_ai(analysis_image_path)
        except Exception as exc:
            print(f"[ai] Vision review failed, continuing detector-only: {exc}")
            ai_analysis = {}

    violations = validate_violations(analyze_violations(detections, ai_analysis))
    annotated_path = generate_annotated_image(original_path, detections, violations, session_id)

    tl_colors = [tl.get("color", "unknown") for tl in detections.get("traffic_lights", [])]
    traffic_light = ai_analysis.get("traffic_light_state") if ai_analysis else None
    if traffic_light not in {"red", "green", "yellow"}:
        traffic_light = "red" if "red" in tl_colors else "yellow" if "yellow" in tl_colors else "green" if "green" in tl_colors else "not_visible"

    processing_ms = (time.time() - started) * 1000
    analysis = {
        "status": "success",
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "processing_time_ms": round(processing_ms, 2),
        "preprocessing": {
            "steps_applied": preprocess.get("steps", []),
            "quality_metrics": preprocess.get("quality_metrics", {}),
        },
        "pipeline": {
            "detector": get_detector_status(),
            "ocr": ocr_status(),
            "vision_review": {"provider": AI_PROVIDER, "configured": AI_PROVIDER != "none"},
        },
        "scene": {
            "description": ai_analysis.get("scene_description", ""),
            "weather": ai_analysis.get("weather_conditions", "unknown"),
            "traffic_light": traffic_light,
        },
        "detection": {
            "total_vehicles": len(detections.get("vehicles", [])),
            "total_pedestrians": len(detections.get("persons", [])),
            "vehicles": detections.get("vehicles", []),
            "persons": detections.get("persons", []),
            "traffic_lights": detections.get("traffic_lights", []),
            "stop_lines": detections.get("stop_lines", []),
            "no_parking_signs": detections.get("no_parking_signs", []),
            "illegal_parking_vehicles": detections.get("illegal_parking_vehicles", []),
            "wrong_side_vehicles": detections.get("wrong_side_vehicles", []),
            "right_side_vehicles": detections.get("right_side_vehicles", []),
            "license_plates": detections.get("license_plates", []),
            "recognized_plates": recognized_plates,
            "scene_geometry": detections.get("scene_geometry", {}),
            "raw_boxes": detections.get("raw_boxes", []),
        },
        "violations": violations,
        "evidence": {
            "original_image": str(original_path),
            "annotated_image": f"/annotated/{session_id}_annotated.jpg" if annotated_path else "",
        },
        "raw_ai_response": ai_analysis,
    }
    return analysis


def store_analysis(analysis: dict[str, Any], original_path: str) -> None:
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO analysis_sessions
            (id, original_image, annotated_image, total_vehicles, total_violations,
             total_pedestrians, processing_time_ms, preprocessing_applied, timestamp, raw_ai_response, review_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                analysis["session_id"],
                original_path,
                str(ANNOTATED_DIR / f"{analysis['session_id']}_annotated.jpg"),
                analysis["detection"]["total_vehicles"],
                len(analysis["violations"]),
                analysis["detection"]["total_pedestrians"],
                analysis["processing_time_ms"],
                json.dumps(analysis["preprocessing"]["steps_applied"]),
                analysis["timestamp"],
                json.dumps(analysis),
                "pending",
            ),
        )
        for violation in analysis["violations"]:
            violation_id = str(uuid.uuid4())[:12]
            violation["record_id"] = violation_id
            violation["review_status"] = "pending"
            bbox = violation.get("bbox_percent", {})
            conn.execute(
                """
                INSERT INTO violations
                (id, image_id, original_image, annotated_image, violation_type, confidence,
                 license_plate, vehicle_type, bbox_x, bbox_y, bbox_w, bbox_h,
                 timestamp, preprocessing_applied, description, evidence, review_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    violation_id,
                    analysis["session_id"],
                    original_path,
                    str(ANNOTATED_DIR / f"{analysis['session_id']}_annotated.jpg"),
                    violation.get("type"),
                    violation.get("confidence", 0),
                    violation.get("license_plate"),
                    violation.get("vehicle_type"),
                    bbox.get("x", 0),
                    bbox.get("y", 0),
                    bbox.get("w", 0),
                    bbox.get("h", 0),
                    analysis["timestamp"],
                    json.dumps(analysis["preprocessing"]["steps_applied"]),
                    violation.get("description", ""),
                    violation.get("evidence", "unknown"),
                    "pending",
                ),
            )
        conn.commit()


def store_video_analysis(analysis: dict[str, Any], original_path: str) -> None:
    annotated_path = analysis.get("evidence", {}).get("annotated_video") or analysis.get("evidence", {}).get("summary_image") or ""
    annotated_db_path = str(ANNOTATED_DIR / Path(annotated_path).name) if annotated_path else ""
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO analysis_sessions
            (id, original_image, annotated_image, total_vehicles, total_violations,
             total_pedestrians, processing_time_ms, preprocessing_applied, timestamp, raw_ai_response, review_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                analysis["session_id"],
                original_path,
                annotated_db_path,
                analysis["detection"]["total_vehicles"],
                len(analysis["violations"]),
                analysis["detection"].get("total_pedestrians", 0),
                analysis["processing_time_ms"],
                json.dumps(["video sampling", "centroid+iou tracking", "calibrated temporal rules"]),
                analysis["timestamp"],
                json.dumps(analysis),
                "pending",
            ),
        )
        for violation in analysis["violations"]:
            violation_id = str(uuid.uuid4())[:12]
            violation["record_id"] = violation_id
            violation["review_status"] = "pending"
            bbox = violation.get("bbox_percent", {})
            conn.execute(
                """
                INSERT INTO violations
                (id, image_id, original_image, annotated_image, violation_type, confidence,
                 license_plate, vehicle_type, bbox_x, bbox_y, bbox_w, bbox_h,
                 timestamp, preprocessing_applied, description, evidence, review_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    violation_id,
                    analysis["session_id"],
                    original_path,
                    annotated_db_path,
                    violation.get("type"),
                    violation.get("confidence", 0),
                    violation.get("license_plate"),
                    violation.get("vehicle_type"),
                    bbox.get("x", 0),
                    bbox.get("y", 0),
                    bbox.get("w", 0),
                    bbox.get("h", 0),
                    analysis["timestamp"],
                    json.dumps(["video tracking"]),
                    violation.get("description", ""),
                    violation.get("evidence", "tracker"),
                    "pending",
                ),
            )
        conn.commit()


@app.get("/")
def root():
    return {
        "status": "running",
        "version": "3.0.0",
        "detector": get_detector_status(),
        "vision_review": {"provider": AI_PROVIDER, "configured": AI_PROVIDER != "none"},
    }


@app.get("/api/status")
async def get_status():
    detector_status = get_detector_status()
    return {
        "engine_ready": detector_status.get("ready", False),
        "detector": detector_status,
        "ai_configured": AI_PROVIDER != "none",
        "provider": AI_PROVIDER,
        "ocr": ocr_status(),
    }


@app.post("/api/config")
async def set_config(api_key: str = Query(...)):
    result = configure_ai(api_key.strip())
    return {"status": "success", **result}


@app.get("/api/calibration")
async def get_calibration():
    return {"status": "success", "calibration": current_calibration()}


def _percent(value: Any, default: float) -> float:
    try:
        return round(max(0.0, min(100.0, float(value))), 3)
    except (TypeError, ValueError):
        return default


def _bool(value: Any) -> bool:
    return bool(value) if isinstance(value, bool) else str(value).strip().lower() in {"1", "true", "yes", "on"}


@app.post("/api/calibration")
async def set_calibration(payload: dict[str, Any] = Body(...)):
    config = load_config_file()
    calibration = config.setdefault("calibration", {})

    calibration["stop_line_rule_enabled"] = _bool(payload.get("stop_line_rule_enabled", calibration.get("stop_line_rule_enabled", False)))
    calibration["stop_line_y_percent"] = _percent(payload.get("stop_line_y_percent"), float(calibration.get("stop_line_y_percent", 60)))
    calibration["red_light_rule_enabled"] = _bool(payload.get("red_light_rule_enabled", calibration.get("red_light_rule_enabled", False)))
    calibration["red_light_vehicle_bottom_y_percent"] = _percent(
        payload.get("red_light_vehicle_bottom_y_percent"),
        float(calibration.get("red_light_vehicle_bottom_y_percent", 52)),
    )
    calibration["illegal_parking_rule_enabled"] = _bool(
        payload.get("illegal_parking_rule_enabled", calibration.get("illegal_parking_rule_enabled", False))
    )
    calibration["wrong_side_enabled"] = _bool(payload.get("wrong_side_enabled", calibration.get("wrong_side_enabled", False)))
    calibration["illegal_parking_dwell_seconds"] = max(
        1.0,
        min(120.0, _percent(payload.get("illegal_parking_dwell_seconds"), float(calibration.get("illegal_parking_dwell_seconds", 8.0)))),
    )
    calibration["tracking_sample_fps"] = max(
        0.25,
        min(5.0, float(payload.get("tracking_sample_fps", calibration.get("tracking_sample_fps", 1.5)) or 1.5)),
    )
    calibration["tracking_max_frames"] = max(
        8,
        min(240, int(float(payload.get("tracking_max_frames", calibration.get("tracking_max_frames", 90)) or 90))),
    )
    calibration["wrong_side_min_travel_percent"] = max(
        2.0,
        min(40.0, _percent(payload.get("wrong_side_min_travel_percent"), float(calibration.get("wrong_side_min_travel_percent", 8.0)))),
    )
    signal_override = str(payload.get("traffic_light_state_override", calibration.get("traffic_light_state_override", "auto")))
    calibration["traffic_light_state_override"] = (
        signal_override if signal_override in {"auto", "red", "yellow", "green", "not_visible"} else "auto"
    )

    direction = str(payload.get("expected_lane_direction", calibration.get("expected_lane_direction", "left_to_right")))
    calibration["expected_lane_direction"] = direction if direction in {"left_to_right", "right_to_left"} else "left_to_right"

    rois = []
    for roi in payload.get("illegal_parking_rois", calibration.get("illegal_parking_rois", [])) or []:
        x = _percent(roi.get("x"), 0)
        y = _percent(roi.get("y"), 0)
        w = max(0.1, min(100.0 - x, _percent(roi.get("w"), 0)))
        h = max(0.1, min(100.0 - y, _percent(roi.get("h"), 0)))
        rois.append({"x": round(x, 3), "y": round(y, 3), "w": round(w, 3), "h": round(h, 3)})
    calibration["illegal_parking_rois"] = rois

    save_config_file(config)
    return {"status": "success", "calibration": calibration}


@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    session_id = str(uuid.uuid4())[:12]
    ext = Path(file.filename or "image.jpg").suffix.lower() or ".jpg"
    if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
        ext = ".jpg"
    original_path = str(UPLOAD_DIR / f"{session_id}{ext}")
    with open(original_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    analysis = await run_analysis(original_path, session_id)
    store_analysis(analysis, original_path)
    return JSONResponse(content={k: v for k, v in analysis.items() if k != "raw_ai_response"})


@app.post("/api/analyze-zip")
async def analyze_zip(file: UploadFile = File(...), max_images: int = Query(MAX_BATCH_IMAGES, ge=1, le=100)):
    ext = Path(file.filename or "").suffix.lower()
    if ext != ".zip":
        raise HTTPException(status_code=400, detail="File must be a .zip archive containing images.")

    batch_id = f"batch_{uuid.uuid4().hex[:10]}"
    zip_path = UPLOAD_DIR / f"{batch_id}.zip"
    started = time.time()
    with open(zip_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    zip_size = zip_path.stat().st_size
    if zip_size <= 0:
        raise HTTPException(status_code=400, detail="Uploaded ZIP is empty.")
    if zip_size > MAX_ZIP_BYTES:
        raise HTTPException(status_code=413, detail=f"ZIP is too large. Limit is {MAX_ZIP_BYTES // (1024 * 1024)} MB.")

    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    try:
        with zipfile.ZipFile(zip_path) as archive:
            image_members: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
            total_uncompressed = 0
            for info in archive.infolist():
                if info.is_dir():
                    continue
                member_path = _safe_zip_image_path(info.filename)
                if member_path is None:
                    continue
                if info.file_size > MAX_BATCH_IMAGE_BYTES:
                    failures.append(
                        {
                            "source_filename": str(member_path),
                            "error": f"Image is too large. Limit is {MAX_BATCH_IMAGE_BYTES // (1024 * 1024)} MB.",
                        }
                    )
                    continue
                total_uncompressed += info.file_size
                if total_uncompressed > MAX_ZIP_UNCOMPRESSED_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"ZIP expands too large. Limit is {MAX_ZIP_UNCOMPRESSED_BYTES // (1024 * 1024)} MB.",
                    )
                image_members.append((info, member_path))

            if not image_members:
                raise HTTPException(status_code=400, detail="ZIP does not contain supported image files.")

            image_members.sort(key=lambda item: str(item[1]).lower())
            selected_members = image_members[:max_images]
            if len(image_members) > max_images:
                failures.append(
                    {
                        "source_filename": "batch_limit",
                        "error": f"Skipped {len(image_members) - max_images} image(s) after the {max_images}-image batch limit.",
                    }
                )

            for index, (info, member_path) in enumerate(selected_members, start=1):
                source_filename = str(member_path)
                extracted_path = UPLOAD_DIR / _safe_batch_image_filename(batch_id, index, source_filename)
                try:
                    with archive.open(info) as source, open(extracted_path, "wb") as target:
                        shutil.copyfileobj(source, target)

                    if cv2.imread(str(extracted_path)) is None:
                        raise ValueError("Image could not be decoded.")

                    session_id = f"{batch_id}_{index:03d}"
                    analysis = await run_analysis(str(extracted_path), session_id)
                    store_analysis(analysis, str(extracted_path))
                    results.append(_compact_batch_analysis(analysis, source_filename))
                except Exception as exc:
                    failures.append({"source_filename": source_filename, "error": str(exc)})
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid ZIP archive.") from exc

    total_violations = sum(item.get("violation_count", 0) for item in results)
    total_vehicles = sum(item.get("detection", {}).get("total_vehicles", 0) for item in results)
    total_plates = sum(len(item.get("detection", {}).get("recognized_plates", []) or []) for item in results)
    images_with_violations = sum(1 for item in results if item.get("violation_count", 0) > 0)

    return JSONResponse(
        content={
            "status": "success",
            "batch_id": batch_id,
            "source_zip": file.filename,
            "requested_limit": max_images,
            "zip_size_bytes": zip_size,
            "processed": len(results),
            "failed": len(failures),
            "total_violations": total_violations,
            "processing_time_ms": round((time.time() - started) * 1000, 2),
            "summary": {
                "images_with_violations": images_with_violations,
                "total_vehicles": total_vehicles,
                "recognized_plates": total_plates,
                "clean_images": max(0, len(results) - images_with_violations),
            },
            "results": results,
            "failures": failures,
        }
    )


@app.post("/api/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    ext = Path(file.filename or "video.mp4").suffix.lower() or ".mp4"
    allowed_ext = {".mp4", ".mov", ".m4v", ".avi", ".webm", ".mkv"}
    is_video_type = bool(file.content_type and file.content_type.startswith("video/"))
    if not is_video_type and ext not in allowed_ext:
        raise HTTPException(status_code=400, detail="File must be a video.")

    session_id = str(uuid.uuid4())[:12]
    if ext not in allowed_ext:
        ext = ".mp4"
    original_path = str(UPLOAD_DIR / f"{session_id}{ext}")
    with open(original_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        analysis = analyze_video_file(original_path, session_id, ANNOTATED_DIR)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    store_video_analysis(analysis, original_path)
    return JSONResponse(content=analysis)


@app.post("/api/violations/{violation_id}/reject")
async def reject_violation(violation_id: str, payload: dict[str, Any] | None = Body(None)):
    comment = str((payload or {}).get("comment", "Rejected during human review")).strip()
    reviewed_at = datetime.now().isoformat()
    with get_db() as conn:
        row = conn.execute("SELECT * FROM violations WHERE id = ?", (violation_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Violation record not found.")
        conn.execute(
            """
            UPDATE violations
            SET review_status = 'rejected', review_comment = ?, reviewed_at = ?
            WHERE id = ?
            """,
            (comment, reviewed_at, violation_id),
        )
        conn.commit()
    return {"status": "success", "violation_id": violation_id, "review_status": "rejected", "reviewed_at": reviewed_at}


@app.post("/api/sessions/{session_id}/approve")
async def approve_session(session_id: str, payload: dict[str, Any] | None = Body(None)):
    payload = payload or {}
    rejected_ids = {str(item) for item in payload.get("rejected_violation_ids", []) if str(item).strip()}
    note = str(payload.get("note", "Human reviewed and approved")).strip()
    reviewed_at = datetime.now().isoformat()

    with get_db() as conn:
        session = conn.execute("SELECT * FROM analysis_sessions WHERE id = ?", (session_id,)).fetchone()
        if session is None:
            raise HTTPException(status_code=404, detail="Analysis session not found.")

        rows = conn.execute("SELECT * FROM violations WHERE image_id = ?", (session_id,)).fetchall()
        valid_ids = {row["id"] for row in rows}
        invalid_ids = sorted(rejected_ids - valid_ids)
        if invalid_ids:
            raise HTTPException(status_code=400, detail=f"Rejected IDs do not belong to this session: {', '.join(invalid_ids)}")

        if rejected_ids:
            placeholders = ",".join("?" for _ in rejected_ids)
            conn.execute(
                f"""
                UPDATE violations
                SET review_status = 'rejected', review_comment = ?, reviewed_at = ?
                WHERE image_id = ? AND id IN ({placeholders})
                """,
                [note or "Rejected during approval review", reviewed_at, session_id, *sorted(rejected_ids)],
            )

        conn.execute(
            """
            UPDATE violations
            SET review_status = 'approved', review_comment = ?, reviewed_at = ?
            WHERE image_id = ? AND review_status != 'rejected'
            """,
            (note or "Approved during human review", reviewed_at, session_id),
        )

        approved_rows = conn.execute(
            "SELECT * FROM violations WHERE image_id = ? AND review_status = 'approved' ORDER BY confidence DESC",
            (session_id,),
        ).fetchall()
        rejected_count = conn.execute(
            "SELECT COUNT(*) FROM violations WHERE image_id = ? AND review_status = 'rejected'",
            (session_id,),
        ).fetchone()[0]

        approved_violations = [_violation_from_db_row(row) for row in approved_rows]
        raw_analysis: dict[str, Any] = {}
        try:
            raw_analysis = json.loads(session["raw_ai_response"] or "{}")
        except json.JSONDecodeError:
            raw_analysis = {}

        reviewed_annotated_path = ""
        original_image = session["original_image"] or ""
        if original_image and Path(original_image).suffix.lower() in IMAGE_EXTENSIONS and Path(original_image).exists():
            reviewed_annotated_path = generate_annotated_image(
                original_image,
                raw_analysis.get("detection", {}),
                approved_violations,
                f"{session_id}_approved",
            )

        final_annotated_path = reviewed_annotated_path or session["annotated_image"] or ""
        raw_analysis["violations"] = approved_violations
        raw_analysis["review"] = {
            "status": "approved",
            "approved_at": reviewed_at,
            "approved_count": len(approved_violations),
            "rejected_count": rejected_count,
            "note": note,
        }
        if final_annotated_path:
            raw_analysis.setdefault("evidence", {})["annotated_image"] = _public_annotated_url(final_annotated_path)

        conn.execute(
            """
            UPDATE analysis_sessions
            SET total_violations = ?, annotated_image = ?, raw_ai_response = ?,
                review_status = 'approved', approved_at = ?, review_note = ?
            WHERE id = ?
            """,
            (len(approved_violations), final_annotated_path, json.dumps(raw_analysis), reviewed_at, note, session_id),
        )
        if final_annotated_path:
            conn.execute(
                "UPDATE violations SET annotated_image = ? WHERE image_id = ? AND review_status = 'approved'",
                (final_annotated_path, session_id),
            )
        conn.commit()

    return {
        "status": "success",
        "session_id": session_id,
        "review_status": "approved",
        "approved_at": reviewed_at,
        "approved_count": len(approved_violations),
        "rejected_count": rejected_count,
        "violations": approved_violations,
        "evidence": {"annotated_image": _public_annotated_url(final_annotated_path)},
    }


@app.get("/api/violations")
async def get_violations(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    violation_type: str | None = Query(None),
    search: str | None = Query(None),
    review_status: str | None = Query(None),
):
    offset = (page - 1) * limit
    with get_db() as conn:
        clauses = []
        params: list[Any] = []
        if violation_type:
            clauses.append("violation_type = ?")
            params.append(violation_type)
        if review_status:
            clauses.append("review_status = ?")
            params.append(review_status)
        if search:
            clauses.append("(license_plate LIKE ? OR description LIKE ? OR evidence LIKE ?)")
            params.extend([f"%{search}%", f"%{search}%", f"%{search}%"])
        where_sql = "WHERE " + " AND ".join(clauses) if clauses else ""
        total = conn.execute(f"SELECT COUNT(*) FROM violations {where_sql}", params).fetchone()[0]
        rows = conn.execute(
            f"SELECT * FROM violations {where_sql} ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            params + [limit, offset],
        ).fetchall()
    return {
        "total": total,
        "page": page,
        "limit": limit,
        "total_pages": (total + limit - 1) // limit,
        "violations": [_record_row_for_api(row) for row in rows],
    }


@app.get("/api/analytics")
async def get_analytics():
    with get_db() as conn:
        total_sessions = conn.execute("SELECT COUNT(*) FROM analysis_sessions").fetchone()[0]
        total_violations = conn.execute("SELECT COUNT(*) FROM violations WHERE review_status = 'approved'").fetchone()[0]
        pending_violations = conn.execute("SELECT COUNT(*) FROM violations WHERE review_status = 'pending'").fetchone()[0]
        rejected_violations = conn.execute("SELECT COUNT(*) FROM violations WHERE review_status = 'rejected'").fetchone()[0]
        by_type = conn.execute(
            """
            SELECT violation_type, COUNT(*)
            FROM violations
            WHERE review_status = 'approved'
            GROUP BY violation_type
            ORDER BY COUNT(*) DESC
            """
        ).fetchall()
        by_vehicle = conn.execute(
            """
            SELECT vehicle_type, COUNT(*)
            FROM violations
            WHERE vehicle_type IS NOT NULL AND review_status = 'approved'
            GROUP BY vehicle_type
            ORDER BY COUNT(*) DESC
            """
        ).fetchall()
        avg_confidence = conn.execute(
            """
            SELECT violation_type, AVG(confidence)
            FROM violations
            WHERE review_status = 'approved'
            GROUP BY violation_type
            """
        ).fetchall()
        recent = conn.execute(
            "SELECT timestamp, total_violations, total_vehicles FROM analysis_sessions ORDER BY timestamp DESC LIMIT 10"
        ).fetchall()
        avg_processing = conn.execute(
            "SELECT AVG(processing_time_ms), MIN(processing_time_ms), MAX(processing_time_ms) FROM analysis_sessions"
        ).fetchone()
        top_plates = conn.execute(
            """
            SELECT license_plate, COUNT(*)
            FROM violations
            WHERE license_plate IS NOT NULL AND license_plate != 'null'
              AND review_status = 'approved'
            GROUP BY license_plate
            ORDER BY COUNT(*) DESC
            LIMIT 10
            """
        ).fetchall()

    return {
        "summary": {
            "total_sessions": total_sessions,
            "total_violations": total_violations,
            "total_images_analyzed": total_sessions,
            "pending_violations": pending_violations,
            "rejected_violations": rejected_violations,
        },
        "violations_by_type": [{"type": row[0], "count": row[1]} for row in by_type],
        "violations_by_vehicle": [{"vehicle": row[0], "count": row[1]} for row in by_vehicle],
        "avg_confidence_by_type": [{"type": row[0], "confidence": round(row[1], 3)} for row in avg_confidence],
        "recent_trend": [{"timestamp": row[0], "violations": row[1], "vehicles": row[2]} for row in recent],
        "performance": {
            "avg_processing_ms": round(avg_processing[0] or 0, 2),
            "min_processing_ms": round(avg_processing[1] or 0, 2),
            "max_processing_ms": round(avg_processing[2] or 0, 2),
        },
        "top_offending_plates": [{"plate": row[0], "count": row[1]} for row in top_plates],
    }


@app.get("/api/analytics/brief")
async def get_analytics_brief():
    analytics = await get_analytics()
    provider = AI_PROVIDER
    generated_at = datetime.now().isoformat()
    try:
        ai_brief = await _ai_analytics_brief(analytics)
        if ai_brief:
            return {
                "status": "success",
                "provider": provider,
                "generated_at": generated_at,
                "brief": ai_brief,
            }
    except Exception as exc:
        print(f"[analytics] AI brief failed, using local brief: {exc}")

    return {
        "status": "success",
        "provider": "local" if provider == "none" else f"{provider}_fallback",
        "generated_at": generated_at,
        "brief": _normalize_brief_payload(_local_analytics_brief(analytics)),
    }


@app.get("/api/sessions")
async def get_sessions(limit: int = Query(20, ge=1, le=100)):
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM analysis_sessions ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
    return {"sessions": [dict(row) for row in rows]}


@app.delete("/api/violations")
async def clear_all():
    with get_db() as conn:
        conn.execute("DELETE FROM violations")
        conn.execute("DELETE FROM analysis_sessions")
        conn.commit()
    return {"status": "cleared"}


@app.get("/api/performance")
async def get_performance_metrics():
    with get_db() as conn:
        total_sessions = conn.execute("SELECT COUNT(*) FROM analysis_sessions").fetchone()[0]
        avg_row = conn.execute(
            "SELECT AVG(processing_time_ms), MIN(processing_time_ms), MAX(processing_time_ms) FROM analysis_sessions"
        ).fetchone()
        review_row = conn.execute(
            """
            SELECT
              SUM(CASE WHEN review_status = 'approved' THEN 1 ELSE 0 END),
              SUM(CASE WHEN review_status = 'pending' THEN 1 ELSE 0 END),
              SUM(CASE WHEN review_status = 'rejected' THEN 1 ELSE 0 END)
            FROM violations
            """
        ).fetchone()
        eval_row = conn.execute(
            "SELECT * FROM evaluation_runs ORDER BY created_at DESC LIMIT 1"
        ).fetchone()

    avg_processing = float(avg_row[0] or 0)
    detector_status = get_detector_status()
    throughput = {
        "avg_processing_ms": round(avg_processing, 2),
        "min_processing_ms": round(avg_row[1] or 0, 2),
        "max_processing_ms": round(avg_row[2] or 0, 2),
        "images_per_minute": round(60000 / max(avg_processing or 1, 1), 1),
        "total_images_processed": total_sessions,
    }
    review = {
        "approved": int(review_row[0] or 0),
        "pending": int(review_row[1] or 0),
        "rejected": int(review_row[2] or 0),
    }
    coverage = [
        {"feature": "Image preprocessing", "status": "ready", "detail": "CLAHE, sharpening, denoising"},
        {"feature": "Vehicle/road-user detection", "status": "ready", "detail": "YOLO ensemble with fallback detector"},
        {"feature": "Helmet non-compliance", "status": "ready", "detail": "Detector and rule grounded"},
        {"feature": "Seatbelt non-compliance", "status": "ready", "detail": "No-seatbelt class plus cabin fallback"},
        {"feature": "Triple riding", "status": "ready", "detail": "Direct class plus rider association"},
        {"feature": "Wrong-side driving", "status": "ready", "detail": "Direct class plus still-image road geometry"},
        {"feature": "Stop-line/red-light", "status": "ready", "detail": "Signal state and stop-line gate"},
        {"feature": "Illegal parking", "status": "ready", "detail": "Direct class, sign evidence, optional ROI"},
        {"feature": "License plate OCR", "status": "ready", "detail": "Plate detector with Tesseract fallback"},
        {"feature": "Human review", "status": "ready", "detail": "Approve/reject workflow"},
        {"feature": "Formal metrics", "status": "pending" if not eval_row else "ready", "detail": "Requires labelled ground truth"},
    ]
    evaluation_readiness = {
        "ground_truth_available": bool(eval_row),
        "schema_endpoint": "/api/evaluation/schema",
        "command": "python backend/scripts/evaluate_dataset.py --images <image_dir> --ground-truth <ground_truth.json> --write-db",
        "required_metrics": ["accuracy", "precision", "recall", "f1_score", "mAP50"],
    }
    operational = {
        "detector_ready": bool(detector_status.get("ready")),
        "ocr_ready": bool(ocr_status().get("enabled")),
        "review_enabled": True,
        "batch_enabled": True,
    }

    if not eval_row:
        return {
            "status": "needs_ground_truth",
            "message": "Accuracy, Precision, Recall, F1-score, and mAP require labelled ground-truth annotations. Use backend/scripts/evaluate_dataset.py after preparing labels.",
            "throughput": throughput,
            "detector": detector_status,
            "ocr": ocr_status(),
            "review": review,
            "coverage": coverage,
            "evaluation_readiness": evaluation_readiness,
            "operational": operational,
        }

    metrics = json.loads(eval_row["metrics_json"])
    metrics["status"] = "evaluated"
    metrics["dataset_name"] = eval_row["dataset_name"]
    metrics["created_at"] = eval_row["created_at"]
    metrics["throughput"] = throughput
    metrics["detector"] = detector_status
    metrics["ocr"] = ocr_status()
    metrics["review"] = review
    metrics["coverage"] = coverage
    metrics["evaluation_readiness"] = evaluation_readiness
    metrics["operational"] = operational
    return metrics


@app.get("/api/evaluation/schema")
async def evaluation_schema():
    return {
        "images": [
            {
                "file_name": "traffic_001.jpg",
                "violations": [
                    {
                        "type": "Helmet Non-compliance",
                        "bbox_percent": {"x": 12.5, "y": 34.0, "w": 10.2, "h": 18.7},
                    }
                ],
            }
        ]
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

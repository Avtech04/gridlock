"""
License plate OCR helpers.

OCR engines are optional. The code tries PaddleOCR, EasyOCR, then pytesseract
when those packages are installed. If no engine exists, the API still runs and
reports that OCR is unavailable.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.yaml"


def _load_config() -> dict[str, Any]:
    if yaml is None or not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


CFG = _load_config().get("ocr", {})
_ENGINE_CACHE: dict[str, Any] = {}


def normalize_plate_text(text: str) -> str | None:
    value = re.sub(r"[^A-Z0-9]", "", str(text).upper())
    if len(value) < 5:
        return None

    def as_digit(char: str) -> str:
        return {
            "O": "0",
            "Q": "0",
            "D": "0",
            "I": "1",
            "L": "1",
            "T": "7",
            "Z": "2",
            "S": "5",
            "B": "8",
            "G": "6",
        }.get(char, char)

    def as_letter(char: str) -> str:
        return {
            "0": "O",
            "1": "I",
            "2": "Z",
            "5": "S",
            "6": "G",
            "8": "B",
        }.get(char, char)

    def canonical_indian(candidate: str) -> str | None:
        candidate = re.sub(r"[^A-Z0-9]", "", candidate.upper())
        for district_len in (2, 1):
            for series_len in (3, 2, 1):
                expected_len = 2 + district_len + series_len + 4
                if len(candidate) != expected_len:
                    continue
                state = "".join(as_letter(ch) for ch in candidate[:2])
                district = "".join(as_digit(ch) for ch in candidate[2 : 2 + district_len])
                series_start = 2 + district_len
                series_end = series_start + series_len
                series = "".join(as_letter(ch) for ch in candidate[series_start:series_end])
                serial = "".join(as_digit(ch) for ch in candidate[series_end:])
                normalized = f"{state}{district}{series}{serial}"
                if re.fullmatch(r"[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}", normalized):
                    return normalized
        return None

    for length in range(min(11, len(value)), 7, -1):
        for start in range(0, len(value) - length + 1):
            normalized = canonical_indian(value[start : start + length])
            if normalized:
                return normalized

    bh = re.search(r"[0-9A-Z]{2}BH[0-9A-Z]{4}[A-Z]{1,2}", value)
    if bh:
        raw = bh.group(0)
        year = "".join(as_digit(ch) for ch in raw[:2])
        digits = "".join(as_digit(ch) for ch in raw[4:8])
        suffix = "".join(as_letter(ch) for ch in raw[8:])
        normalized = f"{year}BH{digits}{suffix}"
        if re.fullmatch(r"[0-9]{2}BH[0-9]{4}[A-Z]{1,2}", normalized):
            return normalized

    indian_patterns = [
        r"[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}",
        r"[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}",
        r"[0-9]{2}BH[0-9]{4}[A-Z]{1,2}",
    ]
    for pattern in indian_patterns:
        match = re.search(pattern, value)
        if match:
            return match.group(0)

    # Demo/test images may include non-Indian plates. Keep this generic path
    # strict enough to avoid words: plate-like strings must mix letters/digits.
    def generic_plate(candidate: str) -> str | None:
        candidate = re.sub(r"[^A-Z0-9]", "", candidate.upper())
        if not 6 <= len(candidate) <= 10:
            return None
        if not re.search(r"[A-Z]", candidate) or not re.search(r"[0-9]", candidate):
            return None
        if re.fullmatch(r"[A-Z0-9]{5,10}", candidate):
            return candidate
        return None

    # Common ASEAN/Vietnam-style plate: 29A33185. OCR often reads A as 4.
    for length in (8, 7):
        for start in range(0, len(value) - length + 1):
            candidate = value[start : start + length]
            if not re.fullmatch(r"[0-9]{3}[0-9]{4,5}", candidate):
                continue
            repaired_letter = {
                "4": "A",
                "8": "B",
                "0": "D",
                "1": "I",
                "5": "S",
                "6": "G",
            }.get(candidate[2])
            if repaired_letter:
                return f"{candidate[:2]}{repaired_letter}{candidate[3:]}"

    for length in range(min(10, len(value)), 5, -1):
        for start in range(0, len(value) - length + 1):
            normalized = generic_plate(value[start : start + length])
            if normalized:
                return normalized

    return None


def _crop_percent(image: Any, bbox: dict[str, float], pad: float = 0.08) -> Any:
    h, w = image.shape[:2]
    x = bbox.get("x", 0) / 100 * w
    y = bbox.get("y", 0) / 100 * h
    bw = bbox.get("w", 0) / 100 * w
    bh = bbox.get("h", 0) / 100 * h
    px = bw * pad
    py = bh * pad
    x1 = max(0, int(x - px))
    y1 = max(0, int(y - py))
    x2 = min(w, int(x + bw + px))
    y2 = min(h, int(y + bh + py))
    return image[y1:y2, x1:x2]


def _plate_crop_variants(crop: Any) -> list[Any]:
    if cv2 is None or crop is None or crop.size == 0:
        return [crop]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    longest = max(gray.shape[:2])
    scale = max(2.0, min(5.0, 420 / max(longest, 1)))
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 5, 35, 35)
    variants = [gray]

    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    variants.append(equalized)

    _, otsu = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.extend([otsu, 255 - otsu])

    adaptive = cv2.adaptiveThreshold(
        equalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7,
    )
    variants.extend([adaptive, 255 - adaptive])
    return variants


def _engine_paddle() -> Any | None:
    if "paddleocr" in _ENGINE_CACHE:
        return _ENGINE_CACHE["paddleocr"]
    try:
        from paddleocr import PaddleOCR

        engine = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)
        _ENGINE_CACHE["paddleocr"] = engine
        return engine
    except Exception as exc:
        _ENGINE_CACHE["paddleocr"] = None
        print(f"[ocr] PaddleOCR unavailable: {exc}")
        return None


def _engine_easyocr() -> Any | None:
    if "easyocr" in _ENGINE_CACHE:
        return _ENGINE_CACHE["easyocr"]
    try:
        import easyocr

        engine = easyocr.Reader(["en"], gpu=False, verbose=False)
        _ENGINE_CACHE["easyocr"] = engine
        return engine
    except Exception as exc:
        _ENGINE_CACHE["easyocr"] = None
        print(f"[ocr] EasyOCR unavailable: {exc}")
        return None


def _read_with_paddle(crop: Any) -> list[dict[str, Any]]:
    engine = _engine_paddle()
    if engine is None:
        return []
    try:
        result = engine.predict(crop)
        texts: list[dict[str, Any]] = []
        for page in result or []:
            data = getattr(page, "json", None) or page
            if isinstance(data, dict):
                rec_texts = data.get("rec_texts", [])
                rec_scores = data.get("rec_scores", [])
                for text, score in zip(rec_texts, rec_scores):
                    texts.append({"text": text, "confidence": float(score), "engine": "paddleocr"})
        return texts
    except Exception as exc:
        print(f"[ocr] PaddleOCR read failed: {exc}")
        return []


def _read_with_easyocr(crop: Any) -> list[dict[str, Any]]:
    engine = _engine_easyocr()
    if engine is None:
        return []
    try:
        return [
            {"text": item[1], "confidence": float(item[2]), "engine": "easyocr"}
            for item in engine.readtext(crop)
        ]
    except Exception as exc:
        print(f"[ocr] EasyOCR read failed: {exc}")
        return []


def _read_with_tesseract(crop: Any) -> list[dict[str, Any]]:
    try:
        import pytesseract

        results: list[dict[str, Any]] = []
        seen: set[str] = set()
        for variant in _plate_crop_variants(crop)[:6]:
            for psm in (7, 8, 13):
                text = pytesseract.image_to_string(
                    variant,
                    config=f"--psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                    timeout=1.2,
                )
                cleaned = normalize_plate_text(text)
                if not cleaned or cleaned in seen:
                    continue
                seen.add(cleaned)
                is_indian = bool(
                    re.fullmatch(r"[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}", cleaned)
                    or re.fullmatch(r"[0-9]{2}BH[0-9]{4}[A-Z]{1,2}", cleaned)
                )
                results.append(
                    {
                        "text": cleaned,
                        "confidence": 0.72 if is_indian else 0.55,
                        "engine": "pytesseract",
                    }
                )
        return results
    except Exception as exc:
        print(f"[ocr] pytesseract unavailable/read failed: {exc}")
        return []


def _bbox_iou(a: dict[str, float], b: dict[str, float]) -> float:
    ax1, ay1 = float(a.get("x", 0)), float(a.get("y", 0))
    ax2, ay2 = ax1 + float(a.get("w", 0)), ay1 + float(a.get("h", 0))
    bx1, by1 = float(b.get("x", 0)), float(b.get("y", 0))
    bx2, by2 = bx1 + float(b.get("w", 0)), by1 + float(b.get("h", 0))
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    area_a = max((ax2 - ax1) * (ay2 - ay1), 0.0)
    area_b = max((bx2 - bx1) * (by2 - by1), 0.0)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _center_inside(inner: dict[str, float], outer: dict[str, float], x_pad: float = 0.0, y_pad: float = 0.0) -> bool:
    ox1, oy1 = float(outer.get("x", 0)), float(outer.get("y", 0))
    ox2 = ox1 + float(outer.get("w", 0))
    oy2 = oy1 + float(outer.get("h", 0))
    cx = float(inner.get("x", 0)) + float(inner.get("w", 0)) / 2
    cy = float(inner.get("y", 0)) + float(inner.get("h", 0)) / 2
    return (ox1 - x_pad) <= cx <= (ox2 + x_pad) and (oy1 - y_pad) <= cy <= (oy2 + y_pad)


def _dedupe_plate_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(candidates, key=lambda item: item.get("confidence", 0.0), reverse=True)
    kept: list[dict[str, Any]] = []
    for candidate in ordered:
        bbox = candidate.get("bbox_percent", {})
        if any(_bbox_iou(bbox, existing.get("bbox_percent", {})) >= 0.35 for existing in kept):
            continue
        kept.append(candidate)
    return kept


def _dedupe_recognized_by_vehicle(recognized: list[dict[str, Any]], vehicles: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not recognized:
        return []
    vehicles = vehicles or []
    ordered = sorted(
        recognized,
        key=lambda item: (float(item.get("detector_confidence", 0.0)), float(item.get("confidence", 0.0))),
        reverse=True,
    )
    kept: list[dict[str, Any]] = []
    occupied_vehicle_ids: set[int] = set()
    for plate in ordered:
        bbox = plate.get("bbox_percent", {})
        vehicle_id = None
        for index, vehicle in enumerate(vehicles):
            if _center_inside(bbox, vehicle.get("bbox_percent", {}), x_pad=2.0, y_pad=2.0):
                vehicle_id = index
                break
        if vehicle_id is not None:
            if vehicle_id in occupied_vehicle_ids:
                continue
            occupied_vehicle_ids.add(vehicle_id)
        elif any(_bbox_iou(bbox, existing.get("bbox_percent", {})) >= 0.18 for existing in kept):
            continue
        kept.append(plate)
    return kept


def _vehicle_crop_bounds(image: Any, vehicle: dict[str, Any]) -> tuple[int, int, int, int]:
    height, width = image.shape[:2]
    bbox = vehicle.get("bbox_percent", {})
    x1 = int(float(bbox.get("x", 0)) / 100 * width)
    y1 = int(float(bbox.get("y", 0)) / 100 * height)
    x2 = int((float(bbox.get("x", 0)) + float(bbox.get("w", 0))) / 100 * width)
    y2 = int((float(bbox.get("y", 0)) + float(bbox.get("h", 0))) / 100 * height)
    y1 = int(y1 + max(0, y2 - y1) * 0.42)
    return max(0, x1), max(0, y1), min(width, x2), min(height, y2)


def _cv_plate_candidates(image: Any, vehicles: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    if cv2 is None or np is None or image is None:
        return []
    height, width = image.shape[:2]
    search_windows: list[tuple[int, int, int, int]] = [(0, int(height * 0.30), width, height)]
    for vehicle in vehicles or []:
        bounds = _vehicle_crop_bounds(image, vehicle)
        if bounds[2] - bounds[0] >= 20 and bounds[3] - bounds[1] >= 12:
            search_windows.append(bounds)

    candidates: list[dict[str, Any]] = []
    for wx1, wy1, wx2, wy2 in search_windows:
        crop = image[wy1:wy2, wx1:wx2]
        if crop.size == 0:
            continue
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 5, 35, 35)
        edges = cv2.Canny(gray, 60, 180)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 9), np.uint8))
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 12 or h <= 5:
                continue
            aspect = w / max(h, 1)
            area = w * h
            if not (1.8 <= aspect <= 7.2):
                continue
            if area < 90 or area > (width * height * 0.025):
                continue
            px = (wx1 + x) / width * 100
            py = (wy1 + y) / height * 100
            pw = w / width * 100
            ph = h / height * 100
            if pw < 1.2 or ph < 0.6 or pw > 22 or ph > 10:
                continue
            candidates.append(
                {
                    "type": "license_plate",
                    "class_name": "cv_plate_candidate",
                    "confidence": round(min(0.66, 0.38 + min(aspect / 7.2, 1.0) * 0.18), 4),
                    "bbox_percent": {"x": round(px, 3), "y": round(py, 3), "w": round(pw, 3), "h": round(ph, 3)},
                    "source": "cv-plate-candidate",
                }
            )
    for vehicle in vehicles or []:
        bbox = vehicle.get("bbox_percent", {})
        vx = float(bbox.get("x", 0))
        vy = float(bbox.get("y", 0))
        vw = float(bbox.get("w", 0))
        vh = float(bbox.get("h", 0))
        if vw < 3.0 or vh < 3.0:
            continue
        for y_factor in (0.56, 0.64, 0.72):
            candidates.append(
                {
                    "type": "license_plate",
                    "class_name": "vehicle_plate_hypothesis",
                    "confidence": 0.395,
                    "bbox_percent": {
                        "x": round(max(0.0, vx + vw * 0.24), 3),
                        "y": round(max(0.0, vy + vh * y_factor), 3),
                        "w": round(min(100.0 - max(0.0, vx + vw * 0.24), vw * 0.52), 3),
                        "h": round(min(100.0 - max(0.0, vy + vh * y_factor), max(vh * 0.12, 0.8)), 3),
                    },
                    "source": "vehicle-plate-hypothesis",
                }
            )

    return _dedupe_plate_candidates(candidates)[:10]


def ocr_status() -> dict[str, Any]:
    engines = {}
    for name in CFG.get("engine_order", ["paddleocr", "easyocr", "pytesseract"]):
        if name == "paddleocr":
            engines[name] = _engine_paddle() is not None
        elif name == "easyocr":
            engines[name] = _engine_easyocr() is not None
        elif name == "pytesseract":
            try:
                import pytesseract  # noqa: F401

                engines[name] = True
            except Exception:
                engines[name] = False
    return {"enabled": bool(CFG.get("enabled", True)), "engines": engines}


def recognize_plates(
    image_path: str,
    plate_detections: list[dict[str, Any]],
    vehicles: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if not CFG.get("enabled", True) or cv2 is None:
        return []

    image = cv2.imread(image_path)
    if image is None:
        return []

    candidate_detections = _dedupe_plate_candidates([*plate_detections, *_cv_plate_candidates(image, vehicles)])
    if not candidate_detections:
        return []

    min_conf = float(CFG.get("min_text_confidence", 0.45))
    engine_order = CFG.get("engine_order", ["paddleocr", "easyocr", "pytesseract"])
    recognized: list[dict[str, Any]] = []

    for detection in candidate_detections:
        crop = _crop_percent(image, detection.get("bbox_percent", {}))
        candidates: list[dict[str, Any]] = []
        for engine in engine_order:
            if engine == "paddleocr":
                candidates.extend(_read_with_paddle(crop))
            elif engine == "easyocr":
                candidates.extend(_read_with_easyocr(crop))
            elif engine == "pytesseract":
                candidates.extend(_read_with_tesseract(crop))

        best = None
        for candidate in candidates:
            plate = normalize_plate_text(candidate.get("text", ""))
            confidence = float(candidate.get("confidence", 0.0))
            if not plate or confidence < min_conf:
                continue
            if best is None or confidence > best["confidence"]:
                best = {
                    "text": plate,
                    "confidence": round(confidence, 4),
                    "engine": candidate.get("engine", "unknown"),
                    "bbox_percent": detection.get("bbox_percent", {}),
                    "detector_confidence": detection.get("confidence", 0.0),
                }
        if best:
            recognized.append(best)

    return _dedupe_recognized_by_vehicle(recognized, vehicles)

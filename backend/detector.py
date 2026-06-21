"""
Detector runtime for traffic-scene evidence.

The detector is designed for hackathon judging and real deployment:
- custom fine-tuned YOLO weights are tried first;
- current Ultralytics pretrained models are used as generic fallbacks;
- the bundled YOLOv3-tiny COCO model is used when no Ultralytics model is
  available, so the demo still runs offline.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

try:
    import cv2
except Exception:  # pragma: no cover - handled at runtime
    cv2 = None

try:
    import numpy as np
except Exception:  # pragma: no cover - handled at runtime
    np = None

try:
    import yaml
except Exception:  # pragma: no cover - handled at runtime
    yaml = None

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - handled at runtime
    YOLO = None


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.yaml"


def _load_config() -> dict[str, Any]:
    if yaml is None or not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


CFG = _load_config()
MODEL_CFG = CFG.get("model", {})
FALLBACK_CFG = CFG.get("opencv_fallback", {})

VEHICLE_CLASSES = {
    "car",
    "motorcycle",
    "motorbike",
    "bike",
    "scooter",
    "bus",
    "truck",
    "bicycle",
    "auto rickshaw",
    "autorickshaw",
    "rickshaw",
    "van",
}
PERSON_CLASSES = {"person", "pedestrian", "rider", "driver"}
TRAFFIC_LIGHT_CLASSES = {"traffic light", "traffic_light", "signal", "red light", "green light", "yellow light"}
STOP_LINE_CLASSES = {"stop line", "stop_line", "zebra crossing", "crosswalk"}
STOP_SIGN_CLASSES = {"stop sign", "stop_sign"}
NO_PARKING_SIGN_CLASSES = {"no stopping", "no stopping sign", "no parking", "no parking sign"}
NO_ENTRY_SIGN_CLASSES = {"no entry", "no entry sign"}
PLATE_CLASSES = {"license plate", "licence plate", "number plate", "plate", "registration plate"}
ILLEGAL_PARKING_CLASSES = {"illegal parking", "illegally parked", "illegally_parked", "parked vehicle"}
WRONG_SIDE_CLASSES = {"wrong side", "wrong-side", "wrong_side", "wrong way", "wrong-way", "wrong_way"}
RIGHT_SIDE_CLASSES = {"right side", "right-side", "right_side", "correct side"}
SAFETY_CLASSES = {
    "helmet",
    "withhelmet",
    "no helmet",
    "without helmet",
    "withouthelmet",
    "no_helmet",
    "seatbelt",
    "seat belt",
    "person seatbelt",
    "person-seatbelt",
    "no seatbelt",
    "no_seatbelt",
    "without seatbelt",
    "person noseatbelt",
    "person-noseatbelt",
    "triple riding",
    "triple_riding",
    "tripple riding",
    "tripple ridding",
    "tripple-ridding",
}


def _norm_name(name: str) -> str:
    return str(name).strip().lower().replace("_", " ").replace("-", " ")


def _resolve_path(path_value: str) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    local_path = BASE_DIR / path
    return str(local_path if local_path.exists() else path)


def _convert_bbox(xyxy: list[float], img_w: int, img_h: int) -> dict[str, float]:
    x1, y1, x2, y2 = xyxy
    return {
        "x": round(max(0.0, float(x1)) / img_w * 100, 3),
        "y": round(max(0.0, float(y1)) / img_h * 100, 3),
        "w": round(max(0.0, float(x2) - float(x1)) / img_w * 100, 3),
        "h": round(max(0.0, float(y2) - float(y1)) / img_h * 100, 3),
    }


def _percent_to_xyxy(bbox: dict[str, float]) -> tuple[float, float, float, float]:
    x1 = float(bbox.get("x", 0))
    y1 = float(bbox.get("y", 0))
    return x1, y1, x1 + float(bbox.get("w", 0)), y1 + float(bbox.get("h", 0))


def _iou(a: dict[str, float], b: dict[str, float]) -> float:
    ax1, ay1, ax2, ay2 = _percent_to_xyxy(a)
    bx1, by1, bx2, by2 = _percent_to_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    area_a = max((ax2 - ax1) * (ay2 - ay1), 0.0)
    area_b = max((bx2 - bx1) * (by2 - by1), 0.0)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _dedupe(entries: list[dict[str, Any]], iou_threshold: float = 0.55) -> list[dict[str, Any]]:
    ordered = sorted(entries, key=lambda item: item.get("confidence", 0.0), reverse=True)
    kept: list[dict[str, Any]] = []
    for entry in ordered:
        duplicate = False
        for existing in kept:
            same_type = entry.get("type") == existing.get("type")
            if same_type and _iou(entry["bbox_percent"], existing["bbox_percent"]) >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(entry)
    return kept


def _bucket_for_class(class_name: str) -> str | None:
    name = _norm_name(class_name)
    if name in VEHICLE_CLASSES:
        return "vehicles"
    if name in PERSON_CLASSES:
        return "persons"
    if name in TRAFFIC_LIGHT_CLASSES:
        return "traffic_lights"
    if name in STOP_LINE_CLASSES:
        return "stop_lines"
    if name in STOP_SIGN_CLASSES:
        return "stop_signs"
    if name in NO_PARKING_SIGN_CLASSES:
        return "no_parking_signs"
    if name in NO_ENTRY_SIGN_CLASSES:
        return "no_entry_signs"
    if name in PLATE_CLASSES:
        return "license_plates"
    if name in ILLEGAL_PARKING_CLASSES:
        return "illegal_parking_vehicles"
    if name in WRONG_SIDE_CLASSES:
        return "wrong_side_vehicles"
    if name in RIGHT_SIDE_CLASSES:
        return "right_side_vehicles"
    if name in SAFETY_CLASSES:
        return "safety"
    return None


def _vehicle_type(class_name: str) -> str:
    name = _norm_name(class_name)
    aliases = {
        "motorbike": "motorcycle",
        "bike": "motorcycle",
        "two wheeler": "motorcycle",
        "scooter": "motorcycle",
        "auto rickshaw": "auto-rickshaw",
        "autorickshaw": "auto-rickshaw",
        "rickshaw": "auto-rickshaw",
        "rikshaw": "auto-rickshaw",
        "auto": "auto-rickshaw",
    }
    return aliases.get(name, name)


def _entry_type(bucket: str, class_name: str) -> str:
    name = _norm_name(class_name)
    if bucket == "vehicles":
        return _vehicle_type(name)
    if bucket == "license_plates":
        return "license_plate"
    if bucket == "traffic_lights":
        return "traffic_light"
    if bucket == "stop_lines":
        return "stop_line"
    if bucket == "no_parking_signs":
        return "no_parking_sign"
    if bucket == "no_entry_signs":
        return "no_entry_sign"
    if bucket == "stop_signs":
        return "stop_sign"
    if bucket == "safety":
        aliases = {
            "withhelmet": "helmet",
            "withouthelmet": "no helmet",
            "without helmet": "no helmet",
            "no helmet": "no helmet",
            "no seatbelt": "no seatbelt",
            "without seatbelt": "no seatbelt",
            "person noseatbelt": "no seatbelt",
            "person no seatbelt": "no seatbelt",
            "person seatbelt": "seatbelt",
            "triple riding": "triple riding",
            "tripple riding": "triple riding",
            "tripple ridding": "triple riding",
        }
        return aliases.get(name, name)
    if bucket == "illegal_parking_vehicles":
        return "illegal_parking"
    if bucket == "wrong_side_vehicles":
        return "wrong_side"
    if bucket == "right_side_vehicles":
        return "right_side"
    return name


def _detect_traffic_light_color(img: Any, bbox_percent: dict[str, float], img_h: int, img_w: int) -> str:
    if cv2 is None or np is None:
        return "unknown"

    x = int(bbox_percent["x"] / 100 * img_w)
    y = int(bbox_percent["y"] / 100 * img_h)
    w = int(bbox_percent["w"] / 100 * img_w)
    h = int(bbox_percent["h"] / 100 * img_h)
    crop = img[max(0, y) : min(img_h, y + h), max(0, x) : min(img_w, x + w)]
    if crop.size == 0 or crop.shape[0] < 6 or crop.shape[1] < 3:
        return "unknown"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    third = max(crop.shape[0] // 3, 1)
    slices = {
        "red": hsv[:third],
        "yellow": hsv[third : 2 * third],
        "green": hsv[2 * third :],
    }

    scores: dict[str, int] = {}
    for color, region in slices.items():
        if region.size == 0:
            scores[color] = 0
            continue
        if color == "red":
            mask_a = cv2.inRange(region, np.array([0, 90, 90]), np.array([10, 255, 255]))
            mask_b = cv2.inRange(region, np.array([170, 90, 90]), np.array([180, 255, 255]))
            mask = cv2.bitwise_or(mask_a, mask_b)
        elif color == "yellow":
            mask = cv2.inRange(region, np.array([15, 80, 90]), np.array([35, 255, 255]))
        else:
            mask = cv2.inRange(region, np.array([38, 50, 60]), np.array([95, 255, 255]))
        scores[color] = int(cv2.countNonZero(mask))

    best = max(scores, key=scores.get)
    min_pixels = max(8, int(crop.shape[0] * crop.shape[1] * 0.015))
    return best if scores[best] >= min_pixels else "unknown"


def _vehicle_visual_attributes(img: Any, bbox_percent: dict[str, float], img_h: int, img_w: int) -> dict[str, Any]:
    if cv2 is None or np is None:
        return {}

    x = int(float(bbox_percent.get("x", 0)) / 100 * img_w)
    y = int(float(bbox_percent.get("y", 0)) / 100 * img_h)
    w = int(float(bbox_percent.get("w", 0)) / 100 * img_w)
    h = int(float(bbox_percent.get("h", 0)) / 100 * img_h)
    if w < 90 or h < 90:
        return {}

    crop = img[max(0, y) : min(img_h, y + h), max(0, x) : min(img_w, x + w)]
    if crop.size == 0:
        return {}

    ch, cw = crop.shape[:2]
    cabin = crop[int(ch * 0.10) : int(ch * 0.48), int(cw * 0.16) : int(cw * 0.78)]
    if cabin.size == 0:
        return {}

    hsv = cv2.cvtColor(cabin, cv2.COLOR_BGR2HSV)
    skin_a = cv2.inRange(hsv, np.array([0, 20, 45]), np.array([28, 190, 255]))
    skin_b = cv2.inRange(hsv, np.array([160, 20, 45]), np.array([180, 180, 255]))
    skin = cv2.bitwise_or(skin_a, skin_b)
    skin_ratio = float(cv2.countNonZero(skin)) / max(cabin.shape[0] * cabin.shape[1], 1)

    # This fallback is deliberately narrow. Normal CCTV angles cannot prove
    # seatbelt use; only a close, centered, frontal cabin crop should create a
    # weak no-seatbelt candidate when the specialist detector is silent.
    occupant_visible = skin_ratio >= 0.014
    width_pct = float(bbox_percent.get("w", 0))
    height_pct = float(bbox_percent.get("h", 0))
    x_pct = float(bbox_percent.get("x", 0))
    close_cabin = (
        width_pct >= 36
        and height_pct >= 38
        and x_pct <= 60
        and (x_pct + width_pct) >= 42
    )
    return {
        "occupant_visible": occupant_visible,
        "cabin_skin_ratio": round(skin_ratio, 4),
        "seatbelt_non_compliance_candidate": bool(occupant_visible and close_cabin),
    }


def _cv_no_parking_entries(img: Any) -> list[dict[str, Any]]:
    if cv2 is None or np is None:
        return []

    img_h, img_w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_a = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([12, 255, 255]))
    mask_b = cv2.inRange(hsv, np.array([168, 70, 50]), np.array([180, 255, 255]))
    mask = cv2.bitwise_or(mask_a, mask_b)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    entries: list[dict[str, Any]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if area < img_w * img_h * 0.00025:
            continue
        if y > img_h * 0.45:
            continue
        aspect = w / max(h, 1)
        fill = area / max(w * h, 1)
        width_pct = w / img_w * 100
        height_pct = h / img_h * 100
        roadside = x <= img_w * 0.18 or x >= img_w * 0.84
        upper_scene = y <= img_h * 0.32
        foreground_board = (
            0.55 <= aspect <= 1.35
            and 14.0 <= width_pct <= 34.0
            and 14.0 <= height_pct <= 42.0
            and 18.0 <= (y / img_h * 100) <= 58.0
            and fill >= 0.24
        )
        roadside_board = (
            roadside
            and upper_scene
            and 0.45 <= aspect <= 1.35
            and 5.0 <= width_pct <= 16.0
            and 5.0 <= height_pct <= 20.0
        )
        if not (roadside_board or foreground_board):
            continue
        if fill < 0.12:
            continue
        bbox = _convert_bbox([x, y, x + w, y + h], img_w, img_h)
        entries.append(
            {
                "type": "no_parking_sign",
                "class_name": "no parking sign",
                "confidence": round(min(0.86, 0.58 + min(fill, 0.45) * 0.45), 4),
                "bbox_percent": bbox,
                "source": "cv-sign",
                "model": "red-circle-sign-heuristic",
            }
        )
    return _dedupe(entries, iou_threshold=0.35)


def _filter_no_parking_entries(signs: list[dict[str, Any]], raw_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    traffic_lights = [
        entry for entry in raw_entries
        if _bucket_for_class(entry.get("class_name", entry.get("type", ""))) == "traffic_lights"
    ]
    road_users = [
        entry
        for entry in raw_entries
        if _bucket_for_class(entry.get("class_name", entry.get("type", ""))) in {"vehicles", "persons"}
    ]
    filtered: list[dict[str, Any]] = []
    for sign in signs:
        bbox = sign.get("bbox_percent", {})
        width = float(bbox.get("w", 0))
        height = float(bbox.get("h", 0))
        if height > width * 2.4:
            continue
        if any(_iou(bbox, tl.get("bbox_percent", {})) >= 0.05 for tl in traffic_lights):
            continue
        if width < 12.0 and height < 12.0:
            sx1, sy1, sx2, sy2 = _percent_to_xyxy(bbox)
            scx = (sx1 + sx2) / 2
            scy = (sy1 + sy2) / 2
            attached_to_road_user = False
            for user in road_users:
                ux1, uy1, ux2, uy2 = _percent_to_xyxy(user.get("bbox_percent", {}))
                if (ux1 - 2.0) <= scx <= (ux2 + 2.0) and (uy1 - 4.0) <= scy <= (uy2 + 2.0):
                    attached_to_road_user = True
                    break
                if _iou(bbox, user.get("bbox_percent", {})) >= 0.01:
                    attached_to_road_user = True
                    break
            if attached_to_road_user:
                continue
        filtered.append(sign)
    return filtered


def _detect_scene_geometry(img: Any) -> dict[str, Any]:
    if cv2 is None or np is None:
        return {}

    img_h, img_w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, np.array([0, 0, 145]), np.array([180, 80, 255]))
    white[: int(img_h * 0.35), :] = 0
    white[int(img_h * 0.90) :, :] = 0
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((5, 15), np.uint8))
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    stripe_boxes: list[dict[str, float]] = []
    lane_arrows: list[dict[str, float]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < img_w * 0.025 or h < img_h * 0.004:
            continue
        aspect = w / max(h, 1)
        width_pct = w / img_w * 100
        height_pct = h / img_h * 100
        if aspect < 2.2 or h > img_h * 0.08:
            if 4.0 <= width_pct <= 16.0 and 4.0 <= height_pct <= 14.0 and 0.9 <= aspect <= 3.8:
                y_pct = (y + h / 2) / img_h * 100
                if 25 <= y_pct <= 85:
                    lane_arrows.append(_convert_bbox([x, y, x + w, y + h], img_w, img_h))
            continue
        y_pct = (y + h / 2) / img_h * 100
        if 35 <= y_pct <= 88:
            stripe_boxes.append(_convert_bbox([x, y, x + w, y + h], img_w, img_h))

    geometry: dict[str, Any] = {}
    if lane_arrows:
        geometry["lane_arrow_markings"] = lane_arrows[:12]

    if len(stripe_boxes) >= 3:
        bands: list[list[dict[str, float]]] = []
        for stripe in sorted(stripe_boxes, key=lambda item: item["y"] + item["h"] / 2):
            center_y = stripe["y"] + stripe["h"] / 2
            placed = False
            for band in bands:
                band_center = sum(item["y"] + item["h"] / 2 for item in band) / len(band)
                if abs(center_y - band_center) <= 7.0:
                    band.append(stripe)
                    placed = True
                    break
            if not placed:
                bands.append([stripe])
        best_band = max(bands, key=lambda band: (len(band), sum(item["w"] for item in band)), default=[])
        if len(best_band) >= 3:
            x_min = min(item["x"] for item in best_band)
            x_max = max(item["x"] + item["w"] for item in best_band)
            if x_max - x_min >= 24.0:
                geometry["stop_line_y_percent"] = round(min(item["y"] + item["h"] / 2 for item in best_band), 2)
                geometry["zebra_stripes"] = best_band[:20]
        geometry["source"] = "cv-road-marking"
    return geometry


class DetectorRuntime:
    def __init__(self) -> None:
        self.backend = "unavailable"
        self.model_name = None
        self.error = None
        self.yolo_model = None
        self.yolo_models: list[dict[str, Any]] = []
        self.cv_net = None
        self.cv_names: list[str] = []
        self._load()

    def _load(self) -> None:
        if YOLO is not None:
            loaded_generic = False
            for candidate in MODEL_CFG.get("candidates", []):
                try:
                    resolved = _resolve_path(candidate)
                    if ("/" in candidate or "\\" in candidate) and not Path(resolved).exists():
                        print(f"[detector] Skipping missing local model: {candidate}")
                        continue
                    is_project_model = candidate.replace("\\", "/").startswith("models/")
                    if loaded_generic and not is_project_model:
                        continue
                    model = YOLO(resolved)
                    self.yolo_models.append({"name": candidate, "model": model, "project_model": is_project_model})
                    if not is_project_model:
                        loaded_generic = True
                    self.yolo_model = model
                    self.backend = "ultralytics-ensemble" if len(self.yolo_models) > 1 else "ultralytics"
                    self.model_name = ", ".join(item["name"] for item in self.yolo_models)
                    self.error = None
                    print(f"[detector] Loaded Ultralytics model: {candidate}")
                except Exception as exc:
                    self.error = str(exc)
                    print(f"[detector] Could not load {candidate}: {exc}")
            if self.yolo_models:
                return

        if FALLBACK_CFG.get("enabled", True):
            self._load_opencv_fallback()

    def _load_opencv_fallback(self) -> None:
        if cv2 is None:
            self.backend = "unavailable"
            self.error = "OpenCV is not installed"
            return
        cfg_path = BASE_DIR / FALLBACK_CFG.get("cfg", "yolov3-tiny.cfg")
        weights_path = BASE_DIR / FALLBACK_CFG.get("weights", "yolov3-tiny.weights")
        names_path = BASE_DIR / FALLBACK_CFG.get("names", "coco.names")
        if not cfg_path.exists() or not weights_path.exists() or not names_path.exists():
            self.backend = "unavailable"
            self.error = "No detector weights are available"
            return
        try:
            self.cv_net = cv2.dnn.readNetFromDarknet(str(cfg_path), str(weights_path))
            with open(names_path, "r", encoding="utf-8") as handle:
                self.cv_names = [line.strip() for line in handle if line.strip()]
            self.backend = "opencv-dnn"
            self.model_name = weights_path.name
            self.error = None
            print("[detector] Loaded OpenCV YOLOv3-tiny fallback")
        except Exception as exc:
            self.backend = "unavailable"
            self.error = str(exc)

    def status(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model": self.model_name,
            "ready": self.backend != "unavailable",
            "error": self.error,
        }

    def detect(self, image_path: str) -> dict[str, Any]:
        if cv2 is None:
            return _empty_result(self.status())
        img = cv2.imread(image_path)
        if img is None:
            result = _empty_result(self.status())
            result["error"] = "Image could not be read"
            return result

        if self.backend == "ultralytics" and self.yolo_model is not None:
            return self._detect_ultralytics(image_path, img)
        if self.backend == "ultralytics-ensemble" and self.yolo_models:
            return self._detect_ultralytics(image_path, img)
        if self.backend == "opencv-dnn" and self.cv_net is not None:
            return self._detect_opencv(img)
        return _empty_result(self.status())

    def _detect_ultralytics(self, image_path: str, img: Any) -> dict[str, Any]:
        img_h, img_w = img.shape[:2]
        conf = float(MODEL_CFG.get("confidence_threshold", 0.35))
        imgsz = int(MODEL_CFG.get("image_size", 960))
        max_det = int(MODEL_CFG.get("max_det", 300))
        augment = bool(MODEL_CFG.get("tta_augment", True))

        raw_entries: list[dict[str, Any]] = []
        model_items = self.yolo_models or [{"name": self.model_name or "ultralytics", "model": self.yolo_model}]
        for model_item in model_items:
            model = model_item["model"]
            source_model = model_item["name"]
            is_project_model = bool(model_item.get("project_model", True))
            try:
                results = model.predict(
                    image_path,
                    conf=conf,
                    imgsz=imgsz,
                    max_det=max_det,
                    augment=augment,
                    verbose=False,
                )
            except TypeError:
                results = model(image_path, verbose=False)

            for result in results:
                names = getattr(result, "names", None) or getattr(model, "names", {})
                for box in result.boxes:
                    cls_id = int(box.cls)
                    class_name = _norm_name(names.get(cls_id, cls_id) if isinstance(names, dict) else names[cls_id])
                    bucket = _bucket_for_class(class_name)
                    if bucket is None:
                        continue
                    if not is_project_model and bucket not in {"vehicles", "persons"}:
                        continue
                    confidence = float(box.conf)
                    bbox = _convert_bbox(box.xyxy[0].tolist(), img_w, img_h)
                    raw_entries.append(self._entry(bucket, class_name, confidence, bbox, img, img_h, img_w, source_model))

        raw_entries.extend(_filter_no_parking_entries(_cv_no_parking_entries(img), raw_entries))
        result = _pack_result(raw_entries, self.status())
        result["scene_geometry"] = _detect_scene_geometry(img)
        return result

    def _detect_opencv(self, img: Any) -> dict[str, Any]:
        img_h, img_w = img.shape[:2]
        input_size = int(FALLBACK_CFG.get("input_size", 416))
        conf_threshold = float(FALLBACK_CFG.get("confidence_threshold", 0.45))
        nms_iou = float(FALLBACK_CFG.get("nms_iou", 0.45))

        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (input_size, input_size), swapRB=True, crop=False)
        self.cv_net.setInput(blob)
        layer_names = self.cv_net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.cv_net.getUnconnectedOutLayers().flatten()]
        outputs = self.cv_net.forward(output_layers)

        boxes_abs: list[list[int]] = []
        confidences: list[float] = []
        class_ids: list[int] = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(scores.argmax())
                confidence = float(scores[class_id])
                class_name = self.cv_names[class_id] if class_id < len(self.cv_names) else str(class_id)
                if confidence < conf_threshold or _bucket_for_class(class_name) is None:
                    continue
                center_x = int(detection[0] * img_w)
                center_y = int(detection[1] * img_h)
                width = int(detection[2] * img_w)
                height = int(detection[3] * img_h)
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                boxes_abs.append([x, y, width, height])
                confidences.append(confidence)
                class_ids.append(class_id)

        raw_entries: list[dict[str, Any]] = []
        indices = cv2.dnn.NMSBoxes(boxes_abs, confidences, conf_threshold, nms_iou)
        for idx in np.array(indices).flatten().tolist() if len(indices) else []:
            x, y, width, height = boxes_abs[idx]
            class_name = self.cv_names[class_ids[idx]]
            bucket = _bucket_for_class(class_name)
            if bucket is None:
                continue
            bbox = _convert_bbox([x, y, x + width, y + height], img_w, img_h)
            raw_entries.append(self._entry(bucket, class_name, confidences[idx], bbox, img, img_h, img_w, self.model_name))

        raw_entries.extend(_filter_no_parking_entries(_cv_no_parking_entries(img), raw_entries))
        result = _pack_result(raw_entries, self.status())
        result["scene_geometry"] = _detect_scene_geometry(img)
        return result

    def _entry(
        self,
        bucket: str,
        class_name: str,
        confidence: float,
        bbox: dict[str, float],
        img: Any,
        img_h: int,
        img_w: int,
        source_model: str | None = None,
    ) -> dict[str, Any]:
        entry = {
            "type": _entry_type(bucket, class_name),
            "class_name": _norm_name(class_name),
            "confidence": round(float(confidence), 4),
            "bbox_percent": bbox,
            "source": self.backend,
            "model": source_model or self.model_name,
        }
        if bucket == "vehicles":
            attrs = _vehicle_visual_attributes(img, bbox, img_h, img_w)
            if attrs:
                entry.update(attrs)
        if bucket == "traffic_lights":
            normalized = _norm_name(class_name)
            if normalized in {"red light", "green light", "yellow light"}:
                entry["color"] = normalized.split()[0]
            else:
                entry["color"] = _detect_traffic_light_color(img, bbox, img_h, img_w)
        if bucket == "safety":
            entry["safety_type"] = _entry_type(bucket, class_name)
        return entry


def _empty_result(status: dict[str, Any]) -> dict[str, Any]:
    return {
        "vehicles": [],
        "persons": [],
        "traffic_lights": [],
        "stop_lines": [],
        "stop_signs": [],
        "no_parking_signs": [],
        "no_entry_signs": [],
        "license_plates": [],
        "illegal_parking_vehicles": [],
        "wrong_side_vehicles": [],
        "right_side_vehicles": [],
        "safety": [],
        "raw_boxes": [],
        "scene_geometry": {},
        "detector": status,
    }


def _pack_result(raw_entries: list[dict[str, Any]], status: dict[str, Any]) -> dict[str, Any]:
    result = _empty_result(status)
    thresholds = {
        "vehicles": float(MODEL_CFG.get("vehicle_confidence", 0.45)),
        "license_plates": float(MODEL_CFG.get("plate_confidence", 0.35)),
        "safety": float(MODEL_CFG.get("safety_confidence", 0.45)),
        "stop_lines": float(MODEL_CFG.get("stop_line_confidence", 0.30)),
        "no_parking_signs": float(MODEL_CFG.get("sign_confidence", 0.35)),
        "no_entry_signs": float(MODEL_CFG.get("sign_confidence", 0.35)),
        "illegal_parking_vehicles": float(MODEL_CFG.get("violation_confidence", 0.30)),
        "wrong_side_vehicles": float(MODEL_CFG.get("violation_confidence", 0.30)),
        "right_side_vehicles": float(MODEL_CFG.get("violation_confidence", 0.35)),
    }

    for entry in _dedupe(raw_entries):
        bucket = _bucket_for_class(entry.get("class_name", entry.get("type", "")))
        if bucket is None:
            continue
        result["raw_boxes"].append(entry)
        min_conf = thresholds.get(bucket, float(MODEL_CFG.get("confidence_threshold", 0.35)))
        if entry.get("confidence", 0.0) >= min_conf:
            result[bucket].append(entry)
    return result


_RUNTIME = DetectorRuntime()


def detect(image_path: str) -> dict[str, Any]:
    return _RUNTIME.detect(image_path)


def get_detector_status() -> dict[str, Any]:
    return _RUNTIME.status()

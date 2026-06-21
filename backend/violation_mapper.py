"""
Violation reasoning layer.

This module deliberately separates observable facts from inferred violations.
Detector-backed facts are trusted most. Vision-model findings are accepted only
when confidence is high and, by default, when they overlap a detected road user.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.yaml"


VALID_VIOLATION_TYPES = {
    "Helmet Non-compliance",
    "Seatbelt Non-compliance",
    "Triple Riding",
    "Wrong-side Driving",
    "Stop-line Violation",
    "Red-light Violation",
    "Illegal Parking",
    "Mobile Phone Use",
    "Overloading",
}


def _load_config() -> dict[str, Any]:
    if yaml is None or not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


CFG = _load_config()
RULE_CFG = CFG.get("violation", {})
CALIBRATION = CFG.get("calibration", {})


def _xyxy(bbox: dict[str, float]) -> tuple[float, float, float, float]:
    x1 = float(bbox.get("x", 0))
    y1 = float(bbox.get("y", 0))
    return x1, y1, x1 + float(bbox.get("w", 0)), y1 + float(bbox.get("h", 0))


def _iou(a: dict[str, float], b: dict[str, float]) -> float:
    ax1, ay1, ax2, ay2 = _xyxy(a)
    bx1, by1, bx2, by2 = _xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    area_a = max((ax2 - ax1) * (ay2 - ay1), 0.0)
    area_b = max((bx2 - bx1) * (by2 - by1), 0.0)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _overlap_fraction(small: dict[str, float], big: dict[str, float]) -> float:
    sx1, sy1, sx2, sy2 = _xyxy(small)
    bx1, by1, bx2, by2 = _xyxy(big)
    ix1, iy1 = max(sx1, bx1), max(sy1, by1)
    ix2, iy2 = min(sx2, bx2), min(sy2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    small_area = max((sx2 - sx1) * (sy2 - sy1), 0.0)
    return intersection / small_area if small_area > 0 else 0.0


def _center_inside(inner: dict[str, float], outer: dict[str, float], x_pad: float = 0.0, y_pad: float = 0.0) -> bool:
    ox1, oy1, ox2, oy2 = _xyxy(outer)
    cx = float(inner.get("x", 0)) + float(inner.get("w", 0)) / 2
    cy = float(inner.get("y", 0)) + float(inner.get("h", 0)) / 2
    return (ox1 - x_pad) <= cx <= (ox2 + x_pad) and (oy1 - y_pad) <= cy <= (oy2 + y_pad)


def _clamp_bbox(bbox: dict[str, float]) -> dict[str, float]:
    x = max(0.0, min(100.0, float(bbox.get("x", 0))))
    y = max(0.0, min(100.0, float(bbox.get("y", 0))))
    w = max(0.1, min(100.0 - x, float(bbox.get("w", 1))))
    h = max(0.1, min(100.0 - y, float(bbox.get("h", 1))))
    return {"x": round(x, 3), "y": round(y, 3), "w": round(w, 3), "h": round(h, 3)}


def _bbox_center(bbox: dict[str, float]) -> tuple[float, float]:
    return (
        float(bbox.get("x", 0)) + float(bbox.get("w", 0)) / 2,
        float(bbox.get("y", 0)) + float(bbox.get("h", 0)) / 2,
    )


def _point_inside_roi(x: float, y: float, roi: dict[str, Any]) -> bool:
    rx = float(roi.get("x", 0))
    ry = float(roi.get("y", 0))
    rw = float(roi.get("w", 0))
    rh = float(roi.get("h", 0))
    return rx <= x <= rx + rw and ry <= y <= ry + rh


def _vehicle_front_beyond_line(bbox: dict[str, float], line_y: float) -> bool:
    top_y = float(bbox.get("y", 0))
    bottom_y = top_y + float(bbox.get("h", 0))
    height = max(float(bbox.get("h", 0)), 0.1)
    overlaps_line = top_y <= line_y <= bottom_y
    just_past_line = bottom_y < line_y and (line_y - bottom_y) <= max(4.0, height * 0.38)
    fully_past_line = top_y > line_y and (top_y - line_y) <= max(28.0, height * 1.15)
    return overlaps_line or just_past_line or fully_past_line


def _implicit_no_parking_zone(sign: dict[str, Any]) -> dict[str, Any]:
    bbox = sign.get("bbox_percent", {})
    x = float(bbox.get("x", 0))
    y = float(bbox.get("y", 0))
    width = float(bbox.get("w", 0))
    height = float(bbox.get("h", 0))
    sx = float(bbox.get("x", 0)) + float(bbox.get("w", 0)) / 2
    sy = float(bbox.get("y", 0)) + float(bbox.get("h", 0))
    if width >= 12.0 and height >= 14.0 and y >= 22.0:
        zone_x = 0.0 if sx < 60.0 else max(0.0, sx - 18.0)
        zone_right = min(100.0, sx + 18.0 if sx >= 60.0 else sx + 22.0)
        return {
            "x": zone_x,
            "y": max(0.0, y - 14.0),
            "w": max(1.0, zone_right - zone_x),
            "h": 100.0 - max(0.0, y - 14.0),
            "_foreground_sign": True,
            "_sign_y": y,
            "_sign_h": height,
        }
    if sx < 35:
        return {"x": 0.0, "y": min(100.0, sy + 2.0), "w": 58.0, "h": max(1.0, 100.0 - sy - 2.0)}
    if sx > 65:
        return {"x": 42.0, "y": min(100.0, sy + 2.0), "w": 58.0, "h": max(1.0, 100.0 - sy - 2.0)}
    return {"x": 0.0, "y": min(100.0, sy + 2.0), "w": 100.0, "h": max(1.0, 100.0 - sy - 2.0)}


def _detected_stop_line_y(stop_lines: list[dict[str, Any]]) -> float | None:
    candidates = []
    for line in stop_lines:
        bbox = line.get("bbox_percent", {})
        width = float(bbox.get("w", 0))
        height = float(bbox.get("h", 0))
        confidence = float(line.get("confidence", 0.0))
        if width < 4.0 or height <= 0:
            continue
        y = float(bbox.get("y", 0)) + height / 2
        candidates.append((confidence, width, y))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return round(candidates[0][2], 2)


def _aligned_with_lane_arrow(vehicle: dict[str, Any], arrow: dict[str, float]) -> bool:
    bbox = vehicle.get("bbox_percent", {})
    vx, vy = _bbox_center(bbox)
    ax, ay = _bbox_center(arrow)
    v_width = float(bbox.get("w", 0))
    v_height = float(bbox.get("h", 0))
    arrow_width = float(arrow.get("w", 0))
    arrow_y = float(arrow.get("y", 0))
    arrow_bottom = arrow_y + float(arrow.get("h", 0))
    plausible_vehicle_size = 6.0 <= v_width <= 18.0 and 8.0 <= v_height <= 25.0
    same_lane = abs(vx - ax) <= max(7.0, arrow_width * 0.9)
    close_to_arrow = float(bbox.get("y", 0)) <= arrow_bottom + 2.0 and (float(bbox.get("y", 0)) + v_height) >= arrow_y - 2.0
    mid_road = 18.0 <= vy <= 62.0
    return plausible_vehicle_size and same_lane and close_to_arrow and mid_road


def _make_violation(
    vtype: str,
    confidence: float,
    bbox: dict[str, float],
    vehicle_type: str,
    description: str,
    license_plate: str | None = None,
    evidence: str = "rules",
) -> dict[str, Any]:
    return {
        "type": vtype,
        "confidence": round(max(0.0, min(float(confidence), 0.99)), 4),
        "bbox_percent": _clamp_bbox(bbox),
        "vehicle_type": vehicle_type,
        "license_plate": license_plate,
        "description": description,
        "evidence": evidence,
    }


def _anchored_confidence(primary: float, anchor: float, primary_weight: float = 0.75) -> float:
    primary = max(0.0, min(float(primary), 1.0))
    anchor = max(0.0, min(float(anchor), 1.0))
    return primary * primary_weight + anchor * (1.0 - primary_weight)


def _vehicle_plate(vehicle: dict[str, Any], plates: list[dict[str, Any]]) -> str | None:
    vehicle_bbox = vehicle.get("bbox_percent", {})
    best = None
    best_score = 0.0
    for plate in plates:
        bbox = plate.get("bbox_percent", {})
        if _center_inside(bbox, vehicle_bbox, x_pad=2.0, y_pad=2.0):
            score = float(plate.get("confidence", 0.0))
            if score > best_score:
                best_score = score
                best = plate.get("text")
    return best


def _riders_for_motorcycle(motorcycle: dict[str, Any], persons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    moto_bbox = motorcycle.get("bbox_percent", {})
    mx1, my1, mx2, my2 = _xyxy(moto_bbox)
    moto_w = max(mx2 - mx1, 0.1)
    moto_h = max(my2 - my1, 0.1)
    riders: list[dict[str, Any]] = []
    for person in persons:
        p_bbox = person.get("bbox_percent", {})
        px1, py1, px2, py2 = _xyxy(p_bbox)
        person_w = max(px2 - px1, 0.1)
        person_h = max(py2 - py1, 0.1)
        overlap = _overlap_fraction(p_bbox, moto_bbox)
        x_overlap = max(0.0, min(px2, mx2) - max(px1, mx1))
        x_overlap_fraction = x_overlap / person_w
        p_center_x = (px1 + px2) / 2
        p_center_y = (py1 + py2) / 2
        p_bottom_y = py2

        # A rider should be centered on the bike, with the lower body reaching
        # into the motorcycle box. This rejects nearby standing pedestrians.
        centered_on_bike = (mx1 - moto_w * 0.15) <= p_center_x <= (mx2 + moto_w * 0.15)
        lower_body_on_bike = (my1 + moto_h * 0.40) <= p_bottom_y <= (my2 + moto_h * 0.22)
        torso_near_bike = (my1 - moto_h * 0.45) <= p_center_y <= (my2 + moto_h * 0.15)
        enough_horizontal_overlap = x_overlap_fraction >= 0.38
        plausible_size = person_h >= moto_h * 0.22 and person_w <= moto_w * 1.15

        if plausible_size and centered_on_bike and lower_body_on_bike and torso_near_bike and (overlap > 0.10 or enough_horizontal_overlap):
            riders.append(person)
    return riders


def _plausible_no_helmet_box(bbox: dict[str, float]) -> bool:
    width = float(bbox.get("w", 0))
    height = float(bbox.get("h", 0))
    area = width * height
    return 0.6 <= width <= 16.0 and 0.6 <= height <= 22.0 and area <= 180.0


def _plausible_rider_sized_no_helmet(helmet_box: dict[str, Any], motorcycle_bbox: dict[str, float]) -> bool:
    bbox = helmet_box.get("bbox_percent", {})
    model_name = str(helmet_box.get("model", "")).replace("\\", "/")
    confidence = float(helmet_box.get("confidence", 0.0))
    width = float(bbox.get("w", 0))
    height = float(bbox.get("h", 0))
    is_multitask_rider_box = "traffic_yolo" in model_name and confidence >= 0.18 and 4.0 <= width <= 22.0 and 12.0 <= height <= 58.0
    is_helmet_model_rider_box = (
        "helmet_violations" in model_name
        and confidence >= 0.50
        and 5.0 <= width <= 24.0
        and 18.0 <= height <= 84.0
    )
    if not (is_multitask_rider_box or is_helmet_model_rider_box):
        return False
    return _center_inside(bbox, motorcycle_bbox, x_pad=5.0, y_pad=8.0) or _overlap_fraction(bbox, motorcycle_bbox) >= 0.12


def _road_user_anchor_for_no_helmet(
    helmet_box: dict[str, Any],
    motorcycles: list[dict[str, Any]],
    persons: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, float]:
    bbox = helmet_box.get("bbox_percent", {})
    anchor, score = _nearest_anchor(bbox, motorcycles)
    if anchor and (
        score >= 0.04
        or _center_inside(anchor.get("bbox_percent", {}), bbox, x_pad=2.0, y_pad=2.0)
        or _center_inside(bbox, anchor.get("bbox_percent", {}), x_pad=6.0, y_pad=8.0)
    ):
        return anchor, score

    anchor, score = _nearest_anchor(bbox, persons)
    if anchor and (
        score >= 0.04
        or _center_inside(anchor.get("bbox_percent", {}), bbox, x_pad=2.0, y_pad=2.0)
        or _center_inside(bbox, anchor.get("bbox_percent", {}), x_pad=6.0, y_pad=8.0)
    ):
        return anchor, score

    return None, 0.0


def _broad_no_helmet_candidate(
    helmet_box: dict[str, Any],
    all_no_helmet: list[dict[str, Any]],
    positive_helmet: list[dict[str, Any]],
    motorcycles: list[dict[str, Any]],
    persons: list[dict[str, Any]],
) -> tuple[bool, dict[str, Any] | None]:
    bbox = helmet_box.get("bbox_percent", {})
    model_name = str(helmet_box.get("model", "")).replace("\\", "/")
    confidence = float(helmet_box.get("confidence", 0.0))
    width = float(bbox.get("w", 0))
    height = float(bbox.get("h", 0))
    area = width * height

    # Some helmet datasets label the whole rider/motorcycle region as
    # WithoutHelmet. These are valid, but only if anchored to a road user.
    plausible_region = 4.0 <= width <= 82.0 and 8.0 <= height <= 99.5 and 45.0 <= area <= 7600.0
    if not plausible_region:
        return False, None

    anchor, _score = _road_user_anchor_for_no_helmet(helmet_box, motorcycles, persons)
    if not anchor:
        return False, None

    strong_specialist = "helmet_violations" in model_name and confidence >= 0.55
    strong_multitask = "traffic_yolo" in model_name and confidence >= 0.55
    supporting_boxes = [
        box
        for box in all_no_helmet
        if float(box.get("confidence", 0.0)) >= 0.30
        and _iou(bbox, box.get("bbox_percent", {})) >= 0.05
    ]
    consensus_specialist = "helmet_violations" in model_name and confidence >= 0.34 and len(supporting_boxes) >= 2

    if not (strong_specialist or strong_multitask or consensus_specialist):
        return False, None

    if consensus_specialist:
        for helmet in positive_helmet:
            if float(helmet.get("confidence", 0.0)) >= 0.70 and _iou(bbox, helmet.get("bbox_percent", {})) >= 0.35:
                return False, None

    return True, anchor


def _rider_sized_no_helmet_without_motorcycle(helmet_box: dict[str, Any]) -> bool:
    bbox = helmet_box.get("bbox_percent", {})
    model_name = str(helmet_box.get("model", "")).replace("\\", "/")
    confidence = float(helmet_box.get("confidence", 0.0))
    width = float(bbox.get("w", 0))
    height = float(bbox.get("h", 0))
    if "traffic_yolo" not in model_name or confidence < 0.18:
        return False
    return 5.0 <= width <= 18.0 and 16.0 <= height <= 58.0


def _nearest_anchor(ai_bbox: dict[str, float], candidates: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, float]:
    best = None
    best_iou = 0.0
    for candidate in candidates:
        score = _iou(ai_bbox, candidate.get("bbox_percent", {}))
        if score > best_iou:
            best_iou = score
            best = candidate
    return best, best_iou


def _deduplicate(violations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    duplicate_iou = float(RULE_CFG.get("duplicate_iou", 0.50))
    unique: list[dict[str, Any]] = []
    for violation in sorted(violations, key=lambda item: item.get("confidence", 0), reverse=True):
        exists = False
        for kept in unique:
            if kept["type"] == violation["type"] and _iou(kept["bbox_percent"], violation["bbox_percent"]) >= duplicate_iou:
                exists = True
                break
        if not exists:
            unique.append(violation)

    max_per_type = {
        "Red-light Violation": 3,
        "Stop-line Violation": 3,
        "Illegal Parking": 3,
        "Seatbelt Non-compliance": 2,
        "Helmet Non-compliance": 3,
        "Triple Riding": 2,
        "Wrong-side Driving": 3,
    }
    counts: dict[str, int] = {}
    limited: list[dict[str, Any]] = []
    for violation in unique:
        vtype = violation.get("type", "")
        count = counts.get(vtype, 0)
        if count >= max_per_type.get(vtype, 5):
            continue
        counts[vtype] = count + 1
        limited.append(violation)
    return limited


def analyze_violations(yolo_detections: dict[str, Any], ai_analysis: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    global CFG, RULE_CFG, CALIBRATION
    CFG = _load_config()
    RULE_CFG = CFG.get("violation", {})
    CALIBRATION = CFG.get("calibration", {})

    vehicles = yolo_detections.get("vehicles", [])
    persons = yolo_detections.get("persons", [])
    traffic_lights = yolo_detections.get("traffic_lights", [])
    stop_lines = yolo_detections.get("stop_lines", [])
    no_parking_signs = yolo_detections.get("no_parking_signs", [])
    illegal_parking_vehicles = yolo_detections.get("illegal_parking_vehicles", [])
    wrong_side_vehicles = yolo_detections.get("wrong_side_vehicles", [])
    safety = yolo_detections.get("safety", [])
    recognized_plates = yolo_detections.get("recognized_plates", [])
    scene_geometry = yolo_detections.get("scene_geometry", {}) or {}

    violations: list[dict[str, Any]] = []
    motorcycles = [v for v in vehicles if v.get("type") in {"motorcycle", "bicycle"}]
    non_motorcycles = [v for v in vehicles if v.get("type") not in {"motorcycle", "bicycle"}]

    no_helmet = [s for s in safety if s.get("safety_type") in {"no helmet", "without helmet"}]
    positive_helmet = [s for s in safety if s.get("safety_type") in {"helmet", "with helmet"}]
    no_seatbelt = [s for s in safety if s.get("safety_type") in {"no seatbelt", "without seatbelt"}]
    triple_riding_evidence = [s for s in safety if s.get("safety_type") in {"triple riding"}]
    used_no_helmet_ids: set[int] = set()

    for evidence_box in illegal_parking_vehicles:
        bbox = evidence_box.get("bbox_percent", {})
        violations.append(
            _make_violation(
                "Illegal Parking",
                min(float(evidence_box.get("confidence", 0.78)) * 0.95, 0.94),
                bbox,
                evidence_box.get("vehicle_type") or evidence_box.get("type", "vehicle"),
                "Detector classified this vehicle as illegally parked",
                _vehicle_plate(evidence_box, recognized_plates),
                "detector",
            )
        )

    for evidence_box in wrong_side_vehicles:
        bbox = evidence_box.get("bbox_percent", {})
        violations.append(
            _make_violation(
                "Wrong-side Driving",
                min(float(evidence_box.get("confidence", 0.78)) * 0.95, 0.94),
                bbox,
                evidence_box.get("vehicle_type") or evidence_box.get("type", "vehicle"),
                "Detector classified this road user as travelling on the wrong side",
                _vehicle_plate(evidence_box, recognized_plates),
                "detector",
            )
        )

    for evidence_box in triple_riding_evidence:
        bbox = evidence_box.get("bbox_percent", {})
        anchor, score = _nearest_anchor(bbox, motorcycles + persons)
        final_bbox = anchor.get("bbox_percent", bbox) if anchor and score >= 0.10 else bbox
        violations.append(
            _make_violation(
                "Triple Riding",
                min(float(evidence_box.get("confidence", 0.75)) * 0.96, 0.97),
                final_bbox,
                "motorcycle",
                "Specialist detector found triple-riding evidence",
                _vehicle_plate(anchor, recognized_plates) if anchor else None,
                "detector",
            )
        )

    for helmet_box in no_helmet:
        helmet_id = id(helmet_box)
        if helmet_id in used_no_helmet_ids:
            continue
        is_candidate, anchor = _broad_no_helmet_candidate(
            helmet_box,
            no_helmet,
            positive_helmet,
            motorcycles,
            persons,
        )
        if not is_candidate:
            continue
        confidence = float(helmet_box.get("confidence", 0.0))
        bbox = helmet_box.get("bbox_percent", {})
        final_bbox = anchor.get("bbox_percent", bbox) if anchor else bbox
        anchor_confidence = float(anchor.get("confidence", 0.80)) if anchor else 0.80
        violations.append(
            _make_violation(
                "Helmet Non-compliance",
                min(max(0.48, _anchored_confidence(confidence, anchor_confidence, primary_weight=0.72)), 0.96),
                final_bbox,
                "motorcycle",
                "Helmet specialist found a rider without helmet",
                _vehicle_plate(anchor, recognized_plates) if anchor else None,
                "detector+road-user-anchor",
            )
        )
        used_no_helmet_ids.add(helmet_id)

    for motorcycle in motorcycles:
        moto_bbox = motorcycle.get("bbox_percent", {})
        plate = _vehicle_plate(motorcycle, recognized_plates)
        riders = _riders_for_motorcycle(motorcycle, persons)
        if len(riders) >= int(RULE_CFG.get("triple_riding_min_riders", 3)):
            violations.append(
                _make_violation(
                    "Triple Riding",
                    min(motorcycle.get("confidence", 0.7) * 0.96, 0.97),
                    moto_bbox,
                    "motorcycle",
                    f"Detected {len(riders)} riders associated with one motorcycle",
                    plate,
                    "detector+rules",
                )
            )

        for helmet_box in no_helmet:
            helmet_id = id(helmet_box)
            if helmet_id in used_no_helmet_ids:
                continue
            helmet_bbox = helmet_box.get("bbox_percent", {})
            rider_sized_evidence = _plausible_rider_sized_no_helmet(helmet_box, moto_bbox)
            if not _plausible_no_helmet_box(helmet_bbox) and not rider_sized_evidence:
                continue
            on_motorcycle = _center_inside(helmet_bbox, moto_bbox, x_pad=4.0, y_pad=6.0)
            on_associated_rider = any(
                _center_inside(helmet_bbox, rider.get("bbox_percent", {}), x_pad=2.0, y_pad=2.0)
                for rider in riders
            )
            if on_motorcycle or on_associated_rider or rider_sized_evidence:
                violations.append(
                    _make_violation(
                        "Helmet Non-compliance",
                        min(
                            _anchored_confidence(
                                helmet_box.get("confidence", 0.75),
                                motorcycle.get("confidence", 0.8),
                            ),
                            0.97,
                        ),
                        moto_bbox,
                        "motorcycle",
                        "Specialist safety detector found a rider without helmet",
                        plate,
                        "detector",
                    )
                )
                used_no_helmet_ids.add(helmet_id)
                break

    for helmet_box in no_helmet:
        helmet_id = id(helmet_box)
        if helmet_id in used_no_helmet_ids or not _rider_sized_no_helmet_without_motorcycle(helmet_box):
            continue
        helmet_bbox = helmet_box.get("bbox_percent", {})
        anchor, score = _nearest_anchor(helmet_bbox, persons)
        anchored_to_person = bool(
            anchor
            and (
                score >= 0.08
                or _center_inside(anchor.get("bbox_percent", {}), helmet_bbox, x_pad=2.0, y_pad=2.0)
                or _center_inside(helmet_bbox, anchor.get("bbox_percent", {}), x_pad=2.0, y_pad=2.0)
            )
        )
        if not anchored_to_person:
            continue
        violations.append(
            _make_violation(
                "Helmet Non-compliance",
                max(0.46, min(float(helmet_box.get("confidence", 0.0)) * 1.25, 0.62)),
                anchor.get("bbox_percent", helmet_bbox) if anchor else helmet_bbox,
                "motorcycle",
                "Low-light safety detector found a rider-sized no-helmet region",
                None,
                "detector+person-anchor",
            )
        )
        used_no_helmet_ids.add(helmet_id)

    for vehicle in non_motorcycles:
        plate = _vehicle_plate(vehicle, recognized_plates)
        for seatbelt_box in no_seatbelt:
            if _center_inside(seatbelt_box.get("bbox_percent", {}), vehicle.get("bbox_percent", {}), x_pad=3.0, y_pad=3.0):
                violations.append(
                    _make_violation(
                        "Seatbelt Non-compliance",
                        min(
                            _anchored_confidence(
                                seatbelt_box.get("confidence", 0.75),
                                vehicle.get("confidence", 0.8),
                            ),
                            0.97,
                        ),
                        vehicle.get("bbox_percent", {}),
                        vehicle.get("type", "vehicle"),
                        "Specialist safety detector found driver/passenger without seatbelt",
                        plate,
                        "detector",
                    )
                )

    for seatbelt_box in no_seatbelt:
        confidence = float(seatbelt_box.get("confidence", 0.0))
        model_name = str(seatbelt_box.get("model", "")).replace("\\", "/")
        if confidence < 0.78 or "traffic_yolo" not in model_name:
            continue
        bbox = seatbelt_box.get("bbox_percent", {})
        if any(_center_inside(bbox, vehicle.get("bbox_percent", {}), x_pad=3.0, y_pad=3.0) for vehicle in non_motorcycles):
            continue
        violations.append(
            _make_violation(
                "Seatbelt Non-compliance",
                min(confidence * 0.95, 0.94),
                bbox,
                "car",
                "High-confidence trained cabin detector found no-seatbelt evidence",
                None,
                "detector-direct",
            )
        )

    for vehicle in non_motorcycles:
        plate = _vehicle_plate(vehicle, recognized_plates)
        if not no_seatbelt and not no_parking_signs and not illegal_parking_vehicles and vehicle.get("seatbelt_non_compliance_candidate"):
            positive_seatbelt_inside = any(
                seatbelt.get("safety_type") in {"seatbelt", "seat belt"}
                and _center_inside(seatbelt.get("bbox_percent", {}), vehicle.get("bbox_percent", {}), x_pad=3.0, y_pad=3.0)
                for seatbelt in safety
            )
            if not positive_seatbelt_inside:
                violations.append(
                    _make_violation(
                        "Seatbelt Non-compliance",
                        min(0.62 + vehicle.get("confidence", 0.7) * 0.18, 0.79),
                        vehicle.get("bbox_percent", {}),
                        vehicle.get("type", "vehicle"),
                        "Close cabin view shows an occupant and no seatbelt evidence was detected",
                        plate,
                        "image-cabin-rule",
                    )
                )

    tl_state = "not_visible"
    if ai_analysis:
        tl_state = ai_analysis.get("traffic_light_state", "not_visible")
    if tl_state not in {"red", "green", "yellow"}:
        colors = [tl.get("color", "unknown") for tl in traffic_lights]
        if "red" in colors:
            tl_state = "red"
        elif "yellow" in colors:
            tl_state = "yellow"
        elif "green" in colors:
            tl_state = "green"
    signal_override = CALIBRATION.get("traffic_light_state_override", "auto")
    if signal_override in {"red", "yellow", "green", "not_visible"}:
        tl_state = signal_override

    auto_line_y = _detected_stop_line_y(stop_lines) or scene_geometry.get("stop_line_y_percent")
    min_auto_line_y = float(CALIBRATION.get("min_auto_stop_line_y_percent", 52.0))
    if auto_line_y is not None and float(auto_line_y) < min_auto_line_y:
        auto_line_y = None
    auto_scene_rules = bool(CALIBRATION.get("auto_scene_rules_enabled", True))

    if tl_state == "red" and (bool(CALIBRATION.get("red_light_rule_enabled", False)) or (auto_scene_rules and auto_line_y)):
        line_y = float(CALIBRATION.get("red_light_vehicle_bottom_y_percent", auto_line_y or 52))
        if not bool(CALIBRATION.get("red_light_rule_enabled", False)) and auto_line_y:
            line_y = float(auto_line_y)
        for vehicle in vehicles:
            bbox = vehicle.get("bbox_percent", {})
            if _vehicle_front_beyond_line(bbox, line_y):
                violations.append(
                    _make_violation(
                        "Red-light Violation",
                        min(vehicle.get("confidence", 0.7) * 0.86, 0.94),
                        bbox,
                        vehicle.get("type", "vehicle"),
                        (
                            "Vehicle front is beyond the visible stop line while signal is red"
                            if auto_line_y
                            else "Vehicle front is beyond the calibrated red-light enforcement line"
                        ),
                        _vehicle_plate(vehicle, recognized_plates),
                        "detector+signal+road-marking" if auto_line_y else "detector+calibration",
                    )
                )

    if bool(CALIBRATION.get("stop_line_rule_enabled", False)) or (tl_state == "red" and auto_scene_rules and auto_line_y):
        line_y = float(CALIBRATION.get("stop_line_y_percent", auto_line_y or 60))
        if not bool(CALIBRATION.get("stop_line_rule_enabled", False)) and auto_line_y:
            line_y = float(auto_line_y)
        for vehicle in vehicles:
            bbox = vehicle.get("bbox_percent", {})
            if _vehicle_front_beyond_line(bbox, line_y):
                violations.append(
                    _make_violation(
                        "Stop-line Violation",
                        min(vehicle.get("confidence", 0.7) * 0.82, 0.92),
                        bbox,
                        vehicle.get("type", "vehicle"),
                        "Vehicle front is beyond the visible/calibrated stop line",
                        _vehicle_plate(vehicle, recognized_plates),
                        "detector+road-marking" if auto_line_y else "detector+calibration",
                    )
                )

    if bool(CALIBRATION.get("illegal_parking_rule_enabled", False)):
        rois = CALIBRATION.get("illegal_parking_rois", []) or []
        for vehicle in vehicles:
            bbox = vehicle.get("bbox_percent", {})
            center_x, center_y = _bbox_center(bbox)
            for roi in rois:
                if _point_inside_roi(center_x, center_y, roi):
                    violations.append(
                        _make_violation(
                            "Illegal Parking",
                            min(vehicle.get("confidence", 0.7) * 0.78, 0.90),
                            bbox,
                            vehicle.get("type", "vehicle"),
                            "Vehicle is inside a calibrated restricted parking zone",
                            _vehicle_plate(vehicle, recognized_plates),
                            "detector+calibration",
                        )
                    )
                    break

    if no_parking_signs and auto_scene_rules:
        implicit_zones = [_implicit_no_parking_zone(sign) for sign in no_parking_signs]
        for vehicle in vehicles:
            bbox = vehicle.get("bbox_percent", {})
            center_x, center_y = _bbox_center(bbox)
            matching_zone = None
            for zone in implicit_zones:
                if not _point_inside_roi(center_x, center_y, zone):
                    continue
                if zone.get("_foreground_sign"):
                    min_foreground_y = float(zone.get("_sign_y", 0.0)) + float(zone.get("_sign_h", 0.0)) * 0.25
                    if center_y < min_foreground_y:
                        continue
                matching_zone = zone
                break
            if matching_zone:
                violations.append(
                    _make_violation(
                        "Illegal Parking",
                        min(vehicle.get("confidence", 0.7) * 0.80, 0.91),
                        bbox,
                        vehicle.get("type", "vehicle"),
                        "Vehicle is in a curb-side restricted area indicated by a visible no-parking sign",
                        _vehicle_plate(vehicle, recognized_plates),
                        "detector+no-parking-sign",
                    )
                )

    if auto_scene_rules and bool(CALIBRATION.get("wrong_side_enabled", True)):
        lane_arrows = scene_geometry.get("lane_arrow_markings", []) or []
        used_vehicle_ids: set[int] = set()
        for arrow in lane_arrows:
            for vehicle in non_motorcycles:
                if id(vehicle) in used_vehicle_ids or float(vehicle.get("confidence", 0.0)) < 0.62:
                    continue
                if _aligned_with_lane_arrow(vehicle, arrow):
                    used_vehicle_ids.add(id(vehicle))
                    violations.append(
                        _make_violation(
                            "Wrong-side Driving",
                            min(vehicle.get("confidence", 0.7) * 0.82, 0.88),
                            vehicle.get("bbox_percent", {}),
                            vehicle.get("type", "vehicle"),
                            "Vehicle is aligned against visible lane-direction arrow evidence in the still image",
                            _vehicle_plate(vehicle, recognized_plates),
                            "detector+road-arrow",
                        )
                    )
                    break

    if ai_analysis:
        ai_min = float(RULE_CFG.get("ai_min_confidence", 0.75))
        require_anchor = bool(RULE_CFG.get("require_yolo_anchor_for_ai", True))
        anchor_iou = float(RULE_CFG.get("yolo_anchor_iou", 0.12))
        anchors = vehicles + persons
        for av in ai_analysis.get("violations", []):
            av_type = av.get("type")
            if av_type not in VALID_VIOLATION_TYPES:
                continue
            if "compliance" in av_type.lower() and "non-compliance" not in av_type.lower():
                continue
            confidence = float(av.get("confidence", 0.0))
            if confidence < ai_min:
                continue
            ai_bbox = av.get("bbox_percent") or {}
            if ai_bbox.get("w", 0) <= 0 or ai_bbox.get("h", 0) <= 0:
                continue
            anchor, score = _nearest_anchor(ai_bbox, anchors)
            if require_anchor and score < anchor_iou:
                continue
            final_bbox = anchor.get("bbox_percent", ai_bbox) if anchor else ai_bbox
            vehicle_type = av.get("vehicle_type") or (anchor.get("type") if anchor else "vehicle")
            violations.append(
                _make_violation(
                    av_type,
                    min(confidence * 0.93, 0.96),
                    final_bbox,
                    vehicle_type,
                    av.get("description", f"High-confidence vision review: {av_type}"),
                    av.get("license_plate"),
                    "vision-review",
                )
            )

    return _deduplicate(violations)

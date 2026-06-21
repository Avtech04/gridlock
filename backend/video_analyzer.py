"""
Video tracking and calibrated rule engine for traffic violations.

The image pipeline is intentionally detection-first. This module handles the
violations that need time: wrong-side driving, stop-line crossing, red-light
crossing, and parking dwell inside a restricted zone.
"""

from __future__ import annotations

import math
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import yaml

from detector import detect, get_detector_status
from ocr import recognize_plates


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.yaml"


def _load_config() -> dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _calibration() -> dict[str, Any]:
    return _load_config().get("calibration", {})


def _bbox_center(bbox: dict[str, float]) -> tuple[float, float]:
    return (
        float(bbox.get("x", 0)) + float(bbox.get("w", 0)) / 2,
        float(bbox.get("y", 0)) + float(bbox.get("h", 0)) / 2,
    )


def _bbox_bottom(bbox: dict[str, float]) -> float:
    return float(bbox.get("y", 0)) + float(bbox.get("h", 0))


def _bbox_top(bbox: dict[str, float]) -> float:
    return float(bbox.get("y", 0))


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


def _point_inside_roi(x: float, y: float, roi: dict[str, Any]) -> bool:
    rx = float(roi.get("x", 0))
    ry = float(roi.get("y", 0))
    rw = float(roi.get("w", 0))
    rh = float(roi.get("h", 0))
    return rx <= x <= rx + rw and ry <= y <= ry + rh


def _implicit_no_parking_zone(sign: dict[str, Any]) -> dict[str, float]:
    bbox = sign.get("bbox_percent", {})
    sx = float(bbox.get("x", 0)) + float(bbox.get("w", 0)) / 2
    sy = float(bbox.get("y", 0)) + float(bbox.get("h", 0))
    if sx < 35:
        return {"x": 0.0, "y": min(100.0, sy + 2.0), "w": 58.0, "h": max(1.0, 100.0 - sy - 2.0)}
    if sx > 65:
        return {"x": 42.0, "y": min(100.0, sy + 2.0), "w": 58.0, "h": max(1.0, 100.0 - sy - 2.0)}
    return {"x": 0.0, "y": min(100.0, sy + 2.0), "w": 100.0, "h": max(1.0, 100.0 - sy - 2.0)}


def _attach_plates_to_vehicles(detections: dict[str, Any], recognized_plates: list[dict[str, Any]]) -> None:
    for vehicle in detections.get("vehicles", []):
        vehicle_bbox = vehicle.get("bbox_percent", {})
        vx1, vy1, vx2, vy2 = _xyxy(vehicle_bbox)
        best = None
        for plate in recognized_plates:
            bbox = plate.get("bbox_percent", {})
            px, py = _bbox_center(bbox)
            if vx1 - 2 <= px <= vx2 + 2 and vy1 - 2 <= py <= vy2 + 2:
                if best is None or plate.get("confidence", 0) > best.get("confidence", 0):
                    best = plate
        if best:
            vehicle["license_plate"] = best.get("text")
            vehicle["plate_confidence"] = best.get("confidence")


def _clamp_bbox(bbox: dict[str, float]) -> dict[str, float]:
    x = max(0.0, min(100.0, float(bbox.get("x", 0))))
    y = max(0.0, min(100.0, float(bbox.get("y", 0))))
    w = max(0.1, min(100.0 - x, float(bbox.get("w", 1))))
    h = max(0.1, min(100.0 - y, float(bbox.get("h", 1))))
    return {"x": round(x, 3), "y": round(y, 3), "w": round(w, 3), "h": round(h, 3)}


def _traffic_light_state(detections: dict[str, Any], calibration: dict[str, Any]) -> str:
    signal_override = calibration.get("traffic_light_state_override", "auto")
    if signal_override in {"red", "yellow", "green", "not_visible"}:
        return signal_override
    colors = [tl.get("color", "unknown") for tl in detections.get("traffic_lights", [])]
    if "red" in colors:
        return "red"
    if "yellow" in colors:
        return "yellow"
    if "green" in colors:
        return "green"
    return "not_visible"


def _resize_frame(frame: Any, max_side: int = 960) -> Any:
    height, width = frame.shape[:2]
    largest = max(height, width)
    if largest <= max_side:
        return frame
    scale = max_side / largest
    return cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)


@dataclass
class Track:
    id: int
    vehicle_type: str
    observations: list[dict[str, Any]] = field(default_factory=list)
    missed: int = 0

    @property
    def last(self) -> dict[str, Any]:
        return self.observations[-1]

    def add(self, observation: dict[str, Any]) -> None:
        self.vehicle_type = observation.get("vehicle_type", self.vehicle_type)
        self.observations.append(observation)
        self.missed = 0

    def avg_confidence(self) -> float:
        if not self.observations:
            return 0.0
        return sum(float(obs.get("confidence", 0)) for obs in self.observations) / len(self.observations)

    def license_plate(self) -> str | None:
        candidates: dict[str, tuple[int, float]] = {}
        for obs in self.observations:
            plate = obs.get("license_plate")
            if not plate:
                continue
            count, confidence = candidates.get(plate, (0, 0.0))
            candidates[plate] = (count + 1, max(confidence, float(obs.get("plate_confidence", 0.0) or 0.0)))
        if not candidates:
            return None
        return sorted(candidates.items(), key=lambda item: (item[1][0], item[1][1]), reverse=True)[0][0]


def _observation(
    vehicle: dict[str, Any],
    sample_index: int,
    frame_index: int,
    timestamp_s: float,
    traffic_light: str,
    no_parking_signs: list[dict[str, Any]],
) -> dict[str, Any]:
    bbox = _clamp_bbox(vehicle.get("bbox_percent", {}))
    center_x, center_y = _bbox_center(bbox)
    sign_zone = None
    for sign in no_parking_signs:
        zone = _implicit_no_parking_zone(sign)
        if _point_inside_roi(center_x, center_y, zone):
            sign_zone = zone
            break
    return {
        "sample_index": sample_index,
        "frame_index": frame_index,
        "time_s": round(timestamp_s, 3),
        "bbox_percent": bbox,
        "center": [round(center_x, 3), round(center_y, 3)],
        "top_y": round(_bbox_top(bbox), 3),
        "bottom_y": round(_bbox_bottom(bbox), 3),
        "confidence": round(float(vehicle.get("confidence", 0.0)), 4),
        "vehicle_type": vehicle.get("type", "vehicle"),
        "license_plate": vehicle.get("license_plate"),
        "plate_confidence": vehicle.get("plate_confidence"),
        "traffic_light": traffic_light,
        "parking_zone_source": "visible no-parking sign" if sign_zone else None,
    }


def _match_tracks(
    tracks: list[Track],
    observations: list[dict[str, Any]],
    max_missed: int = 3,
    max_distance: float = 13.5,
) -> list[dict[str, Any]]:
    for track in tracks:
        track.missed += 1

    pairs: list[tuple[float, int, int]] = []
    for track_index, track in enumerate(tracks):
        if not track.observations or track.missed > max_missed + 1:
            continue
        tx, ty = track.last["center"]
        tbbox = track.last["bbox_percent"]
        for obs_index, obs in enumerate(observations):
            ox, oy = obs["center"]
            distance = math.hypot(tx - ox, ty - oy)
            overlap = _iou(tbbox, obs["bbox_percent"])
            type_penalty = 0.0 if track.vehicle_type == obs["vehicle_type"] else 2.0
            score = distance + (1.0 - overlap) * 4.0 + type_penalty
            if distance <= max_distance or overlap >= 0.18:
                pairs.append((score, track_index, obs_index))

    assigned_tracks: set[int] = set()
    assigned_obs: set[int] = set()
    assignments: list[dict[str, Any]] = []
    for _, track_index, obs_index in sorted(pairs, key=lambda item: item[0]):
        if track_index in assigned_tracks or obs_index in assigned_obs:
            continue
        track = tracks[track_index]
        obs = observations[obs_index]
        track.add(obs)
        assigned_tracks.add(track_index)
        assigned_obs.add(obs_index)
        assignments.append({"track_id": track.id, **obs})

    next_id = max((track.id for track in tracks), default=0) + 1
    for obs_index, obs in enumerate(observations):
        if obs_index in assigned_obs:
            continue
        track = Track(id=next_id, vehicle_type=obs["vehicle_type"])
        track.add(obs)
        tracks.append(track)
        assignments.append({"track_id": track.id, **obs})
        next_id += 1

    tracks[:] = [track for track in tracks if track.missed <= max_missed or len(track.observations) >= 2]
    return assignments


def _event(
    vtype: str,
    confidence: float,
    track: Track,
    observation: dict[str, Any],
    description: str,
    evidence: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "type": vtype,
        "confidence": round(max(0.0, min(float(confidence), 0.99)), 4),
        "bbox_percent": _clamp_bbox(observation.get("bbox_percent", {})),
        "vehicle_type": track.vehicle_type,
        "license_plate": observation.get("license_plate") or track.license_plate(),
        "description": description,
        "evidence": evidence,
        "track_id": track.id,
        "frame_index": observation.get("frame_index"),
        "sample_index": observation.get("sample_index"),
        "time_s": observation.get("time_s"),
    }
    if extra:
        payload.update(extra)
    return payload


def _line_crossing(track: Track, line_y: float, require_red: bool = False) -> dict[str, Any] | None:
    previous: dict[str, Any] | None = None
    for obs in track.observations:
        if previous is None:
            previous = obs
            continue
        prev_front = float(previous.get("top_y", previous.get("bottom_y", 0)))
        current_front = float(obs.get("top_y", obs.get("bottom_y", 0)))
        current_bottom = float(obs.get("bottom_y", 0))
        crossed_front = (prev_front > line_y >= current_front) or (prev_front < line_y <= current_front)
        overlaps_line = current_front <= line_y <= current_bottom
        crossed = crossed_front or overlaps_line
        red_ok = not require_red or obs.get("traffic_light") == "red" or previous.get("traffic_light") == "red"
        if crossed and red_ok:
            return obs
        previous = obs
    return None


def _movement_for_direction(track: Track, expected_direction: str) -> tuple[bool, float, str, str, float, float]:
    first = track.observations[0]
    last = track.observations[-1]
    dx = float(last["center"][0]) - float(first["center"][0])
    dy = float(last["center"][1]) - float(first["center"][1])
    if expected_direction == "right_to_left":
        return dx > 0, abs(dx), "left-to-right" if dx > 0 else "right-to-left", "right-to-left", dx, dy
    if expected_direction == "top_to_bottom":
        return dy < 0, abs(dy), "bottom-to-top" if dy < 0 else "top-to-bottom", "top-to-bottom", dx, dy
    if expected_direction == "bottom_to_top":
        return dy > 0, abs(dy), "top-to-bottom" if dy > 0 else "bottom-to-top", "bottom-to-top", dx, dy
    return dx < 0, abs(dx), "right-to-left" if dx < 0 else "left-to-right", "left-to-right", dx, dy


def _generate_events(tracks: list[Track], calibration: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    expected_direction = calibration.get("expected_lane_direction", "left_to_right")
    min_travel = float(calibration.get("wrong_side_min_travel_percent", 8.0))
    dwell_threshold = float(calibration.get("illegal_parking_dwell_seconds", 8.0))
    stationary_limit = float(calibration.get("parking_stationary_max_travel_percent", 4.0))

    for track in tracks:
        if len(track.observations) < 2:
            continue

        first = track.observations[0]
        last = track.observations[-1]
        duration = float(last.get("time_s", 0)) - float(first.get("time_s", 0))
        wrong_direction, travel, actual, expected, dx, dy = _movement_for_direction(track, expected_direction)
        avg_conf = track.avg_confidence()

        if calibration.get("wrong_side_enabled") and duration >= 1.0 and travel >= min_travel:
            if wrong_direction:
                confidence = min(0.94, 0.50 + avg_conf * 0.28 + min(travel / 60, 0.16))
                events.append(
                    _event(
                        "Wrong-side Driving",
                        confidence,
                        track,
                        last,
                        f"Track {track.id} moved {actual} by {travel:.1f}% while lane expects {expected}",
                        "tracker+calibration",
                        {
                            "duration_s": round(duration, 2),
                            "movement_dx_percent": round(dx, 2),
                            "movement_dy_percent": round(dy, 2),
                            "expected_direction": expected,
                            "actual_direction": actual,
                        },
                    )
                )

        if calibration.get("stop_line_rule_enabled"):
            line_y = float(calibration.get("stop_line_y_percent", 60))
            crossing = _line_crossing(track, line_y, require_red=False)
            if crossing:
                confidence = min(0.92, 0.48 + avg_conf * 0.38)
                events.append(
                    _event(
                        "Stop-line Violation",
                        confidence,
                        track,
                        crossing,
                        f"Track {track.id} crossed calibrated stop line at Y={line_y:.1f}%",
                        "tracker+calibration",
                        {"line_y_percent": round(line_y, 2)},
                    )
                )

        if calibration.get("red_light_rule_enabled"):
            line_y = float(calibration.get("red_light_vehicle_bottom_y_percent", 52))
            crossing = _line_crossing(track, line_y, require_red=True)
            if crossing:
                confidence = min(0.94, 0.52 + avg_conf * 0.38)
                events.append(
                    _event(
                        "Red-light Violation",
                        confidence,
                        track,
                        crossing,
                        f"Track {track.id} crossed red-light gate at Y={line_y:.1f}% while signal was red",
                        "tracker+signal+calibration",
                        {"line_y_percent": round(line_y, 2), "traffic_light": "red"},
                    )
                )

        if calibration.get("illegal_parking_rule_enabled"):
            for roi_index, roi in enumerate(calibration.get("illegal_parking_rois", []) or [], start=1):
                segment_start: dict[str, Any] | None = None
                segment_end: dict[str, Any] | None = None
                for obs in track.observations:
                    cx, cy = obs["center"]
                    if _point_inside_roi(cx, cy, roi):
                        if segment_start is None:
                            segment_start = obs
                        segment_end = obs
                    else:
                        if segment_start and segment_end:
                            dwell = float(segment_end["time_s"]) - float(segment_start["time_s"])
                            if dwell >= dwell_threshold:
                                break
                        segment_start = None
                        segment_end = None
                if not segment_start or not segment_end:
                    continue
                dwell = float(segment_end["time_s"]) - float(segment_start["time_s"])
                if dwell < dwell_threshold:
                    continue
                confidence = min(0.93, 0.48 + avg_conf * 0.32 + min(dwell / max(dwell_threshold, 1), 1.5) * 0.08)
                events.append(
                    _event(
                        "Illegal Parking",
                        confidence,
                        track,
                        segment_end,
                        f"Track {track.id} stayed inside no-parking zone {roi_index} for {dwell:.1f}s",
                        "tracker+dwell+calibration",
                        {
                            "duration_s": round(dwell, 2),
                            "dwell_threshold_s": round(dwell_threshold, 2),
                            "roi_index": roi_index,
                        },
                    )
                )
                break

        if calibration.get("illegal_parking_rule_enabled") or any(obs.get("parking_zone_source") for obs in track.observations):
            sign_observations = [obs for obs in track.observations if obs.get("parking_zone_source")]
            if sign_observations:
                start = sign_observations[0]
                end = sign_observations[-1]
                dwell = float(end["time_s"]) - float(start["time_s"])
                sx, sy = start["center"]
                ex, ey = end["center"]
                movement = math.hypot(float(ex) - float(sx), float(ey) - float(sy))
                if dwell >= dwell_threshold and movement <= stationary_limit:
                    confidence = min(0.92, 0.50 + avg_conf * 0.30 + min(dwell / max(dwell_threshold, 1), 1.5) * 0.08)
                    events.append(
                        _event(
                            "Illegal Parking",
                            confidence,
                            track,
                            end,
                            f"Track {track.id} remained near a visible no-parking sign for {dwell:.1f}s",
                            "tracker+dwell+no-parking-sign",
                            {
                                "duration_s": round(dwell, 2),
                                "movement_percent": round(movement, 2),
                                "dwell_threshold_s": round(dwell_threshold, 2),
                            },
                        )
                    )

    unique: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any]] = set()
    for event in sorted(events, key=lambda item: item.get("confidence", 0), reverse=True):
        key = (event.get("track_id"), event.get("type"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(event)
    return sorted(unique, key=lambda item: (item.get("time_s") or 0, item.get("track_id") or 0, item.get("type") or ""))


def _draw_label(image: Any, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.46
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    y = max(th + 8, y)
    cv2.rectangle(image, (x, y - th - 7), (x + tw + 6, y + 2), color, -1)
    cv2.putText(image, text, (x + 3, y - 4), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _draw_percent_box(image: Any, bbox: dict[str, float], color: tuple[int, int, int], thickness: int, label: str) -> None:
    height, width = image.shape[:2]
    x = int(float(bbox.get("x", 0)) * width / 100)
    y = int(float(bbox.get("y", 0)) * height / 100)
    bw = int(float(bbox.get("w", 0)) * width / 100)
    bh = int(float(bbox.get("h", 0)) * height / 100)
    if bw <= 1 or bh <= 1:
        return
    cv2.rectangle(image, (x, y), (x + bw, y + bh), color, thickness)
    _draw_label(image, label, x, y, color)


def _draw_calibration(image: Any, calibration: dict[str, Any]) -> None:
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
            _draw_label(image, f"No parking {index}", x1, max(y1 - 8, 18), (255, 80, 0))


def _annotate_frame(
    frame: Any,
    calibration: dict[str, Any],
    assignments: list[dict[str, Any]],
    no_parking_signs: list[dict[str, Any]],
    events: list[dict[str, Any]],
    timestamp_s: float,
) -> Any:
    output = frame.copy()
    _draw_calibration(output, calibration)
    for sign in no_parking_signs:
        _draw_percent_box(output, sign.get("bbox_percent", {}), (255, 80, 0), 2, "No parking sign")
    event_keys = {(event.get("track_id"), event.get("sample_index")) for event in events}
    for assignment in assignments:
        track_id = assignment["track_id"]
        is_event = (track_id, assignment.get("sample_index")) in event_keys
        color = (0, 0, 255) if is_event else (36, 160, 90)
        label = f"ID {track_id} {assignment.get('vehicle_type', 'vehicle')} {assignment.get('confidence', 0) * 100:.0f}%"
        if assignment.get("license_plate"):
            label = f"{label} {assignment['license_plate']}"
        _draw_percent_box(output, assignment["bbox_percent"], color, 3 if is_event else 1, label)

    stamp = f"{timestamp_s:.1f}s"
    cv2.rectangle(output, (8, output.shape[0] - 30), (210, output.shape[0] - 6), (0, 0, 0), -1)
    cv2.putText(output, f"Gridlock video {stamp}", (14, output.shape[0] - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return output


def _write_video_artifacts(
    frame_records: list[dict[str, Any]],
    events: list[dict[str, Any]],
    calibration: dict[str, Any],
    annotated_dir: Path,
    session_id: str,
    output_fps: float,
) -> dict[str, str]:
    if not frame_records:
        return {"annotated_video": "", "summary_image": ""}

    annotated_dir.mkdir(exist_ok=True)
    first = frame_records[0]["image"]
    height, width = first.shape[:2]
    video_path = annotated_dir / f"{session_id}_tracked.webm"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"VP80"), max(output_fps, 1.0), (width, height))

    event_sample = events[0].get("sample_index") if events else frame_records[min(len(frame_records) - 1, len(frame_records) // 2)]["sample_index"]
    summary_frame = None
    for record in frame_records:
        annotated = _annotate_frame(
            record["image"],
            calibration,
            record["assignments"],
            record.get("no_parking_signs", []),
            events,
            record["time_s"],
        )
        if writer.isOpened():
            writer.write(annotated)
        if record["sample_index"] == event_sample and summary_frame is None:
            summary_frame = annotated
    writer.release()

    summary_path = annotated_dir / f"{session_id}_video_summary.jpg"
    cv2.imwrite(str(summary_path), summary_frame if summary_frame is not None else first, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return {
        "annotated_video": f"/annotated/{session_id}_tracked.webm" if video_path.exists() and video_path.stat().st_size > 0 else "",
        "summary_image": f"/annotated/{session_id}_video_summary.jpg",
    }


def analyze_video_file(video_path: str, session_id: str, annotated_dir: Path) -> dict[str, Any]:
    started = time.time()
    calibration = _calibration()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("Video could not be opened.")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration_s = total_frames / source_fps if total_frames > 0 and source_fps > 0 else 0.0

    target_sample_fps = max(0.25, min(float(calibration.get("tracking_sample_fps", 1.5)), 5.0))
    max_samples = max(8, min(int(calibration.get("tracking_max_frames", 90)), 240))
    stride = max(1, int(round(source_fps / target_sample_fps)))

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"gridlock_video_{session_id}_"))
    tracks: list[Track] = []
    frame_records: list[dict[str, Any]] = []
    traffic_counts = {"red": 0, "yellow": 0, "green": 0, "not_visible": 0}
    sampled = 0
    frame_index = 0

    try:
        while sampled < max_samples:
            if total_frames > 0 and frame_index >= total_frames:
                break
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                break

            frame = _resize_frame(frame)
            timestamp_s = frame_index / source_fps if source_fps > 0 else float(sampled)
            frame_path = tmp_dir / f"frame_{sampled:04d}.jpg"
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])

            detections = detect(str(frame_path))
            recognized_plates = recognize_plates(str(frame_path), detections.get("license_plates", []), detections.get("vehicles", []))
            _attach_plates_to_vehicles(detections, recognized_plates)
            traffic_light = _traffic_light_state(detections, calibration)
            traffic_counts[traffic_light] = traffic_counts.get(traffic_light, 0) + 1

            observations = [
                _observation(vehicle, sampled, frame_index, timestamp_s, traffic_light, detections.get("no_parking_signs", []))
                for vehicle in detections.get("vehicles", [])
            ]
            assignments = _match_tracks(tracks, observations)
            frame_records.append(
                {
                    "sample_index": sampled,
                    "frame_index": frame_index,
                    "time_s": round(timestamp_s, 3),
                    "image": frame,
                    "assignments": assignments,
                    "recognized_plates": recognized_plates,
                    "no_parking_signs": detections.get("no_parking_signs", []),
                    "scene_geometry": detections.get("scene_geometry", {}),
                    "traffic_light": traffic_light,
                    "vehicle_count": len(observations),
                }
            )

            sampled += 1
            frame_index += stride
    finally:
        capture.release()
        shutil.rmtree(tmp_dir, ignore_errors=True)

    stable_tracks = [track for track in tracks if len(track.observations) >= 2]
    events = _generate_events(stable_tracks, calibration)
    artifacts = _write_video_artifacts(frame_records, events, calibration, annotated_dir, session_id, target_sample_fps)

    dominant_light = max(traffic_counts, key=traffic_counts.get) if any(traffic_counts.values()) else "not_visible"
    track_summaries = [
        {
            "track_id": track.id,
            "vehicle_type": track.vehicle_type,
            "frames_seen": len(track.observations),
            "start_time_s": track.observations[0]["time_s"],
            "end_time_s": track.observations[-1]["time_s"],
            "duration_s": round(float(track.observations[-1]["time_s"]) - float(track.observations[0]["time_s"]), 2),
            "movement_dx_percent": round(float(track.observations[-1]["center"][0]) - float(track.observations[0]["center"][0]), 2),
            "movement_dy_percent": round(float(track.observations[-1]["center"][1]) - float(track.observations[0]["center"][1]), 2),
            "avg_confidence": round(track.avg_confidence(), 4),
            "license_plate": track.license_plate(),
        }
        for track in stable_tracks
    ]

    processing_ms = (time.time() - started) * 1000
    return {
        "status": "success",
        "media_type": "video",
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "processing_time_ms": round(processing_ms, 2),
        "video": {
            "duration_s": round(duration_s, 2),
            "source_fps": round(source_fps, 3),
            "source_frames": total_frames,
            "source_width": source_width,
            "source_height": source_height,
            "sampled_frames": len(frame_records),
            "sample_fps": target_sample_fps,
            "frame_stride": stride,
        },
        "pipeline": {
            "detector": get_detector_status(),
            "tracking": {
                "method": "centroid+iou",
                "stable_tracks": len(stable_tracks),
                "parking_dwell_seconds": float(calibration.get("illegal_parking_dwell_seconds", 8.0)),
            },
        },
        "scene": {
            "traffic_light": dominant_light,
            "traffic_light_frame_counts": traffic_counts,
        },
        "detection": {
            "total_vehicles": len(stable_tracks),
            "total_pedestrians": 0,
            "tracks": track_summaries,
        },
        "violations": events,
        "evidence": {
            "original_video": str(video_path),
            **artifacts,
        },
        "calibration": calibration,
    }

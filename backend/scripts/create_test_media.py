from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = ROOT / "test_media" / "images"
VIDEO_DIR = ROOT / "test_media" / "videos"


def copy_file(src: str, dst_name: str) -> Path:
    dst = IMAGE_DIR / dst_name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / src, dst)
    return dst


def video_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    if path.suffix.lower() == ".webm":
        return cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"VP80"), fps, size)
    return cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)


def resize_max(frame: np.ndarray, max_width: int = 960) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / width
    return cv2.resize(frame, (max_width, int(height * scale)), interpolation=cv2.INTER_AREA)


def write_shift_video(src: Path, dst: Path, shifts: list[tuple[int, int]], fps: float = 2.0) -> None:
    image = cv2.imread(str(src))
    if image is None:
        raise RuntimeError(f"Could not read {src}")
    image = resize_max(image)
    height, width = image.shape[:2]
    writer = video_writer(dst, fps, (width, height))
    for shift_x, shift_y in shifts:
        matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        frame = cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REFLECT)
        writer.write(frame)
    writer.release()


def write_static_video(src: Path, dst: Path, frames: int = 18, fps: float = 2.0) -> None:
    image = cv2.imread(str(src))
    if image is None:
        raise RuntimeError(f"Could not read {src}")
    image = resize_max(image)
    height, width = image.shape[:2]
    writer = video_writer(dst, fps, (width, height))
    for _ in range(frames):
        writer.write(image)
    writer.release()


def write_real_video_clip(src: Path, dst: Path, seconds: float = 8.0, output_fps: float = 12.0) -> None:
    capture = cv2.VideoCapture(str(src))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {src}")
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, round(source_fps / output_fps))
    max_source_frames = int(source_fps * seconds)
    writer: cv2.VideoWriter | None = None
    frame_index = 0
    written = 0
    while frame_index < max_source_frames:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % stride == 0:
            frame = resize_max(frame)
            if writer is None:
                height, width = frame.shape[:2]
                writer = video_writer(dst, output_fps, (width, height))
            writer.write(frame)
            written += 1
        frame_index += 1
    capture.release()
    if writer is not None:
        writer.release()
    if written == 0:
        raise RuntimeError(f"No frames written for {src}")


def main() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    for old_name in [
        "01_illegal_parking_static_9s.mp4",
        "02_wrong_side_right_to_left_6s.mp4",
        "03_stopline_redlight_crossing_down_6s.mp4",
        "04_real_red_light_dataset_video.mp4",
    ]:
        (VIDEO_DIR / old_name).unlink(missing_ok=True)

    base = copy_file(
        "data/processed/specialist_detector/images/val/helmet_BikesHelmets217.png",
        "01_helmet_no_compliance_real.png",
    )
    copy_file(
        "data/raw_extra/traffic_extra_datasets/traffic_two_wheeler/master_traffic_violation_dataset/valid/images/"
        "dst_8ed867_video_20240825_112943_mp4-0301_jpg.rf.61760ab4ac61fed4d0a4b870f645cdce.jpg",
        "02_triple_riding_real_future_colab.jpg",
    )
    copy_file("test_images/generated/03_triple_riding.png", "02_triple_riding_demo.png")
    copy_file("test_images/generated/09_license_plate_mh01ab1234.png", "03_license_plate_demo.png")
    copy_file("test_images/generated/07_seatbelt_non_compliance.png", "04_seatbelt_camera_angle_demo.png")
    copy_file("test_images/generated/10_negative_compliant_green_light.png", "05_negative_compliant_demo.png")
    parking = copy_file("test_images/generated/04_illegal_parking_no_parking_sign.png", "06_parking_roi_preview_frame.png")

    write_static_video(parking, VIDEO_DIR / "01_illegal_parking_stationary_scene_9s.webm", frames=18, fps=2.0)
    write_shift_video(base, VIDEO_DIR / "02_wrong_side_synthetic_tracker_demo.webm", [(0 - i * 9, 0) for i in range(14)], fps=2.0)
    write_shift_video(base, VIDEO_DIR / "03_stopline_redlight_synthetic_tracker_demo.webm", [(0, -95 + i * 10) for i in range(14)], fps=2.0)

    source_original = ROOT / "data/raw_extra/traffic_extra_datasets/red_light_plate_video/traffic_video_original.mp4"
    if source_original.exists():
        write_real_video_clip(source_original, VIDEO_DIR / "04_real_traffic_video_original_8s.webm")
    source_red = ROOT / "data/raw_extra/traffic_extra_datasets/red_light_plate_video/traffic_video_modified.mp4"
    if source_red.exists():
        write_real_video_clip(source_red, VIDEO_DIR / "05_real_traffic_signal_overlay_8s.webm")

    print(f"Created test media in {ROOT / 'test_media'}")


if __name__ == "__main__":
    main()

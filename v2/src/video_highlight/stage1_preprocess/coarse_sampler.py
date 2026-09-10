"""顺序解码视频并按真实解码时间进行 2 FPS 粗采样。"""

from __future__ import annotations

from bisect import bisect_right
from pathlib import Path
from typing import Any

import cv2

from video_highlight.common.exceptions import ArtifactValidationError
from video_highlight.contracts.schema_versions import STAGE1_SCHEMA_VERSION

from .frame_writer import write_jpeg
from .timestamp_map import FrameClock, SamplingSchedule


def _scene_id(timestamp_sec: float, scene_end_times: list[float]) -> int:
    return min(bisect_right(scene_end_times, timestamp_sec), max(0, len(scene_end_times) - 1))


def sample_video(
    video_id: str,
    video_path: str | Path,
    output_dir: str | Path,
    scenes: list[dict[str, Any]],
    metadata: dict[str, Any],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """返回抽样帧映射和本次完整解码统计。"""
    capture = cv2.VideoCapture(str(Path(video_path).resolve()))
    if not capture.isOpened():
        raise ArtifactValidationError(f"OpenCV 无法打开视频: {video_path}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sample_fps = float(config.get("fps", 2.0))
    schedule = SamplingSchedule(sample_fps)
    clock = FrameClock(float(metadata["fps"]))
    scene_ends = [float(scene["end_sec"]) for scene in scenes]
    samples: list[dict[str, Any]] = []
    decoded_frames = 0
    last_timestamp = 0.0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index = decoded_frames
            timestamp = clock.resolve(frame_index, float(capture.get(cv2.CAP_PROP_POS_MSEC)))
            last_timestamp = timestamp
            decoded_frames += 1 # TODO 此处的decoded_frames与frame_index有脱裤子放屁嫌疑
            if not schedule.due(timestamp):
                continue
            scheduled = schedule.consume(timestamp)
            sample_id = len(samples)
            file_name = f"{sample_id:06d}.jpg"
            image_path = output / file_name
            width, height = write_jpeg(
                frame,
                image_path,
                max_side=int(config.get("max_side", 960)),
                quality=int(config.get("jpeg_quality", 90)),
            )
            samples.append(
                {
                    "schema_version": STAGE1_SCHEMA_VERSION,
                    "video_id": video_id,
                    "sample_id": sample_id,
                    "scheduled_sec": scheduled,
                    "timestamp_sec": timestamp,
                    "original_frame": frame_index,
                    "scene_id": _scene_id(timestamp, scene_ends),
                    "image_path": f"coarse_frames/{file_name}",
                    "width": width,
                    "height": height,
                }
            )
    finally:
        capture.release()
    if decoded_frames == 0 or not samples:
        raise ArtifactValidationError(f"视频未解码出有效帧: {video_path}")
    return samples, {
        "decoded_frame_count": decoded_frames,
        "last_decoded_timestamp_sec": last_timestamp,
        "sample_count": len(samples),
        "sample_fps": sample_fps,
    }

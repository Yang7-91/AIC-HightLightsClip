"""仅对 Stage 2 候选区间进行高帧率精解码。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from video_highlight.common.exceptions import ArtifactValidationError


@dataclass(frozen=True, slots=True)
class DecodedCandidate:
    """候选窗口的等时间间隔灰度帧及其原视频坐标。"""

    candidate_id: str
    timestamps_sec: np.ndarray
    frame_indices: np.ndarray
    gray_frames: np.ndarray
    sample_fps: float


def _resize_gray(frame: np.ndarray, max_side: int) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale < 1.0:
        frame = cv2.resize(
            frame,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def decode_candidate(
    candidate: dict[str, Any],
    metadata: dict[str, Any],
    config: dict[str, Any],
) -> DecodedCandidate:
    """按目标采样率解码候选；保留真实解码帧号用于最终帧映射。"""

    source_path = Path(str(metadata.get("source_path", "")))
    if not source_path.is_file():
        raise ArtifactValidationError(f"Stage 1 metadata.source_path 不存在: {source_path}")
    fps = float(metadata["fps"])
    frame_count = int(metadata["frame_count"])
    duration = float(metadata["duration_sec"])
    sample_fps = float(config.get("fps", 10.0))
    max_side = int(config.get("max_side", 320))
    context = max(0.0, float(config.get("context_sec", 0.0)))
    if not math.isfinite(sample_fps) or sample_fps <= 0:
        raise ValueError("decode.fps 必须为正数")
    if max_side <= 0:
        raise ValueError("decode.max_side 必须为正整数")

    start_sec = max(0.0, float(candidate["start_sec"]) - context)
    end_sec = min(duration, float(candidate["end_sec"]) + context)
    if end_sec <= start_sec:
        raise ArtifactValidationError(f"候选区间非法: {candidate.get('candidate_id')}")

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise ArtifactValidationError(f"OpenCV 无法打开视频: {source_path}")
    # 从稍早一帧开始，避免关键帧 seek 落到候选起点之后。
    seek_frame = max(0, min(frame_count - 1, int(math.floor(start_sec * fps)) - 1))
    capture.set(cv2.CAP_PROP_POS_FRAMES, seek_frame)
    step_sec = 1.0 / sample_fps
    next_sample_sec = start_sec
    timestamps: list[float] = []
    frame_indices: list[int] = []
    gray_frames: list[np.ndarray] = []
    previous_timestamp = -1.0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index = max(0, int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1)
            position_msec = float(capture.get(cv2.CAP_PROP_POS_MSEC))
            fallback = frame_index / fps
            timestamp = position_msec / 1000.0 if math.isfinite(position_msec) and position_msec > 0 else fallback
            if timestamp <= previous_timestamp:
                timestamp = fallback
            previous_timestamp = timestamp
            if timestamp + 1e-6 < start_sec:
                continue
            if timestamp >= end_sec - 1e-9:
                break
            if timestamp + (0.5 / fps) < next_sample_sec:
                continue
            timestamps.append(timestamp)
            frame_indices.append(frame_index)
            gray_frames.append(_resize_gray(frame, max_side))
            while next_sample_sec <= timestamp + 1e-6:
                next_sample_sec += step_sec
    finally:
        capture.release()

    if not gray_frames:
        raise ArtifactValidationError(f"候选未解码出采样帧: {candidate.get('candidate_id')}")
    return DecodedCandidate(
        candidate_id=str(candidate["candidate_id"]),
        timestamps_sec=np.asarray(timestamps, dtype=np.float64),
        frame_indices=np.asarray(frame_indices, dtype=np.int64),
        gray_frames=np.stack(gray_frames),
        sample_fps=sample_fps,
    )

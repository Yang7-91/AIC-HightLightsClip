"""纯 OpenCV 主体初始化、光流传播和中心降级跟踪后端。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from video_highlight.common.exceptions import ArtifactValidationError, ConfigurationError

from .keyframe_selector import reinitialization_frames
from .prompt_generator import subject_point
from .subject_selector import select_subject_box
from .track_monitor import track_is_valid


@dataclass(frozen=True, slots=True)
class TrackPoint:
    frame: int
    subject_box: list[float]
    confidence: float
    source: str


class SubjectTracker(Protocol):
    def track(
        self,
        video_path: Path,
        interval: dict[str, Any],
        frame_size: tuple[int, int],
        scenes: list[dict[str, Any]],
    ) -> list[TrackPoint]: ...


class CenterSubjectTracker:
    """不打开视频的确定性中心主体后端，主要用于降级与链路测试。"""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def track(self, video_path: Path, interval: dict[str, Any], frame_size: tuple[int, int], scenes: list[dict[str, Any]]) -> list[TrackPoint]:
        del video_path, scenes
        width, height = frame_size
        box_w = width * float(self.config.get("initial_width_ratio", 0.28))
        box_h = height * float(self.config.get("initial_height_ratio", 0.38))
        box = [(width - box_w) * 0.5, (height - box_h) * 0.5, (width + box_w) * 0.5, (height + box_h) * 0.5]
        return [TrackPoint(frame, list(box), 0.0, "center") for frame in range(int(interval["start_frame"]), int(interval["end_frame"]))]


class OpenCVSubjectTracker:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @staticmethod
    def _features(gray: np.ndarray, box: list[float], maximum: int) -> np.ndarray | None:
        mask = np.zeros(gray.shape, dtype=np.uint8)
        x1, y1, x2, y2 = (int(round(value)) for value in box)
        mask[max(0, y1):min(gray.shape[0], y2), max(0, x1):min(gray.shape[1], x2)] = 255
        return cv2.goodFeaturesToTrack(gray, mask=mask, maxCorners=maximum, qualityLevel=0.01, minDistance=5, blockSize=7)

    def track(self, video_path: Path, interval: dict[str, Any], frame_size: tuple[int, int], scenes: list[dict[str, Any]]) -> list[TrackPoint]:
        if not video_path.is_file():
            raise ArtifactValidationError(f"源视频不存在: {video_path}")
        start, end = int(interval["start_frame"]), int(interval["end_frame"])
        resets = reinitialization_frames(interval, scenes, int(self.config.get("reinitialize_every_frames", 0)))
        point = subject_point(interval)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ArtifactValidationError(f"OpenCV 无法打开视频: {video_path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        previous_gray: np.ndarray | None = None
        previous_box: list[float] | None = None
        previous_points: np.ndarray | None = None
        output: list[TrackPoint] = []
        maximum_points = int(self.config.get("max_corners", 120))
        try:
            for frame_index in range(start, end):
                ok, frame = capture.read()
                if not ok:
                    raise ArtifactValidationError(f"视频在 frame={frame_index} 提前结束")
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if frame_index in resets or previous_gray is None or previous_box is None:
                    box, confidence, source = select_subject_box(frame, point, self.config)
                    points = self._features(gray, box, maximum_points)
                else:
                    if previous_points is None or len(previous_points) < 4:
                        previous_points = self._features(previous_gray, previous_box, maximum_points)
                    box, confidence, source = list(previous_box), 0.0, "optical_flow"
                    points = None
                    if previous_points is not None and len(previous_points):
                        tracked, status, _ = cv2.calcOpticalFlowPyrLK(previous_gray, gray, previous_points, None, winSize=(21, 21), maxLevel=3)
                        if tracked is not None and status is not None:
                            valid = status.reshape(-1).astype(bool)
                            old, new = previous_points.reshape(-1, 2)[valid], tracked.reshape(-1, 2)[valid]
                            if len(new) >= 4:
                                delta = np.median(new - old, axis=0)
                                box = [previous_box[0] + float(delta[0]), previous_box[1] + float(delta[1]), previous_box[2] + float(delta[0]), previous_box[3] + float(delta[1])]
                                width, height = frame_size
                                box_w, box_h = box[2] - box[0], box[3] - box[1]
                                cx = min(max((box[0] + box[2]) * 0.5, box_w * 0.5), width - box_w * 0.5)
                                cy = min(max((box[1] + box[3]) * 0.5, box_h * 0.5), height - box_h * 0.5)
                                box = [cx - box_w * 0.5, cy - box_h * 0.5, cx + box_w * 0.5, cy + box_h * 0.5]
                                confidence = float(len(new) / max(len(previous_points), 1))
                                points = new.reshape(-1, 1, 2).astype(np.float32)
                    if not track_is_valid(previous_box, box, frame_size, confidence, self.config):
                        box, confidence, source = select_subject_box(frame, point, self.config)
                        points = self._features(gray, box, maximum_points)
                        source = "reinitialized_" + source
                output.append(TrackPoint(frame_index, list(box), confidence, source))
                previous_gray, previous_box, previous_points = gray, list(box), points
        finally:
            capture.release()
        return output


def build_subject_tracker(config: dict[str, Any]) -> SubjectTracker:
    backend = str(config.get("backend", "opencv")).lower()
    if backend == "opencv":
        return OpenCVSubjectTracker(config)
    if backend == "center":
        return CenterSubjectTracker(config)
    if backend == "sam2":
        from .sam2_adapter import SAM2SubjectTracker
        return SAM2SubjectTracker(config)
    raise ConfigurationError(f"未知 Stage 4 tracking.backend: {backend}")

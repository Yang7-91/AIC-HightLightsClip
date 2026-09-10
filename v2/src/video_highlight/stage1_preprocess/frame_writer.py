"""粗采样图像缩放和 JPEG 持久化。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from video_highlight.common.exceptions import ArtifactValidationError


def resize_max_side(frame: np.ndarray, max_side: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if max_side <= 0 or max(height, width) <= max_side:
        return frame
    scale = max_side / max(height, width)
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(frame, target, interpolation=cv2.INTER_AREA)


def write_jpeg(frame: np.ndarray, path: str | Path, max_side: int, quality: int) -> tuple[int, int]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    resized = resize_max_side(frame, max_side)
    quality = min(100, max(1, int(quality)))
    encoded, buffer = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not encoded:
        raise ArtifactValidationError(f"无法编码采样帧: {target}")
    try:
        target.write_bytes(buffer.tobytes())
    except OSError as error:
        raise ArtifactValidationError(f"无法写入采样帧: {target}") from error
    height, width = resized.shape[:2]
    return width, height

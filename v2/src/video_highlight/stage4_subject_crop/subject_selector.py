"""根据 Stage 3 点提示或中心偏置视觉显著性初始化主体框。"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _clamp_box(box: tuple[float, float, float, float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = box
    x1, x2 = max(0.0, min(float(width - 1), x1)), max(1.0, min(float(width), x2))
    y1, y2 = max(0.0, min(float(height - 1), y1)), max(1.0, min(float(height), y2))
    if x2 <= x1:
        x2 = min(float(width), x1 + 1.0)
    if y2 <= y1:
        y2 = min(float(height), y1 + 1.0)
    return [x1, y1, x2, y2]


def select_subject_box(
    frame: np.ndarray,
    point: tuple[float, float] | None,
    config: dict[str, Any],
) -> tuple[list[float], float, str]:
    """返回 ``(xyxy 主体框, 置信度, 来源)``。"""

    height, width = frame.shape[:2]
    box_w = width * float(config.get("initial_width_ratio", 0.28))
    box_h = height * float(config.get("initial_height_ratio", 0.38))
    if point is not None:
        center_x, center_y = point[0] * width, point[1] * height
        return _clamp_box((center_x - box_w / 2, center_y - box_h / 2, center_x + box_w / 2, center_y + box_h / 2), width, height), 0.85, "stage3_point"

    max_side = int(config.get("saliency_max_side", 320))
    scale = min(1.0, max_side / max(height, width))
    small = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    saliency = cv2.GaussianBlur(cv2.magnitude(gradient_x, gradient_y), (0, 0), 5.0)
    small_h, small_w = gray.shape
    yy, xx = np.mgrid[0:small_h, 0:small_w]
    center_bias = np.exp(-(((xx - small_w / 2) / max(small_w * 0.55, 1.0)) ** 2 + ((yy - small_h / 2) / max(small_h * 0.55, 1.0)) ** 2))
    weighted = saliency * center_bias.astype(np.float32)
    total = float(weighted.sum())
    if total > 1e-6:
        center_x = float((weighted * xx).sum() / total) / scale
        center_y = float((weighted * yy).sum() / total) / scale
        confidence = min(0.65, total / max(1.0, small_w * small_h * 20.0))
        source = "visual_saliency"
    else:
        center_x, center_y, confidence, source = width * 0.5, height * 0.5, 0.0, "frame_center"
    return _clamp_box((center_x - box_w / 2, center_y - box_h / 2, center_x + box_w / 2, center_y + box_h / 2), width, height), confidence, source

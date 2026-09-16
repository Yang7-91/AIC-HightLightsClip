"""主体覆盖、中心性、留白和尺度的逐帧构图代价。"""

from __future__ import annotations

from typing import Any


def _intersection_area(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """
    计算a,b两个矩形的相交面积，未相交则为零
    """
    ax1, ay1, aw, ah = a
    bx1, by1, bx2, by2 = b
    return max(0.0, min(ax1 + aw, bx2) - max(ax1, bx1)) * max(0.0, min(ay1 + ah, by2) - max(ay1, by1))


def composition_cost(
    crop: tuple[float, float, float, float],
    subject_box: list[float] | tuple[float, float, float, float],
    frame_size: tuple[int, int],
    config: dict[str, Any],
) -> float:
    """按权重计算当下新裁剪框的三种cost的和：与原裁剪框的相交程度、居中程度、缩放倍数"""
    x, y, width, height = crop
    sx1, sy1, sx2, sy2 = map(float, subject_box)
    subject_area = max(1.0, (sx2 - sx1) * (sy2 - sy1))
    uncovered = 1.0 - min(1.0, _intersection_area(crop, (sx1, sy1, sx2, sy2)) / subject_area)
    subject_cx, subject_cy = (sx1 + sx2) * 0.5, (sy1 + sy2) * 0.5
    normalized_dx = abs(subject_cx - (x + width * 0.5)) / max(width, 1.0)
    normalized_dy = abs(subject_cy - (y + height * 0.5)) / max(height, 1.0)
    frame_width = max(1.0, float(frame_size[0]))
    zoom_cost = 1.0 - min(1.0, width / frame_width)
    weights = config.get("weights", {})
    return (
        float(weights.get("uncovered", 0.55)) * uncovered
        + float(weights.get("centering", 0.25)) * (normalized_dx + normalized_dy)
        + float(weights.get("zoom", 0.20)) * zoom_cost
    )

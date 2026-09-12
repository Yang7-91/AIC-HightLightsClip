"""目标比例构图框的尺寸、边界和最终整数约束。"""

from __future__ import annotations

import math
from typing import Sequence


def maximum_crop_width(frame_size: tuple[int, int], target_ratio: tuple[float, float]) -> int:
    width, height = frame_size
    target_w, target_h = target_ratio
    return max(1, min(width, int(math.floor(height * target_w / target_h + 1e-9))))


def legal_crop_from_state(
    center_x: float,
    center_y: float,
    crop_width: float,
    frame_size: tuple[int, int],
    target_ratio: tuple[float, float],
) -> tuple[float, float, float, float]:
    """将 ``(中心 x, 中心 y, 宽度)`` 裁剪为画面内合法的 ``xywh``。"""

    frame_w, frame_h = frame_size
    target_w, target_h = target_ratio
    max_width = maximum_crop_width(frame_size, target_ratio)
    width = max(1.0, min(float(max_width), float(crop_width)))
    height = width * target_h / target_w
    x = max(0.0, min(float(frame_w) - width, float(center_x) - width * 0.5))
    y = max(0.0, min(float(frame_h) - height, float(center_y) - height * 0.5))
    return x, y, width, height


def finalize_bbox(
    crop: Sequence[float],
    frame_size: tuple[int, int],
    target_ratio: tuple[float, float],
) -> list[int]:
    """统一取整后再次限界，返回比赛要求的整数 ``[x, y, w]``。"""

    if len(crop) < 3 or not all(math.isfinite(float(value)) for value in crop[:3]):
        raise ValueError("构图状态必须至少包含三个有限数值")
    x, y, width = float(crop[0]), float(crop[1]), float(crop[2])
    legal = legal_crop_from_state(x + width * 0.5, y + (width * target_ratio[1] / target_ratio[0]) * 0.5, width, frame_size, target_ratio)
    frame_w, frame_h = frame_size
    target_w, target_h = target_ratio
    integer_width = max(1, min(maximum_crop_width(frame_size, target_ratio), int(math.floor(legal[2] + 1e-9))))
    inferred_height = integer_width * target_h / target_w
    integer_x = max(0, min(frame_w - integer_width, int(round(legal[0]))))
    integer_y = max(0, min(int(math.floor(frame_h - inferred_height + 1e-9)), int(round(legal[1]))))
    return [integer_x, integer_y, integer_width]

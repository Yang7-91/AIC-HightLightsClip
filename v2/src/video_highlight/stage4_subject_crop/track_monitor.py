"""检测主体框越界、面积异常和中心突跳。"""

from __future__ import annotations

import math
from typing import Any


def track_is_valid(
    previous: list[float] | None,
    current: list[float] | None,
    frame_size: tuple[int, int],
    confidence: float,
    config: dict[str, Any],
) -> bool:
    if current is None or len(current) != 4 or confidence < float(config.get("min_confidence", 0.15)):
        return False
    width, height = frame_size
    x1, y1, x2, y2 = map(float, current)
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ratio = area / max(1.0, width * height)
    if not (0.0 <= x1 < x2 <= width and 0.0 <= y1 < y2 <= height):
        return False
    if not (float(config.get("min_area_ratio", 0.001)) <= ratio <= float(config.get("max_area_ratio", 0.75))):
        return False
    if previous is None:
        return True
    px1, py1, px2, py2 = map(float, previous)
    previous_area = max(1.0, (px2 - px1) * (py2 - py1))
    area_factor = max(area / previous_area, previous_area / max(area, 1.0))
    center_jump = math.hypot((x1 + x2 - px1 - px2) * 0.5, (y1 + y2 - py1 - py2) * 0.5) / max(1.0, math.hypot(width, height))
    return area_factor <= float(config.get("max_area_change", 3.0)) and center_jump <= float(config.get("max_center_jump", 0.20))

"""从 Stage 3 主体提示生成跟踪后端可用的归一化点。"""

from __future__ import annotations

from typing import Any


def subject_point(interval: dict[str, Any]) -> tuple[float, float] | None:
    value = interval.get("subject_point")
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        return None
    return x, y

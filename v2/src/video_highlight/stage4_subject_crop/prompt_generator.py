"""从 Stage 3.5 稀疏逐帧预测中选择跟踪后端可用的归一化点。"""

from __future__ import annotations

from typing import Any


def subject_point(interval: dict[str, Any], frame: int | None = None) -> tuple[float, float] | None:
    rows = [row for row in interval.get("subject_points", []) if row.get("subject_point") is not None]
    if not rows:
        return None
    target = int(interval["start_frame"]) if frame is None else int(frame)
    # 精确采样帧优先；镜头重置或跟踪失败发生在非采样帧时，使用时间上最近的可靠点。
    chosen = min(rows, key=lambda row: abs(int(row["frame"]) - target))
    value = chosen.get("subject_point")
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        return None
    return x, y


def subject_point_frames(interval: dict[str, Any]) -> set[int]:
    """返回具有非空模型点的原始帧号，用作 OpenCV 跟踪强制重初始化位置。"""

    return {int(row["frame"]) for row in interval.get("subject_points", []) if row.get("subject_point") is not None}

"""最终 ``[x,y,w]`` 的确定性取整及取整后边界修正。"""

from __future__ import annotations

import math
from collections.abc import Sequence

from video_highlight.common.exceptions import ArtifactValidationError


def _integer(value: float, mode: str) -> int:
    if mode == "round":
        return int(round(value))
    if mode == "floor":
        return int(math.floor(value))
    raise ArtifactValidationError(f"未知坐标取整模式: {mode}")


def quantize_bbox(
    bbox: Sequence[object],
    frame_size: tuple[int, int],
    target_ratio: tuple[int, int],
    mode: str = "round",
) -> tuple[list[int], bool]:
    """把数值框转成画面内合法的整数 ``[x,y,w]``。

    Stage 4 正常情况下已经输出整数；这里仍执行最后一道确定性防线，以兼容历史
    Stage 4 产物中的浮点数，并处理取整后恰好越过右/下边界的情况。

    Returns:
        ``(合法整数框, 是否发生数值变化或边界修正)``。
    """

    if not isinstance(bbox, (list, tuple)) or len(bbox) != 3:
        raise ArtifactValidationError("Stage 4 bboxes 必须是 [x,y,w]")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in bbox):
        raise ArtifactValidationError("Stage 4 bboxes 只能包含数值")
    raw = tuple(float(value) for value in bbox)
    if not all(math.isfinite(value) for value in raw):
        raise ArtifactValidationError("Stage 4 bboxes 不能包含 NaN 或 Infinity")
    if raw[2] <= 0:
        raise ArtifactValidationError("Stage 4 构图框宽度必须大于 0")

    frame_w, frame_h = frame_size
    ratio_w, ratio_h = target_ratio
    if frame_w <= 0 or frame_h <= 0 or ratio_w <= 0 or ratio_h <= 0:
        raise ArtifactValidationError("视频尺寸和目标比例必须为正数")

    # 最大合法宽度同时受原图宽度以及“按目标比例换算后的高度”限制。
    maximum_width = max(1, min(frame_w, int(math.floor(frame_h * ratio_w / ratio_h + 1e-9))))
    width = max(1, min(maximum_width, _integer(raw[2], mode)))
    inferred_height = width * ratio_h / ratio_w
    maximum_y = max(0, int(math.floor(frame_h - inferred_height + 1e-9)))
    x = max(0, min(frame_w - width, _integer(raw[0], mode)))
    y = max(0, min(maximum_y, _integer(raw[1], mode)))
    result = [x, y, width]
    changed = any(float(result[index]) != raw[index] for index in range(3))
    return result, changed

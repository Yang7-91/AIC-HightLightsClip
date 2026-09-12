"""解析模型逐帧主体点响应并补齐遗漏帧。"""

from __future__ import annotations

import json
import math
from typing import Any

from video_highlight.common.exceptions import ArtifactValidationError


def parse_predictions(text: str, sample_count: int) -> tuple[list[dict[str, Any]], list[int]]:
    try:
        root = json.loads(text)
    except json.JSONDecodeError as error:
        raise ArtifactValidationError(f"Stage 3.5 响应不是合法 JSON: {error}") from error
    predictions = root.get("predictions") if isinstance(root, dict) else None
    if not isinstance(predictions, list):
        raise ArtifactValidationError("Stage 3.5 响应缺少 predictions 数组")
    by_index: dict[int, dict[str, Any]] = {}
    for item in predictions:
        if not isinstance(item, dict):
            raise ArtifactValidationError("prediction 必须是对象")
        index = int(item.get("sample_index", -1))
        if not 0 <= index < sample_count:
            raise ArtifactValidationError(f"未知 sample_index: {index}")
        if index in by_index:
            raise ArtifactValidationError(f"重复 sample_index: {index}")
        point = item.get("subject_point")
        if point is not None:
            if not isinstance(point, list) or len(point) != 2:
                raise ArtifactValidationError(f"sample_index={index} 的 subject_point 非 [x,y]")
            point = [float(point[0]), float(point[1])]
            if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in point):
                raise ArtifactValidationError(f"sample_index={index} 的坐标不在 [0,1]")
        confidence = float(item.get("confidence", 0.0))
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ArtifactValidationError(f"sample_index={index} 的 confidence 非法")
        visibility = str(item.get("visibility", "not_found"))
        if visibility not in {"visible", "occluded", "not_found"}:
            raise ArtifactValidationError(f"sample_index={index} 的 visibility 非法")
        if visibility == "visible" and point is None:
            raise ArtifactValidationError(f"sample_index={index} 标记 visible 却没有坐标")
        by_index[index] = {
            "subject_point": point,
            "confidence": confidence,
            "visibility": visibility,
            "reason": str(item.get("reason", "")),
        }
    missing = [index for index in range(sample_count) if index not in by_index]
    # 不因少量漏答丢弃整段：缺失项显式降级为 not_found，并在 diagnostics 中记录。
    for index in missing:
        by_index[index] = {
            "subject_point": None,
            "confidence": 0.0,
            "visibility": "not_found",
            "reason": "model_omitted_sample",
        }
    return [by_index[index] for index in range(sample_count)], missing

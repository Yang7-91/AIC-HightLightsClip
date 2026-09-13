"""解析模型逐帧主体点响应并补齐遗漏帧。"""

from __future__ import annotations

import json
import math
from typing import Any

from video_highlight.common.exceptions import ArtifactValidationError
from .frame_sampler import SampledFrame


def  _join_error_prediction(error_predictions:list[dict],error_item:dict,error_reason:str):
    error_prediction = {
        "error_reason": error_reason,
        **error_item
    }
    error_predictions.append(error_prediction)

def parse_predictions(text: str, frames: list[SampledFrame], use_batch:bool=False) -> tuple[list[dict[str, Any]], list[int],list[dict]]:
    sample_count = len(frames)
    sample_start = min(frame.sample_index for frame in frames)
    # 致命错误，无法修复，直接抛出异常。后续可能需要做降级处理修复，保证一定有后续步骤可用的输出
    # 当前默认模型需要能够输出正确格式的结果，预测坐标可以缺失、错误，否则直接视为模型忽略该帧
    try:
        root = json.loads(text)
    except json.JSONDecodeError as error:
        raise ArtifactValidationError(f"Stage 3.5 响应不是合法 JSON: {error}") from error
    predictions = root.get("predictions") if isinstance(root, dict) else None
    if not isinstance(predictions, list):
        raise ArtifactValidationError("Stage 3.5 响应缺少 predictions 数组")

    error_predictions = []
    by_index: dict[int, dict[str, Any]] = {}
    for item in predictions:
        if not isinstance(item, dict):
            raise ArtifactValidationError("prediction 必须是对象")
        index = int(item.get("sample_index", -1))
        if not sample_start <= index < sample_start+sample_count:
            _join_error_prediction(error_predictions, item,error_reason=f"未知 sample_index: {index}")
            continue
        if index in by_index:
            _join_error_prediction(error_predictions, item, error_reason=f"重复 sample_index: {index}")
            continue
        point = item.get("subject_point")
        if point is not None:
            if not isinstance(point, list) or len(point) != 2:
                _join_error_prediction(error_predictions, item, error_reason=f"sample_index={index} 的 subject_point 非 [x,y]")
                continue
            point = [float(point[0]), float(point[1])]
            if not all(math.isfinite(value) and value>0 for value in point):
                _join_error_prediction(error_predictions, item,error_reason=f"sample_index={index} 的坐标{point}非有效数")
                continue
            # 预先设定的归一化防御处理
            def _norm(v, size):
                if v <= 1.5:
                    n = v
                elif v <= 1000.0:
                    n = v / 1000.0
                elif size:
                    n = v / float(size)
                else:
                    n = v / 1000.0
                return max(0.0, min(1.0, n))
            point[0] = _norm(point[0], frames[index-sample_start].width)
            point[1] = _norm(point[1], frames[index-sample_start].height)
        else:
            _join_error_prediction(error_predictions, item, error_reason=f"sample_index={index} 的坐标{point}为空")
            continue

        confidence = float(item.get("confidence", 0.0))
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            # raise ArtifactValidationError(f"sample_index={index} 的 confidence 非法") # 暂不处理
            confidence = 0.5
        visibility = str(item.get("visibility", "not_found"))
        if visibility not in {"visible", "occluded", "not_found"}:
            raise ArtifactValidationError(f"sample_index={index} 的 visibility 非法")
        # if visibility == "visible" and point is None:
        #     raise ArtifactValidationError(f"sample_index={index} 标记 visible 却没有坐标")
        by_index[index] = {
            "subject_point": point,
            "confidence": confidence,
            "visibility": visibility,
            "reason": str(item.get("reason", "")),
        }
        if not use_batch:  # 得到第一个值后直接赋值,并立马退出遍历
            break
    missing = [index for index in range(sample_start,sample_start+sample_count) if index not in by_index]
    # 不因少量漏答丢弃整段：缺失项显式降级为 not_found，并在 diagnostics 中记录。
    for index in missing:
        by_index[index] = {
            "subject_point": None,
            "confidence": 0.0,
            "visibility": "not_found",
            "reason": "model_omitted_sample",
        }
    return [by_index[index] for index in range(sample_start,sample_start+sample_count)], missing, error_predictions

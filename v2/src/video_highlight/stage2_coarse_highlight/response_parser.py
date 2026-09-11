"""
将大模型文本稳健解析为 Stage 2 片段分析结果。

此解析高度依赖prompt_builder.py的OUTPUT_SCHEMA_OPENAI，任何改动都可能导致解析崩溃失效。
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from video_highlight.common.exceptions import ArtifactValidationError
from video_highlight.contracts.schema_versions import STAGE2_SCHEMA_VERSION

from .segment_loader import LoadedSegment


class ResponseParseError(ArtifactValidationError):
    """模型响应不满足 JSON 或字段契约。"""


def _balanced_json_object(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE).strip()
    start = cleaned.find("{")
    if start < 0:
        raise ResponseParseError("响应中没有 JSON 对象")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(cleaned)):
        char = cleaned[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start : index + 1]
    raise ResponseParseError("响应中的 JSON 对象未闭合")


def _score(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ResponseParseError(f"{field} 不是数值") from error
    if not math.isfinite(number):
        raise ResponseParseError(f"{field} 不是有限数")
    return min(1.0, max(0.0, number))


def _required_offset(value: Any, field: str, duration: float, clamp: bool) -> float:
    """解析候选的必填相对时间，并按配置处理越界值。"""

    if value is None:
        raise ResponseParseError(f"{field} 不能为空")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ResponseParseError(f"{field} 不是数值") from error
    if not math.isfinite(number):
        raise ResponseParseError(f"{field} 不是有限数")
    if 0.0 <= number <= duration:
        return number
    if clamp:
        return min(duration, max(0.0, number))
    raise ResponseParseError(f"{field}={number} 超出片段范围 [0,{duration}]")


def _required_text(value: Any, field: str) -> str:
    """读取 JSON Schema 中要求的字符串字段，不隐式转换其他类型。"""

    if not isinstance(value, str):
        raise ResponseParseError(f"{field} 不是字符串")
    return value.strip()


def _nullable_text(value: Any, field: str) -> str | None:
    """读取允许为 null 的字符串字段。"""

    if value is None:
        return None
    if not isinstance(value, str):
        raise ResponseParseError(f"{field} 不是字符串或 null")
    return value.strip() or None


def _evidence_sample_ids(value: Any, segment: LoadedSegment, field: str) -> list[int]:
    """校验证据帧数组，并丢弃不属于当前分析窗口的 sample_id。"""

    if not isinstance(value, list):
        raise ResponseParseError(f"{field} 不是整数数组")
    if len(value) > 8:
        raise ResponseParseError(f"{field} 最多包含 8 个元素")
    if any(not isinstance(item, int) or isinstance(item, bool) for item in value):
        raise ResponseParseError(f"{field} 只能包含整数")
    allowed_sample_ids = {int(row["sample_id"]) for row in segment.samples}
    return sorted(set(value) & allowed_sample_ids)


def _parse_candidate(
    payload: Any,
    segment: LoadedSegment,
    candidate_index: int,
    clamp: bool,
) -> dict[str, Any]:
    """把 ``candidates[]`` 中的一项转换为带绝对时间的内部候选。"""

    field_prefix = f"candidates[{candidate_index}]"
    if not isinstance(payload, dict):
        raise ResponseParseError(f"{field_prefix} 不是 JSON 对象")

    start_offset = _required_offset(
        payload.get("start_offset_sec"),
        f"{field_prefix}.start_offset_sec",
        segment.duration_sec,
        clamp,
    )
    end_offset = _required_offset(
        payload.get("end_offset_sec"),
        f"{field_prefix}.end_offset_sec",
        segment.duration_sec,
        clamp,
    )
    if end_offset <= start_offset:
        raise ResponseParseError(
            f"{field_prefix}.end_offset_sec 必须大于 start_offset_sec"
        )

    category = _nullable_text(payload.get("category"), f"{field_prefix}.category")
    allowed_categories = {
        "action_peak",
        "emotion_peak",
        "humor",
        "insight",
        "conflict",
        "visual_spectacle",
        "other",
    }
    if category is not None and category not in allowed_categories:
        raise ResponseParseError(f"{field_prefix}.category 不在允许的枚举中: {category}")

    return {
        # candidate_index 在单个分析窗口内稳定标识候选，并用于生成不重复的原始 ID。
        "candidate_index": candidate_index,
        "summary": _required_text(payload.get("summary"), f"{field_prefix}.summary"),
        "highlight_score": _score(
            payload.get("highlight_score"), f"{field_prefix}.highlight_score"
        ),
        "completeness_score": _score(
            payload.get("completeness_score"), f"{field_prefix}.completeness_score"
        ),
        "start_offset_sec": start_offset,
        "end_offset_sec": end_offset,
        # Stage 3 消费原视频绝对时间，因此在解析阶段完成偏移换算。
        "start_sec": segment.start_sec + start_offset,
        "end_sec": segment.start_sec + end_offset,
        "category": category,
        "subject": _nullable_text(payload.get("subject"), f"{field_prefix}.subject"),
        "reason": _required_text(payload.get("reason"), f"{field_prefix}.reason"),
        "evidence_sample_ids": _evidence_sample_ids(
            payload.get("evidence_sample_ids"), segment, f"{field_prefix}.evidence_sample_ids"
        ),
    }


def parse_response(text: str, segment: LoadedSegment, config: dict[str, Any]) -> dict[str, Any]:
    """解析 vLLM 结构化输出，并返回一个包含多个候选的片段分析记录。

    返回值仍是一条 ``analyses_segment_results.jsonl`` 记录，因此恢复执行可以继续
    使用 ``segment_id`` 判断片段是否完成；真正的高光项位于 ``candidates`` 数组，
    数组中每一项都包含相对时间和换算后的原视频绝对时间。
    """

    try:
        payload = json.loads(_balanced_json_object(text))
    except json.JSONDecodeError as error:
        raise ResponseParseError(f"响应 JSON 解析失败: {error}") from error
    if not isinstance(payload, dict):
        raise ResponseParseError("响应 JSON 根节点不是对象")

    has_highlight = payload.get("has_highlight")
    if not isinstance(has_highlight, bool):
        raise ResponseParseError("has_highlight 不是布尔值")
    candidate_payloads = payload.get("candidates")
    if not isinstance(candidate_payloads, list):
        raise ResponseParseError("candidates 不是数组")
    if has_highlight != bool(candidate_payloads):
        raise ResponseParseError(
            "has_highlight 必须与 candidates 是否非空保持一致"
        )

    clamp = bool(config.get("clamp_out_of_range_times", True))
    candidates = [
        _parse_candidate(item, segment, index, clamp)
        for index, item in enumerate(candidate_payloads)
    ]

    return {
        "schema_version": STAGE2_SCHEMA_VERSION,
        "video_id": segment.video_id,
        "segment_id": segment.segment_id,
        "segment_start_sec": segment.start_sec,
        "segment_end_sec": segment.end_sec,
        "has_highlight": has_highlight,
        "candidates": candidates,
    }

"""构建 Qwen3.5-4B 的系统提示和片段分析指令。"""

from __future__ import annotations

import json
from typing import Any

from .segment_loader import LoadedSegment

# OUTPUT_SCHEMA = {
#     "has_highlight": "boolean",
#     "summary": "string",
#     "highlight_score": "number, 0..1",
#     "completeness_score": "number, 0..1",
#     "start_offset_sec": "number or null",
#     "end_offset_sec": "number or null",
#     "category": "string or null",
#     "subject": "string or null",
#     "reason": "short string",
#     "evidence_sample_ids": "integer array using sample_id values from the frame timeline",
# }
OUTPUT_SCHEMA_OPENAI = {
    "type": "json_schema",
    "json_schema": {
        "name": "video_highlight_analysis",
        "schema": {
            "type": "object",
            "properties": {
                "has_highlight": {
                    "type": "boolean",
                },
                "no_highlight_reason": {
                    "type": ["string","null"],
                    "maxLength": 80,
                    "description": "没有高光时的简短原因解释"
                },
                "candidates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "summary": {
                                "type": "string",
                                "maxLength": 80,
                                "description": "片段内容的简短概述，用于区分不同片段"
                            },
                            "highlight_score": {
                                "type": "number",
                                "description": "该片段存在高光的置信度",
                                "minimum": 0,
                                "maximum": 1
                            },
                            "completeness_score": {
                                "type": "number",
                                "description": "高光事件在片段内的完整程度；1 表示完整，0 表示完全被截断",
                                "minimum": 0,
                                "maximum": 1
                            },
                            "start_offset_sec": {
                                "type": ["number"]
                            },
                            "end_offset_sec": {
                                "type": ["number"]
                            },
                            "category": {
                              "type": ["string", "null"],
                              "enum": ["action_peak","emotion_peak","humor","insight","conflict","visual_spectacle","other",None],
                              "description": "高光触发原因：action_peak=动作峰值，emotion_peak=情绪峰值，humor=搞笑，insight=金句/观点，conflict=冲突，visual_spectacle=视觉奇观，other=其他；无法判断时返回 null"
                            },
                            "subject": {
                                "type": ["string", "null"],
                                "maxLength": 40,
                                "description": "高光中的主要人物、动物或对象；无法确定时返回 null"
                            },
                            "reason": {
                                "type": "string",
                                "maxLength": 100,
                                "description": "简短的高光判断理由"
                             },
                            "evidence_sample_ids": {
                                "type": "array",
                                "items": {
                                    "type": "integer"
                                },
                                "maxItems": 8,
                                "description": "最多 8 个关键 sample_id；无高光时为空数组"
                            }
                        },
                        "required": [
                            "summary",
                            "highlight_score",
                            "completeness_score",
                            "start_offset_sec",
                            "end_offset_sec",
                            "category",
                            "subject",
                            "reason",
                            "evidence_sample_ids"
                        ],
                        "additionalProperties": False
                    },
                    "minItems": 0,
                    "description": "高光候选列表；若片段中没有任何高光，返回空数组 []"
                },
            },
            "required": [
                "has_highlight",
                "candidates",
            ],
            "additionalProperties": False
        }
    }
}


def build_prompts(
    segment: LoadedSegment,
    context: dict[str, Any],
    prompt_config: dict[str, Any],
    visual_fps: float,
) -> tuple[str, str]:
    system_prompt = str(prompt_config["system_prompt"]).strip()
    values = {
        "video_id": segment.video_id,
        "segment_id": segment.segment_id,
        "segment_start_sec": f"{segment.start_sec:.3f}",
        "segment_end_sec": f"{segment.end_sec:.3f}",
        "segment_duration_sec": f"{segment.duration_sec:.3f}",
        "visual_fps": f"{visual_fps:.3f}",
        "frame_timeline": context["frame_timeline"],
        "audio_events": context["audio_events"],
        "audio_timeline": context["audio_timeline"],
        "asr_timeline": context["asr_timeline"],
        # 保留旧版/自定义提示模板使用的占位符。当前正式模板依靠
        # response_format 的 JSON Schema 约束，因此可以不显式引用它。
        # "output_schema": json.dumps(OUTPUT_SCHEMA, ensure_ascii=False, indent=2),
    }
    user_prompt = str(prompt_config["user_template"]).format_map(values).strip()
    return system_prompt, user_prompt

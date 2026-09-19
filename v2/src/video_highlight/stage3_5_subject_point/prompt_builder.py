"""构造逐帧多主体观察提示、图像内容和 OpenAI JSON Schema。"""

from __future__ import annotations

import base64
from typing import Any

from .frame_sampler import SampledFrame


OUTPUT_SCHEMA_OPENAI: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "subject_observation_predictions",
        "schema": {
            "type": "object",
            "properties": {
                "predictions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "sample_index": {"type": "integer", "minimum": 0},
                            "group_mode": {"type": "string", "enum": ["single", "multiple"]},
                            "grounding_phrases": {
                                "type": "array",
                                "items": {"type": "string", "minLength": 1, "maxLength": 80},
                                "maxItems": 8,
                            },
                            "targets": {
                                "type": "array",
                                "maxItems": 12,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "target_id": {"type": "string", "minLength": 1, "maxLength": 40},
                                        "description": {"type": "string", "maxLength": 80},
                                        "grounding_phrase": {"type": "string", "minLength": 1, "maxLength": 80},
                                        "subject_point": {
                                            "type": ["array", "null"],
                                            "items": {"type": "number"},
                                            "minItems": 2,
                                            "maxItems": 2,
                                        },
                                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                                        "visibility": {"type": "string", "enum": ["visible", "occluded", "not_found"]},
                                    },
                                    "required": [
                                        "target_id", "description", "grounding_phrase",
                                        "subject_point", "confidence", "visibility"
                                    ],
                                    "additionalProperties": False,
                                },
                            },
                            "reason": {"type": "string"},
                        },
                        "required": ["sample_index", "group_mode", "grounding_phrases", "targets", "reason"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["predictions"],
            "additionalProperties": False,
        },
    },
}


DEFAULT_SYSTEM_PROMPT = """你是视频逐帧多主体定位器。对每张图识别为了完整呈现用户指定高光主体而必须保留的所有人物、动物或物体。
每个目标输出归一化视觉中心 [x,y]；不可见或无法可靠定位时为 null。grounding_phrase 与 grounding_phrases 必须是简短、具体、全小写的英文名词短语，适合开放词汇目标检测，不要写动作句子；可用由具体到宽泛的多个短语。target_id 应根据身份或外观生成简短稳定标识，同一区间中尽量复用。单主体用 group_mode=single，需要同时保留多个目标才能完整构图时用 multiple。禁止为了填满数组而加入背景目标。每个 sample_index 必须且只能返回一次，只返回满足 JSON Schema 的对象。"""


def build_prompt(video_id: str, interval: dict[str, Any], frames: list[SampledFrame],use_batch:bool=False) -> str:
    """根据采样帧信息构造提示词，use_sequence默认为False，此时提示词只会让模型每次只单独判断一帧"""
    subject = interval.get("subject") or "画面中的主要高光主体"
    interval_id = interval["interval_id"]

    if not use_batch:
        # 单帧独立判断模式：通常 frames 只包含当前这一帧
        if not frames:
            raise ValueError("frames 不能为空")
        row = frames[0]
        return (
            f"video_id={video_id}\n"
            f"interval_id={interval_id}\n"
            f"subject={subject}\n"
            f"当前帧：sample_index={row.sample_index}, "
            f"original_frame={row.frame}, "
            f"timestamp_sec={row.timestamp_sec:.6f}\n"
            "请列出为了完整呈现该高光主体必须保留的所有目标。不要在预设主体不可见时擅自改成无关主体。"
            "每个可见目标返回归一化中心，并生成可供 Grounding DINO 使用的英文名词短语。"
            "返回示例："
            '{"predictions":[{"sample_index":34,"group_mode":"multiple",'
            '"grounding_phrases":["basketball player","person"],"targets":['
            '{"target_id":"player_red","description":"红衣球员","grounding_phrase":"basketball player",'
            '"subject_point":[0.68,0.52],"confidence":0.99,"visibility":"visible"}],'
            '"reason":"主体清晰可见"}]}'
        )
    timeline = "\n".join(
        f"- sample_index={row.sample_index}, original_frame={row.frame}, timestamp_sec={row.timestamp_sec:.6f}"
        for row in frames
    )
    subject = interval.get("subject") or "画面中的主要高光主体"
    return (
        f"video_id={video_id}\ninterval_id={interval['interval_id']}\n"
        f"subject={subject}\n采样时间线（后续图像严格按此顺序排列）：\n{timeline}\n"
        "请独立判断每一帧，列出为了完整呈现指定高光主体必须保留的所有目标。"
        "同一对象跨帧尽量复用 target_id；可见目标必须返回中心，无法可靠定位时返回 null，禁止改选无关主体。"
    )


def build_user_content(prompt: str, frames: list[SampledFrame]) -> list[dict[str, Any]]:
    """图像只以进程内 data URL 发送；调用完成后不会持久化 Base64。"""

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for row in frames:
        content.append({"type": "text", "text": f"sample_index={row.sample_index}"})
        data = base64.b64encode(row.jpeg_bytes).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}})
    return content

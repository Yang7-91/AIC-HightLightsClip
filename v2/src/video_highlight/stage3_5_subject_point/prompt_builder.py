"""构造逐帧主体点提示、图像内容和 OpenAI JSON Schema。"""

from __future__ import annotations

import base64
from typing import Any

from .frame_sampler import SampledFrame


OUTPUT_SCHEMA_OPENAI: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "subject_point_predictions",
        "schema": {
            "type": "object",
            "properties": {
                "predictions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "sample_index": {"type": "integer", "minimum": 0},
                            "subject_point": {
                                "type": ["array", "null"],
                                "items": {"type": "number"},
                                "minItems": 2,
                                "maxItems": 2,
                            },
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            "visibility": {"type": "string", "enum": ["visible", "occluded", "not_found"]},
                            "reason": {"type": "string"},
                        },
                        "required": ["sample_index", "subject_point", "confidence", "visibility", "reason"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["predictions"],
            "additionalProperties": False,
        },
    },
}


DEFAULT_SYSTEM_PROMPT = """你是视频逐帧主体定位器。对用户按 sample_index 顺序提供的每张图，定位指定主体的视觉中心，如指定主体不可见但有其他可见主体，则以其他可见主体为准
。坐标为相对该图像宽高归一化的 [x,y]，左上角为 [0,0]，右下角为 [1,1]。每个 sample_index 必须且只能返回一次；主体无可靠定位时返回 null，禁止猜测。只返回满足 JSON Schema 的对象。"""


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
            """请判断该帧中主体是否可见，如果预设主体不可见，但有其他可见主体，则以其他可见主体为准。visible 时必须返回主体中心；occluded/not_found 可返回 null。返回json示例：
            {"predictions": [{"sample_index": 34,"subject_point": [0.68,0.52],"confidence": 0.99,"visibility": "visible","reason": "主体清晰可见，无遮挡"}]}"""
        )
    timeline = "\n".join(
        f"- sample_index={row.sample_index}, original_frame={row.frame}, timestamp_sec={row.timestamp_sec:.6f}"
        for row in frames
    )
    subject = interval.get("subject") or "画面中的主要高光主体"
    return (
        f"video_id={video_id}\ninterval_id={interval['interval_id']}\n"
        f"subject={subject}\n采样时间线（后续图像严格按此顺序排列）：\n{timeline}\n"
        "请独立判断每一帧，如果预设主体不可见，但有其他可见主体，则以其他可见主体为准。visible 时必须返回主体中心；occluded/not_found 可返回 null。"
    )


def build_user_content(prompt: str, frames: list[SampledFrame]) -> list[dict[str, Any]]:
    """图像只以进程内 data URL 发送；调用完成后不会持久化 Base64。"""

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for row in frames:
        content.append({"type": "text", "text": f"sample_index={row.sample_index}"})
        data = base64.b64encode(row.jpeg_bytes).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}})
    return content

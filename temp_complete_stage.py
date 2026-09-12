#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 V2 Stage 2 粗高光结果转换为比赛逐帧重构图结果。

本脚本复用 ``baseline_qwen.py`` 原始 Stage 2 的思路：

1. 读取 Stage 2 每个视频的 ``candidates.jsonl``；
2. 防御性合并仍有重叠的候选（默认不跨空白间隔继续合并）；
3. 把秒级区间转换为原视频帧区间；
4. 根据目标比例和源尺寸确定唯一的最大内接裁剪框尺寸；
5. 按固定帧间隔抽取关键帧，通过 Qwen-VL 预测重要主体的归一化中心点；
6. 把裁剪窗放到预测中心并夹紧在源画面内；
7. 在关键帧裁剪框之间线性插值，生成高光区间内每一帧的 ``[x, y, w]``；
8. 生成最终 ``predictions.jsonl``、模型原始响应和区间诊断 JSONL。

这里的“裁剪”指生成比赛提交所需的逐帧裁剪框，不会重新编码或导出 MP4。
Stage 2 候选中的 ``subject`` 会作为可选提示传给主体定位模型，但最终坐标仍
以关键帧图像的模型预测为准。某些关键帧调用失败时会使用其余成功关键帧插值；
若一个区间全部失败，则默认回退到画面中心，保证输出完整。

示例：
    python temp_complete_stage.py `
      --video-id 0

处理索引中的全部视频：
    python temp_complete_stage.py `
      --out temp_complete_predictions.jsonl

处理其他 Stage 2 实验目录：
    python temp_complete_stage.py `
      --stage2-dir v2/runs/another-run/stage2
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, TextIO


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INDEX = SCRIPT_DIR / "test_index.json"
DEFAULT_VIDEO_DIR = Path("F:/datasets/video-clip/video")
DEFAULT_OUTPUT = SCRIPT_DIR / "temp_complete_predictions.jsonl"
DEFAULT_STAGE2_DIR = SCRIPT_DIR / "v2/runs/full/stage2"
DEFAULT_QWEN_API_BASE = "http://172.25.254.120:8000/v1"
DEFAULT_QWEN_MODEL = "QuantTrio/Qwen3.5-4B-AWQ"


class ValidationError(RuntimeError):
    """输入数据不满足快速闭环所需契约。"""


@dataclass(frozen=True, slots=True)
class VideoMeta:
    """生成逐帧裁剪结果所需的最小视频元数据。"""

    frame_count: int
    fps: float
    width: int
    height: int

    @property
    def duration_sec(self) -> float:
        return self.frame_count / self.fps


@dataclass(frozen=True, slots=True)
class Interval:
    """由一个或多个 Stage 2 候选合并得到的秒级高光区间。"""

    start_sec: float
    end_sec: float
    candidate_ids: tuple[str, ...]
    max_score: float
    subjects: tuple[str, ...]


def load_index(path: Path) -> list[tuple[str, tuple[float, float]]]:
    """读取测试索引并返回 ``[(video_id, (target_w, target_h)), ...]``。"""

    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise ValidationError(f"测试索引不存在: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValidationError(f"测试索引不是合法 JSON: {path}: {exc}") from exc
    if not isinstance(value, list):
        raise ValidationError(f"测试索引根节点必须是数组: {path}")

    result: list[tuple[str, tuple[float, float]]] = []
    seen: set[str] = set()
    for position, item in enumerate(value, start=1):
        if not isinstance(item, dict) or "video_id" not in item:
            raise ValidationError(f"测试索引第 {position} 项缺少 video_id")
        video_id = str(item["video_id"])
        if video_id in seen:
            raise ValidationError(f"测试索引存在重复 video_id: {video_id}")
        ratio = item.get("targetRatioWH", [16, 9])
        if not isinstance(ratio, list) or len(ratio) != 2:
            raise ValidationError(f"{video_id}.targetRatioWH 必须是 [w, h]")
        try:
            target_w, target_h = float(ratio[0]), float(ratio[1])
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"{video_id}.targetRatioWH 必须是数值") from exc
        if not all(math.isfinite(v) and v > 0 for v in (target_w, target_h)):
            raise ValidationError(f"{video_id}.targetRatioWH 必须是有限正数")
        seen.add(video_id)
        result.append((video_id, (target_w, target_h)))
    return result


def resolve_stage2_videos_dir(stage2_dir: Path) -> Path:
    """兼容传入 ``stage2`` 根目录或直接传入其 ``videos`` 子目录。"""

    root = stage2_dir.resolve()
    if root.name == "videos" and root.is_dir():
        return root
    videos_dir = root / "videos"
    if videos_dir.is_dir():
        return videos_dir
    raise ValidationError(
        f"找不到 Stage 2 videos 目录: {root}；请传入 stage2 或 stage2/videos"
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL；异常信息包含具体文件和行号，便于定位坏产物。"""

    records: list[dict[str, Any]] = []
    try:
        handle = path.open("r", encoding="utf-8-sig")
    except FileNotFoundError as exc:
        raise ValidationError(f"Stage 2 候选文件不存在: {path}") from exc
    with handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"JSONL 解析失败: {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValidationError(f"JSONL 每行必须是对象: {path}:{line_number}")
            records.append(value)
    return records


def read_video_meta(video_path: Path) -> VideoMeta:
    """参考 baseline_qwen，使用 OpenCV 仅读取视频头部元信息。"""

    try:
        import cv2
    except ImportError as exc:
        raise ValidationError("缺少 opencv-python，请先在 video-clip 环境中安装") from exc

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValidationError(f"无法打开视频: {video_path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()

    # FPS=0 通常表示容器信息不可用。这里不静默使用 25 FPS，避免提交帧号错位。
    if frame_count <= 0 or not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
        raise ValidationError(
            f"视频元数据无效: {video_path} "
            f"frames={frame_count}, fps={fps}, size={width}x{height}"
        )
    return VideoMeta(frame_count=frame_count, fps=fps, width=width, height=height)


def candidate_intervals(
    video_id: str,
    candidates: Iterable[dict[str, Any]],
    duration_sec: float,
    min_score: float,
) -> list[Interval]:
    """校验、过滤并裁定 Stage 2 候选的合法时间范围。"""

    intervals: list[Interval] = []
    for index, row in enumerate(candidates):
        row_video_id = str(row.get("video_id", video_id))
        if row_video_id != video_id:
            raise ValidationError(
                f"{video_id} 的候选文件混入了其他视频: {row_video_id}"
            )
        try:
            start_sec = float(row["start_sec"])
            end_sec = float(row["end_sec"])
            score = float(row.get("coarse_score", 0.0))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError(f"{video_id} 第 {index + 1} 个候选字段无效") from exc
        if not all(math.isfinite(v) for v in (start_sec, end_sec, score)):
            raise ValidationError(f"{video_id} 第 {index + 1} 个候选包含非有限数值")
        if score < min_score:
            continue

        # 防御性处理模型或旧产物中颠倒的端点，并限制在真实视频时长内。
        start_sec, end_sec = sorted((start_sec, end_sec))
        start_sec = max(0.0, min(start_sec, duration_sec))
        end_sec = max(0.0, min(end_sec, duration_sec))
        if end_sec <= start_sec:
            continue
        candidate_id = str(row.get("candidate_id", f"{video_id}_raw_{index:04d}"))
        subject = str(row.get("subject") or "").strip()
        intervals.append(
            Interval(
                start_sec,
                end_sec,
                (candidate_id,),
                score,
                (subject,) if subject else (),
            )
        )
    return intervals


def merge_intervals(intervals: Iterable[Interval], max_gap_sec: float) -> list[Interval]:
    """合并重叠区间，以及间隔不超过 ``max_gap_sec`` 的相邻区间。"""

    if not math.isfinite(max_gap_sec) or max_gap_sec < 0:
        raise ValidationError("max_gap_sec 必须是有限非负数")
    ordered = sorted(intervals, key=lambda item: (item.start_sec, item.end_sec))
    merged: list[Interval] = []
    for current in ordered:
        if not merged or current.start_sec > merged[-1].end_sec + max_gap_sec:
            merged.append(current)
            continue
        previous = merged[-1]
        merged[-1] = Interval(
            start_sec=previous.start_sec,
            end_sec=max(previous.end_sec, current.end_sec),
            candidate_ids=tuple(dict.fromkeys(previous.candidate_ids + current.candidate_ids)),
            max_score=max(previous.max_score, current.max_score),
            subjects=tuple(dict.fromkeys(previous.subjects + current.subjects)),
        )
    return merged


def seconds_to_frame_ranges(
    intervals: Iterable[Interval], fps: float, frame_count: int
) -> list[tuple[int, int]]:
    """按 baseline_qwen 的 round 规则逐一转换为闭区间帧号。"""

    ranges: list[tuple[int, int]] = []
    for interval in intervals:
        start_frame = int(round(interval.start_sec * fps))
        end_frame = int(round(interval.end_sec * fps))
        start_frame = max(0, min(start_frame, frame_count - 1))
        end_frame = max(0, min(end_frame, frame_count - 1))
        if end_frame >= start_frame:
            ranges.append((start_frame, end_frame))
    return ranges


def compute_crop_size(
    source_width: int,
    source_height: int,
    target_width: float,
    target_height: float,
) -> tuple[int, int]:
    """计算完整落在源画面内、且符合目标比例的最大裁剪尺寸。"""

    if min(source_width, source_height) <= 0 or min(target_width, target_height) <= 0:
        raise ValidationError("源尺寸和目标画幅必须为正数")
    target_ratio = target_width / target_height
    source_ratio = source_width / source_height
    if source_ratio >= target_ratio:
        # 提交只写宽度，评测端会按比例反算高度。宽度必须向下取整，否则像
        # 9:16@1920x1080 会因 round(607.5)=608 而反算出 1080.44，造成越界。
        crop_width = min(int(math.floor(source_height * target_ratio + 1e-9)), source_width)
    else:
        crop_width = source_width
    # 内部用 ceil 表示反算高度实际占用的像素范围，夹紧 y 时更保守。
    crop_height = min(
        int(math.ceil(crop_width / target_ratio - 1e-9)),
        source_height,
    )
    return max(1, crop_width), max(1, crop_height)


def centered_crop_box(meta: VideoMeta, target_ratio: tuple[float, float]) -> tuple[int, int, int, int]:
    """以画面中心 ``(0.5, 0.5)`` 放置最大内接裁剪框，并限制到边界内。"""

    crop_width, crop_height = compute_crop_size(
        meta.width, meta.height, target_ratio[0], target_ratio[1]
    )
    x = int(round((meta.width - crop_width) / 2.0))
    y = int(round((meta.height - crop_height) / 2.0))
    x = max(0, min(x, meta.width - crop_width))
    y = max(0, min(y, meta.height - crop_height))
    return x, y, crop_width, crop_height


def center_to_box(
    center_x: float,
    center_y: float,
    meta: VideoMeta,
    crop_width: int,
    crop_height: int,
) -> tuple[int, int, int, int]:
    """把 0~1 主体中心转换为裁剪框，并把窗口夹紧到源画面内。"""

    center_x = max(0.0, min(1.0, float(center_x)))
    center_y = max(0.0, min(1.0, float(center_y)))
    x = int(round(center_x * meta.width - crop_width / 2.0))
    y = int(round(center_y * meta.height - crop_height / 2.0))
    x = max(0, min(x, meta.width - crop_width))
    y = max(0, min(y, meta.height - crop_height))
    return x, y, crop_width, crop_height


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_focus_norm(text: str, width: int, height: int) -> tuple[float, float] | None:
    """解析 Qwen 输出，兼容 0~1、0~1000 和绝对像素坐标。"""

    candidate: tuple[float, float] | None = None
    objects: list[Any] = []
    try:
        objects.append(json.loads(text))
    except (json.JSONDecodeError, TypeError):
        pass
    for match in re.finditer(r"\{[^{}]*\}", text, re.DOTALL):
        try:
            objects.append(json.loads(match.group(0)))
        except json.JSONDecodeError:
            continue
    for value in objects:
        if not isinstance(value, dict):
            continue
        for key in ("center", "subject_center", "focus", "point", "cxcy"):
            point = value.get(key)
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                try:
                    candidate = float(point[0]), float(point[1])
                except (TypeError, ValueError):
                    continue
    if candidate is None:
        numbers = _NUMBER_RE.findall(text)
        if len(numbers) >= 2:
            candidate = float(numbers[-2]), float(numbers[-1])
    if candidate is None or not all(math.isfinite(value) for value in candidate):
        return None

    def normalize(value: float, size: int) -> float:
        if -0.01 <= value <= 1.5:
            normalized = value
        elif -1.0 <= value <= 1000.0:
            normalized = value / 1000.0
        else:
            normalized = value / float(size)
        return max(0.0, min(1.0, normalized))

    return normalize(candidate[0], width), normalize(candidate[1], height)


class QwenFocusClient:
    """仅负责关键帧主体中心预测的 OpenAI 兼容 Qwen-VL 客户端。"""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout_sec: float,
        max_retries: int,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ValidationError("缺少 openai 包，请先在 video-clip 环境中安装") from exc
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url.rstrip("/") + "/",
            timeout=timeout_sec,
            max_retries=max_retries,
        )
        self.model = model

    def healthcheck(self) -> None:
        """确认接口可达，且服务端暴露了配置的模型名。"""

        try:
            model_ids = {str(item.id) for item in self.client.models.list().data}
        except Exception as exc:
            raise ValidationError(f"Qwen API 健康检查失败: {exc}") from exc
        if self.model not in model_ids:
            raise ValidationError(
                f"Qwen API 未暴露模型 {self.model!r}；当前模型: {sorted(model_ids)}"
            )

    def predict_focus(
        self,
        bgr_frame: Any,
        target_ratio: tuple[float, float],
        subject_hint: str,
        max_new_tokens: int,
        enable_thinking: bool,
        thinking_token_budget: int,
        jpeg_quality: int,
    ) -> dict[str, Any]:
        """发送单张关键帧，返回可直接写入 JSONL 的普通字典。"""

        try:
            import cv2
        except ImportError as exc:
            raise ValidationError("缺少 opencv-python，请先在 video-clip 环境中安装") from exc
        ok, encoded = cv2.imencode(
            ".jpg", bgr_frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        )
        if not ok:
            raise ValidationError("关键帧 JPEG 编码失败")
        image_url = "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")
        target_w, target_h = target_ratio
        hint = subject_hint.strip()
        hint_line = (
            f"The coarse highlight detector suggests the subject is: {hint[:300]}.\n"
            if hint
            else ""
        )
        prompt = (
            f"This video frame will be cropped to {target_w:g}:{target_h:g}.\n"
            f"{hint_line}"
            "Locate the center of the most important visible subject or region that must remain "
            "inside the crop. Return normalized integer coordinates from 0 to 1000, where x=0 "
            "is left, x=1000 is right, y=0 is top, and y=1000 is bottom.\n"
            "Return only one JSON object, for example: {\"center\": [500, 500]}"
        )
        schema = {
            "type": "object",
            "properties": {
                "center": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                }
            },
            "required": ["center"],
            "additionalProperties": False,
        }
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            max_completion_tokens=max_new_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "focus", "strict": True, "schema": schema},
            },
            extra_body={
                "chat_template_kwargs": {"enable_thinking": enable_thinking},
                "thinking_token_budget": thinking_token_budget,
            },
        )
        choice = completion.choices[0]
        message = choice.message
        reasoning = getattr(message, "reasoning", None)
        if reasoning is None:
            reasoning = getattr(message, "reasoning_content", None)
        usage = getattr(completion, "usage", None)
        if usage is not None and hasattr(usage, "model_dump"):
            usage = usage.model_dump(mode="json")
        elif usage is not None and not isinstance(usage, (dict, str, int, float, bool)):
            usage = str(usage)
        return {
            "content": message.content,
            "reasoning": reasoning if isinstance(reasoning, (str, type(None))) else str(reasoning),
            "finish_reason": choice.finish_reason,
            "usage": usage,
            "request_id": getattr(completion, "id", None),
        }


def keyframe_ids(start_frame: int, end_frame: int, stride: int) -> list[int]:
    """按固定步长采样，并保证区间首尾帧一定被包含。"""

    frames = list(range(start_frame, end_frame + 1, max(1, stride)))
    if not frames or frames[-1] != end_frame:
        frames.append(end_frame)
    return frames


def extract_frames(video_path: Path, frame_ids: Iterable[int]) -> dict[int, Any]:
    """从最小到最大目标帧顺序解码，避免对每个关键帧重复随机 seek。"""

    try:
        import cv2
    except ImportError as exc:
        raise ValidationError("缺少 opencv-python，请先在 video-clip 环境中安装") from exc
    wanted = sorted(set(int(value) for value in frame_ids))
    if not wanted:
        return {}
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValidationError(f"无法打开视频: {video_path}")
    result: dict[int, Any] = {}
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, wanted[0])
        wanted_set = set(wanted)
        frame_id = wanted[0]
        while frame_id <= wanted[-1]:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_id in wanted_set:
                result[frame_id] = frame
            frame_id += 1
    finally:
        capture.release()
    return result


def densify_boxes(
    start_frame: int,
    end_frame: int,
    key_frames: list[int],
    key_boxes: list[tuple[int, int, int, int]],
) -> dict[int, tuple[int, int, int, int]]:
    """在相邻关键帧裁剪框间线性插值，得到区间内每一帧的框。"""

    if not key_frames:
        return {}
    points = sorted(zip(key_frames, key_boxes), key=lambda item: item[0])
    frames = [item[0] for item in points]
    boxes = [item[1] for item in points]
    dense: dict[int, tuple[int, int, int, int]] = {}
    left_index = 0
    for frame_id in range(start_frame, end_frame + 1):
        while left_index + 1 < len(frames) and frames[left_index + 1] < frame_id:
            left_index += 1
        if frame_id <= frames[0]:
            dense[frame_id] = boxes[0]
        elif frame_id >= frames[-1]:
            dense[frame_id] = boxes[-1]
        else:
            right_index = left_index + 1
            left_frame, right_frame = frames[left_index], frames[right_index]
            ratio = (frame_id - left_frame) / float(right_frame - left_frame)
            dense[frame_id] = tuple(
                int(round(boxes[left_index][axis] + (boxes[right_index][axis] - boxes[left_index][axis]) * ratio))
                for axis in range(4)
            )
    return dense


def reframe_interval(
    model: QwenFocusClient,
    video_path: Path,
    video_id: str,
    interval: Interval,
    frame_range: tuple[int, int],
    target_ratio: tuple[float, float],
    meta: VideoMeta,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """预测一个高光区间的关键帧中心，并插值得到逐帧比赛框。"""

    start_frame, end_frame = frame_range
    requested_frames = keyframe_ids(start_frame, end_frame, args.crop_stride)
    decoded_frames = extract_frames(video_path, requested_frames)
    crop_width, crop_height = compute_crop_size(
        meta.width, meta.height, target_ratio[0], target_ratio[1]
    )
    subject_hint = " / ".join(interval.subjects)
    successful_frames: list[int] = []
    successful_boxes: list[tuple[int, int, int, int]] = []
    raw_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for frame_id in requested_frames:
        frame = decoded_frames.get(frame_id)
        if frame is None:
            failures.append({"frame": frame_id, "error": "关键帧解码失败"})
            continue
        try:
            raw = model.predict_focus(
                frame,
                target_ratio,
                subject_hint,
                args.focus_max_new_tokens,
                args.focus_enable_thinking,
                args.focus_thinking_budget,
                args.jpeg_quality,
            )
            raw_records.append(
                {
                    "video_id": video_id,
                    "frame": frame_id,
                    "stage": "subject_focus",
                    "candidate_ids": list(interval.candidate_ids),
                    "subject_hint": subject_hint or None,
                    "raw": raw,
                }
            )
            content = raw.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("模型返回 content 为空或不是字符串")
            center = parse_focus_norm(content, meta.width, meta.height)
            if center is None:
                raise ValueError(f"无法从模型输出解析主体中心: {content[:200]!r}")
            successful_frames.append(frame_id)
            successful_boxes.append(
                center_to_box(center[0], center[1], meta, crop_width, crop_height)
            )
        except Exception as exc:
            failures.append({"frame": frame_id, "error": str(exc)})
            if args.focus_failure == "error":
                raise ValidationError(f"主体定位失败: {video_id}#{frame_id}: {exc}") from exc

    used_center_fallback = False
    if not successful_frames:
        used_center_fallback = True
        fallback_box = center_to_box(0.5, 0.5, meta, crop_width, crop_height)
        successful_frames = [start_frame, end_frame]
        successful_boxes = [fallback_box, fallback_box]
    dense = densify_boxes(
        start_frame, end_frame, successful_frames, successful_boxes
    )
    predictions = [
        {
            "frame": frame_id,
            "bboxes": [dense[frame_id][0], dense[frame_id][1], dense[frame_id][2]],
        }
        for frame_id in range(start_frame, end_frame + 1)
        if frame_id in dense
    ]
    diagnostic = {
        "candidate_ids": list(interval.candidate_ids),
        "subject_hints": list(interval.subjects),
        "requested_keyframes": requested_frames,
        "successful_keyframes": successful_frames,
        "keyframe_boxes_xywh": [list(box) for box in successful_boxes],
        "focus_failures": failures,
        "used_center_fallback": used_center_fallback,
    }
    return predictions, raw_records, diagnostic


def _open_atomic_target(path: Path) -> tuple[TextIO, Path]:
    """在目标目录创建临时文件，全部视频成功后再原子替换正式输出。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    return handle, Path(handle.name)


def _warn(message: str) -> None:
    print(f"[warning] {message}", file=sys.stderr)


def _target_ratio_for_output(target_ratio: tuple[float, float]) -> list[int | float]:
    """整数比例保持整数形式，非整数比例则保留浮点值。"""

    result: list[int | float] = []
    for value in target_ratio:
        result.append(int(value) if value.is_integer() else value)
    return result


def run(args: argparse.Namespace) -> tuple[int, int]:
    """执行 Stage 2 候选到逐帧重构图的闭环。"""

    if not math.isfinite(args.min_score):
        raise ValidationError("min_score 必须是有限数值")
    if not math.isfinite(args.merge_gap_sec) or args.merge_gap_sec < 0:
        raise ValidationError("merge_gap_sec 必须是有限非负数")
    if args.num_videos < 0:
        raise ValidationError("num_videos 不能小于 0")
    if args.crop_stride <= 0:
        raise ValidationError("crop_stride 必须大于 0")
    if args.focus_max_new_tokens <= 0 or args.focus_thinking_budget < 0:
        raise ValidationError("主体定位 token 参数无效")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValidationError("jpeg_quality 必须在 1~100")

    stage2_videos_dir = resolve_stage2_videos_dir(args.stage2_dir)
    index = load_index(args.index)
    if args.video_id:
        selected_ids = set(args.video_id)
        known_ids = {video_id for video_id, _ in index}
        unknown_ids = selected_ids - known_ids
        if unknown_ids:
            raise ValidationError(f"--video-id 不在测试索引中: {sorted(unknown_ids)}")
        index = [item for item in index if item[0] in selected_ids]
    if args.available_only:
        index = [
            item
            for item in index
            if (stage2_videos_dir / item[0] / "_SUCCESS.json").is_file()
        ]
    if args.num_videos > 0:
        index = index[: args.num_videos]
    if not index:
        raise ValidationError("没有需要处理的视频")

    output_path = args.out.resolve()
    intervals_path = (
        args.intervals_out.resolve()
        if args.intervals_out is not None
        else output_path.with_name(f"{output_path.stem}.intervals.jsonl")
    )
    raw_path = (
        args.raw_out.resolve()
        if args.raw_out is not None
        else output_path.with_name(f"{output_path.stem}.raw.jsonl")
    )
    if len({output_path, intervals_path, raw_path}) != 3:
        raise ValidationError("--out、--intervals-out 和 --raw-out 必须指向不同文件")

    model = QwenFocusClient(
        args.qwen_api_base,
        args.qwen_api_key,
        args.qwen_model_name,
        args.qwen_timeout,
        args.qwen_max_retries,
    )
    if not args.no_healthcheck:
        model.healthcheck()

    prediction_handle, prediction_tmp = _open_atomic_target(output_path)
    interval_handle, interval_tmp = _open_atomic_target(intervals_path)
    raw_handle, raw_tmp = _open_atomic_target(raw_path)
    total_predictions = 0
    total_model_calls = 0

    try:
        with prediction_handle, interval_handle, raw_handle:
            for position, (video_id, target_ratio) in enumerate(index, start=1):
                extension = args.video_extension
                if extension and not extension.startswith("."):
                    extension = f".{extension}"
                video_path = args.video_dir.resolve() / f"{video_id}{extension}"
                candidate_path = stage2_videos_dir / video_id / "candidates.jsonl"
                output_ratio = _target_ratio_for_output(target_ratio)
                predictions: list[dict[str, Any]] = []
                diagnostic: dict[str, Any] = {
                    "video_id": video_id,
                    "targetRatioWH": output_ratio,
                    "source_candidate_count": 0,
                    "merged_intervals": [],
                    "frame_ranges": [],
                    "crop_size_wh": None,
                    "reframing": [],
                    "video_meta": None,
                }

                try:
                    success_path = candidate_path.parent / "_SUCCESS.json"
                    if not success_path.is_file():
                        raise ValidationError(f"Stage 2 没有成功标记: {success_path}")
                    candidates = read_jsonl(candidate_path)
                    diagnostic["source_candidate_count"] = len(candidates)
                    if candidates:
                        if not video_path.is_file():
                            raise ValidationError(f"视频文件不存在: {video_path}")
                        meta = read_video_meta(video_path)
                        raw_intervals = candidate_intervals(
                            video_id,
                            candidates,
                            meta.duration_sec,
                            args.min_score,
                        )
                        merged_intervals = merge_intervals(raw_intervals, args.merge_gap_sec)
                        frame_ranges = seconds_to_frame_ranges(
                            merged_intervals, meta.fps, meta.frame_count
                        )
                        if len(frame_ranges) != len(merged_intervals):
                            raise ValidationError(
                                f"{video_id} 的候选区间无法转换为有效帧区间"
                            )
                        crop_width, crop_height = compute_crop_size(
                            meta.width, meta.height, target_ratio[0], target_ratio[1]
                        )
                        prediction_by_frame: dict[int, dict[str, Any]] = {}
                        reframing_diagnostics: list[dict[str, Any]] = []
                        for interval, frame_range in zip(merged_intervals, frame_ranges):
                            interval_predictions, raw_records, reframe_diagnostic = reframe_interval(
                                model,
                                video_path,
                                video_id,
                                interval,
                                frame_range,
                                target_ratio,
                                meta,
                                args,
                            )
                            for record in raw_records:
                                raw_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                            total_model_calls += len(raw_records)
                            for prediction in interval_predictions:
                                prediction_by_frame[int(prediction["frame"])] = prediction
                            reframing_diagnostics.append(
                                {
                                    "start_sec": interval.start_sec,
                                    "end_sec": interval.end_sec,
                                    "frame_range": list(frame_range),
                                    **reframe_diagnostic,
                                }
                            )
                        predictions = [
                            prediction_by_frame[frame_id]
                            for frame_id in sorted(prediction_by_frame)
                        ]
                        diagnostic.update(
                            {
                                "merged_intervals": [
                                    {
                                        "start_sec": item.start_sec,
                                        "end_sec": item.end_sec,
                                        "candidate_ids": list(item.candidate_ids),
                                        "max_score": item.max_score,
                                        "subjects": list(item.subjects),
                                    }
                                    for item in merged_intervals
                                ],
                                "frame_ranges": [list(item) for item in frame_ranges],
                                "crop_size_wh": [crop_width, crop_height],
                                "reframing": reframing_diagnostics,
                                "video_meta": {
                                    "frame_count": meta.frame_count,
                                    "fps": meta.fps,
                                    "width": meta.width,
                                    "height": meta.height,
                                    "duration_sec": meta.duration_sec,
                                },
                            }
                        )
                except ValidationError as exc:
                    if args.strict:
                        raise
                    diagnostic["warning"] = str(exc)
                    _warn(str(exc))

                record = {
                    "video_id": video_id,
                    "targetRatioWH": output_ratio,
                    "predictions": predictions,
                }
                prediction_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                interval_handle.write(json.dumps(diagnostic, ensure_ascii=False) + "\n")
                prediction_handle.flush()
                interval_handle.flush()
                raw_handle.flush()
                total_predictions += len(predictions)
                focus_failures = sum(
                    len(item.get("focus_failures", []))
                    for item in diagnostic["reframing"]
                )
                print(
                    f"[{position}/{len(index)}] video={video_id} "
                    f"candidates={diagnostic['source_candidate_count']} "
                    f"merged={len(diagnostic['merged_intervals'])} "
                    f"frames={len(predictions)} focus_failures={focus_failures}"
                )
        os.replace(prediction_tmp, output_path)
        os.replace(interval_tmp, intervals_path)
        os.replace(raw_tmp, raw_path)
    except BaseException:
        prediction_handle.close()
        interval_handle.close()
        raw_handle.close()
        prediction_tmp.unlink(missing_ok=True)
        interval_tmp.unlink(missing_ok=True)
        raw_tmp.unlink(missing_ok=True)
        raise

    print(
        f"完成: {len(index)} 个视频，{total_predictions} 条逐帧预测，"
        f"{total_model_calls} 次有效模型响应 -> {output_path}"
    )
    print(f"区间与裁剪框诊断 -> {intervals_path}")
    print(f"主体定位模型原始响应 -> {raw_path}")
    return len(index), total_predictions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="读取 V2 Stage 2 候选，用 Qwen 关键帧主体中心生成逐帧裁剪框",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stage2-dir",
        type=Path,
        default=DEFAULT_STAGE2_DIR,
        help="Stage 2 根目录或其 videos 子目录",
    )
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX, help="测试索引 JSON")
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR, help="原始视频目录")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT, help="最终 predictions.jsonl")
    parser.add_argument(
        "--intervals-out",
        type=Path,
        default=None,
        help="区间诊断 JSONL；默认放在 --out 同目录",
    )
    parser.add_argument(
        "--raw-out",
        type=Path,
        default=None,
        help="主体定位原始响应 JSONL；默认放在 --out 同目录",
    )
    parser.add_argument(
        "--video-id",
        action="append",
        default=[],
        help="只处理指定 video_id，可重复提供",
    )
    parser.add_argument("--num-videos", type=int, default=0, help="只处理索引前 N 项，0 表示不限")
    parser.add_argument(
        "--available-only",
        action="store_true",
        help="只处理存在 Stage 2 _SUCCESS.json 的视频，适合冒烟验证",
    )
    parser.add_argument(
        "--merge-gap-sec",
        type=float,
        default=0.0,
        help="Stage 2 已完成主合并；这里只防御性合并重叠区间，可按需允许额外间隔",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="丢弃 coarse_score 低于该值的候选",
    )
    parser.add_argument("--video-extension", default=".mp4", help="源视频扩展名")
    parser.add_argument(
        "--crop-stride",
        type=int,
        default=15,
        help="主体定位关键帧步长（原视频帧数）；区间首尾帧总会被包含",
    )
    parser.add_argument("--qwen-api-base", default=DEFAULT_QWEN_API_BASE)
    parser.add_argument(
        "--qwen-api-key",
        default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
    )
    parser.add_argument("--qwen-model-name", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--qwen-timeout", type=float, default=300.0)
    parser.add_argument("--qwen-max-retries", type=int, default=2)
    parser.add_argument(
        "--focus-max-new-tokens",
        type=int,
        default=256,
        help="每张关键帧主体中心响应的最大 token 数",
    )
    parser.add_argument(
        "--focus-enable-thinking",
        action="store_true",
        help="主体中心定位开启思考；默认关闭以减少延迟和 JSON 截断",
    )
    parser.add_argument(
        "--focus-thinking-budget",
        type=int,
        default=1024,
        help="开启主体定位思考时的 thinking token 预算",
    )
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument(
        "--focus-failure",
        choices=("center", "error"),
        default="center",
        help="关键帧定位失败策略：继续并在全失败时居中，或立即报错",
    )
    parser.add_argument(
        "--no-healthcheck",
        action="store_true",
        help="启动时不检查 Qwen API 和模型名",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="遇到缺失或无效产物立即失败；默认记录空 predictions 后继续",
    )
    return parser


def main() -> int:
    try:
        run(build_parser().parse_args())
    except (ValidationError, OSError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

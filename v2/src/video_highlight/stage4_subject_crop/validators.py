"""Stage 4 配置、Stage 1/3 输入和逐帧构图输出校验。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from video_highlight.common.atomic_io import read_jsonl
from video_highlight.common.exceptions import ArtifactValidationError


def _read_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ArtifactValidationError(f"缺少文件: {path}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"JSON 根节点不是对象: {path}")
    return value


def list_stage3_video_ids(stage3_dir: str | Path) -> list[str]:
    videos = Path(stage3_dir).resolve() / "videos"
    if not videos.is_dir():
        raise ArtifactValidationError(f"Stage 3 videos 目录不存在: {videos}")
    return sorted(row.name for row in videos.iterdir() if row.is_dir() and not row.name.startswith(".") and (row / "_SUCCESS.json").is_file())


def load_video_inputs(
    stage1_dir: str | Path, stage3_dir: str | Path, video_id: str
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    stage1_video = Path(stage1_dir).resolve() / "videos" / video_id
    stage3_video = Path(stage3_dir).resolve() / "videos" / video_id
    if not (stage1_video / "_SUCCESS.json").is_file():
        raise ArtifactValidationError(f"Stage 1 视频没有成功标记: {stage1_video}")
    if not (stage3_video / "_SUCCESS.json").is_file():
        raise ArtifactValidationError(f"Stage 3 视频没有成功标记: {stage3_video}")
    metadata = _read_object(stage1_video / "metadata.json")
    scenes = read_jsonl(stage1_video / "scenes.jsonl")
    intervals = read_jsonl(stage3_video / "refined_intervals.jsonl")
    frame_count = int(metadata["frame_count"])
    previous_end = -1
    for interval in intervals:
        start, end = int(interval["start_frame"]), int(interval["end_frame"])
        if str(interval.get("video_id")) != video_id or not (0 <= start < end <= frame_count):
            raise ArtifactValidationError(f"Stage 3 区间非法: {interval.get('interval_id')}")
        if start < previous_end:
            raise ArtifactValidationError("Stage 3 区间重叠或未排序")
        previous_end = end
    return metadata, scenes, intervals


def validate_config(config: dict[str, Any]) -> None:
    for key in ("runtime", "tracking", "crop_candidates", "composition", "optimizer", "smoothing"):
        if not isinstance(config.get(key), dict):
            raise ArtifactValidationError(f"Stage 4 配置缺少对象字段: {key}")
    if str(config["runtime"].get("interval_error_policy", "center")) not in {"center", "error"}:
        raise ArtifactValidationError("runtime.interval_error_policy 只能是 center 或 error")


def validate_crops(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    width = int(metadata.get("display_width", metadata["width"]))
    height = int(metadata.get("display_height", metadata["height"]))
    frame_count = int(metadata["frame_count"])
    ratio = metadata.get("targetRatioWH", [16, 9])
    target_w, target_h = float(ratio[0]), float(ratio[1])
    previous_frame = -1
    for row in rows:
        frame = row.get("frame")
        box = row.get("bboxes")
        if not isinstance(frame, int) or not (0 <= frame < frame_count) or frame <= previous_frame:
            raise ArtifactValidationError(f"Stage 4 frame 非法或重复: {frame}")
        previous_frame = frame
        if not isinstance(box, list) or len(box) != 3 or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in box):
            raise ArtifactValidationError(f"frame={frame} 的 bboxes 必须是三个有限数值")
        x, y, crop_w = map(float, box)
        crop_h = crop_w * target_h / target_w
        if not (x >= 0 and y >= 0 and crop_w > 0 and x + crop_w <= width + 1e-6 and y + crop_h <= height + 1e-6):
            raise ArtifactValidationError(f"frame={frame} 构图框越界: {box}")


def validate_stage4_artifacts(video_dir: str | Path) -> dict[str, int]:
    root = Path(video_dir)
    required = ("crops.jsonl", "tracks.jsonl", "diagnostics.jsonl")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ArtifactValidationError(f"Stage 4 缺少产物: {', '.join(missing)}")
    return {"artifact_files": len(required)}

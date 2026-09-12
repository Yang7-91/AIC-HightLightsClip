"""Stage 3.5 的配置、上游输入和持久化结果校验。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from video_highlight.common.atomic_io import read_jsonl
from video_highlight.common.exceptions import ArtifactValidationError

from .frame_sampler import plan_sample_frames


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


def load_video_inputs(stage1_dir: str | Path, stage3_dir: str | Path, video_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    stage1_video = Path(stage1_dir).resolve() / "videos" / video_id
    stage3_video = Path(stage3_dir).resolve() / "videos" / video_id
    if not (stage1_video / "_SUCCESS.json").is_file():
        raise ArtifactValidationError(f"Stage 1 视频没有成功标记: {stage1_video}")
    if not (stage3_video / "_SUCCESS.json").is_file():
        raise ArtifactValidationError(f"Stage 3 视频没有成功标记: {stage3_video}")
    metadata = _read_object(stage1_video / "metadata.json")
    intervals = read_jsonl(stage3_video / "refined_intervals.jsonl")
    if str(metadata.get("video_id")) != video_id:
        raise ArtifactValidationError("Stage 1 metadata.video_id 与目录不一致")
    frame_count = int(metadata["frame_count"])
    previous_end = -1
    for row in intervals:
        start, end = int(row["start_frame"]), int(row["end_frame"])
        if str(row.get("video_id")) != video_id or not 0 <= start < end <= frame_count:
            raise ArtifactValidationError(f"Stage 3 区间非法: {row.get('interval_id')}")
        if start < previous_end:
            raise ArtifactValidationError("Stage 3 区间重叠或未排序")
        previous_end = end
    return metadata, intervals


def validate_config(config: dict[str, Any]) -> None:
    for key in ("api", "generation", "sampling", "parsing", "runtime"):
        if not isinstance(config.get(key), dict):
            raise ArtifactValidationError(f"Stage 3.5 配置缺少对象字段: {key}")
    mode = str(config["runtime"].get("mode", "predict"))
    if mode not in {"predict", "passthrough"}:
        raise ArtifactValidationError("runtime.mode 只能是 predict 或 passthrough")
    sample_fps = float(config["sampling"].get("fps", 2.0))
    if not math.isfinite(sample_fps) or sample_fps <= 0:
        raise ArtifactValidationError("sampling.fps 必须是正有限数")
    quality = int(config["sampling"].get("jpeg_quality", 85))
    if not 1 <= quality <= 100:
        raise ArtifactValidationError("sampling.jpeg_quality 必须在 1..100")


def validate_points(points: list[dict[str, Any]], intervals: list[dict[str, Any]], metadata: dict[str, Any], sample_fps: float) -> None:
    bounds = {str(row["interval_id"]): (int(row["start_frame"]), int(row["end_frame"])) for row in intervals}
    seen: set[tuple[str, int]] = set()
    actual_by_interval: dict[str, list[tuple[int, int]]] = {interval_id: [] for interval_id in bounds}
    for row in points:
        interval_id, frame = str(row["interval_id"]), int(row["frame"])
        if interval_id not in bounds or not bounds[interval_id][0] <= frame < bounds[interval_id][1]:
            raise ArtifactValidationError(f"主体点帧越界: {interval_id}/{frame}")
        key = (interval_id, int(row["sample_index"]))
        if key in seen:
            raise ArtifactValidationError(f"重复主体点样本: {key}")
        seen.add(key)
        actual_by_interval[interval_id].append((int(row["sample_index"]), frame))
        point = row.get("subject_point")
        if point is not None and (not isinstance(point, list) or len(point) != 2 or not all(0 <= float(value) <= 1 for value in point)):
            raise ArtifactValidationError(f"非法归一化主体点: {key}")
    if int(metadata["frame_count"]) <= 0:
        raise ArtifactValidationError("frame_count 必须大于 0")
    fps = float(metadata["fps"])
    for interval_id, (start, end) in bounds.items():
        expected_frames = plan_sample_frames(start, end, fps, sample_fps)
        actual = sorted(actual_by_interval[interval_id])
        if actual != list(enumerate(expected_frames)):
            raise ArtifactValidationError(f"{interval_id} 的主体点未完整覆盖计划采样帧")


def validate_artifacts(video_dir: str | Path) -> dict[str, int]:
    root = Path(video_dir)
    required = ("enriched_intervals.jsonl", "subject_points.jsonl", "requests.jsonl", "raw_responses.jsonl", "diagnostics.jsonl")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ArtifactValidationError(f"Stage 3.5 缺少产物: {', '.join(missing)}")
    return {"artifact_files": len(required)}

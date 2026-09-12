"""Stage 3 配置、上游产物和输出契约校验。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from video_highlight.common.atomic_io import read_jsonl
from video_highlight.common.exceptions import ArtifactValidationError


def read_json_object(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise ArtifactValidationError(f"缺少文件: {target}")
    value = json.loads(target.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"JSON 根节点不是对象: {target}")
    return value


def list_stage2_video_ids(stage2_dir: str | Path) -> list[str]:
    videos_dir = Path(stage2_dir).resolve() / "videos"
    if not videos_dir.is_dir():
        raise ArtifactValidationError(f"Stage 2 videos 目录不存在: {videos_dir}")
    return sorted(
        row.name for row in videos_dir.iterdir()
        if row.is_dir() and not row.name.startswith(".") and (row / "_SUCCESS.json").is_file()
    )


def load_video_inputs(
    stage1_dir: str | Path, stage2_dir: str | Path, video_id: str
) -> tuple[Path, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    stage1_video = Path(stage1_dir).resolve() / "videos" / video_id
    stage2_video = Path(stage2_dir).resolve() / "videos" / video_id
    if not (stage1_video / "_SUCCESS.json").is_file():
        raise ArtifactValidationError(f"Stage 1 视频没有成功标记: {stage1_video}")
    if not (stage2_video / "_SUCCESS.json").is_file():
        raise ArtifactValidationError(f"Stage 2 视频没有成功标记: {stage2_video}")
    metadata = read_json_object(stage1_video / "metadata.json")
    if str(metadata.get("video_id")) != video_id:
        raise ArtifactValidationError(f"metadata.video_id 与目录不一致: {stage1_video}")
    scenes = read_jsonl(stage1_video / "scenes.jsonl")
    candidates = read_jsonl(stage2_video / "candidates.jsonl")
    duration = float(metadata["duration_sec"])
    for row in candidates:
        if str(row.get("video_id")) != video_id:
            raise ArtifactValidationError(f"Stage 2 candidate.video_id 不一致: {row.get('candidate_id')}")
        start, end = float(row["start_sec"]), float(row["end_sec"])
        if not (0.0 <= start < end <= duration + 1e-6):
            raise ArtifactValidationError(f"Stage 2 候选越界: {row.get('candidate_id')} [{start},{end})")
    return stage1_video, metadata, scenes, candidates


def validate_config(config: dict[str, Any]) -> None:
    for key in ("runtime", "decode", "temporal", "boundary", "empty_gate", "merging"):
        if not isinstance(config.get(key), dict):
            raise ArtifactValidationError(f"Stage 3 配置缺少对象字段: {key}")
    if str(config["runtime"].get("mode", "refine")) not in {"refine", "passthrough"}:
        raise ArtifactValidationError("runtime.mode 只能是 refine 或 passthrough")


def validate_intervals(intervals: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    frame_count = int(metadata["frame_count"])
    seen: set[str] = set()
    previous_end = -1
    for row in intervals:
        if "subject_point" in row:
            raise ArtifactValidationError("Stage 3 refined interval 禁止包含 subject_point")
        interval_id = str(row["interval_id"])
        if interval_id in seen:
            raise ArtifactValidationError(f"重复 interval_id: {interval_id}")
        seen.add(interval_id)
        start, end = int(row["start_frame"]), int(row["end_frame"])
        if not (0 <= start < end <= frame_count):
            raise ArtifactValidationError(f"Stage 3 帧区间越界: {interval_id} [{start},{end})")
        if start < previous_end:
            raise ArtifactValidationError("Stage 3 区间重叠或未按帧号排序")
        previous_end = end
        for key in ("start_sec", "end_sec", "coarse_score", "temporal_score"):
            if not math.isfinite(float(row[key])):
                raise ArtifactValidationError(f"{interval_id}.{key} 不是有限数")


def validate_stage3_artifacts(video_dir: str | Path) -> dict[str, int]:
    root = Path(video_dir)
    required = ("refined_intervals.jsonl", "diagnostics.jsonl")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ArtifactValidationError(f"Stage 3 缺少产物: {', '.join(missing)}")
    return {"artifact_files": len(required)}

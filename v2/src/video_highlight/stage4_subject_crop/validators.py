"""Stage 4 配置、Stage 1/3.5 输入和逐帧构图输出校验。"""

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


def list_stage3_5_video_ids(stage3_5_dir: str | Path) -> list[str]:
    videos = Path(stage3_5_dir).resolve() / "videos"
    if not videos.is_dir():
        raise ArtifactValidationError(f"Stage 3.5 videos 目录不存在: {videos}")
    return sorted((row.name for row in videos.iterdir() if row.is_dir() and not row.name.startswith(".") and (row / "_SUCCESS.json").is_file()),
                  key=lambda name: int(name) if name.isdigit() else name,) # 保证有序排列


def load_video_inputs(
    stage1_dir: str | Path, stage3_5_dir: str | Path, video_id: str,project_paths_config:dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    加载stage4的输入数据，同时进行数据校验。数据校验时，以enriched_intervals.jsonl为准，逐项核对subject_points.jsonl数据合法性

    Returns:
        metadata: stage1获取的视频的元信息
        scenes: stage1获取的镜头信息
        intervals: stage3_5获取的个高光区间的所有采样帧的中心主体预测信息
    """
    stage1_video = Path(stage1_dir).resolve() / "videos" / video_id
    stage3_5_video = Path(stage3_5_dir).resolve() / "videos" / video_id
    if not (stage1_video / "_SUCCESS.json").is_file():
        raise ArtifactValidationError(f"Stage 1 视频没有成功标记: {stage1_video}")
    if not (stage3_5_video / "_SUCCESS.json").is_file():
        raise ArtifactValidationError(f"Stage 3.5 视频没有成功标记: {stage3_5_video}")
    metadata = _read_object(stage1_video / "metadata.json")
    # 当切换环境后，视频路径发生改变，此时默认使用配置文件路径，默认mp4
    if not Path(metadata["source_path"]).is_file():
        metadata["source_path"] = str(Path(project_paths_config["video_root"]) / (video_id + ".mp4"))
    scenes = read_jsonl(stage1_video / "scenes.jsonl")
    intervals = read_jsonl(stage3_5_video / "enriched_intervals.jsonl")
    point_rows = read_jsonl(stage3_5_video / "subject_points.jsonl")
    points_by_interval: dict[str, list[dict[str, Any]]] = {}
    for point in point_rows:
        interval_id = str(point.get("interval_id", ""))
        if str(point.get("video_id")) != video_id:
            raise ArtifactValidationError(f"Stage 3.5 主体点 video_id 不一致: {interval_id}")
        value = point.get("subject_point")
        if value is not None and (not isinstance(value, list) or len(value) != 2 or not all(0.0 <= float(axis) <= 1.0 for axis in value)):
            raise ArtifactValidationError(f"Stage 3.5 主体点非法: {interval_id}/{point.get('frame')}")
        points_by_interval.setdefault(interval_id, []).append(point)
    frame_count = int(metadata["frame_count"])
    previous_end = -1
    for interval in intervals:
        start, end = int(interval["start_frame"]), int(interval["end_frame"])
        if str(interval.get("video_id")) != video_id or not (0 <= start < end <= frame_count):
            raise ArtifactValidationError(f"Stage 3.5 区间非法: {interval.get('interval_id')}")
        if start < previous_end:
            raise ArtifactValidationError("Stage 3.5 区间重叠或未排序")
        previous_end = end
        interval_id = str(interval["interval_id"])
        interval["subject_points"] = sorted(points_by_interval.pop(interval_id, []), key=lambda row: int(row["frame"]))
        expected_count = int(interval.get("subject_point_sample_count", len(interval["subject_points"])))
        if len(interval["subject_points"]) != expected_count:
            raise ArtifactValidationError(f"Stage 3.5 主体点数量与区间摘要不一致: {interval_id}")
        sample_indices = [int(point["sample_index"]) for point in interval["subject_points"]]
        if sample_indices != list(range(len(sample_indices))):
            raise ArtifactValidationError(f"Stage 3.5 sample_index 不连续或顺序异常: {interval_id}")
        for point in interval["subject_points"]:
            if not start <= int(point["frame"]) < end:
                raise ArtifactValidationError(f"Stage 3.5 主体点不属于区间: {interval_id}/{point['frame']}")
    if points_by_interval:
        raise ArtifactValidationError(f"Stage 3.5 主体点引用未知区间: {sorted(points_by_interval)}")
    return metadata, scenes, intervals


def validate_config(config: dict[str, Any]) -> None:
    for key in ("runtime", "tracking", "crop_candidates", "composition", "optimizer", "smoothing"):
        if not isinstance(config.get(key), dict):
            raise ArtifactValidationError(f"Stage 4 配置缺少对象字段: {key}")
    if str(config["runtime"].get("interval_error_policy", "center")) not in {"center", "error"}:
        raise ArtifactValidationError("runtime.interval_error_policy 只能是 center 或 error")
    fixed_maximum = config["crop_candidates"].get("fixed_maximum", False)
    if not isinstance(fixed_maximum, bool):
        raise ArtifactValidationError("crop_candidates.fixed_maximum 必须是布尔值")
    bypass_interpolated = config["smoothing"].get("bypass_for_interpolated_qwen", True)
    if not isinstance(bypass_interpolated, bool):
        raise ArtifactValidationError("smoothing.bypass_for_interpolated_qwen 必须是布尔值")
    distance = float(config["tracking"].get("anchor_max_center_distance_ratio", 0.20))
    if not 0.0 <= distance <= 1.0:
        raise ArtifactValidationError("tracking.anchor_max_center_distance_ratio 必须在 [0,1] 内")


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

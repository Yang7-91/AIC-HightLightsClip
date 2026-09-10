"""
根据固定窗口和镜头边界规划 Stage 2 分析片段。
三次处理：
    1.固定长度分片
    2.取固定长度后，尝试在分片终点附近的镜头切换点滑动
    3.每个分片允许部分重叠
"""

from __future__ import annotations

from typing import Any

from video_highlight.contracts.schema_versions import STAGE1_SCHEMA_VERSION


def _nearest_boundary(target: float, boundaries: list[float], tolerance: float, minimum: float) -> float | None:
    candidates = [value for value in boundaries if value >= minimum and abs(value - target) <= tolerance]
    return min(candidates, key=lambda value: (abs(value - target), value)) if candidates else None


def plan_segments(
    video_id: str,
    duration_sec: float,
    scenes: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    window = float(config.get("window_sec", 24.0))
    overlap = window * float(config.get("overlap_ratio", 0.25))
    snap = float(config.get("boundary_snap_sec", 2.0))
    minimum = float(config.get("min_window_sec", 6.0))
    if duration_sec <= 0 or window <= 0 or overlap < 0 or overlap >= window:
        raise ValueError("片段窗口配置非法")
    boundaries = [float(scene["end_sec"]) for scene in scenes[:-1]] # 排除视频最后一个镜头分隔区间
    result: list[dict[str, Any]] = []
    start = 0.0
    while start < duration_sec - 1e-6:
        desired_end = min(duration_sec, start + window)
        actual_end = desired_end
        if desired_end < duration_sec: # 如果此次固定分片终点还没有到视频终点
            snapped = _nearest_boundary(desired_end, boundaries, snap, start + minimum) # 尝试在分片终点向附近的镜头切换点滑动
            if snapped is not None:
                actual_end = snapped
        sample_ids = [
            int(sample["sample_id"])
            for sample in samples
            if start <= float(sample["timestamp_sec"]) < actual_end + (1e-6 if actual_end >= duration_sec else 0.0)
        ]
        scene_ids = [
            int(scene["scene_id"])
            for scene in scenes
            if float(scene["end_sec"]) > start and float(scene["start_sec"]) < actual_end
        ]
        result.append(
            {
                "schema_version": STAGE1_SCHEMA_VERSION,
                "video_id": video_id,
                "segment_id": len(result),
                "start_sec": start,
                "end_sec": actual_end,
                "sample_ids": sample_ids,
                "scene_ids": scene_ids,
            }
        )
        if actual_end >= duration_sec - 1e-6:
            break
        next_start = max(0.0, actual_end - overlap)
        start = next_start if next_start > start + 1e-6 else actual_end
    return result

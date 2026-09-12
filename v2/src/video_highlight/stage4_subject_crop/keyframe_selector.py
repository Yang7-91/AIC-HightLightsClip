"""从 Stage 1 镜头边界生成区间内的跟踪重初始化帧。"""

from __future__ import annotations

from typing import Any


def reinitialization_frames(
    interval: dict[str, Any], scenes: list[dict[str, Any]], stride_frames: int
) -> set[int]:
    start, end = int(interval["start_frame"]), int(interval["end_frame"])
    frames = {start}
    for scene in scenes:
        scene_start = int(scene.get("start_frame", -1))
        if start < scene_start < end:
            frames.add(scene_start)
    if stride_frames > 0:
        frames.update(range(start + stride_frames, end, stride_frames))
    return frames

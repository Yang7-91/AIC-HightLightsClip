"""从 Stage 1 镜头边界生成区间内的跟踪重初始化帧。"""

from __future__ import annotations

from typing import Any


def interval_scene_spans(
    interval: dict[str, Any], scenes: list[dict[str, Any]]
) -> list[tuple[int, int]]:
    """按 Stage 1 硬切边界返回区间内左闭右开的镜头子段。

    跟踪和构图必须使用同一组边界。把切分逻辑放在这里，可以避免出现“构图已经
    按镜头重置，但跟踪状态仍跨镜头传播”的隐蔽错误。

    Returns:
        根据硬镜头镜头边界切割后的帧区间元组
    """

    start, end = int(interval["start_frame"]), int(interval["end_frame"])
    cuts = sorted(
        {
            int(scene["start_frame"])
            for scene in scenes
            if scene.get("start_frame") is not None
            and start < int(scene["start_frame"]) < end # 查看高光区间里有多少个镜头起始点，这些镜头起始点就是边界
        }
    )
    boundaries = [start, *cuts, end]
    return list(zip(boundaries[:-1], boundaries[1:], strict=True))


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

"""秒级/精细采样边界到原始视频半开帧区间的确定性映射。"""

from __future__ import annotations

import math
from typing import Any

from .boundary_decoder import BoundaryDecision
from .candidate_decoder import DecodedCandidate


def map_passthrough(candidate: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    """不改变 Stage 2 秒边界，只补充 Stage 3 必需的原始帧坐标。"""

    fps = float(metadata["fps"])
    frame_count = int(metadata["frame_count"])
    start_sec = float(candidate["start_sec"])
    end_sec = float(candidate["end_sec"])
    start_frame = max(0, min(frame_count - 1, int(math.floor(start_sec * fps + 1e-9))))
    end_frame = max(start_frame + 1, min(frame_count, int(math.ceil(end_sec * fps - 1e-9))))
    return {
        "start_sec": start_sec,
        "end_sec": end_sec,
        "start_frame": start_frame,
        "end_frame": end_frame,
    }


def map_refined(
    decoded: DecodedCandidate,
    decision: BoundaryDecision,
    candidate: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    fps = float(metadata["fps"])
    frame_count = int(metadata["frame_count"])
    start_frame = int(decoded.frame_indices[decision.start_index])
    sample_span_frames = max(1, int(round(fps / decoded.sample_fps)))
    end_frame = int(decoded.frame_indices[decision.end_index]) + sample_span_frames
    start_frame = max(0, min(frame_count - 1, start_frame))
    end_frame = max(start_frame + 1, min(frame_count, end_frame))
    return {
        "start_sec": max(float(candidate["start_sec"]), float(decoded.timestamps_sec[decision.start_index])),
        "end_sec": min(float(candidate["end_sec"]), end_frame / fps),
        "start_frame": start_frame,
        "end_frame": end_frame,
    }

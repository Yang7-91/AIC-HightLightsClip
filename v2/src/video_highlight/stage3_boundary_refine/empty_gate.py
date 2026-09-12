"""无需训练权重的保守视频级无高光门控。"""

from __future__ import annotations

from typing import Any


def apply_empty_gate(
    intervals: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not intervals:
        return [], {"applied": bool(config.get("enabled", True)), "is_empty": True, "reason": "no_candidates"}
    if not bool(config.get("enabled", True)):
        return intervals, {"applied": False, "is_empty": False, "reason": "disabled"}
    max_coarse = max(float(row.get("coarse_score", 0.0)) for row in intervals)
    max_temporal = max(float(row.get("temporal_score", 0.0)) for row in intervals)
    is_empty = (
        max_coarse < float(config.get("max_coarse_threshold", 0.30))
        and max_temporal < float(config.get("max_temporal_threshold", 0.25))
    )
    return ([] if is_empty else intervals), {
        "applied": True,
        "is_empty": is_empty,
        "reason": "scores_below_thresholds" if is_empty else "candidate_retained",
        "max_coarse_score": max_coarse,
        "max_temporal_score": max_temporal,
    }

"""合并边界细化后仍重叠或仅有极小间隔的区间。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def _merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    anchor = left if float(left.get("coarse_score", 0.0)) >= float(right.get("coarse_score", 0.0)) else right
    result = deepcopy(anchor)
    result["start_frame"] = min(int(left["start_frame"]), int(right["start_frame"]))
    result["end_frame"] = max(int(left["end_frame"]), int(right["end_frame"]))
    result["start_sec"] = min(float(left["start_sec"]), float(right["start_sec"]))
    result["end_sec"] = max(float(left["end_sec"]), float(right["end_sec"]))
    result["source_candidate_ids"] = sorted(set(left["source_candidate_ids"] + right["source_candidate_ids"]))
    result["source_segment_ids"] = sorted(set(left.get("source_segment_ids", []) + right.get("source_segment_ids", [])))
    result["coarse_score"] = max(float(left.get("coarse_score", 0.0)), float(right.get("coarse_score", 0.0)))
    result["temporal_score"] = max(float(left.get("temporal_score", 0.0)), float(right.get("temporal_score", 0.0)))
    reasons = [str(item.get("reason", "")).strip() for item in (left, right)]
    result["reason"] = " | ".join(dict.fromkeys(value for value in reasons if value))
    result["refine_status"] = "merged"
    return result


def merge_intervals(
    video_id: str,
    intervals: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    ordered = sorted((deepcopy(row) for row in intervals), key=lambda row: (int(row["start_frame"]), int(row["end_frame"])))
    if bool(config.get("enabled", True)):
        max_gap = max(0, int(config.get("max_gap_frames", 2)))
        require_same_category = bool(config.get("require_same_category", True))
        merged: list[dict[str, Any]] = []
        for row in ordered:
            compatible = not require_same_category or row.get("category") == (merged[-1].get("category") if merged else None)
            if merged and compatible and int(row["start_frame"]) <= int(merged[-1]["end_frame"]) + max_gap:
                merged[-1] = _merge(merged[-1], row)
            else:
                merged.append(row)
        ordered = merged
    for index, row in enumerate(ordered):
        row["interval_id"] = f"{video_id}_interval_{index:04d}"
        row["frame_count"] = int(row["end_frame"]) - int(row["start_frame"])
        row["frame_interval"] = "[start_frame,end_frame)"
    return ordered

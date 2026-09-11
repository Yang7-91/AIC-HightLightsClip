"""扩展、合并重叠候选并生成确定性 candidate_id。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def _merge_pair(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    anchor = left if float(left["coarse_score"]) >= float(right["coarse_score"]) else right
    merged = deepcopy(anchor)
    merged["start_sec"] = min(float(left["start_sec"]), float(right["start_sec"]))
    merged["end_sec"] = max(float(left["end_sec"]), float(right["end_sec"]))
    merged["source_segment_ids"] = sorted(
        {int(value) for value in left["source_segment_ids"] + right["source_segment_ids"]}
    )
    for key in (
        "coarse_score",
        "semantic_score",
        "event_score",
        "audio_score",
        "quality_score",
        "stability_score",
    ):
        merged[key] = max(float(left[key]), float(right[key]))
    reasons = [str(value).strip() for value in (left.get("reason"), right.get("reason")) if str(value or "").strip()]
    merged["reason"] = " | ".join(dict.fromkeys(reasons))
    return merged


def merge_candidates(
    video_id: str,
    candidates: list[dict[str, Any]],
    duration_sec: float,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    直接粗暴合并，只要满足重叠区间要求，就直接合并，各种评估分数取两者最大值
    """
    before = float(config.get("expand_before_sec", 1.0))
    after = float(config.get("expand_after_sec", 1.0))
    max_gap = float(config.get("max_merge_gap_sec", 1.0))
    min_duration = float(config.get("min_duration_sec", 0.5))
    expanded: list[dict[str, Any]] = []
    for candidate in candidates:
        item = deepcopy(candidate)
        item["start_sec"] = max(0.0, float(item["start_sec"]) - before)
        item["end_sec"] = min(duration_sec, float(item["end_sec"]) + after)
        if item["end_sec"] - item["start_sec"] >= min_duration:
            expanded.append(item)
    expanded.sort(key=lambda item: (float(item["start_sec"]), float(item["end_sec"])))
    merged: list[dict[str, Any]] = []
    for candidate in expanded:
        if merged and float(candidate["start_sec"]) <= float(merged[-1]["end_sec"]) + max_gap:
            merged[-1] = _merge_pair(merged[-1], candidate)
        else:
            merged.append(candidate)
    for index, candidate in enumerate(merged):
        candidate["candidate_id"] = f"{video_id}_candidate_{index:04d}"
    return merged

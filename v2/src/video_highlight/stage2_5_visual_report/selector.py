"""MHS-VIS-0 示例选择器：挑出最适合展示"大段疑似多高光"的候选。"""

from __future__ import annotations

from typing import Any, Mapping

SELECTION_VERSION = "mhs_vis0_selector.v1"


def _duration_score(duration_sec: float) -> float:
    return min(max(duration_sec, 0.0), 60.0) / 60.0


def _peak_score(num_peaks: int) -> float:
    return min(num_peaks, 5) / 5.0


def select_visual_examples(
    candidates_by_video: Mapping[str, list[dict[str, Any]]],
    durations_by_video: Mapping[str, float],
    config: Mapping[str, Any],
    *,
    peaks_by_candidate: Mapping[str, list[float]] | None = None,
    subsegments_by_candidate: Mapping[str, list[dict[str, Any]]] | None = None,
    manual_video_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """确定性选择展示样本。

    优先级：时长长 / 多峰（motion 峰代理）/ proposed subsegments > 1 / coverage 大；
    同一视频最多 ``max_candidates_per_video`` 个，总视频数最多 ``max_videos`` 个。
    """
    selection_cfg = config["selection"]
    max_videos = int(selection_cfg["max_videos"])
    max_per_video = int(selection_cfg["max_candidates_per_video"])
    min_duration = float(selection_cfg["min_candidate_duration_sec"])
    allow_manual = [str(item) for item in (manual_video_ids or selection_cfg.get("allow_manual_video_ids", []))]
    peaks_by_candidate = peaks_by_candidate or {}
    subsegments_by_candidate = subsegments_by_candidate or {}

    def candidate_score(candidate: dict[str, Any], video_duration: float) -> tuple[float, list[str]]:
        reasons: list[str] = []
        score = 0.0
        if candidate["duration_sec"] >= min_duration:
            score += 0.5
            reasons.append("long_candidate")
        score += 0.3 * _duration_score(candidate["duration_sec"])
        num_peaks = len(peaks_by_candidate.get(candidate["candidate_id"], []))
        if num_peaks >= 2:
            score += 0.5 * _peak_score(num_peaks)
            reasons.append("multi_peak")
        subsegments = subsegments_by_candidate.get(candidate["candidate_id"], [])
        if len(subsegments) > 1:
            score += 0.3
            reasons.append("multi_subsegment")
        if video_duration > 0:
            coverage = candidate["duration_sec"] / video_duration
            if coverage >= 0.5:
                score += 0.2
                reasons.append("high_coverage")
            score += 0.1 * min(coverage, 1.0)
        return score, reasons

    scored: list[dict[str, Any]] = []
    for video_id in sorted(candidates_by_video):
        video_duration = float(durations_by_video.get(video_id, 0.0))
        ordered = sorted(
            candidates_by_video[video_id],
            key=lambda item: (item["start_sec"], item["candidate_id"]),
        )
        for candidate in ordered:
            score, reasons = candidate_score(candidate, video_duration)
            scored.append(
                {
                    "video_id": video_id,
                    "candidate_id": candidate["candidate_id"],
                    "candidate_start_sec": candidate["start_sec"],
                    "candidate_end_sec": candidate["end_sec"],
                    "duration_sec": candidate["duration_sec"],
                    "score": candidate.get("score"),
                    "coverage_ratio": (
                        candidate["duration_sec"] / video_duration if video_duration > 0 else 0.0
                    ),
                    "num_motion_peaks": len(peaks_by_candidate.get(candidate["candidate_id"], [])),
                    "num_proposed_subsegments": len(
                        subsegments_by_candidate.get(candidate["candidate_id"], [])
                    ),
                    "_score": score,
                    "reason": "+".join(reasons) if reasons else "fallback_selection",
                }
            )
    if allow_manual:
        manual_rank = {video_id: index for index, video_id in enumerate(allow_manual)}
        scored = [row for row in scored if row["video_id"] in manual_rank]
        scored.sort(
            key=lambda row: (
                manual_rank[row["video_id"]],
                -row["_score"],
                row["video_id"],
                row["candidate_start_sec"],
                row["candidate_id"],
            )
        )
    else:
        scored.sort(
            key=lambda row: (
                -row["_score"],
                row["video_id"],
                row["candidate_start_sec"],
                row["candidate_id"],
            )
        )
    selected: list[dict[str, Any]] = []
    per_video: dict[str, int] = {}
    for row in scored:
        if row["video_id"] in per_video and per_video[row["video_id"]] >= max_per_video:
            continue
        if row["video_id"] not in per_video and len(per_video) >= max_videos:
            continue
        per_video[row["video_id"]] = per_video.get(row["video_id"], 0) + 1
        row = dict(row)
        row.pop("_score", None)
        selected.append(row)
    # 保持评分优先顺序（最典型的例子在前）；同分已在 scored 排序中按 (video_id, start) 稳定。
    return selected

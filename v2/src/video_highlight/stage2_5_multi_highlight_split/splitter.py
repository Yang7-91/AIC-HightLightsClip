"""MHS-1 拆分器：eventness 峰-谷分析 + 保守切分 + guards。"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


class SplitterError(RuntimeError):
    """eventness 输入非法或拆分计算失败。"""


def find_event_peaks(
    bins: list[Mapping[str, Any]],
    *,
    peak_min_distance_sec: float,
    relative_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """在 eventness 曲线上找局部峰（间隔受限、阈值受限）。"""
    if len(bins) < 3:
        return []
    scores = [float(item["eventness_score"]) for item in bins]
    maximum = max(scores)
    if maximum <= 1e-9:
        return []
    threshold = max(maximum * relative_threshold, float(np.median(scores)))
    peaks: list[dict[str, Any]] = []
    for index in range(1, len(bins) - 1):
        value = scores[index]
        if value < threshold:
            continue
        if value >= scores[index - 1] and value >= scores[index + 1]:
            t_sec = float(bins[index]["start_sec"])
            if peaks and t_sec - peaks[-1]["t_sec"] < peak_min_distance_sec:
                # 保留更高者
                if value > peaks[-1]["score"]:
                    peaks[-1] = {"t_sec": t_sec, "score": value, "bin_index": index}
                continue
            peaks.append({"t_sec": t_sec, "score": value, "bin_index": index})
    return peaks


def find_valleys_between_peaks(
    bins: list[Mapping[str, Any]],
    peaks: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """相邻峰之间的最低点（含分位阈值校验）。"""
    valleys: list[dict[str, Any]] = []
    for left, right in zip(peaks, peaks[1:]):
        lo = int(left["bin_index"])
        hi = int(right["bin_index"])
        if hi - lo < 2:
            continue
        segment = bins[lo + 1 : hi]
        if not segment:
            continue
        scores = [float(item["eventness_score"]) for item in segment]
        min_index = int(np.argmin(scores))
        valleys.append(
            {
                "t_sec": float(segment[min_index]["start_sec"]),
                "score": float(scores[min_index]),
                "bin_index": lo + 1 + min_index,
                "left_peak": float(left["t_sec"]),
                "right_peak": float(right["t_sec"]),
            }
        )
    return valleys


def propose_subsegments_from_peaks(
    candidate_start: float,
    candidate_end: float,
    peaks: list[Mapping[str, Any]],
    valleys: list[Mapping[str, Any]],
    *,
    valley_quantile_max: float,
    all_scores: list[float],
    merge_gap_sec: float = 0.0,
    valley_drop_sec: float = 0.0,
) -> list[dict[str, float]]:
    """在显著低谷处切分候选：每个切点两侧各丢弃 ``valley_drop_sec / 2``。

    ``merge_gap_sec`` 用于合并距离过近的切点（只保留一个）。
    返回的每个子段只保留事件内容，谷区不再被覆盖。
    """
    if not peaks or not valleys:
        return []
    if not all_scores:
        return []
    valley_threshold = float(np.quantile(all_scores, valley_quantile_max))
    cut_points = sorted(
        valley["t_sec"] for valley in valleys if valley["score"] <= valley_threshold
    )
    deduped: list[float] = []
    for cut in cut_points:
        if deduped and cut - deduped[-1] < merge_gap_sec:
            continue
        deduped.append(cut)
    if not deduped:
        return []
    half_drop = max(0.0, valley_drop_sec) / 2.0
    edges = [candidate_start] + deduped + [candidate_end]
    segments: list[dict[str, float]] = []
    for index, (lo, hi) in enumerate(zip(edges, edges[1:])):
        seg_start = lo + (half_drop if index > 0 else 0.0)
        seg_end = hi - (half_drop if index < len(edges) - 2 else 0.0)
        if seg_end - seg_start <= 1e-9:
            continue
        segments.append({"start_sec": float(seg_start), "end_sec": float(seg_end)})
    return segments


def apply_split_guards(
    parent_start: float,
    parent_end: float,
    segments: list[dict[str, float]],
    *,
    pad_sec: float,
    merge_gap_sec: float,
    min_subsegment_duration_sec: float,
    max_subsegments_per_candidate: int,
    max_coverage_reduction_ratio: float,
    video_duration: float,
) -> dict[str, Any]:
    """对拟拆分结果应用全部 guards；不确定时返回 identity。

    返回决策 dict：``{"action": "split"|"fallback_identity"|"identity",
    "segments": [...], "fallback_reason": str|None, "coverage_reduction_ratio": float}``
    """
    parent_duration = parent_end - parent_start
    identity = {
        "action": "identity",
        "segments": [{"start_sec": parent_start, "end_sec": parent_end}],
        "fallback_reason": None,
        "coverage_reduction_ratio": 0.0,
    }
    if parent_duration <= 0 or not segments:
        return identity

    adjusted: list[dict[str, float]] = []
    for segment in segments:
        start = segment["start_sec"]
        end = segment["end_sec"]
        # pad 只作用于候选外边界（切点边由 valley 位置决定，不加 pad，避免谷区被回填）。
        if abs(start - parent_start) <= 1e-9:
            start = max(parent_start, start - pad_sec)
        if abs(end - parent_end) <= 1e-9:
            end = min(parent_end, end + pad_sec)
        adjusted.append({"start_sec": start, "end_sec": end})
    adjusted.sort(key=lambda item: item["start_sec"])
    # 丢弃过短段；保持分离（不跨丢弃区合并，避免回填谷区）。
    kept = [
        segment
        for segment in adjusted
        if segment["end_sec"] - segment["start_sec"] >= min_subsegment_duration_sec
    ]
    if not kept:
        return {
            "action": "fallback_identity",
            "segments": [{"start_sec": parent_start, "end_sec": parent_end}],
            "fallback_reason": "all_subsegments_below_min_duration",
            "coverage_reduction_ratio": 0.0,
        }
    if len(kept) > max_subsegments_per_candidate:
        kept = kept[:max_subsegments_per_candidate]
    if len(kept) < 2:
        return {
            "action": "fallback_identity",
            "segments": [{"start_sec": parent_start, "end_sec": parent_end}],
            "fallback_reason": "fewer_than_two_valid_subsegments",
            "coverage_reduction_ratio": 0.0,
        }
    covered = sum(segment["end_sec"] - segment["start_sec"] for segment in kept)
    coverage_reduction = max(0.0, 1.0 - covered / parent_duration)
    if coverage_reduction > max_coverage_reduction_ratio + 1e-9:
        return {
            "action": "fallback_identity",
            "segments": [{"start_sec": parent_start, "end_sec": parent_end}],
            "fallback_reason": "coverage_reduction_exceeds_cap",
            "coverage_reduction_ratio": coverage_reduction,
        }
    for segment in kept:
        if segment["start_sec"] < 0 or segment["end_sec"] > video_duration + 1e-9:
            return {
                "action": "fallback_identity",
                "segments": [{"start_sec": parent_start, "end_sec": parent_end}],
                "fallback_reason": "subsegment_outside_video",
                "coverage_reduction_ratio": coverage_reduction,
            }
        if segment["end_sec"] - segment["start_sec"] <= 1e-9:
            return {
                "action": "fallback_identity",
                "segments": [{"start_sec": parent_start, "end_sec": parent_end}],
                "fallback_reason": "empty_subsegment",
                "coverage_reduction_ratio": coverage_reduction,
            }
    return {
        "action": "split",
        "segments": kept,
        "fallback_reason": None,
        "coverage_reduction_ratio": coverage_reduction,
    }


def propose_split(
    candidate: Mapping[str, Any],
    bins: list[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    config_name: str,
    video_duration: float,
) -> dict[str, Any]:
    """对单个候选运行完整拆分决策（不使用 weak-reference）。"""
    filter_cfg = config["candidate_filter"]
    split_cfg = config["splitter"]["configs"][config_name]
    parent_start = float(candidate["start_sec"])
    parent_end = float(candidate["end_sec"])
    duration = parent_end - parent_start
    base_decision = {
        "action": "identity",
        "segments": [{"start_sec": parent_start, "end_sec": parent_end}],
        "fallback_reason": None,
        "coverage_reduction_ratio": 0.0,
        "num_peaks": 0,
        "num_valleys": 0,
    }
    if duration < float(filter_cfg["min_candidate_duration_sec"]):
        base_decision["fallback_reason"] = "candidate_too_short"
        return base_decision
    if len(bins) < 3:
        base_decision["fallback_reason"] = "too_few_bins"
        return base_decision
    scores = [float(item["eventness_score"]) for item in bins]
    if max(scores) - min(scores) <= 1e-9:
        base_decision["fallback_reason"] = "constant_eventness"
        return base_decision
    peaks = find_event_peaks(
        bins,
        peak_min_distance_sec=float(split_cfg["peak_min_distance_sec"]),
    )
    base_decision["num_peaks"] = len(peaks)
    if len(peaks) < int(filter_cfg["min_peak_count_for_split"]):
        base_decision["fallback_reason"] = "insufficient_peaks"
        return base_decision
    valleys = find_valleys_between_peaks(bins, peaks)
    base_decision["num_valleys"] = len(valleys)
    proposed = propose_subsegments_from_peaks(
        parent_start,
        parent_end,
        peaks,
        valleys,
        valley_quantile_max=float(split_cfg["valley_quantile_max"]),
        all_scores=scores,
        merge_gap_sec=float(split_cfg["merge_gap_sec"]),
        valley_drop_sec=float(split_cfg["merge_gap_sec"]),
    )
    if len(proposed) < 2:
        base_decision["fallback_reason"] = "valley_not_prominent"
        return base_decision
    decision = apply_split_guards(
        parent_start,
        parent_end,
        proposed,
        pad_sec=float(split_cfg["pad_sec"]),
        merge_gap_sec=float(split_cfg["merge_gap_sec"]),
        min_subsegment_duration_sec=float(split_cfg["min_subsegment_duration_sec"]),
        max_subsegments_per_candidate=int(split_cfg["max_subsegments_per_candidate"]),
        max_coverage_reduction_ratio=float(split_cfg["max_coverage_reduction_ratio"]),
        video_duration=video_duration,
    )
    decision["num_peaks"] = len(peaks)
    decision["num_valleys"] = len(valleys)
    return decision

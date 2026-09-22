"""MHS-1 评估：对 weak-reference 的 duration-based P/R/F1/tIoU/coverage（事后评估）。"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

_EPS = 1e-9


def _merge_intervals(segments: list[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted((s, e) for s, e in segments if e > s)
    merged: list[tuple[float, float]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1] + _EPS:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def video_metrics(
    predicted: list[tuple[float, float]],
    references: list[tuple[float, float]],
) -> dict[str, float]:
    """merged per-video duration 指标（与 Stage 4 轮次评估口径一致）。"""
    predicted = _merge_intervals(predicted)
    references = [(s, e) for s, e in references if e > s]
    pred_dur = sum(e - s for s, e in predicted)
    ref_dur = sum(e - s for s, e in references)
    intersection = 0.0
    for ps, pe in predicted:
        for rs, re_ in references:
            intersection += max(0.0, min(pe, re_) - max(ps, rs))
    union = pred_dur + ref_dur - intersection
    precision = intersection / pred_dur if pred_dur > _EPS else 0.0
    recall = intersection / ref_dur if ref_dur > _EPS else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > _EPS else 0.0
    iou = intersection / union if union > _EPS else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "temporal_iou": iou,
        "prediction_duration_sec": pred_dur,
        "reference_duration_sec": ref_dur,
        "intersection_sec": intersection,
    }


def evaluate_segments(
    segments_by_video: Mapping[str, list[tuple[float, float]]],
    references_by_video: Mapping[str, list[tuple[float, float]]],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """per-video mean 聚合。"""
    per_video: dict[str, dict[str, float]] = {}
    for video_id in sorted(segments_by_video):
        per_video[video_id] = video_metrics(
            segments_by_video[video_id], references_by_video.get(video_id, [])
        )
    keys = ("precision", "recall", "f1", "temporal_iou", "prediction_duration_sec", "reference_duration_sec")
    aggregate = {
        key: _finite(sum(row[key] for row in per_video.values()) / len(per_video))
        for key in keys
        if per_video
    }
    return aggregate, per_video


def _finite(value: float) -> float:
    return float(value) if math.isfinite(value) else 0.0


def compare_aggregates(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    *,
    baseline_segments: int,
    candidate_segments: int,
    num_videos: int,
    split_rate: float,
    fallback_rate: float,
    gate: Mapping[str, Any],
    new_zero_recall: int,
) -> dict[str, Any]:
    deltas = {
        "delta_precision": candidate["precision"] - baseline["precision"],
        "delta_recall": candidate["recall"] - baseline["recall"],
        "delta_f1": candidate["f1"] - baseline["f1"],
        "delta_temporal_iou": candidate["temporal_iou"] - baseline["temporal_iou"],
        "delta_coverage": (
            candidate["prediction_duration_sec"] - baseline["prediction_duration_sec"]
        ),
    }
    checks = {
        "recall_delta_min": deltas["delta_recall"] >= float(gate["recall_delta_min"]),
        "precision_delta_min": deltas["delta_precision"] >= float(gate["precision_delta_min"]),
        "f1_or_tiou_delta_min": (
            deltas["delta_f1"] >= float(gate["f1_or_tiou_delta_min"])
            or deltas["delta_temporal_iou"] >= float(gate["f1_or_tiou_delta_min"])
        ),
        "coverage_delta_max": (
            candidate["prediction_duration_sec"] - baseline["prediction_duration_sec"]
            <= float(gate["coverage_delta_max"]) + 1e-12
        ),
        "new_zero_recall": new_zero_recall <= int(gate["new_zero_recall_allowed"]),
    }
    return {
        "baseline": dict(baseline),
        "candidate": dict(candidate),
        "deltas": deltas,
        "num_candidates": candidate_segments,
        "baseline_segments": baseline_segments,
        "segments_per_video": candidate_segments / num_videos if num_videos else 0.0,
        "split_rate": split_rate,
        "fallback_rate": fallback_rate,
        "gate_checks": checks,
        "gate": "PASS" if all(checks.values()) else "FAIL",
    }


def count_new_zero_recall(
    baseline_per_video: Mapping[str, Mapping[str, float]],
    candidate_per_video: Mapping[str, Mapping[str, float]],
) -> int:
    """候选评估中新出现的 recall==0 视频数（相对 baseline）。"""
    count = 0
    for video_id, row in candidate_per_video.items():
        base = baseline_per_video.get(video_id)
        if base is None:
            continue
        if row["recall"] <= 1e-9 and base["recall"] > 1e-9:
            count += 1
    return count


def gate_summary_for_note(status: str) -> str:
    return "PASS" if status == "PASS" else "FAIL"

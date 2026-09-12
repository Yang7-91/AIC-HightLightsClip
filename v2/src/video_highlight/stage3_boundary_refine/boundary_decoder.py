"""将帧级高光概率和边界概率解码成一个保守精细区间。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .feature_extractor import FeatureSequence
from .temporal_backend import TemporalScores


@dataclass(frozen=True, slots=True)
class BoundaryDecision:
    start_index: int
    end_index: int
    temporal_score: float
    boundary_confidence: float
    status: str


def _best_component(mask: np.ndarray, scores: np.ndarray) -> tuple[int, int]:
    best = (0, len(mask) - 1)
    best_score = -1.0
    start: int | None = None
    for index, active in enumerate(np.append(mask, False)):
        if active and start is None:
            start = index
        elif not active and start is not None:
            end = index - 1
            score = float(scores[start : end + 1].sum())
            if score > best_score:
                best, best_score = (start, end), score
            start = None
    return best


def decode_boundaries(
    features: FeatureSequence,
    scores: TemporalScores,
    candidate: dict[str, Any],
    config: dict[str, Any],
) -> BoundaryDecision:
    count = len(features.timestamps_sec)
    if count == 0:
        raise ValueError("边界解码不能接受空序列")
    temporal_score = float(scores.highlight.max())
    dynamic_range = float(scores.highlight.max() - scores.highlight.min())
    if count < 3 or dynamic_range < float(config.get("min_dynamic_range", 0.08)):
        return BoundaryDecision(0, count - 1, temporal_score, 0.0, "preserved_low_contrast")

    quantile = float(config.get("core_quantile", 0.65))
    threshold = max(
        float(config.get("core_threshold", 0.50)),
        float(np.quantile(scores.highlight, min(1.0, max(0.0, quantile)))),
    )
    mask = scores.highlight >= threshold
    if not bool(mask.any()):
        return BoundaryDecision(0, count - 1, temporal_score, 0.0, "preserved_no_core")
    core_start, core_end = _best_component(mask, scores.highlight)
    timestamps = features.timestamps_sec
    max_trim = max(0.0, float(config.get("max_trim_sec", 1.5)))
    latest_start = float(candidate["start_sec"]) + max_trim
    earliest_end = float(candidate["end_sec"]) - max_trim

    start_region = np.arange(0, core_start + 1)
    start_index = int(start_region[np.argmax(scores.start[start_region])])
    end_region = np.arange(core_end, count)
    end_index = int(end_region[np.argmax(scores.end[end_region])])
    # 无训练规则只允许有限幅度向内收缩，防止运动弱但语义强的片段被误删。
    start_index = min(start_index, int(np.searchsorted(timestamps, latest_start, side="right") - 1))
    start_index = max(0, start_index)
    earliest_end_index = int(np.searchsorted(timestamps, earliest_end, side="left"))
    end_index = max(end_index, min(count - 1, earliest_end_index))
    minimum_duration = max(0.0, float(config.get("min_duration_sec", 0.5)))
    if end_index <= start_index or timestamps[end_index] - timestamps[start_index] < minimum_duration:
        return BoundaryDecision(0, count - 1, temporal_score, 0.0, "preserved_short_decode")
    confidence = float((scores.start[start_index] + scores.end[end_index]) * 0.5)
    return BoundaryDecision(start_index, end_index, temporal_score, confidence, "refined")

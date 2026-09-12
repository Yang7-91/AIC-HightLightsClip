"""按实践方案中的权重计算高召回粗候选分数。"""

from __future__ import annotations

from typing import Any

from video_highlight.contracts.schema_versions import STAGE2_SCHEMA_VERSION

from .segment_loader import LoadedSegment


def _audio_score(segment: LoadedSegment) -> float:
    event_scores = [float(row.get("score", 0.0)) for row in segment.audio_events]
    return min(1.0, max(event_scores, default=0.0))


def build_candidate(
    analysis: dict[str, Any],
    segment: LoadedSegment,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    if analysis.get("start_sec") is None or analysis.get("end_sec") is None:
        return None
    weights = config.get("weights", {})
    semantic = float(analysis["highlight_score"])
    event = float(analysis["completeness_score"])
    # audio = _audio_score(segment)
    audio = 0.0 # 目前直接置零
    quality = float(config.get("neutral_quality_score", 0.5))
    stability = float(config.get("neutral_stability_score", 0.5))
    components = {
        "semantic": semantic,
        "event": event,
        "audio": audio,
        "quality": quality,
        "stability": stability,
    }
    weight_sum = sum(float(weights.get(name, 0.0)) for name in components)
    if weight_sum <= 0:
        raise ValueError("scoring.weights 权重和必须大于 0")
    coarse = sum(float(weights.get(name, 0.0)) * value for name, value in components.items()) / weight_sum
    if coarse < float(config.get("min_candidate_score", 0.35)):
        return None
    return {
        "schema_version": STAGE2_SCHEMA_VERSION,
        "video_id": segment.video_id,
        "candidate_id": (
            f"{segment.video_id}_raw_{segment.segment_id:06d}_"
            f"{int(analysis.get('candidate_index', 0)):02d}"
        ),
        "start_sec": float(analysis["start_sec"]),
        "end_sec": float(analysis["end_sec"]),
        "coarse_score": coarse,
        "semantic_score": semantic,
        "event_score": event,
        "audio_score": audio,
        "quality_score": quality,
        "stability_score": stability,
        "source_segment_ids": [segment.segment_id],
        "subject": analysis.get("subject"),
        "category": analysis.get("category"),
        "reason": analysis.get("reason", ""),
    }

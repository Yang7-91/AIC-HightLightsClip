"""Stage 2 粗高光候选的持久化数据契约。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


class SerializableRecord:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class SegmentAnalysis(SerializableRecord):
    schema_version: str
    video_id: str
    segment_id: int
    segment_start_sec: float
    segment_end_sec: float
    has_highlight: bool
    summary: str
    highlight_score: float
    completeness_score: float
    start_offset_sec: float | None
    end_offset_sec: float | None
    start_sec: float | None
    end_sec: float | None
    subject: str | None
    subject_point: list[float] | None
    category: str | None
    reason: str
    evidence_sample_ids: list[int]


@dataclass(frozen=True, slots=True)
class HighlightCandidate(SerializableRecord):
    schema_version: str
    video_id: str
    candidate_id: str
    start_sec: float
    end_sec: float
    coarse_score: float
    semantic_score: float
    event_score: float
    audio_score: float
    quality_score: float
    stability_score: float
    source_segment_ids: list[int]
    subject: str | None
    subject_point: list[float] | None
    category: str | None
    reason: str


@dataclass(frozen=True, slots=True)
class SubjectHint(SerializableRecord):
    schema_version: str
    video_id: str
    candidate_id: str
    subject: str | None
    subject_point: list[float] | None
    source_segment_ids: list[int]

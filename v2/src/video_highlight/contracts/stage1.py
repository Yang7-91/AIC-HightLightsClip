"""Stage 1 持久化输出数据契约。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


class SerializableRecord:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class SampledFrame(SerializableRecord):
    schema_version: str
    video_id: str
    sample_id: int
    scheduled_sec: float
    timestamp_sec: float
    original_frame: int
    scene_id: int
    image_path: str
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class Scene(SerializableRecord):
    schema_version: str
    video_id: str
    scene_id: int
    start_sec: float
    end_sec: float
    start_frame: int
    end_frame: int
    transition_type: str
    transition_score: float


@dataclass(frozen=True, slots=True)
class AnalysisSegment(SerializableRecord):
    schema_version: str
    video_id: str
    segment_id: int
    start_sec: float
    end_sec: float
    sample_ids: list[int]
    scene_ids: list[int]


@dataclass(frozen=True, slots=True)
class AudioEvent(SerializableRecord):
    schema_version: str
    video_id: str
    event_id: int
    start_sec: float
    end_sec: float
    label: str
    score: float
    evidence: dict[str, float]

"""MHS-1 输出 schema：与 Stage 2 candidates 尽量兼容。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

MHS1_SCHEMA_VERSION = "stage2_5.mhs1_candidate.v1"


@dataclass(frozen=True, slots=True)
class Mhs1Candidate:
    video_id: str
    source_candidate_id: str
    mhs1_candidate_id: str
    start_sec: float
    end_sec: float
    score: float | None
    reason: str | None
    method: str
    parent_start_sec: float
    parent_end_sec: float
    split_decision: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(
        default_factory=lambda: {
            "uses_weak_reference_for_decision": False,
            "heldout_access": False,
            "training": False,
        }
    )

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["schema_version"] = MHS1_SCHEMA_VERSION
        row["duration_sec"] = self.end_sec - self.start_sec
        return row


def make_mhs1_candidate(
    *,
    video_id: str,
    source_candidate_id: str,
    segment_index: int,
    start_sec: float,
    end_sec: float,
    parent_start_sec: float,
    parent_end_sec: float,
    score: float | None,
    reason: str | None,
    method: str,
    split_decision: dict[str, Any],
) -> Mhs1Candidate:
    return Mhs1Candidate(
        video_id=video_id,
        source_candidate_id=source_candidate_id,
        mhs1_candidate_id=f"{source_candidate_id}::mhs1-{method}-{segment_index}",
        start_sec=float(start_sec),
        end_sec=float(end_sec),
        score=score,
        reason=reason,
        method=method,
        parent_start_sec=float(parent_start_sec),
        parent_end_sec=float(parent_end_sec),
        split_decision=dict(split_decision),
    )

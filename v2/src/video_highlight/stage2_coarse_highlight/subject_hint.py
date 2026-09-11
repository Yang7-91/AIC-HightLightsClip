"""从最终粗候选生成 Stage 4 可消费的主体提示。"""

from __future__ import annotations

from typing import Any

from video_highlight.contracts.schema_versions import STAGE2_SCHEMA_VERSION


def build_subject_hints(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "schema_version": STAGE2_SCHEMA_VERSION,
            "video_id": candidate["video_id"],
            "candidate_id": candidate["candidate_id"],
            "subject": candidate.get("subject"),
            "subject_point": candidate.get("subject_point"),
            "source_segment_ids": candidate["source_segment_ids"],
        }
        for candidate in candidates
    ]

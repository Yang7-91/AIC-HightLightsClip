"""从最终粗候选生成纯语义主体提示，供后续 Stage 3.5 定位主体。"""

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
            "source_segment_ids": candidate["source_segment_ids"],
        }
        for candidate in candidates
    ]

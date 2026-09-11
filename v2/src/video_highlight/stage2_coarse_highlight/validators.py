"""Stage 2 输入、候选区间和持久化文件校验。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from video_highlight.common.exceptions import ArtifactValidationError


def validate_config(config: dict[str, Any]) -> None:
    for key in ("api", "generation", "video_input", "context", "scoring", "merging", "runtime"):
        if not isinstance(config.get(key), dict):
            raise ArtifactValidationError(f"Stage 2 配置缺少对象字段: {key}")
    if config["runtime"].get("backend", "openai") == "openai":
        for key in ("base_url", "api_key", "model"):
            if not str(config["api"].get(key, "")).strip():
                raise ArtifactValidationError(f"api.{key} 不能为空")


def validate_candidates(candidates: list[dict[str, Any]], duration_sec: float) -> None:
    previous_end = -1.0
    ids: set[str] = set()
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        if candidate_id in ids:
            raise ArtifactValidationError(f"重复 candidate_id: {candidate_id}")
        ids.add(candidate_id)
        start = float(candidate["start_sec"])
        end = float(candidate["end_sec"])
        if not (0.0 <= start < end <= duration_sec + 1e-6):
            raise ArtifactValidationError(f"候选时间范围非法: {candidate_id} [{start},{end})")
        if start < previous_end - 1e-6:
            raise ArtifactValidationError("合并后的候选仍存在重叠或未按时间排序")
        previous_end = end
        score = float(candidate["coarse_score"])
        if not 0.0 <= score <= 1.0:
            raise ArtifactValidationError(f"候选分数非法: {candidate_id}={score}")


def validate_stage2_artifacts(video_dir: str | Path) -> dict[str, int]:
    root = Path(video_dir)
    required = ["requests.jsonl", "raw_responses.jsonl", "analyses_segment_results.jsonl", "candidates.jsonl", "subject_hints.jsonl"]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ArtifactValidationError(f"Stage 2 缺少产物: {', '.join(missing)}")
    return {"artifact_files": len(required)}

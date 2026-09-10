"""Stage 1 索引和持久化产物校验。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from video_highlight.common.exceptions import ArtifactValidationError


def validate_index_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ArtifactValidationError("输入索引为空")
    seen: set[str] = set()
    for row in rows:
        if "video_id" not in row:
            raise ArtifactValidationError("索引记录缺少 video_id")
        video_id = str(row["video_id"])
        if video_id in seen:
            raise ArtifactValidationError(f"索引存在重复 video_id: {video_id}")
        seen.add(video_id)


def validate_stage1_artifacts(video_dir: str | Path, has_audio: bool, audio_enabled: bool) -> dict[str, int]:
    root = Path(video_dir)
    required = ["metadata.json", "scenes.jsonl", "segments.jsonl", "sample_map.jsonl"]
    missing = [name for name in required if not (root / name).is_file()]
    images = list((root / "coarse_frames").glob("*.jpg"))
    if not images:
        missing.append("coarse_frames/*.jpg")
    if has_audio and audio_enabled and not (root / "audio.wav").is_file():
        missing.append("audio.wav")
    if missing:
        raise ArtifactValidationError(f"Stage 1 缺少产物: {', '.join(missing)}")
    return {"sample_images": len(images)}

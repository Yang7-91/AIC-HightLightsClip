"""通过 FFmpeg 提取连续、单声道 PCM 音轨。"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from video_highlight.common.exceptions import ExternalToolError


def extract_audio(
    video_path: str | Path,
    output_path: str | Path,
    sample_rate: int = 16000,
    ffmpeg_bin: str = "ffmpeg",
) -> dict[str, Any]:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(Path(video_path).resolve()),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(target),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if completed.returncode != 0:
        target.unlink(missing_ok=True)
        raise ExternalToolError(f"FFmpeg 音轨提取失败: {completed.stderr.strip()}")
    return {"status": "extracted", "path": target.name, "sample_rate": sample_rate, "channels": 1}

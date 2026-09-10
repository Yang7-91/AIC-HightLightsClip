"""通过 FFprobe 读取视频、视频流和音频流元信息。"""

from __future__ import annotations

import json
import math
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

from video_highlight.common.exceptions import ArtifactValidationError, ExternalToolError
from video_highlight.common.hashing import file_sha256
from video_highlight.contracts.schema_versions import STAGE1_SCHEMA_VERSION


def _fraction(value: object) -> float | None:
    try:
        text = str(value)
        if not text or text == "0/0":
            return None
        result = float(Fraction(text))
        return result if math.isfinite(result) and result > 0 else None
    except (ValueError, ZeroDivisionError):
        return None


def _number(value: object) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _rotation(stream: dict[str, Any]) -> int:
    raw = stream.get("tags", {}).get("rotate")
    if raw is None:
        for side_data in stream.get("side_data_list", []):
            if "rotation" in side_data:
                raw = side_data["rotation"]
                break
    try:
        return int(round(float(raw or 0))) % 360
    except (TypeError, ValueError):
        return 0


def probe_video(
    video_id: str,
    video_path: str | Path,
    ffprobe_bin: str = "ffprobe",
    compute_sha256: bool = True,
) -> dict[str, Any]:
    """返回后续阶段所需的确定性视频元信息。"""
    path = Path(video_path).resolve()
    if not path.is_file():
        raise ArtifactValidationError(f"视频不存在: {path}")
    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if completed.returncode != 0:
        raise ExternalToolError(f"FFprobe 读取失败: {completed.stderr.strip()}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ExternalToolError("FFprobe 返回了非法 JSON") from error

    streams = payload.get("streams", [])
    video_streams = [stream for stream in streams if stream.get("codec_type") == "video"]
    audio_streams = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if not video_streams:
        raise ArtifactValidationError(f"文件中没有视频流: {path}")
    video = video_streams[0]
    format_info = payload.get("format", {})
    fps = _fraction(video.get("avg_frame_rate")) or _fraction(video.get("r_frame_rate"))
    duration = _number(video.get("duration")) or _number(format_info.get("duration"))
    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)
    rotation = _rotation(video)
    display_width, display_height = (height, width) if rotation in {90, 270} else (width, height)
    frame_count_raw = video.get("nb_frames")
    try:
        frame_count = int(frame_count_raw) if frame_count_raw not in {None, "N/A"} else None
    except (TypeError, ValueError):
        frame_count = None
    if frame_count is None and fps and duration:
        frame_count = int(round(fps * duration))
    if width <= 0 or height <= 0 or not fps or duration is None or duration <= 0:
        raise ArtifactValidationError(f"视频关键元信息非法: {path}")

    return {
        "schema_version": STAGE1_SCHEMA_VERSION,
        "video_id": video_id,
        "source_path": str(path),
        "file_name": path.name,
        "file_size_bytes": path.stat().st_size,
        "sha256": file_sha256(path) if compute_sha256 else None,
        "duration_sec": duration,
        "fps": fps,
        "frame_count": frame_count,
        "time_base": video.get("time_base"),
        "width": width,
        "height": height,
        "display_width": display_width,
        "display_height": display_height,
        "rotation": rotation,
        "video_codec": video.get("codec_name"),
        "pixel_format": video.get("pix_fmt"),
        "has_audio": bool(audio_streams),
        "audio_streams": [
            {
                "index": stream.get("index"),
                "codec": stream.get("codec_name"),
                "sample_rate": int(stream["sample_rate"]) if stream.get("sample_rate") else None,
                "channels": stream.get("channels"),
                "duration_sec": _number(stream.get("duration")),
                "time_base": stream.get("time_base"),
            }
            for stream in audio_streams
        ],
    }

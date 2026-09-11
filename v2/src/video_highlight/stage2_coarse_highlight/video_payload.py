"""构建 OpenAI 兼容接口所需的视频 URL 或 Base64 data URL。"""

from __future__ import annotations

import base64
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

from video_highlight.common.exceptions import ArtifactValidationError, ExternalToolError

from .segment_loader import LoadedSegment


@dataclass(frozen=True, slots=True)
class PreparedVideoPayload:
    content_item: dict[str, Any]
    descriptor: dict[str, Any]


def _format_url(template: str, segment: LoadedSegment) -> str:
    source_path = str(segment.metadata.get("source_path", ""))
    values = {
        "video_id": segment.video_id,
        "video_id_urlencoded": quote(segment.video_id, safe=""),
        "segment_id": segment.segment_id,
        "start_sec": f"{segment.start_sec:.3f}",
        "end_sec": f"{segment.end_sec:.3f}",
        "start_ms": int(round(segment.start_sec * 1000)),
        "end_ms": int(round(segment.end_sec * 1000)),
        "source_name": Path(source_path).name,
        "source_path": source_path,
    }
    try:
        return template.format_map(values)
    except KeyError as error:
        raise ArtifactValidationError(f"video_input.url_template 包含未知占位符: {error}") from error


def _concat_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "'\\''")


def _encode_frames(segment: LoadedSegment, config: dict[str, Any], cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = cache_dir / f"segment_{segment.segment_id:06d}.mp4"
    frame_list = cache_dir / f"segment_{segment.segment_id:06d}.frames.txt"
    fps = float(config.get("fps", 2.0))
    if fps <= 0:
        raise ArtifactValidationError("video_input.fps 必须大于 0")
    frame_duration = 1.0 / fps
    lines: list[str] = []
    for frame_path in segment.frame_paths:
        lines.append(f"file '{_concat_path(frame_path)}'")
        lines.append(f"duration {frame_duration:.9f}")
    # concat demuxer 需要重复最后一帧，才能正确应用最后一个 duration。
    lines.append(f"file '{_concat_path(segment.frame_paths[-1])}'")
    frame_list.write_text("\n".join(lines) + "\n", encoding="utf-8")
    command = [
        str(config.get("ffmpeg_bin", "ffmpeg")),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(frame_list),
        "-an",
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
        "-r",
        str(fps),
        "-c:v",
        str(config.get("codec", "libx264")),
        "-preset",
        str(config.get("preset", "veryfast")),
        "-crf",
        str(config.get("crf", 28)),
        "-pix_fmt",
        str(config.get("pixel_format", "yuv420p")),
        "-movflags",
        "+faststart",
        str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    frame_list.unlink(missing_ok=True)
    if completed.returncode != 0 or not output.is_file():
        output.unlink(missing_ok=True)
        raise ExternalToolError(f"低帧率视频编码失败: {completed.stderr.strip()}")
    return output


def prepare_video_payload(
    segment: LoadedSegment,
    config: dict[str, Any],
    cache_dir: str | Path,
) -> PreparedVideoPayload:
    """根据配置返回可直接放入 messages.content 的 video_url 项。"""
    mode = str(config.get("mode", "data_url")).lower()
    if mode == "url":
        template = str(config.get("url_template", "")).strip()
        if not template:
            raise ArtifactValidationError("video_input.mode=url 时必须配置 url_template")
        url = _format_url(template, segment)
        scheme = urlparse(url).scheme.lower()
        allowed = {str(value).lower() for value in config.get("allowed_url_schemes", ["http", "https"])}
        if scheme not in allowed:
            raise ArtifactValidationError(f"视频 URL scheme={scheme!r} 不在允许列表 {sorted(allowed)} 中")
        item: dict[str, Any] = {"type": "video_url", "video_url": {"url": url}}
        if bool(config.get("include_time_range_fields", False)):
            # 仅在当前 vLLM/Qwen 媒体解析器明确支持时启用；标准 OpenAI 字段不保证处理该范围。
            item["video_start"] = segment.start_sec
            item["video_end"] = segment.end_sec
        return PreparedVideoPayload(
            content_item=item,
            descriptor={"mode": "url", "url": url, "frame_count": len(segment.frame_paths)},
        )

    if mode != "data_url":
        raise ArtifactValidationError(f"未知 video_input.mode: {mode}")
    encoded = _encode_frames(segment, config, Path(cache_dir))
    size_bytes = encoded.stat().st_size
    max_bytes = int(float(config.get("max_payload_mib", 64.0)) * 1024 * 1024)
    if size_bytes > max_bytes:
        encoded.unlink(missing_ok=True)
        raise ArtifactValidationError(
            f"编码视频 {size_bytes / 1024 / 1024:.2f} MiB 超过 max_payload_mib={max_bytes / 1024 / 1024:.2f}"
        )
    data_url = "data:video/mp4;base64," + base64.b64encode(encoded.read_bytes()).decode("ascii")
    descriptor = {
        "mode": "data_url",
        "encoded_size_bytes": size_bytes,
        "frame_count": len(segment.frame_paths),
        "fps": float(config.get("fps", 2.0)),
        "encoded_file": str(encoded) if config.get("keep_encoded_video", False) else None,
    }
    if not bool(config.get("keep_encoded_video", False)):
        encoded.unlink(missing_ok=True)
    return PreparedVideoPayload(
        content_item={"type": "video_url", "video_url": {"url": data_url}},
        descriptor=descriptor,
    )

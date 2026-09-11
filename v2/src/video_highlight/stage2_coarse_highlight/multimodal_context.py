"""把 Stage 1 视觉时间线和音频证据压缩成提示词上下文。"""

from __future__ import annotations

from typing import Any

from .segment_loader import LoadedSegment


def _evenly(values: list[Any], maximum: int) -> list[Any]:
    if maximum <= 0 or len(values) <= maximum:
        return values
    if maximum == 1:
        return [values[len(values) // 2]]
    indices = [round(index * (len(values) - 1) / (maximum - 1)) for index in range(maximum)]
    return [values[index] for index in indices]


def _none_text() -> str:
    return "（无可用信息）"


def build_multimodal_context(segment: LoadedSegment, config: dict[str, Any]) -> dict[str, Any]:
    """生成紧凑的文本时间线，并保留用于构建视频载荷的帧路径。"""
    frame_lines = []
    for ordinal, sample in enumerate(segment.samples):
        timestamp = float(sample["timestamp_sec"])
        frame_lines.append(
            f"F{ordinal:03d}: sample_id={int(sample['sample_id'])}, "
            f"absolute={timestamp:.3f}s, offset={timestamp - segment.start_sec:.3f}s, "
            f"original_frame={int(sample['original_frame'])}, scene_id={int(sample['scene_id'])}"
        )

    min_event_score = float(config.get("audio_event_min_score", 0.25))
    ranked_events = [row for row in segment.audio_events if float(row.get("score", 0.0)) >= min_event_score]
    ranked_events.sort(key=lambda row: (-float(row.get("score", 0.0)), float(row.get("start_sec", 0.0))))
    ranked_events = ranked_events[: int(config.get("max_audio_events", 24))]
    ranked_events.sort(key=lambda row: float(row.get("start_sec", 0.0)))
    event_lines = [
        f"[{float(row['start_sec']):.3f},{float(row['end_sec']):.3f})s "
        f"{row.get('label', 'unknown')} score={float(row.get('score', 0.0)):.3f}"
        for row in ranked_events
    ]

    audio_rows = _evenly(segment.audio_timeline, int(config.get("max_audio_bins", 64)))
    audio_lines = [
        f"[{float(row['start_sec']):.3f},{float(row['end_sec']):.3f})s "
        f"rms={float(row.get('rms_dbfs', -120.0)):.1f}dBFS, "
        f"peak={float(row.get('peak', 0.0)):.3f}, flux={float(row.get('spectral_flux', 0.0)):.4f}, "
        f"events={','.join(str(value) for value in row.get('event_labels', [])) or 'none'}"
        for row in audio_rows
    ]

    transcripts = segment.transcripts[: int(config.get("max_asr_segments", 32))]
    asr_lines = [
        f"[{float(row.get('start_sec', 0.0)):.3f},{float(row.get('end_sec', 0.0)):.3f})s "
        f"{str(row.get('text', '')).strip()}"
        for row in transcripts
        if str(row.get("text", "")).strip()
    ]
    return {
        "frame_timeline": "\n".join(frame_lines) or _none_text(),
        "audio_events": "\n".join(event_lines) or _none_text(),
        "audio_timeline": "\n".join(audio_lines) or _none_text(),
        "asr_timeline": "\n".join(asr_lines) or _none_text(),
        "selected_frame_count": len(segment.frame_paths),
        "audio_event_count": len(ranked_events),
        "audio_bin_count": len(audio_rows),
        "asr_segment_count": len(asr_lines),
    }

"""把声学特征、通用声音事件与 ASR 组织为统一时间线。"""

from __future__ import annotations

from typing import Any

from video_highlight.contracts.schema_versions import STAGE1_SCHEMA_VERSION


def build_audio_timeline(
    video_id: str,
    feature_rows: list[dict[str, Any]],
    events: list[dict[str, Any]],
    transcripts: list[dict[str, object]],
) -> list[dict[str, Any]]:
    timeline: list[dict[str, Any]] = []
    for row in feature_rows:
        start, end = float(row["start_sec"]), float(row["end_sec"])
        labels = [
            str(event["label"])
            for event in events
            if float(event["end_sec"]) > start and float(event["start_sec"]) < end
        ]
        texts = [
            str(item.get("text", ""))
            for item in transcripts
            if float(item.get("end_sec", 0.0)) > start and float(item.get("start_sec", 0.0)) < end
        ]
        timeline.append(
            {
                "schema_version": STAGE1_SCHEMA_VERSION,
                "video_id": video_id,
                "bin_id": int(row["bin_id"]),
                "start_sec": start,
                "end_sec": end,
                "rms_dbfs": float(row["rms_dbfs"]),
                "peak": float(row["peak"]),
                "spectral_flux": float(row["spectral_flux"]),
                "event_labels": labels,
                "asr_text": " ".join(text for text in texts if text),
            }
        )
    return timeline

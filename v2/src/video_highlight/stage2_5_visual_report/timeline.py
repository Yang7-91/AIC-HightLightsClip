"""MHS-VIS-0 时间轴数据与 SVG 渲染（纯 SVG，不依赖 matplotlib）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np


class TimelineError(RuntimeError):
    """运动/镜头/峰值计算失败。"""


def compute_motion_and_shots(
    video_path: str | Path,
    start_sec: float,
    end_sec: float,
    *,
    sample_stride_sec: float = 0.5,
    shot_hist_threshold: float = 0.35,
    min_shot_gap_sec: float = 1.0,
) -> dict[str, Any]:
    """逐点采样计算运动强度曲线与镜头切换候选（确定性）。

    motion 为相邻采样帧的灰度差（0-1 归一化）；shot 为直方图差异超阈值的采样点。
    """
    import math

    source = Path(video_path)
    if not source.is_file():
        raise TimelineError(f"video not found: {source}")
    capture = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG)
    if not capture.isOpened():
        raise TimelineError(f"OpenCV could not open video: {source}")
    samples: list[dict[str, float]] = []
    previous_gray: np.ndarray | None = None
    previous_hist: np.ndarray | None = None
    t = start_sec
    try:
        while t <= end_sec + 1e-9:
            capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            small = cv2.resize(frame, (64, 48), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
            cv2.normalize(hist, hist)
            motion = 0.0
            hist_diff = 0.0
            if previous_gray is not None:
                motion = float(np.mean(cv2.absdiff(previous_gray, gray))) / 255.0
                if previous_hist is not None:
                    hist_diff = float(
                        cv2.compareHist(previous_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                    )
            samples.append({"t_sec": round(t, 3), "motion": motion, "hist_diff": hist_diff})
            previous_gray = gray
            previous_hist = hist
            t += sample_stride_sec
    finally:
        capture.release()
    if len(samples) < 2:
        raise TimelineError(f"too few sampled frames in [{start_sec}, {end_sec}]")
    shot_boundaries: list[float] = []
    for sample in samples[1:]:
        if sample["hist_diff"] >= shot_hist_threshold:
            if shot_boundaries and sample["t_sec"] - shot_boundaries[-1] < min_shot_gap_sec:
                continue
            shot_boundaries.append(sample["t_sec"])
    return {"samples": samples, "shot_boundaries_sec": shot_boundaries}


def find_motion_peaks(
    samples: list[Mapping[str, float]],
    *,
    min_gap_sec: float = 3.0,
    relative_threshold: float = 0.5,
) -> list[float]:
    """在运动曲线上找局部峰（高于相对阈值且间距受限）。"""
    if not samples:
        return []
    values = [float(sample["motion"]) for sample in samples]
    maximum = max(values)
    if maximum <= 1e-9:
        return []
    threshold = maximum * relative_threshold
    peaks: list[float] = []
    for index in range(1, len(samples) - 1):
        value = values[index]
        if value < threshold:
            continue
        if value >= values[index - 1] and value >= values[index + 1]:
            t_sec = float(samples[index]["t_sec"])
            if peaks and t_sec - peaks[-1] < min_gap_sec:
                continue
            peaks.append(t_sec)
    return peaks


def build_timeline_data(
    candidate: Mapping[str, Any],
    video_duration_sec: float,
    *,
    motion_samples: list[Mapping[str, float]],
    shot_boundaries: list[float],
    motion_peaks: list[float],
    subsegments: list[Mapping[str, Any]] | None = None,
    eventness_bins: list[Mapping[str, Any]] | None = None,
    audio_peaks: list[float] | None = None,
    yolo_density: list[Mapping[str, float]] | None = None,
    vlm_scores: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """构建渲染所需的时间轴数据（JSON 可序列化）。"""
    return {
        "video_duration_sec": float(video_duration_sec),
        "candidate": {
            "candidate_id": candidate["candidate_id"],
            "start_sec": float(candidate["start_sec"]),
            "end_sec": float(candidate["end_sec"]),
            "duration_sec": float(candidate["duration_sec"]),
            "score": candidate.get("score"),
        },
        "subsegments": [
            {
                "start_sec": float(segment["start_sec"]),
                "end_sec": float(segment["end_sec"]),
                "label": str(segment.get("label", "")),
            }
            for segment in (subsegments or [])
        ],
        "motion": [
            {"t_sec": float(sample["t_sec"]), "value": float(sample["motion"])}
            for sample in motion_samples
        ],
        "eventness": [
            {
                "t_sec": float(item.get("t_sec", item.get("start_sec", 0.0))),
                "value": float(item.get("value", item.get("eventness", 0.0))),
            }
            for item in (eventness_bins or [])
        ],
        "eventness_available": bool(eventness_bins),
        "shot_boundaries_sec": [float(value) for value in shot_boundaries],
        "motion_peaks_sec": [float(value) for value in motion_peaks],
        "audio_peaks_sec": [float(value) for value in (audio_peaks or [])],
        "yolo_density": [
            {"t_sec": float(item["t_sec"]), "value": float(item["value"])}
            for item in (yolo_density or [])
        ],
        "vlm_scores": [dict(item) for item in (vlm_scores or [])],
        "diagnostic_only": True,
        "deployable_method": False,
    }


_PALETTE = ("#22c55e", "#f59e0b", "#3b82f6", "#ec4899", "#14b8a6", "#a855f7")


def render_timeline_svg(
    timeline: Mapping[str, Any],
    *,
    width_px: int = 1400,
    height_px: int = 170,
) -> str:
    """把时间轴数据渲染成内联 SVG 字符串。"""
    pad_left = 24.0
    pad_right = 24.0
    track_top = 66.0
    track_height = 26.0
    curve_top = 18.0
    curve_height = 38.0
    duration = float(timeline["video_duration_sec"]) or 1.0
    plot_width = max(width_px - pad_left - pad_right, 10.0)

    def x_of(t_sec: float) -> float:
        return pad_left + max(0.0, min(float(t_sec), duration)) / duration * plot_width

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width_px} {height_px}" '
        f'width="100%" style="max-width:{width_px}px" role="img">',
        f'<rect x="0" y="0" width="{width_px}" height="{height_px}" fill="#0b1220" rx="8"/>',
    ]
    # 原始 candidate 灰条
    candidate = timeline["candidate"]
    x0 = x_of(candidate["start_sec"])
    x1 = x_of(candidate["end_sec"])
    parts.append(
        f'<rect x="{x0:.1f}" y="{track_top:.1f}" width="{max(x1 - x0, 1):.1f}" '
        f'height="{track_height}" fill="#64748b" opacity="0.85" rx="4"/>'
    )
    parts.append(
        f'<text x="{(x0 + x1) / 2:.1f}" y="{track_top + 17:.1f}" text-anchor="middle" '
        f'font-size="11" fill="#e2e8f0">candidate {candidate["duration_sec"]:.1f}s</text>'
    )
    # proposed subsegments 彩条
    for index, segment in enumerate(timeline["subsegments"]):
        sx0 = x_of(segment["start_sec"])
        sx1 = x_of(segment["end_sec"])
        color = _PALETTE[index % len(_PALETTE)]
        parts.append(
            f'<rect x="{sx0:.1f}" y="{track_top + track_height + 6:.1f}" '
            f'width="{max(sx1 - sx0, 1):.1f}" height="12" fill="{color}" rx="3" opacity="0.95"/>'
        )
    # 曲线（eventness 优先，否则 motion proxy）
    series = timeline["eventness"] if timeline["eventness_available"] else timeline["motion"]
    series_label = "eventness" if timeline["eventness_available"] else "motion proxy"
    if series:
        values = [item["value"] for item in series]
        maximum = max(values) if max(values) > 1e-9 else 1.0
        points = " ".join(
            f"{x_of(item['t_sec']):.1f},{curve_top + curve_height - (item['value'] / maximum) * curve_height:.1f}"
            for item in series
        )
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="#38bdf8" stroke-width="1.6"/>'
        )
        parts.append(
            f'<text x="{pad_left:.1f}" y="{curve_top - 4:.1f}" font-size="10" '
            f'fill="#7dd3fc">{series_label}</text>'
        )
    # shot boundary 竖线
    for shot in timeline["shot_boundaries_sec"]:
        sx = x_of(shot)
        parts.append(
            f'<line x1="{sx:.1f}" y1="{curve_top:.1f}" x2="{sx:.1f}" '
            f'y2="{track_top + track_height + 20:.1f}" stroke="#f97316" stroke-width="1" opacity="0.8"/>'
        )
    # 运动峰标记
    for peak in timeline["motion_peaks_sec"]:
        px = x_of(peak)
        parts.append(
            f'<circle cx="{px:.1f}" cy="{curve_top + 4:.1f}" r="3" fill="#facc15"/>'
        )
    parts.append("</svg>")
    return "".join(parts)

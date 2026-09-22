"""MHS-1 eventness 曲线：frame difference / shot boundary / audio energy 逐 bin 信号。"""

from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

_SIGNAL_KEYS = ("frame_difference", "shot_boundary", "audio_energy")


class EventnessError(RuntimeError):
    """视频/音频无法读取或信号计算失败。"""


def compute_frame_difference_curve(
    video_path: str | Path,
    start_sec: float,
    end_sec: float,
    *,
    sample_fps: float = 2.0,
    bin_sec: float = 1.0,
) -> tuple[list[dict[str, float]], list[float]]:
    """逐采样点帧差 + 直方图差，再聚合到 bin；返回 (bins, shot timestamps)。"""
    source = Path(video_path)
    if not source.is_file():
        raise EventnessError(f"video not found: {source}")
    capture = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG)
    if not capture.isOpened():
        raise EventnessError(f"OpenCV could not open video: {source}")
    stride = 1.0 / max(sample_fps, 0.1)
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
            frame_difference = 0.0
            histogram_difference = 0.0
            if previous_gray is not None:
                frame_difference = float(np.mean(cv2.absdiff(previous_gray, gray))) / 255.0
            if previous_hist is not None:
                histogram_difference = float(
                    cv2.compareHist(previous_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                )
            samples.append(
                {
                    "t_sec": round(t, 3),
                    "frame_difference": frame_difference,
                    "histogram_difference": histogram_difference,
                }
            )
            previous_gray = gray
            previous_hist = hist
            t += stride
    finally:
        capture.release()
    if len(samples) < 2:
        raise EventnessError(f"too few sampled frames in [{start_sec}, {end_sec}]")

    # 自适应 shot 阈值（median + 3*MAD 与 p85 的较大者）
    hist_values = np.asarray([item["histogram_difference"] for item in samples], dtype=float)
    median = float(np.median(hist_values))
    mad = float(np.median(np.abs(hist_values - median)))
    threshold = max(median + 3.0 * 1.4826 * mad, float(np.percentile(hist_values, 85)), 1e-6)
    shot_times: list[float] = []
    for item in samples:
        if item["histogram_difference"] > threshold:
            if shot_times and item["t_sec"] - shot_times[-1] < 1.0:
                continue
            shot_times.append(item["t_sec"])

    bins: list[dict[str, float]] = []
    index = 0
    bin_start = start_sec
    while bin_start < end_sec - 1e-9:
        bin_end = min(bin_start + bin_sec, end_sec)
        inside = [
            item
            for item in samples
            if bin_start - 1e-9 <= item["t_sec"] < bin_end + 1e-9
        ]
        frame_difference = (
            float(np.mean([item["frame_difference"] for item in inside])) if inside else 0.0
        )
        shot_hits = sum(1 for item in inside if item["histogram_difference"] > threshold)
        bins.append(
            {
                "bin_index": index,
                "start_sec": round(bin_start, 3),
                "end_sec": round(bin_end, 3),
                "frame_difference": frame_difference,
                "shot_boundary": 1.0 if shot_hits > 0 else 0.0,
            }
        )
        index += 1
        bin_start = bin_end
    return bins, shot_times


def compute_audio_energy_curve(
    video_path: str | Path,
    start_sec: float,
    end_sec: float,
    *,
    bin_sec: float = 1.0,
    ffmpeg_bin: str = "ffmpeg",
    sample_rate: int = 8000,
) -> tuple[list[float], bool]:
    """用 ffmpeg 提取音频并聚合每 bin RMS 能量；无音轨时返回 (全 0, False)。"""
    duration = end_sec - start_sec
    if duration <= 0:
        return [], False
    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_sec:.3f}",
        "-i",
        str(video_path),
        "-t",
        f"{duration:.3f}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "pipe:1",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, check=False)
    except FileNotFoundError:
        return [], False
    if completed.returncode != 0 or not completed.stdout:
        return [], False
    samples = np.frombuffer(completed.stdout, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return [], False
    per_bin = max(1, int(round(sample_rate * bin_sec)))
    energies: list[float] = []
    for offset in range(0, samples.size, per_bin):
        chunk = samples[offset : offset + per_bin]
        if chunk.size == 0:
            continue
        rms = float(np.sqrt(np.mean(chunk * chunk))) / 32768.0
        energies.append(rms)
    return energies, True


def smooth_eventness(scores: list[float], window_sec: float, bin_sec: float) -> list[float]:
    """滑动平均平滑（窗口为奇数 bin 数）。"""
    if not scores:
        return []
    window = max(1, int(round(window_sec / max(bin_sec, 1e-6))))
    if window % 2 == 0:
        window += 1
    if window <= 1:
        return list(scores)
    half = window // 2
    smoothed: list[float] = []
    for index in range(len(scores)):
        lo = max(0, index - half)
        hi = min(len(scores), index + half + 1)
        smoothed.append(float(np.mean(scores[lo:hi])))
    return smoothed


def normalize_eventness(scores: list[float]) -> list[float]:
    """minmax 归一化到 [0, 1]；常数序列返回全 0。"""
    if not scores:
        return []
    array = np.asarray(scores, dtype=float)
    low = float(array.min())
    high = float(array.max())
    if high - low <= 1e-12:
        return [0.0 for _ in scores]
    return [float((value - low) / (high - low)) for value in scores]


def build_eventness_bins(
    video_id: str,
    candidate_id: str,
    *,
    bins: list[dict[str, float]],
    audio_energies: list[float],
    audio_available: bool,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """合成 eventness bins（含加权总分）。"""
    eventness_cfg = config["eventness"]
    weights = eventness_cfg["weights"]
    signals = eventness_cfg["signals"]
    raw_frame = [item["frame_difference"] for item in bins]
    raw_shot = [item["shot_boundary"] for item in bins]
    if audio_available and audio_energies:
        raw_audio = [
            audio_energies[index] if index < len(audio_energies) else 0.0
            for index in range(len(bins))
        ]
    else:
        raw_audio = [0.0 for _ in bins]
    norm_frame = normalize_eventness(smooth_eventness(raw_frame, float(eventness_cfg["smooth_window_sec"]), float(eventness_cfg["bin_sec"])))
    norm_shot = normalize_eventness(smooth_eventness(raw_shot, float(eventness_cfg["smooth_window_sec"]), float(eventness_cfg["bin_sec"])))
    norm_audio = normalize_eventness(smooth_eventness(raw_audio, float(eventness_cfg["smooth_window_sec"]), float(eventness_cfg["bin_sec"])))
    effective_weights = {
        "frame_difference": float(weights["frame_difference"]) if signals["frame_difference"] else 0.0,
        "shot_boundary": float(weights["shot_boundary"]) if signals["shot_boundary"] else 0.0,
        "audio_energy": float(weights["audio_energy"]) if (signals["audio_energy"] and audio_available) else 0.0,
    }
    total_weight = sum(effective_weights.values())
    if total_weight <= 0:
        effective_weights = {
            "frame_difference": 1.0,
            "shot_boundary": 0.0,
            "audio_energy": 0.0,
        }
        total_weight = 1.0
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(bins):
        score = (
            effective_weights["frame_difference"] * norm_frame[index]
            + effective_weights["shot_boundary"] * norm_shot[index]
            + effective_weights["audio_energy"] * norm_audio[index]
        ) / total_weight
        rows.append(
            {
                "video_id": video_id,
                "candidate_id": candidate_id,
                "bin_index": int(item["bin_index"]),
                "start_sec": float(item["start_sec"]),
                "end_sec": float(item["end_sec"]),
                "frame_difference": float(item["frame_difference"]),
                "shot_boundary": float(item["shot_boundary"]),
                "audio_energy": float(raw_audio[index]),
                "eventness_score": float(score) if math.isfinite(score) else 0.0,
                "audio_available": bool(audio_available),
            }
        )
    return rows

"""提取规则基线和未来 TCN 共用的帧级时序特征。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .candidate_decoder import DecodedCandidate

FEATURE_NAMES = (
    "motion",
    "histogram_change",
    "brightness",
    "sharpness",
    "audio_energy",
    "audio_flux",
    "scene_boundary",
    "coarse_score",
)


@dataclass(frozen=True, slots=True)
class FeatureSequence:
    timestamps_sec: np.ndarray
    frame_indices: np.ndarray
    values: np.ndarray
    names: tuple[str, ...] = FEATURE_NAMES


def load_audio_features(stage1_video_dir: str | Path) -> dict[str, np.ndarray] | None:
    path = Path(stage1_video_dir) / "audio_features.npz"
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _aligned_audio(
    timestamps: np.ndarray,
    audio: dict[str, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if not audio or len(audio.get("start_sec", [])) == 0:
        zeros = np.zeros(len(timestamps), dtype=np.float32)
        return zeros, zeros.copy()
    starts = np.asarray(audio["start_sec"], dtype=np.float64)
    ends = np.asarray(audio["end_sec"], dtype=np.float64)
    centers = (starts + ends) * 0.5
    indices = np.searchsorted(centers, timestamps, side="left")
    indices = np.clip(indices, 0, len(centers) - 1)
    left = np.maximum(indices - 1, 0)
    use_left = np.abs(centers[left] - timestamps) < np.abs(centers[indices] - timestamps)
    indices = np.where(use_left, left, indices)
    rms = np.asarray(audio.get("rms_dbfs", np.zeros(len(centers))), dtype=np.float32)[indices]
    # 把常见的 [-80, 0] dBFS 映射到 [0, 1]，越界值安全裁剪。
    energy = np.clip((rms + 80.0) / 80.0, 0.0, 1.0)
    flux = np.asarray(audio.get("spectral_flux", np.zeros(len(centers))), dtype=np.float32)[indices]
    return energy.astype(np.float32), flux.astype(np.float32)


def extract_features(
    decoded: DecodedCandidate,
    candidate: dict[str, Any],
    scenes: list[dict[str, Any]],
    audio: dict[str, np.ndarray] | None,
    config: dict[str, Any],
) -> FeatureSequence:
    """生成运动、画质、音频、镜头边界和粗分数组成的特征矩阵。"""

    frames = decoded.gray_frames
    count = len(frames)
    motion = np.zeros(count, dtype=np.float32)
    histogram_change = np.zeros(count, dtype=np.float32)
    brightness = np.zeros(count, dtype=np.float32)
    sharpness = np.zeros(count, dtype=np.float32)
    previous_hist: np.ndarray | None = None
    for index, gray in enumerate(frames):
        brightness[index] = float(gray.mean()) / 255.0
        sharpness[index] = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).reshape(-1)
        hist /= max(float(hist.sum()), 1e-8)
        if index:
            motion[index] = float(cv2.absdiff(gray, frames[index - 1]).mean()) / 255.0
        if previous_hist is not None:
            histogram_change[index] = float(cv2.compareHist(previous_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
        previous_hist = hist

    audio_energy, audio_flux = _aligned_audio(decoded.timestamps_sec, audio)
    boundary_radius = max(1e-3, float(config.get("scene_boundary_radius_sec", 0.4)))
    boundaries = np.asarray(
        sorted({float(scene["start_sec"]) for scene in scenes[1:]}), dtype=np.float64
    )
    if len(boundaries):
        distances = np.min(
            np.abs(decoded.timestamps_sec[:, None] - boundaries[None, :]), axis=1
        )
        scene_boundary = np.exp(-distances / boundary_radius).astype(np.float32)
    else:
        scene_boundary = np.zeros(count, dtype=np.float32)
    coarse = np.full(count, float(candidate.get("coarse_score", 0.0)), dtype=np.float32)
    values = np.column_stack(
        [motion, histogram_change, brightness, sharpness, audio_energy, audio_flux, scene_boundary, coarse]
    ).astype(np.float32)
    return FeatureSequence(decoded.timestamps_sec, decoded.frame_indices, values)

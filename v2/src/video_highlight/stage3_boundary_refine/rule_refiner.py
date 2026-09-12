"""无需训练权重的保守时序打分基线。"""

from __future__ import annotations

from typing import Any

import numpy as np

from .feature_extractor import FEATURE_NAMES, FeatureSequence


def _robust_unit(values: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.zeros_like(values, dtype=np.float32)
    low, high = np.percentile(values, [10.0, 90.0])
    if float(high - low) < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    if window == 1 or len(values) < 2:
        return values.astype(np.float32)
    kernel = np.ones(window, dtype=np.float32) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def compute_rule_scores(
    features: FeatureSequence,
    candidate: dict[str, Any],
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """把低成本特征转换为类似 TCN 四个输出头的概率。"""

    columns = {name: features.values[:, index] for index, name in enumerate(FEATURE_NAMES)}
    motion = np.maximum(_robust_unit(columns["motion"]), _robust_unit(columns["histogram_change"]))
    audio = np.maximum(_robust_unit(columns["audio_energy"]), _robust_unit(columns["audio_flux"]))
    sharpness = _robust_unit(np.log1p(np.maximum(columns["sharpness"], 0.0)))
    weights = config.get("weights", {})
    activity = (
        float(weights.get("motion", 0.55)) * motion
        + float(weights.get("audio", 0.20)) * audio
        + float(weights.get("quality", 0.10)) * sharpness
        + float(weights.get("scene", 0.15)) * columns["scene_boundary"]
    )
    weight_sum = sum(float(weights.get(name, default)) for name, default in (
        ("motion", 0.55), ("audio", 0.20), ("quality", 0.10), ("scene", 0.15)
    ))
    activity = activity / max(weight_sum, 1e-8)
    activity = _smooth(activity, int(config.get("smooth_window", 5)))

    coarse = float(candidate.get("coarse_score", 0.0))
    coarse_weight = float(config.get("coarse_weight", 0.30))
    # 对金句等低运动类别降低活动依赖，避免规则基线误删静态语义高光。
    if str(candidate.get("category", "")) in {"insight", "emotion_peak", "humor"}:
        coarse_weight = max(coarse_weight, float(config.get("semantic_category_coarse_weight", 0.55)))
    highlight = np.clip((1.0 - coarse_weight) * activity + coarse_weight * coarse, 0.0, 1.0)
    highlight = _smooth(highlight, int(config.get("smooth_window", 5)))

    gradient = np.diff(highlight, prepend=highlight[0])
    positive = _robust_unit(np.maximum(gradient, 0.0))
    negative = _robust_unit(np.maximum(-gradient, 0.0))
    scene = np.clip(columns["scene_boundary"], 0.0, 1.0)
    boundary_scene_weight = float(config.get("boundary_scene_weight", 0.35))
    start = np.clip((1.0 - boundary_scene_weight) * positive + boundary_scene_weight * scene, 0.0, 1.0)
    end = np.clip((1.0 - boundary_scene_weight) * negative + boundary_scene_weight * scene, 0.0, 1.0)
    empty_probability = float(np.clip(1.0 - max(float(highlight.max()), coarse), 0.0, 1.0))
    return highlight.astype(np.float32), start.astype(np.float32), end.astype(np.float32), empty_probability

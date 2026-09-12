"""规则基线与可选 TCN 的统一帧级推理接口。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from video_highlight.common.exceptions import ConfigurationError

from .feature_extractor import FeatureSequence


@dataclass(frozen=True, slots=True)
class TemporalScores:
    highlight: np.ndarray
    start: np.ndarray
    end: np.ndarray
    empty_probability: float


class TemporalBackend(Protocol):
    def predict(self, features: FeatureSequence, candidate: dict[str, Any]) -> TemporalScores: ...


class RuleTemporalBackend:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def predict(self, features: FeatureSequence, candidate: dict[str, Any]) -> TemporalScores:
        from .rule_refiner import compute_rule_scores

        highlight, start, end, empty_probability = compute_rule_scores(
            features, candidate, self.config
        )
        return TemporalScores(highlight, start, end, empty_probability)


def build_temporal_backend(config: dict[str, Any]) -> TemporalBackend:
    name = str(config.get("backend", "rule")).lower()
    if name == "rule":
        return RuleTemporalBackend(config.get("rule", {}))
    if name == "tcn":
        from .tcn_inference import TorchScriptTCNBackend

        return TorchScriptTCNBackend(config.get("tcn", {}))
    raise ConfigurationError(f"未知 Stage 3 temporal.backend: {name}")

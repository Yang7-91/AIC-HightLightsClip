"""已有 TorchScript TCN 权重的纯推理适配，不包含训练逻辑。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from video_highlight.common.exceptions import ConfigurationError

from .feature_extractor import FeatureSequence
from .temporal_backend import TemporalScores


def _probability(values: Any) -> np.ndarray:
    array = values.detach().float().cpu().numpy().reshape(-1)
    if len(array) and (float(array.min()) < 0.0 or float(array.max()) > 1.0):
        array = 1.0 / (1.0 + np.exp(-array))
    return np.clip(array, 0.0, 1.0).astype(np.float32)


class TorchScriptTCNBackend:
    """加载外部训练好的模型；仅在配置 backend=tcn 时要求安装 PyTorch。"""

    def __init__(self, config: dict[str, Any]) -> None:
        checkpoint = Path(str(config.get("checkpoint", "")))
        if not checkpoint.is_file():
            raise ConfigurationError(f"TCN checkpoint 不存在: {checkpoint}")
        try:
            import torch
        except ImportError as error:
            raise ConfigurationError("使用 TCN 后端需要在 video-clip 环境安装 PyTorch") from error
        self.torch = torch
        requested = str(config.get("device", "cuda"))
        self.device = requested if requested != "cuda" or torch.cuda.is_available() else "cpu"
        self.model = torch.jit.load(str(checkpoint), map_location=self.device).eval()

    def predict(self, features: FeatureSequence, candidate: dict[str, Any]) -> TemporalScores:
        del candidate
        tensor = self.torch.from_numpy(features.values.T[None]).to(self.device)
        with self.torch.inference_mode():
            output = self.model(tensor)
        if isinstance(output, dict):
            highlight, start, end, empty = (
                output["highlight"], output["start"], output["end"], output["empty"]
            )
        elif isinstance(output, (tuple, list)) and len(output) == 4:
            highlight, start, end, empty = output
        else:
            raise ConfigurationError("TCN 输出必须是四元组或包含 highlight/start/end/empty 的字典")
        scores = TemporalScores(
            _probability(highlight),
            _probability(start),
            _probability(end),
            float(_probability(empty)[0]),
        )
        expected = len(features.timestamps_sec)
        if any(len(values) != expected for values in (scores.highlight, scores.start, scores.end)):
            raise ConfigurationError("TCN 帧级输出长度与输入序列不一致")
        return scores

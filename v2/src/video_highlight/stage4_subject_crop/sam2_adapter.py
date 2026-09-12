"""官方 SAM2 视频预测器的可选懒加载适配器。"""

from __future__ import annotations

from contextlib import ExitStack, nullcontext
from pathlib import Path
import tempfile
from typing import Any

import cv2
import numpy as np

from video_highlight.common.exceptions import ConfigurationError

from .prompt_generator import subject_point
from .subject_selector import select_subject_box
from .subject_tracker import TrackPoint


class SAM2SubjectTracker:
    def __init__(self, config: dict[str, Any]) -> None:
        checkpoint = Path(str(config.get("checkpoint", "")))
        model_config = str(config.get("model_config", ""))
        if not checkpoint.is_file() or not model_config:
            raise ConfigurationError("SAM2 后端要求有效的 tracking.checkpoint 和 model_config")
        try:
            import torch
            from sam2.build_sam import build_sam2_video_predictor
        except ImportError as error:
            raise ConfigurationError("SAM2 后端需要安装官方 sam2 包和兼容的 PyTorch") from error
        self.torch = torch
        self.device = str(config.get("device", "cuda"))
        self.amp_dtype = str(config.get("amp_dtype", "bfloat16"))
        self.predictor = build_sam2_video_predictor(model_config, str(checkpoint), device=self.device)
        self.config = config

    def _contexts(self) -> ExitStack:
        stack = ExitStack()
        stack.enter_context(self.torch.inference_mode())
        if self.device.startswith("cuda") and self.amp_dtype != "none":
            stack.enter_context(self.torch.autocast("cuda", dtype=getattr(self.torch, self.amp_dtype)))
        else:
            stack.enter_context(nullcontext())
        return stack

    def track(self, video_path: Path, interval: dict[str, Any], frame_size: tuple[int, int], scenes: list[dict[str, Any]]) -> list[TrackPoint]:
        del scenes
        start, end = int(interval["start_frame"]), int(interval["end_frame"])
        with tempfile.TemporaryDirectory(prefix="stage4-sam2-") as directory:
            capture = cv2.VideoCapture(str(video_path))
            capture.set(cv2.CAP_PROP_POS_FRAMES, start)
            first = None
            for local_index, _ in enumerate(range(start, end)):
                ok, frame = capture.read()
                if not ok:
                    capture.release()
                    raise RuntimeError("SAM2 区间解码提前结束")
                if first is None:
                    first = frame.copy()
                if not cv2.imwrite(str(Path(directory) / f"{local_index:06d}.jpg"), frame):
                    capture.release()
                    raise RuntimeError("SAM2 临时帧写入失败")
            capture.release()
            anchor, _, _ = select_subject_box(first, subject_point(interval, start), self.config)
            state = self.predictor.init_state(video_path=directory)
            results: dict[int, TrackPoint] = {}
            with self._contexts():
                self.predictor.add_new_points_or_box(inference_state=state, frame_idx=0, obj_id=1, box=np.asarray(anchor, dtype=np.float32))
                for local_frame, object_ids, logits in self.predictor.propagate_in_video(state):
                    mask = (logits[0].detach().float().cpu().numpy().squeeze() > 0).astype(np.uint8)
                    # 不假设预测器一定返回原图分辨率；统一映射回 Stage 1 记录的显示尺寸。
                    frame_width, frame_height = frame_size
                    if mask.shape != (frame_height, frame_width):
                        mask = cv2.resize(mask, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
                    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
                    if count <= 1:
                        continue
                    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                    x, y, width, height, area = stats[index]
                    frame = start + int(local_frame)
                    results[frame] = TrackPoint(frame, [float(x), float(y), float(x + width), float(y + height)], min(1.0, float(area) / max(1.0, width * height)), "sam2")
            if hasattr(self.predictor, "reset_state"):
                self.predictor.reset_state(state)
            if len(results) != end - start:
                raise RuntimeError("SAM2 未覆盖高光区间全部帧")
            return [results[frame] for frame in range(start, end)]

"""Hugging Face Grounding DINO 推理适配器；仅在配置启用时加载重依赖。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from video_highlight.common.exceptions import ConfigurationError

from .grounding_selector import GroundedDetection


class GroundingDINOAdapter:
    def __init__(self, config: dict[str, Any]) -> None:
        try:
            import torch
            from PIL import Image
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as error:
            raise ConfigurationError(
                "启用 Grounding DINO 需要安装 torch、Pillow 和 transformers"
            ) from error
        model_name = str(config.get("model", "")).strip()
        if not model_name:
            raise ConfigurationError("tracking.grounding.model 不能为空")
        local_only = bool(config.get("local_files_only", True))
        if local_only and not Path(model_name).exists():
            raise ConfigurationError(f"Grounding DINO 本地模型不存在: {model_name}")
        self.torch = torch
        self.Image = Image
        self.device = str(config.get("device", "cuda"))
        self.processor = AutoProcessor.from_pretrained(model_name, local_files_only=local_only)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_name, local_files_only=local_only
        ).to(self.device)
        self.model.eval()
        self.box_threshold = float(config.get("box_threshold", 0.25))
        self.text_threshold = float(config.get("text_threshold", 0.25))

    @staticmethod
    def build_prompt(phrases: list[str]) -> str:
        normalized: list[str] = []
        seen: set[str] = set()
        for value in phrases:
            phrase = str(value).strip().lower().rstrip(". ")
            if phrase and phrase not in seen:
                normalized.append(phrase)
                seen.add(phrase)
        return " . ".join(normalized) + (" ." if normalized else "")

    def detect(self, frame_bgr: np.ndarray, phrases: list[str]) -> list[GroundedDetection]:
        prompt = self.build_prompt(phrases)
        if not prompt:
            return []
        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self.Image.fromarray(rgb)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.get("input_ids"),
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(height, width)],
        )
        if not results:
            return []
        result = results[0]
        boxes = result.get("boxes", [])
        scores = result.get("scores", [])
        labels = result.get("text_labels", result.get("labels", []))
        output: list[GroundedDetection] = []
        for box, score, label in zip(boxes, scores, labels, strict=True):
            values = box.detach().float().cpu().tolist() if hasattr(box, "detach") else list(box)
            confidence = float(score.detach().float().cpu().item()) if hasattr(score, "detach") else float(score)
            x1, y1, x2, y2 = values
            x1, x2 = max(0.0, min(float(width), x1)), max(0.0, min(float(width), x2))
            y1, y2 = max(0.0, min(float(height), y1)), max(0.0, min(float(height), y2))
            if x2 <= x1 or y2 <= y1:
                continue
            output.append(GroundedDetection((x1, y1, x2, y2), confidence, str(label)))
        return output

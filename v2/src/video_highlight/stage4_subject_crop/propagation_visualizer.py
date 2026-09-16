"""SAM2 传播过程的低频率调试图与可选视频输出。"""

from __future__ import annotations

import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Sequence

import cv2
import numpy as np

from video_highlight.common.exceptions import ArtifactValidationError

from .subject_tracker import TrackPoint


class VisualizationError(ArtifactValidationError):
    """可视化产物读取、编码或写入失败。"""


def _read_image(path: Path) -> np.ndarray | None:
    """绕开 Windows OpenCV 对非 ASCII 路径支持不完整的问题。"""

    try:
        payload = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if payload.size == 0:
        return None
    return cv2.imdecode(payload, cv2.IMREAD_COLOR)


def _write_jpeg(path: Path, image: np.ndarray) -> bool:
    """使用 ``imencode + tofile`` 向中文路径安全写入 JPEG。"""

    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        return False
    try:
        encoded.tofile(str(path))
    except OSError:
        return False
    return True


class PropagationVisualizer:
    """按配置采样并持久化一个镜头内的 SAM2 传播可视化。

    采样只决定哪些帧被绘制，不改变 SAM2 的逐帧传播。锚点和窗口回退帧可绕过
    普通采样频率强制保存。视频在镜头完成后由最终图片按绝对帧号排序编码，因而
    同一帧先产生 SAM2 结果、后因窗口异常改为回退时，不会在视频中出现重复帧。
    """

    def __init__(
        self,
        config: dict[str, Any],
        output_dir: Path | None,
        source_fps: float,
        span_start: int,
    ) -> None:
        self.config = config
        self.source_fps = source_fps if math.isfinite(source_fps) and source_fps > 0 else 30.0
        self.span_start = span_start
        self.enabled = bool(config.get("enabled", False)) and output_dir is not None
        self.save_images = bool(config.get("save_images", True))
        self.write_video = bool(config.get("write_video", False))
        self.output_dir = output_dir
        self._video_frame_dir: Path | None = None
        if self.enabled and (self.save_images or self.write_video):
            assert output_dir is not None
            output_dir.mkdir(parents=True, exist_ok=True)
            if self.write_video and not self.save_images:
                self._video_frame_dir = output_dir / ".video_frames"
                self._video_frame_dir.mkdir(parents=True, exist_ok=True)

    @property
    def active(self) -> bool:
        return self.enabled and (self.save_images or self.write_video)

    def should_save(self, frame: int, *, is_anchor: bool, is_fallback: bool) -> bool:
        """判断绝对帧是否满足普通采样或强制保留条件。"""

        if not self.active:
            return False
        if is_anchor and bool(self.config.get("always_save_anchor_frames", True)):
            return True
        if is_fallback and bool(self.config.get("always_save_fallback_frames", True)):
            return True
        sample_fps = float(self.config.get("sample_fps", 2.0))
        # 用采样桶而不是 round(fps/sample_fps) 固定步长，可更准确处理 29.97 FPS。
        relative = max(0, frame - self.span_start)
        current_bucket = math.floor(relative * sample_fps / self.source_fps + 1e-9)
        if relative == 0:
            return True
        previous_bucket = math.floor((relative - 1) * sample_fps / self.source_fps + 1e-9)
        return current_bucket > previous_bucket

    @staticmethod
    def _draw_box(
        image: np.ndarray,
        box: Sequence[float],
        color: tuple[int, int, int],
        thickness: int,
    ) -> None:
        x1, y1, x2, y2 = (int(round(value)) for value in box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    def _render(
        self,
        frame: np.ndarray,
        prediction: TrackPoint,
        mask: np.ndarray | None,
        qwen_point: tuple[float, float] | None,
        prompt_box: Sequence[float] | None,
        is_anchor: bool,
        is_fallback: bool,
    ) -> np.ndarray:
        canvas = frame.copy()
        height, width = canvas.shape[:2]
        if mask is not None:
            if mask.shape != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            selected = mask.astype(bool)
            if np.any(selected):
                alpha = float(self.config.get("mask_alpha", 0.35))
                color = np.asarray([60, 210, 60], dtype=np.float32)
                pixels = canvas[selected].astype(np.float32)
                canvas[selected] = np.clip(pixels * (1.0 - alpha) + color * alpha, 0, 255).astype(np.uint8)
        if prompt_box is not None:
            self._draw_box(canvas, prompt_box, (0, 215, 255), 2)
        self._draw_box(canvas, prediction.subject_box, (0, 0, 255) if is_fallback else (255, 120, 0), 3)
        x1, y1, x2, y2 = prediction.subject_box
        center = (int(round((x1 + x2) * 0.5)), int(round((y1 + y2) * 0.5)))
        cv2.drawMarker(canvas, center, (255, 255, 255), cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
        if qwen_point is not None:
            qx = min(width - 1, max(0, int(round(qwen_point[0] * (width - 1)))))
            qy = min(height - 1, max(0, int(round(qwen_point[1] * (height - 1)))))
            cv2.circle(canvas, (qx, qy), 8, (255, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, (qx, qy), 13, (255, 255, 255), 2, cv2.LINE_AA)
        flags = ["ANCHOR"] if is_anchor else []
        if is_fallback:
            flags.append("FALLBACK")
        label = (
            f"frame={prediction.frame} source={prediction.source} "
            f"conf={prediction.confidence:.3f} {' '.join(flags)}"
        ).strip()
        cv2.rectangle(canvas, (0, 0), (min(width - 1, 1100), 42), (0, 0, 0), -1)
        cv2.putText(canvas, label, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
        return canvas

    def save(
        self,
        frame_index: int,
        source_frame_path: Path,
        prediction: TrackPoint,
        *,
        mask: np.ndarray | None = None,
        qwen_point: tuple[float, float] | None = None,
        prompt_box: Sequence[float] | None = None,
        is_anchor: bool = False,
        is_fallback: bool = False,
    ) -> None:
        if not self.should_save(frame_index, is_anchor=is_anchor, is_fallback=is_fallback):
            return
        frame = _read_image(source_frame_path)
        if frame is None:
            raise VisualizationError(f"无法读取 SAM2 可视化源帧: {source_frame_path}")
        canvas = self._render(frame, prediction, mask, qwen_point, prompt_box, is_anchor, is_fallback)
        assert self.output_dir is not None
        target_dir = self.output_dir if self.save_images else self._video_frame_dir
        assert target_dir is not None
        target = target_dir / f"frame_{frame_index:06d}.jpg"
        if not _write_jpeg(target, canvas):
            raise VisualizationError(f"SAM2 可视化图片写入失败: {target}")

    def close(self) -> None:
        """按最终帧图编码镜头调试视频，并清理仅供编码使用的隐藏图片。"""

        if not self.active or not self.write_video:
            return
        assert self.output_dir is not None
        frame_dir = self.output_dir if self.save_images else self._video_frame_dir
        assert frame_dir is not None
        images = sorted(frame_dir.glob("frame_*.jpg"))
        if not images:
            if self._video_frame_dir is not None:
                shutil.rmtree(self._video_frame_dir, ignore_errors=True)
            return
        first = _read_image(images[0])
        if first is None:
            raise VisualizationError(f"无法读取可视化视频首帧: {images[0]}")
        height, width = first.shape[:2]
        output_fps = min(self.source_fps, float(self.config.get("sample_fps", 2.0)))
        # VideoWriter 在 Windows 下同样可能无法创建中文路径。先编码到系统临时
        # ASCII 路径，完成后再由 pathlib 移动到正式输出目录。
        descriptor, temporary_name = tempfile.mkstemp(prefix="stage4-sam2-viz-", suffix=".mp4")
        os.close(descriptor)
        temporary_video = Path(temporary_name)
        temporary_video.unlink(missing_ok=True)
        writer = cv2.VideoWriter(
            str(temporary_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            max(0.1, output_fps),
            (width, height),
        )
        if not writer.isOpened():
            temporary_video.unlink(missing_ok=True)
            raise VisualizationError(f"无法创建 SAM2 可视化视频: {self.output_dir / 'propagation.mp4'}")
        try:
            for path in images:
                image = _read_image(path)
                if image is None:
                    raise VisualizationError(f"无法读取可视化视频帧: {path}")
                if image.shape[:2] != (height, width):
                    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
                writer.write(image)
        finally:
            writer.release()
        final_video = self.output_dir / "propagation.mp4"
        final_video.unlink(missing_ok=True)
        shutil.move(str(temporary_video), str(final_video))
        if self._video_frame_dir is not None:
            shutil.rmtree(self._video_frame_dir, ignore_errors=True)

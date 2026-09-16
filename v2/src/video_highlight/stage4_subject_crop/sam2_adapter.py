"""官方 SAM2 视频预测器适配器：镜头隔离、Qwen 锚点纠偏和窗口级降级。"""

from __future__ import annotations

from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
import math
from pathlib import Path
import tempfile
from typing import Any

import cv2
import numpy as np

from video_highlight.common.exceptions import ArtifactValidationError, ConfigurationError

from .keyframe_selector import interval_scene_spans
from .propagation_visualizer import PropagationVisualizer, VisualizationError
from .subject_selector import select_subject_box
from .subject_tracker import CenterSubjectTracker, TrackPoint


@dataclass(frozen=True, slots=True)
class AnchorWindow:
    """一个镜头内由当前条件帧覆盖的左闭右开 SAM2 传播窗口。"""

    start: int
    end: int
    point: tuple[float, float] | None
    is_qwen_anchor: bool


def _valid_point_rows(interval: dict[str, Any], start: int, end: int) -> list[dict[str, Any]]:
    """筛出当前镜头内坐标合法且非空的 Stage 3.5 观测。"""

    rows: list[dict[str, Any]] = []
    for row in interval.get("subject_points", []):
        frame = int(row.get("frame", -1))
        value = row.get("subject_point")
        if not (start <= frame < end and isinstance(value, (list, tuple)) and len(value) == 2):
            continue
        try:
            point = float(value[0]), float(value[1])
        except (TypeError, ValueError):
            continue
        if 0.0 <= point[0] <= 1.0 and 0.0 <= point[1] <= 1.0:
            rows.append({**row, "_point": point})
    return sorted(rows, key=lambda row: int(row["frame"]))


def anchor_windows(
    interval: dict[str, Any], span_start: int, span_end: int
) -> list[AnchorWindow]:
    """以每个有效 Qwen 点为边界，建立不跨镜头的单向传播窗口。

    镜头起点不一定恰好是 2 FPS 采样帧，因此会建立一个虚拟起始锚点，并使用
    当前镜头内时间最近的 Qwen 点。若整个镜头都没有点，则返回 ``point=None``，
    后续由视觉初始化尝试启动，失败时才使用固定画面中心。
    """

    rows = _valid_point_rows(interval, span_start, span_end)
    by_frame = {int(row["frame"]): row["_point"] for row in rows}
    boundaries = [span_start, *sorted(frame for frame in by_frame if frame > span_start), span_end]
    windows: list[AnchorWindow] = []
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        if start in by_frame:
            point = by_frame[start]
            is_qwen_anchor = True
        elif rows:
            nearest = min(rows, key=lambda row: abs(int(row["frame"]) - start))
            point = nearest["_point"]
            is_qwen_anchor = False
        else:
            point = None
            is_qwen_anchor = False
        windows.append(AnchorWindow(start, end, point, is_qwen_anchor))
    return windows


class SAM2SubjectTracker:
    """在单镜头状态内按相邻 Qwen 锚点窗口执行 SAM2 单向传播。"""

    def __init__(self, config: dict[str, Any], visualization_config: dict[str, Any] | None = None) -> None:
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
        self.visualization_config = visualization_config or {}

    def _contexts(self) -> ExitStack:
        stack = ExitStack()
        stack.enter_context(self.torch.inference_mode())
        if self.device.startswith("cuda") and self.amp_dtype != "none":
            stack.enter_context(self.torch.autocast("cuda", dtype=getattr(self.torch, self.amp_dtype)))
        else:
            stack.enter_context(nullcontext())
        return stack

    @staticmethod
    def _write_scene_frames(video_path: Path, start: int, end: int, directory: Path) -> None:
        """顺序解码一个镜头子段，供一次 SAM2 ``init_state`` 复用。"""

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ArtifactValidationError(f"OpenCV 无法打开视频: {video_path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        try:
            for local_index, frame_index in enumerate(range(start, end)):
                ok, frame = capture.read()
                if not ok:
                    raise ArtifactValidationError(f"SAM2 镜头解码在 frame={frame_index} 提前结束")
                if not cv2.imwrite(str(directory / f"{local_index:06d}.jpg"), frame):
                    raise ArtifactValidationError(f"SAM2 临时帧写入失败: frame={frame_index}")
        finally:
            capture.release()

    @staticmethod
    def _logits_to_mask(logits: Any, frame_size: tuple[int, int]) -> np.ndarray:
        """把 GPU logits 一次性转换为与显示画面尺寸一致的二值 Mask。"""

        mask = (logits[0].detach().float().cpu().numpy().squeeze() > 0).astype(np.uint8)
        frame_width, frame_height = frame_size
        if mask.shape != (frame_height, frame_width):
            mask = cv2.resize(mask, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
        return mask

    def _mask_track_point(
        self,
        absolute_frame: int,
        mask: np.ndarray,
        frame_size: tuple[int, int],
        prompt_point: tuple[float, float] | None,
        source: str,
    ) -> TrackPoint | None:
        """把 SAM2 logits 转为主体框；锚点帧优先选择包含 Qwen 点的连通域。"""

        frame_width, frame_height = frame_size
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if count <= 1:
            return None
        component = 0
        if prompt_point is not None:
            px = min(frame_width - 1, max(0, int(round(prompt_point[0] * (frame_width - 1)))))
            py = min(frame_height - 1, max(0, int(round(prompt_point[1] * (frame_height - 1)))))
            component = int(labels[py, px])
        if component <= 0:
            component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, width, height, area = stats[component]
        box = [float(x), float(y), float(x + width), float(y + height)]
        confidence = min(1.0, float(area) / max(1.0, float(width * height)))
        return TrackPoint(absolute_frame, box, confidence, source)

    def _anchor_is_valid(
        self,
        prediction: TrackPoint | None,
        point: tuple[float, float] | None,
        frame_size: tuple[int, int],
    ) -> bool:
        """检查锚点 Mask 是否包含或足够靠近 Qwen 中心点。"""

        if prediction is None:
            return False
        if point is None:
            return True
        frame_width, frame_height = frame_size
        px, py = point[0] * frame_width, point[1] * frame_height
        x1, y1, x2, y2 = prediction.subject_box
        if x1 <= px <= x2 and y1 <= py <= y2:
            return True
        center_x, center_y = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        distance_ratio = math.hypot(center_x - px, center_y - py) / max(
            1.0, math.hypot(frame_width, frame_height)
        )
        return distance_ratio <= float(self.config.get("anchor_max_center_distance_ratio", 0.20))

    def _add_anchor_prompt(
        self,
        state: Any,
        local_frame: int,
        absolute_frame: int,
        frame: np.ndarray,
        point: tuple[float, float] | None,
        frame_size: tuple[int, int],
    ) -> TrackPoint:
        """先用 Qwen 中心小框提示；校验失败时追加正点提示进行一次纠偏。"""

        anchor_box, _, _ = select_subject_box(frame, point, self.config)
        _, _, logits = self.predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=local_frame,
            obj_id=1,
            box=np.asarray(anchor_box, dtype=np.float32),
        )
        mask = self._logits_to_mask(logits, frame_size)
        prediction = self._mask_track_point(absolute_frame, mask, frame_size, point, "sam2_anchor_box")
        if self._anchor_is_valid(prediction, point, frame_size):
            return prediction
        if point is not None and bool(self.config.get("correction_with_point_prompt", True)):
            width, height = frame_size
            points = np.asarray([[point[0] * width, point[1] * height]], dtype=np.float32)
            labels = np.asarray([1], dtype=np.int32)
            _, _, logits = self.predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=local_frame,
                obj_id=1,
                points=points,
                labels=labels,
                clear_old_points=False,
            )
            mask = self._logits_to_mask(logits, frame_size)
            prediction = self._mask_track_point(absolute_frame, mask, frame_size, point, "sam2_anchor_corrected")
            if self._anchor_is_valid(prediction, point, frame_size):
                return prediction
        raise RuntimeError(f"SAM2 锚点 Mask 未通过 Qwen 中心校验: frame={absolute_frame}")

    @staticmethod
    def _fallback_point(point: TrackPoint) -> TrackPoint:
        """标明仅当前 SAM2 窗口使用了 Qwen 插值/中心降级。"""

        return TrackPoint(
            point.frame,
            list(point.subject_box),
            min(0.5, point.confidence),
            f"sam2_window_fallback_{point.source}",
        )

    def _track_scene(
        self,
        video_path: Path,
        interval: dict[str, Any],
        frame_size: tuple[int, int],
        span_start: int,
        span_end: int,
        fallback_by_frame: dict[int, TrackPoint],
        visualization_dir: Path | None,
        source_fps: float,
        scene_index: int,
    ) -> list[TrackPoint]:
        """在一个 SAM2 状态内依次处理相邻锚点窗口，绝不跨镜头复用记忆。"""

        with tempfile.TemporaryDirectory(prefix="stage4-sam2-scene-") as directory_name:
            directory = Path(directory_name)
            self._write_scene_frames(video_path, span_start, span_end, directory)
            scene_visualization_dir = None
            if visualization_dir is not None:
                scene_visualization_dir = (
                    visualization_dir
                    / str(interval["interval_id"])
                    / f"scene_{scene_index:03d}_{span_start}_{span_end}"
                )
            visualizer = PropagationVisualizer(
                self.visualization_config,
                scene_visualization_dir,
                source_fps,
                span_start,
            )
            state = self.predictor.init_state(
                video_path=str(directory),
                offload_video_to_cpu=bool(self.config.get("offload_video_to_cpu", True)),
                offload_state_to_cpu=bool(self.config.get("offload_state_to_cpu", False)),
                async_loading_frames=bool(self.config.get("async_loading_frames", False)),
            )
            results: dict[int, TrackPoint] = {}

            def apply_fallback(window: AnchorWindow, only_frame: int | None = None) -> None:
                """写入窗口级回退结果，并按配置保存没有 Mask 的诊断图。"""

                frame_range = (
                    range(only_frame, only_frame + 1)
                    if only_frame is not None
                    else range(window.start, window.end)
                )
                for frame_index in frame_range:
                    prediction = self._fallback_point(fallback_by_frame[frame_index])
                    results[frame_index] = prediction
                    is_anchor = frame_index == window.start and window.is_qwen_anchor
                    visualizer.save(
                        frame_index,
                        directory / f"{frame_index - span_start:06d}.jpg",
                        prediction,
                        qwen_point=window.point if frame_index == window.start else None,
                        is_anchor=is_anchor,
                        is_fallback=True,
                    )

            try:
                with self._contexts():
                    for window in anchor_windows(interval, span_start, span_end):
                        local_start = window.start - span_start
                        frame = cv2.imread(str(directory / f"{local_start:06d}.jpg"))
                        if frame is None:
                            # 只降级当前窗口；下一 Qwen 锚点仍有机会恢复 SAM2。
                            apply_fallback(window)
                            continue
                        try:
                            self._add_anchor_prompt(
                                state,
                                local_start,
                                window.start,
                                frame,
                                window.point,
                                frame_size,
                            )
                            # max_frame_num_to_track 在官方实现中是相对起点的最大偏移，
                            # 因而窗口长度 L 对应 L-1；生成器会包含条件帧本身。
                            for local_frame, _, logits in self.predictor.propagate_in_video(
                                state,
                                start_frame_idx=local_start,
                                max_frame_num_to_track=max(0, window.end - window.start - 1),
                                reverse=False,
                            ):
                                absolute_frame = span_start + int(local_frame)
                                if not window.start <= absolute_frame < window.end:
                                    continue
                                prompt = window.point if absolute_frame == window.start else None
                                mask = self._logits_to_mask(logits, frame_size)
                                prediction = self._mask_track_point(
                                    absolute_frame,
                                    mask,
                                    frame_size,
                                    prompt,
                                    "sam2_anchor" if absolute_frame == window.start else "sam2_propagated",
                                )
                                if prediction is not None:
                                    results[absolute_frame] = prediction
                                    is_anchor = absolute_frame == window.start and window.is_qwen_anchor
                                    prompt_box = None
                                    if absolute_frame == window.start:
                                        prompt_frame = frame
                                        prompt_box, _, _ = select_subject_box(
                                            prompt_frame, window.point, self.config
                                        )
                                    visualizer.save(
                                        absolute_frame,
                                        directory / f"{int(local_frame):06d}.jpg",
                                        prediction,
                                        mask=mask,
                                        qwen_point=window.point if absolute_frame == window.start else None,
                                        prompt_box=prompt_box,
                                        is_anchor=is_anchor,
                                        is_fallback=False,
                                    )
                        except VisualizationError:
                            # 调试产物失败不能伪装成跟踪失败，否则会把正确的 SAM2
                            # 结果替换为 Qwen 回退，同时再次写图并掩盖真正原因。
                            raise
                        except Exception:
                            # CUDA/Mask/锚点异常的影响限制在当前相邻锚点窗口。已经成功的
                            # 镜头帧保留，下一锚点仍会重新输入 Qwen 观测进行纠偏。
                            apply_fallback(window)
                            continue
                        # SAM2 偶发不返回某一帧 Mask 时只补该帧，而非令整个区间回退。
                        for frame_index in range(window.start, window.end):
                            if frame_index not in results:
                                apply_fallback(window, only_frame=frame_index)
            finally:
                if hasattr(self.predictor, "reset_state"):
                    self.predictor.reset_state(state)
                visualizer.close()
            return [results[frame] for frame in range(span_start, span_end)]

    def track(
        self,
        video_path: Path,
        interval: dict[str, Any],
        frame_size: tuple[int, int],
        scenes: list[dict[str, Any]],
        visualization_dir: Path | None = None,
    ) -> list[TrackPoint]:
        if not video_path.is_file():
            raise ArtifactValidationError(f"源视频不存在: {video_path}")
        # Qwen 线性插值是逐窗口降级轨迹；它自身同样按镜头切分，不跨硬切插值。
        fallback = CenterSubjectTracker(self.config).track(
            Path(), interval, frame_size, scenes
        )
        fallback_by_frame = {point.frame: point for point in fallback}
        capture = cv2.VideoCapture(str(video_path))
        source_fps = float(capture.get(cv2.CAP_PROP_FPS)) if capture.isOpened() else 0.0
        capture.release()
        output: list[TrackPoint] = []
        for scene_index, (span_start, span_end) in enumerate(interval_scene_spans(interval, scenes)):
            output.extend(
                self._track_scene(
                    video_path,
                    interval,
                    frame_size,
                    span_start,
                    span_end,
                    fallback_by_frame,
                    visualization_dir,
                    source_fps,
                    scene_index,
                )
            )
        expected = list(range(int(interval["start_frame"]), int(interval["end_frame"])))
        if [point.frame for point in output] != expected:
            raise RuntimeError("SAM2 镜头窗口没有完整覆盖高光区间")
        return output

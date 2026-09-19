"""Grounding DINO 多目标锚定 + SAM2 镜头内窗口传播。"""

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
from video_highlight.contracts.schema_versions import STAGE4_SCHEMA_VERSION

from .grounding_selector import (
    GroundedDetection,
    ScoredDetection,
    associate_object_ids,
    box_iou,
    select_detections,
)
from .keyframe_selector import interval_scene_spans
from .prompt_generator import observation_points, subject_observations
from .propagation_visualizer import PropagationVisualizer, VisualizationError
from .subject_selector import select_subject_box
from .subject_tracker import CenterSubjectTracker, TrackPoint


@dataclass(frozen=True, slots=True)
class AnchorWindow:
    """一个镜头内由当前 Stage 3.5 观察覆盖的左闭右开传播窗口。"""

    start: int
    end: int
    observation: dict[str, Any] | None
    is_qwen_anchor: bool

    @property
    def points(self) -> list[tuple[float, float]]:
        return [] if self.observation is None else observation_points(self.observation)

    @property
    def grounding_phrases(self) -> list[str]:
        if self.observation is None:
            return []
        values = self.observation.get("grounding_phrases", [])
        return [str(value) for value in values if str(value).strip()]

    @property
    def group_mode(self) -> str:
        if self.observation is None:
            return "single"
        return str(self.observation.get("group_mode", "multiple" if len(self.points) > 1 else "single"))

    @property
    def point(self) -> tuple[float, float] | None:
        """保留旧测试和可视化使用的群体中心属性。"""

        points = self.points
        if not points:
            return None
        return (
            (min(point[0] for point in points) + max(point[0] for point in points)) * 0.5,
            (min(point[1] for point in points) + max(point[1] for point in points)) * 0.5,
        )


def anchor_windows(interval: dict[str, Any], span_start: int, span_end: int) -> list[AnchorWindow]:
    """用每条 Stage 3.5 观察建立窗口，镜头起点使用最近观察作为虚拟锚点。"""

    rows = [
        row for row in subject_observations(interval)
        if span_start <= int(row.get("frame", -1)) < span_end
        and (
            observation_points(row)
            or any(str(value).strip() for value in row.get("grounding_phrases", []))
        )
    ]
    by_frame = {int(row["frame"]): row for row in rows}
    boundaries = [span_start, *sorted(frame for frame in by_frame if frame > span_start), span_end]
    windows: list[AnchorWindow] = []
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        if start in by_frame:
            observation = by_frame[start]
            is_qwen_anchor = True
        elif rows:
            observation = min(rows, key=lambda row: abs(int(row["frame"]) - start))
            is_qwen_anchor = False
        else:
            observation = None
            is_qwen_anchor = False
        windows.append(AnchorWindow(start, end, observation, is_qwen_anchor))
    return windows


def _union_box(boxes: list[list[float] | tuple[float, ...]]) -> list[float] | None:
    if not boxes:
        return None
    return [
        min(float(box[0]) for box in boxes),
        min(float(box[1]) for box in boxes),
        max(float(box[2]) for box in boxes),
        max(float(box[3]) for box in boxes),
    ]


def _mask_box(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


class SAM2SubjectTracker:
    """用 Grounding DINO 选择多个语义对象，并在每个锚点窗口用 SAM2 传播。"""

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
        grounding_config = config.get("grounding", {})
        self.grounding_config = grounding_config if isinstance(grounding_config, dict) else {}
        self.grounder = None
        if bool(self.grounding_config.get("enabled", False)):
            from .grounding_dino_adapter import GroundingDINOAdapter

            self.grounder = GroundingDINOAdapter(self.grounding_config)
        self.last_grounding_records: list[dict[str, Any]] = []

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
        mask = (logits.detach().float().cpu().numpy().squeeze() > 0).astype(np.uint8)
        width, height = frame_size
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        return mask

    def _active_masks(
        self,
        object_ids: Any,
        logits: Any,
        active_ids: set[int],
        frame_size: tuple[int, int],
    ) -> tuple[np.ndarray, dict[int, list[float]]]:
        width, height = frame_size
        union = np.zeros((height, width), dtype=np.uint8)
        boxes: dict[int, list[float]] = {}
        ids = [
            int(value.detach().cpu().item()) if hasattr(value, "detach")
            else int(value.item()) if hasattr(value, "item")
            else int(value)
            for value in object_ids
        ]
        for index, object_id in enumerate(ids):
            if object_id not in active_ids:
                continue
            mask = self._logits_to_mask(logits[index], frame_size)
            box = _mask_box(mask)
            if box is None:
                continue
            union |= mask
            boxes[object_id] = box
        return union, boxes

    @staticmethod
    def _track_point(
        frame: int, mask: np.ndarray, source: str, object_ids: tuple[int, ...] = ()
    ) -> TrackPoint | None:
        box = _mask_box(mask)
        if box is None:
            return None
        area = float(mask.sum())
        box_area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
        return TrackPoint(
            frame, box, min(1.0, area / box_area), source,
            max(1, len(object_ids)), object_ids
        )

    def _anchor_is_valid(
        self,
        prediction: TrackPoint | None,
        mask: np.ndarray,
        points: list[tuple[float, float]],
        selected_boxes: list[tuple[float, ...]],
        frame_size: tuple[int, int],
    ) -> bool:
        if prediction is None:
            return False
        width, height = frame_size
        if points:
            hits = 0
            for point in points:
                px = min(width - 1, max(0, int(round(point[0] * (width - 1)))))
                py = min(height - 1, max(0, int(round(point[1] * (height - 1)))))
                hits += int(mask[py, px] > 0)
            if hits / len(points) < float(self.config.get("anchor_min_point_coverage", 0.5)):
                center_x = (prediction.subject_box[0] + prediction.subject_box[2]) * 0.5
                center_y = (prediction.subject_box[1] + prediction.subject_box[3]) * 0.5
                nearest = min(
                    math.hypot(center_x - point[0] * width, center_y - point[1] * height)
                    for point in points
                ) / max(1.0, math.hypot(width, height))
                if nearest > float(self.config.get("anchor_max_center_distance_ratio", 0.20)):
                    return False
        grounded_union = _union_box([list(box) for box in selected_boxes])
        if grounded_union is not None:
            grounded_area = max(1.0, (grounded_union[2] - grounded_union[0]) * (grounded_union[3] - grounded_union[1]))
            x1 = max(grounded_union[0], prediction.subject_box[0])
            y1 = max(grounded_union[1], prediction.subject_box[1])
            x2 = min(grounded_union[2], prediction.subject_box[2])
            y2 = min(grounded_union[3], prediction.subject_box[3])
            coverage = max(0.0, x2 - x1) * max(0.0, y2 - y1) / grounded_area
            if coverage < float(self.config.get("grounding_min_coverage", 0.10)):
                return False
        return True

    def _fallback_detections(
        self, frame: np.ndarray, points: list[tuple[float, float]]
    ) -> list[GroundedDetection]:
        if points:
            return [
                GroundedDetection(tuple(select_subject_box(frame, point, self.config)[0]), 0.5, "qwen target")
                for point in points
            ]
        box, confidence, source = select_subject_box(frame, None, self.config)
        return [GroundedDetection(tuple(box), confidence, source)]

    def _resolve_detections(
        self,
        frame: np.ndarray,
        window: AnchorWindow,
        previous_boxes: dict[int, list[float]],
        frame_size: tuple[int, int],
    ) -> tuple[list[GroundedDetection], list[GroundedDetection], list[ScoredDetection], str, str | None]:
        raw: list[GroundedDetection] = []
        scored: list[ScoredDetection] = []
        error_message: str | None = None
        if self.grounder is not None and window.grounding_phrases:
            try:
                raw = self.grounder.detect(frame, window.grounding_phrases)
                selected, scored = select_detections(
                    raw,
                    window.points,
                    previous_boxes,
                    window.group_mode,
                    frame_size,
                    self.grounding_config,
                )
                if selected:
                    return selected, raw, scored, "grounding_dino", None
            except Exception as error:
                error_message = f"{type(error).__name__}: {error}"
                if str(self.grounding_config.get("error_policy", "qwen")) == "error":
                    raise
        return self._fallback_detections(frame, window.points), raw, scored, "qwen_fallback", error_message

    @staticmethod
    def _point_assignments(
        points: list[tuple[float, float]],
        assignments: list[tuple[int, GroundedDetection]],
        frame_size: tuple[int, int],
    ) -> dict[int, list[tuple[float, float]]]:
        width, height = frame_size
        output: dict[int, list[tuple[float, float]]] = {object_id: [] for object_id, _ in assignments}
        for point in points:
            px, py = point[0] * width, point[1] * height
            choices: list[tuple[float, int]] = []
            for object_id, detection in assignments:
                box = detection.box_xyxy
                contains = 0.0 if box[0] <= px <= box[2] and box[1] <= py <= box[3] else 1.0
                center = ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)
                choices.append((contains + math.hypot(px - center[0], py - center[1]), object_id))
            if choices:
                output[min(choices)[1]].append(point)
        return output

    def _add_anchor_prompts(
        self,
        state: Any,
        local_frame: int,
        absolute_frame: int,
        assignments: list[tuple[int, GroundedDetection]],
        points: list[tuple[float, float]],
        frame_size: tuple[int, int],
        source_prefix: str,
    ) -> tuple[TrackPoint, np.ndarray, dict[int, list[float]]]:
        object_ids: Any = []
        logits: Any = []
        active_ids = {object_id for object_id, _ in assignments}
        for object_id, detection in assignments:
            _, object_ids, logits = self.predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=local_frame,
                obj_id=object_id,
                box=np.asarray(detection.box_xyxy, dtype=np.float32),
            )
        union_mask, object_boxes = self._active_masks(object_ids, logits, active_ids, frame_size)
        ordered_ids = tuple(sorted(active_ids))
        prediction = self._track_point(
            absolute_frame, union_mask, f"sam2_{source_prefix}_anchor", ordered_ids
        )
        selected_boxes = [detection.box_xyxy for _, detection in assignments]
        if self._anchor_is_valid(prediction, union_mask, points, selected_boxes, frame_size):
            assert prediction is not None
            return prediction, union_mask, object_boxes

        # 将每个 Qwen 正点追加到离它最近的已分配对象，不清除已有框提示。
        width, height = frame_size
        for object_id, assigned_points in self._point_assignments(points, assignments, frame_size).items():
            if not assigned_points:
                continue
            coordinates = np.asarray(
                [[point[0] * width, point[1] * height] for point in assigned_points], dtype=np.float32
            )
            labels = np.ones((len(assigned_points),), dtype=np.int32)
            _, object_ids, logits = self.predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=local_frame,
                obj_id=object_id,
                points=coordinates,
                labels=labels,
                clear_old_points=False,
            )
        union_mask, object_boxes = self._active_masks(object_ids, logits, active_ids, frame_size)
        prediction = self._track_point(
            absolute_frame, union_mask, f"sam2_{source_prefix}_corrected", ordered_ids
        )
        if self._anchor_is_valid(prediction, union_mask, points, selected_boxes, frame_size):
            assert prediction is not None
            return prediction, union_mask, object_boxes
        raise RuntimeError(f"SAM2 多目标锚点 Mask 校验失败: frame={absolute_frame}")

    @staticmethod
    def _fallback_point(point: TrackPoint) -> TrackPoint:
        return TrackPoint(
            point.frame,
            list(point.subject_box),
            min(0.5, point.confidence),
            f"sam2_window_fallback_{point.source}",
            point.object_count,
            point.object_ids,
        )

    @staticmethod
    def _detection_dict(detection: GroundedDetection) -> dict[str, Any]:
        return {
            "box_xyxy": [float(value) for value in detection.box_xyxy],
            "score": float(detection.score),
            "phrase": detection.phrase,
        }

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
        with tempfile.TemporaryDirectory(prefix="stage4-sam2-scene-") as directory_name:
            directory = Path(directory_name)
            self._write_scene_frames(video_path, span_start, span_end, directory)
            scene_visualization_dir = None
            if visualization_dir is not None:
                scene_visualization_dir = visualization_dir / str(interval["interval_id"]) / f"scene_{scene_index:03d}_{span_start}_{span_end}"
            visualizer = PropagationVisualizer(
                self.visualization_config, scene_visualization_dir, source_fps, span_start
            )
            state = self.predictor.init_state(
                video_path=str(directory),
                offload_video_to_cpu=bool(self.config.get("offload_video_to_cpu", True)),
                offload_state_to_cpu=bool(self.config.get("offload_state_to_cpu", False)),
                async_loading_frames=bool(self.config.get("async_loading_frames", False)),
            )
            results: dict[int, TrackPoint] = {}
            previous_object_boxes: dict[int, list[float]] = {}
            next_object_id = 1

            def apply_fallback(window: AnchorWindow, only_frame: int | None = None) -> None:
                frame_range = range(only_frame, only_frame + 1) if only_frame is not None else range(window.start, window.end)
                for frame_index in frame_range:
                    prediction = self._fallback_point(fallback_by_frame[frame_index])
                    results[frame_index] = prediction
                    visualizer.save(
                        frame_index,
                        directory / f"{frame_index - span_start:06d}.jpg",
                        prediction,
                        qwen_point=window.point if frame_index == window.start else None,
                        is_anchor=frame_index == window.start and window.is_qwen_anchor,
                        is_fallback=True,
                    )

            try:
                with self._contexts():
                    for window in anchor_windows(interval, span_start, span_end):
                        local_start = window.start - span_start
                        frame = cv2.imread(str(directory / f"{local_start:06d}.jpg"))
                        if frame is None:
                            apply_fallback(window)
                            continue
                        grounding_source = "unresolved"
                        try:
                            selected, raw, scored, grounding_source, grounding_error = self._resolve_detections(
                                frame, window, previous_object_boxes, frame_size
                            )
                            assignments, next_object_id = associate_object_ids(
                                selected,
                                previous_object_boxes,
                                next_object_id,
                                frame_size,
                                self.grounding_config,
                            )
                            active_ids = {object_id for object_id, _ in assignments}
                            anchor_prediction, anchor_mask, anchor_object_boxes = self._add_anchor_prompts(
                                state,
                                local_start,
                                window.start,
                                assignments,
                                window.points,
                                frame_size,
                                grounding_source,
                            )
                            previous_object_boxes = anchor_object_boxes
                            self.last_grounding_records.append({
                                "schema_version": STAGE4_SCHEMA_VERSION,
                                "video_id": interval["video_id"],
                                "interval_id": interval["interval_id"],
                                "scene_index": scene_index,
                                "frame": window.start,
                                "window_end": window.end,
                                "is_qwen_anchor": window.is_qwen_anchor,
                                "group_mode": window.group_mode,
                                "qwen_targets": [] if window.observation is None else list(window.observation.get("targets", [])),
                                "qwen_points": [list(point) for point in window.points],
                                "grounding_phrases": window.grounding_phrases,
                                "raw_detections": [self._detection_dict(row) for row in raw],
                                "scored_detections": [{
                                    **self._detection_dict(row.detection),
                                    "total_score": row.total_score,
                                    "point_score": row.point_score,
                                    "temporal_score": row.temporal_score,
                                } for row in scored],
                                "selected_objects": [{
                                    "object_id": object_id,
                                    **self._detection_dict(detection),
                                } for object_id, detection in assignments],
                                "source": grounding_source,
                                "error": grounding_error,
                                "status": "accepted",
                            })
                            results[window.start] = anchor_prediction
                            visualizer.save(
                                window.start,
                                directory / f"{local_start:06d}.jpg",
                                anchor_prediction,
                                mask=anchor_mask,
                                qwen_point=window.point,
                                prompt_box=_union_box([list(detection.box_xyxy) for _, detection in assignments]),
                                is_anchor=window.is_qwen_anchor,
                                is_fallback=False,
                            )
                            for local_frame, object_ids, logits in self.predictor.propagate_in_video(
                                state,
                                start_frame_idx=local_start,
                                max_frame_num_to_track=max(0, window.end - window.start - 1),
                                reverse=False,
                            ):
                                absolute_frame = span_start + int(local_frame)
                                if not window.start <= absolute_frame < window.end:
                                    continue
                                union_mask, object_boxes = self._active_masks(object_ids, logits, active_ids, frame_size)
                                prediction = self._track_point(
                                    absolute_frame,
                                    union_mask,
                                    f"sam2_{grounding_source}_anchor" if absolute_frame == window.start else f"sam2_{grounding_source}_propagated",
                                    tuple(sorted(active_ids)),
                                )
                                if prediction is None:
                                    continue
                                results[absolute_frame] = prediction
                                previous_object_boxes = object_boxes or previous_object_boxes
                                visualizer.save(
                                    absolute_frame,
                                    directory / f"{int(local_frame):06d}.jpg",
                                    prediction,
                                    mask=union_mask,
                                    qwen_point=window.point if absolute_frame == window.start else None,
                                    is_anchor=absolute_frame == window.start and window.is_qwen_anchor,
                                    is_fallback=False,
                                )
                        except VisualizationError:
                            raise
                        except Exception as error:
                            self.last_grounding_records.append({
                                "schema_version": STAGE4_SCHEMA_VERSION,
                                "video_id": interval["video_id"],
                                "interval_id": interval["interval_id"],
                                "scene_index": scene_index,
                                "frame": window.start,
                                "window_end": window.end,
                                "qwen_targets": [] if window.observation is None else list(window.observation.get("targets", [])),
                                "qwen_points": [list(point) for point in window.points],
                                "grounding_phrases": window.grounding_phrases,
                                "source": grounding_source,
                                "status": "window_fallback",
                                "error": f"{type(error).__name__}: {error}",
                            })
                            apply_fallback(window)
                            continue
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
        self.last_grounding_records = []
        fallback = CenterSubjectTracker(self.config).track(Path(), interval, frame_size, scenes)
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

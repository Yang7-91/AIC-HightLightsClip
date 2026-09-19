"""Grounding DINO 多框的空间筛选、去重和跨锚点对象 ID 关联。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True, slots=True)
class GroundedDetection:
    box_xyxy: tuple[float, float, float, float]
    score: float
    phrase: str


@dataclass(frozen=True, slots=True)
class ScoredDetection:
    detection: GroundedDetection
    total_score: float
    point_score: float
    temporal_score: float


def box_iou(left: tuple[float, ...] | list[float], right: tuple[float, ...] | list[float]) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left[2]) - float(left[0])) * max(0.0, float(left[3]) - float(left[1]))
    right_area = max(0.0, float(right[2]) - float(right[0])) * max(0.0, float(right[3]) - float(right[1]))
    return intersection / max(1e-9, left_area + right_area - intersection)


def _center_score(
    left: tuple[float, ...] | list[float], right: tuple[float, ...] | list[float],
    frame_size: tuple[int, int]
) -> float:
    left_center = ((float(left[0]) + float(left[2])) * 0.5, (float(left[1]) + float(left[3])) * 0.5)
    right_center = ((float(right[0]) + float(right[2])) * 0.5, (float(right[1]) + float(right[3])) * 0.5)
    diagonal = max(1.0, math.hypot(*frame_size))
    distance = math.hypot(left_center[0] - right_center[0], left_center[1] - right_center[1]) / diagonal
    return math.exp(-distance / 0.12)


def _point_score(
    box: tuple[float, ...], points: list[tuple[float, float]], frame_size: tuple[int, int]
) -> float:
    if not points:
        return 0.5
    width, height = frame_size
    box_center = ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)
    diagonal = max(1.0, math.hypot(width, height))
    scores: list[float] = []
    for point in points:
        px, py = point[0] * width, point[1] * height
        if box[0] <= px <= box[2] and box[1] <= py <= box[3]:
            scores.append(1.0)
        else:
            distance = math.hypot(px - box_center[0], py - box_center[1]) / diagonal
            scores.append(math.exp(-distance / 0.10))
    return max(scores)


def _deduplicate(detections: list[GroundedDetection], threshold: float) -> list[GroundedDetection]:
    output: list[GroundedDetection] = []
    for detection in sorted(detections, key=lambda row: row.score, reverse=True):
        if any(box_iou(detection.box_xyxy, kept.box_xyxy) >= threshold for kept in output):
            continue
        output.append(detection)
    return output


def score_detections(
    detections: list[GroundedDetection],
    qwen_points: list[tuple[float, float]],
    previous_boxes: dict[int, list[float]],
    frame_size: tuple[int, int],
    config: dict[str, Any],
) -> list[ScoredDetection]:
    detection_weight = float(config.get("detection_weight", 0.40))
    point_weight = float(config.get("point_weight", 0.35))
    temporal_weight = float(config.get("temporal_weight", 0.25))
    rows: list[ScoredDetection] = []
    for detection in _deduplicate(detections, float(config.get("nms_iou", 0.85))):
        point = _point_score(detection.box_xyxy, qwen_points, frame_size)
        temporal = max(
            (
                0.65 * box_iou(detection.box_xyxy, previous)
                + 0.35 * _center_score(detection.box_xyxy, previous, frame_size)
                for previous in previous_boxes.values()
            ),
            default=0.5,
        )
        total = detection_weight * detection.score + point_weight * point + temporal_weight * temporal
        rows.append(ScoredDetection(detection, total, point, temporal))
    return sorted(rows, key=lambda row: row.total_score, reverse=True)


def select_detections(
    detections: list[GroundedDetection],
    qwen_points: list[tuple[float, float]],
    previous_boxes: dict[int, list[float]],
    group_mode: str,
    frame_size: tuple[int, int],
    config: dict[str, Any],
) -> tuple[list[GroundedDetection], list[ScoredDetection]]:
    """选择单个或多个主体框；multiple 模式优先让不同 Qwen 点认领不同框。"""

    scored = score_detections(detections, qwen_points, previous_boxes, frame_size, config)
    threshold = float(config.get("selection_threshold", 0.45))
    maximum = max(1, int(config.get("max_objects", 8)))
    eligible = [row for row in scored if row.total_score >= threshold]
    if not eligible:
        return [], scored
    if group_mode != "multiple":
        return [eligible[0].detection], scored

    selected: list[GroundedDetection] = []
    used: set[int] = set()
    width, height = frame_size
    # 每个 Qwen 点优先认领一个包含它或离它最近的独立检测框。
    for point in qwen_points:
        px, py = point[0] * width, point[1] * height
        choices: list[tuple[float, int, ScoredDetection]] = []
        for index, row in enumerate(eligible):
            if index in used:
                continue
            box = row.detection.box_xyxy
            individual_point_score = _point_score(box, [point], frame_size)
            if individual_point_score < float(config.get("claim_min_point_score", 0.20)):
                continue
            contains = 1.0 if box[0] <= px <= box[2] and box[1] <= py <= box[3] else 0.0
            center = ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)
            distance = math.hypot(px - center[0], py - center[1]) / max(1.0, math.hypot(width, height))
            choices.append((contains * 2.0 + row.total_score - distance, index, row))
        if choices:
            _, index, row = max(choices, key=lambda value: value[0])
            used.add(index)
            selected.append(row.detection)

    # Qwen 可能漏掉群体中的成员；允许加入语义得分高且空间关系合理的额外框。
    minimum_point = float(config.get("multiple_min_point_score", 0.35))
    for index, row in enumerate(eligible):
        if len(selected) >= maximum:
            break
        if index in used:
            continue
        if not qwen_points or row.point_score >= minimum_point:
            selected.append(row.detection)
            used.add(index)
    return selected[:maximum], scored


def associate_object_ids(
    detections: list[GroundedDetection],
    previous_boxes: dict[int, list[float]],
    next_object_id: int,
    frame_size: tuple[int, int],
    config: dict[str, Any],
) -> tuple[list[tuple[int, GroundedDetection]], int]:
    """将本锚点多框贪心匹配到上一窗口对象，未匹配框分配新 ID。"""

    assigned: list[tuple[int, GroundedDetection]] = []
    available = set(previous_boxes)
    threshold = float(config.get("association_threshold", 0.30))
    for detection in detections:
        matches = [
            (
                0.65 * box_iou(detection.box_xyxy, previous_boxes[obj_id])
                + 0.35 * _center_score(detection.box_xyxy, previous_boxes[obj_id], frame_size),
                obj_id,
            )
            for obj_id in available
        ]
        if matches and max(matches)[0] >= threshold:
            _, object_id = max(matches)
            available.remove(object_id)
        else:
            object_id = next_object_id
            next_object_id += 1
        assigned.append((object_id, detection))
    return assigned, next_object_id

"""输入索引、Stage 1 元数据及最终比赛 JSONL 的严格校验。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from video_highlight.common.exceptions import ArtifactValidationError


def _reject_constant(value: str) -> None:
    raise ArtifactValidationError(f"JSON 中不允许出现 {value}")


def _read_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ArtifactValidationError(f"缺少文件: {path}")
    value = json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"JSON 根节点不是对象: {path}")
    return value


def _normalize_ratio(value: object, location: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ArtifactValidationError(f"{location}.targetRatioWH 必须是 [w,h]")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ArtifactValidationError(f"{location}.targetRatioWH 必须是两个整数")
    ratio = tuple(int(item) for item in value)
    if any(float(value[index]) != ratio[index] for index in range(2)) or min(ratio) <= 0:
        raise ArtifactValidationError(f"{location}.targetRatioWH 必须是两个正整数")
    return ratio  # type: ignore[return-value]


def load_input_index(path: str | Path) -> list[dict[str, object]]:
    """读取 JSON/JSONL 视频索引，保留其原始行顺序。"""

    target = Path(path)
    if not target.is_file():
        raise ArtifactValidationError(f"输入索引不存在: {target}")
    if target.suffix.lower() == ".jsonl":
        rows = [
            json.loads(line, parse_constant=_reject_constant)
            for line in target.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
    else:
        payload = json.loads(target.read_text(encoding="utf-8-sig"), parse_constant=_reject_constant)
        rows = payload if isinstance(payload, list) else payload.get("videos", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise ArtifactValidationError("输入索引必须是非空对象数组或 JSONL")
    normalized: list[dict[str, object]] = []
    seen: set[str] = set()
    for position, row in enumerate(rows):
        video_id = str(row.get("video_id", "")).strip()
        if not video_id or video_id in seen:
            raise ArtifactValidationError(f"输入索引第 {position + 1} 项 video_id 缺失或重复")
        seen.add(video_id)
        ratio = _normalize_ratio(row.get("targetRatioWH"), f"索引 video_id={video_id}")
        normalized.append({"video_id": video_id, "targetRatioWH": list(ratio)})
    return normalized


def load_stage1_metadata(
    stage1_dir: str | Path, index_rows: list[dict[str, object]], require_success: bool = True
) -> dict[str, dict[str, Any]]:
    """按索引加载 Stage 1 元数据，用于帧范围、比例和空间边界复核。"""

    videos_dir = Path(stage1_dir).resolve() / "videos"
    result: dict[str, dict[str, Any]] = {}
    for index_row in index_rows:
        video_id = str(index_row["video_id"])
        video_dir = videos_dir / video_id
        if require_success and not (video_dir / "_SUCCESS.json").is_file():
            raise ArtifactValidationError(f"Stage 1 视频没有成功标记: {video_dir}")
        metadata = _read_object(video_dir / "metadata.json")
        if str(metadata.get("video_id")) != video_id:
            raise ArtifactValidationError(f"Stage 1 metadata.video_id 与索引不一致: {video_id}")
        index_ratio = tuple(index_row["targetRatioWH"])
        metadata_ratio = _normalize_ratio(metadata.get("targetRatioWH"), f"Stage 1 video_id={video_id}")
        if metadata_ratio != index_ratio:
            raise ArtifactValidationError(f"video_id={video_id} 的索引比例与 Stage 1 不一致")
        result[video_id] = metadata
    return result


def validate_submission_rows(
    rows: list[dict[str, Any]],
    index_rows: list[dict[str, object]],
    metadata_by_id: dict[str, dict[str, Any]],
    strict_fields: bool = True,
) -> dict[str, int]:
    """验证一视频一行、索引顺序、字段、帧号以及按比例推导后的空间边界。"""

    if len(rows) != len(index_rows):
        raise ArtifactValidationError(f"提交行数 {len(rows)} 与索引视频数 {len(index_rows)} 不一致")
    total_predictions = 0
    empty_video_count = 0
    for position, (row, index_row) in enumerate(zip(rows, index_rows, strict=True), start=1):
        video_id = str(index_row["video_id"])
        if strict_fields and set(row) != {"video_id", "targetRatioWH", "predictions"}:
            raise ArtifactValidationError(f"提交第 {position} 行顶层字段不符合要求")
        if not isinstance(row.get("video_id"), str) or row["video_id"] != video_id:
            raise ArtifactValidationError(f"提交第 {position} 行 video_id 或顺序与索引不一致")
        ratio = _normalize_ratio(row.get("targetRatioWH"), f"提交 video_id={video_id}")
        if ratio != tuple(index_row["targetRatioWH"]):
            raise ArtifactValidationError(f"video_id={video_id} 的提交比例与索引不一致")
        predictions = row.get("predictions")
        if not isinstance(predictions, list):
            raise ArtifactValidationError(f"video_id={video_id}.predictions 必须是数组")

        metadata = metadata_by_id[video_id]
        frame_count = int(metadata["frame_count"])
        frame_w = int(metadata.get("display_width", metadata["width"]))
        frame_h = int(metadata.get("display_height", metadata["height"]))
        previous_frame = -1
        for prediction in predictions:
            if not isinstance(prediction, dict):
                raise ArtifactValidationError(f"video_id={video_id} 的 prediction 必须是对象")
            if strict_fields and set(prediction) != {"frame", "bboxes"}:
                raise ArtifactValidationError(f"video_id={video_id} 的 prediction 含未知或缺失字段")
            frame = prediction.get("frame")
            if isinstance(frame, bool) or not isinstance(frame, int) or not (0 <= frame < frame_count):
                raise ArtifactValidationError(f"video_id={video_id} 存在越界或非整数 frame: {frame}")
            if frame <= previous_frame:
                raise ArtifactValidationError(f"video_id={video_id} 的 frame 未严格升序或重复")
            previous_frame = frame
            bbox = prediction.get("bboxes")
            if not isinstance(bbox, list) or len(bbox) != 3 or any(
                isinstance(value, bool) or not isinstance(value, int) for value in bbox
            ):
                raise ArtifactValidationError(f"video_id={video_id}, frame={frame} 的 bboxes 必须是三个整数")
            x, y, width = bbox
            height = width * ratio[1] / ratio[0]
            if not all(math.isfinite(float(value)) for value in bbox) or not (
                x >= 0
                and y >= 0
                and width > 0
                and x + width <= frame_w
                and y + height <= frame_h + 1e-6
            ):
                raise ArtifactValidationError(f"video_id={video_id}, frame={frame} 的构图框非法或越界")
        total_predictions += len(predictions)
        empty_video_count += int(not predictions)
    return {
        "video_count": len(rows),
        "prediction_count": total_predictions,
        "empty_video_count": empty_video_count,
    }


def read_submission_file(path: str | Path) -> list[dict[str, Any]]:
    """严格逐行解析提交文件；空白行也视为格式错误。"""

    target = Path(path)
    if not target.is_file():
        raise ArtifactValidationError(f"提交文件不存在: {target}")
    physical_lines = target.read_text(encoding="utf-8-sig").splitlines()
    if any(not line.strip() for line in physical_lines):
        raise ArtifactValidationError("提交 JSONL 不能包含空白行")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(physical_lines, start=1):
        try:
            value = json.loads(line, parse_constant=_reject_constant)
        except (json.JSONDecodeError, ArtifactValidationError) as error:
            raise ArtifactValidationError(f"提交第 {line_number} 行不是合法 JSON: {error}") from error
        if not isinstance(value, dict):
            raise ArtifactValidationError(f"提交第 {line_number} 行不是 JSON 对象")
        rows.append(value)
    return rows


def validate_submission_file(
    path: str | Path,
    index_rows: list[dict[str, object]],
    metadata_by_id: dict[str, dict[str, Any]],
    strict_fields: bool = True,
) -> dict[str, int]:
    return validate_submission_rows(read_submission_file(path), index_rows, metadata_by_id, strict_fields)

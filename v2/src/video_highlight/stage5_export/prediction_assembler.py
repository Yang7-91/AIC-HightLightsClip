"""把 Stage 4 调试型逐帧记录收敛为比赛提交记录。"""

from __future__ import annotations

from typing import Any

from video_highlight.common.exceptions import ArtifactValidationError
from video_highlight.contracts.submission import FinalPrediction, SubmissionRow

from .coordinate_quantizer import quantize_bbox


def assemble_submission_row(
    video_id: str,
    target_ratio: tuple[int, int],
    crop_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    quantization_mode: str,
) -> tuple[dict[str, object], dict[str, int]]:
    """组装一个视频，并删除 Stage 4 的 interval/置信度等非提交字段。"""

    frame_size = (
        int(metadata.get("display_width", metadata["width"])),
        int(metadata.get("display_height", metadata["height"])),
    )
    frame_count = int(metadata["frame_count"])
    predictions: list[FinalPrediction] = []
    repaired_count = 0

    # Stage 4 正常输出已经有序，但最终导出仍自行排序，避免文件拼接顺序影响提交。
    for row in sorted(crop_rows, key=lambda value: int(value.get("frame", -1))):
        if str(row.get("video_id")) != video_id:
            raise ArtifactValidationError(f"Stage 4 crop.video_id 不一致: {video_id}")
        frame = row.get("frame")
        if isinstance(frame, bool) or not isinstance(frame, int) or not (0 <= frame < frame_count):
            raise ArtifactValidationError(f"video_id={video_id} 存在非法 frame: {frame}")
        bbox, changed = quantize_bbox(
            row.get("bboxes", []), frame_size, target_ratio, quantization_mode
        )
        repaired_count += int(changed)
        predictions.append(FinalPrediction(frame=frame, bboxes=tuple(bbox)))

    frames = [prediction.frame for prediction in predictions]
    if len(frames) != len(set(frames)):
        raise ArtifactValidationError(f"video_id={video_id} 的 Stage 4 构图包含重复 frame")

    result = SubmissionRow(video_id, target_ratio, tuple(predictions)).to_dict()
    return result, {
        "prediction_count": len(predictions),
        "repaired_bbox_count": repaired_count,
    }

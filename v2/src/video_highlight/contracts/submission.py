"""Stage 5 比赛提交 JSONL 的强类型数据契约。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FinalPrediction:
    """一帧最终预测；框高由目标比例推导，因此只保存 ``[x, y, w]``。"""

    frame: int
    bboxes: tuple[int, int, int]

    def to_dict(self) -> dict[str, object]:
        return {"frame": self.frame, "bboxes": list(self.bboxes)}


@dataclass(frozen=True, slots=True)
class SubmissionRow:
    """提交文件中的一行，对应输入索引中的一个视频。"""

    video_id: str
    target_ratio_wh: tuple[int, int]
    predictions: tuple[FinalPrediction, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "video_id": self.video_id,
            "targetRatioWH": list(self.target_ratio_wh),
            "predictions": [prediction.to_dict() for prediction in self.predictions],
        }

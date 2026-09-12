"""Stage 5 坐标量化、索引顺序和最终提交契约测试。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from video_highlight.common.atomic_io import write_json, write_jsonl
from video_highlight.common.exceptions import ArtifactValidationError
from video_highlight.stage5_export.coordinate_quantizer import quantize_bbox
from video_highlight.stage5_export.pipeline import run_stage5
from video_highlight.stage5_export.submission_validator import validate_submission_rows


def stage5_config() -> dict:
    return {
        "output": {"filename": "submission.jsonl"},
        "quantization": {"mode": "round"},
        "validation": {
            "require_stage1_run_success": True,
            "require_stage1_video_success": True,
            "require_stage4_run_success": True,
            "require_stage4_video_success": True,
            "strict_fields": True,
        },
    }


class QuantizerTests(unittest.TestCase):
    def test_rounding_is_relimited_to_portrait_frame(self) -> None:
        bbox, changed = quantize_bbox([999.8, -2.2, 700.4], (1080, 1080), (9, 16))
        self.assertTrue(changed)
        self.assertEqual(bbox, [473, 0, 607])
        self.assertLessEqual(bbox[1] + bbox[2] * 16 / 9, 1080 + 1e-6)


class SubmissionPipelineTests(unittest.TestCase):
    @staticmethod
    def _prepare_video(root: Path, video_id: str, crops: list[dict]) -> None:
        stage1_video = root / "stage1/videos" / video_id
        stage4_video = root / "stage4/videos" / video_id
        stage1_video.mkdir(parents=True)
        stage4_video.mkdir(parents=True)
        write_json(
            stage1_video / "metadata.json",
            {
                "video_id": video_id,
                "width": 720,
                "height": 1280,
                "display_width": 720,
                "display_height": 1280,
                "frame_count": 20,
                "targetRatioWH": [16, 9],
            },
        )
        write_json(stage1_video / "_SUCCESS.json", {"status": "success"})
        write_jsonl(stage4_video / "crops.jsonl", crops)
        write_json(stage4_video / "_SUCCESS.json", {"status": "success"})

    def test_pipeline_preserves_index_order_and_empty_video(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index = [
                {"video_id": "b", "targetRatioWH": [16, 9]},
                {"video_id": "a", "targetRatioWH": [16, 9]},
            ]
            write_json(root / "index.json", index)
            write_json(root / "stage1/_SUCCESS.json", {"status": "success"})
            write_json(root / "stage4/_SUCCESS.json", {"status": "success"})
            self._prepare_video(
                root,
                "b",
                [
                    {"video_id": "b", "frame": 3, "bboxes": [0.4, 5.6, 720.0]},
                    {"video_id": "b", "frame": 1, "bboxes": [0, 2, 720]},
                ],
            )
            self._prepare_video(root, "a", [])
            summary = run_stage5(
                root / "index.json",
                root / "stage1",
                root / "stage4",
                root / "stage5",
                stage5_config(),
            )
            rows = [
                json.loads(line)
                for line in (root / "stage5/submission.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([row["video_id"] for row in rows], ["b", "a"])
            self.assertEqual([item["frame"] for item in rows[0]["predictions"]], [1, 3])
            self.assertEqual(rows[1]["predictions"], [])
            self.assertEqual(summary["video_count"], 2)
            self.assertEqual(summary["empty_video_count"], 1)
            self.assertTrue((root / "stage5/_SUCCESS.json").is_file())

    def test_validator_rejects_duplicate_frame(self) -> None:
        index = [{"video_id": "0", "targetRatioWH": [16, 9]}]
        metadata = {
            "0": {
                "width": 720,
                "height": 1280,
                "frame_count": 10,
            }
        }
        rows = [{
            "video_id": "0",
            "targetRatioWH": [16, 9],
            "predictions": [
                {"frame": 1, "bboxes": [0, 0, 720]},
                {"frame": 1, "bboxes": [0, 0, 720]},
            ],
        }]
        with self.assertRaises(ArtifactValidationError):
            validate_submission_rows(rows, index, metadata)


if __name__ == "__main__":
    unittest.main()

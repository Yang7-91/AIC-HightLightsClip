"""Stage 4 构图几何、轨迹平滑和独立流水线测试。"""

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
from video_highlight.stage4_subject_crop.boundary_limiter import finalize_bbox, legal_crop_from_state
from video_highlight.stage4_subject_crop.pipeline import run_stage4
from video_highlight.stage4_subject_crop.pipeline import _planning_spans
from video_highlight.stage4_subject_crop.trajectory_smoother import smooth_trajectory


def center_config() -> dict:
    return {
        "runtime": {"interval_error_policy": "center"},
        "tracking": {"backend": "center", "initial_width_ratio": 0.25, "initial_height_ratio": 0.4},
        "crop_candidates": {"scales": [0.7, 1.0], "offsets": [0.0], "subject_margins": [0.2, 0.2, 0.2, 0.2]},
        "composition": {"weights": {"uncovered": 0.55, "centering": 0.25, "zoom": 0.2}},
        "optimizer": {"center_weight": 0.2, "scale_weight": 0.12},
        "smoothing": {"center_alpha": 0.25, "width_alpha": 0.15, "max_center_step_ratio": 0.04},
    }


class GeometryTests(unittest.TestCase):
    def test_portrait_crop_is_legal_after_integer_rounding(self) -> None:
        crop = legal_crop_from_state(5_000, -100, 5_000, (1920, 1080), (9, 16))
        x, y, width = finalize_bbox(crop, (1920, 1080), (9, 16))
        self.assertGreater(width, 0)
        self.assertGreaterEqual(x, 0)
        self.assertGreaterEqual(y, 0)
        self.assertLessEqual(x + width, 1920)
        self.assertLessEqual(y + width * 16 / 9, 1080 + 1e-6)

    def test_smoothing_preserves_length(self) -> None:
        crops = [(0.0, 0.0, 400.0, 225.0), (300.0, 100.0, 500.0, 281.25)]
        result = smooth_trajectory(crops, (1280, 720), (16, 9), {"center_alpha": 0.2, "width_alpha": 0.2, "max_center_step_ratio": 0.02})
        self.assertEqual(len(result), 2)

    def test_planning_spans_split_at_stage1_scene_cut(self) -> None:
        interval = {"start_frame": 2, "end_frame": 12}
        scenes = [
            {"start_frame": 0, "end_frame": 7},
            {"start_frame": 7, "end_frame": 20},
        ]
        self.assertEqual(_planning_spans(interval, scenes), [(2, 7), (7, 12)])


class CenterPipelineTests(unittest.TestCase):
    def test_stage4_consumes_stage1_and_stage3_without_stage2(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stage1_video = root / "stage1/videos/0"
            stage3_video = root / "stage3/videos/0"
            stage1_video.mkdir(parents=True)
            stage3_video.mkdir(parents=True)
            write_json(stage1_video / "metadata.json", {"video_id": "0", "source_path": "not-used.mp4", "width": 720, "height": 1280, "display_width": 720, "display_height": 1280, "frame_count": 20, "fps": 30.0, "targetRatioWH": [16, 9]})
            write_jsonl(stage1_video / "scenes.jsonl", [{"scene_id": 0, "start_frame": 0, "end_frame": 20}])
            write_json(stage1_video / "_SUCCESS.json", {"status": "success"})
            write_jsonl(stage3_video / "refined_intervals.jsonl", [{"video_id": "0", "interval_id": "0_interval_0000", "start_frame": 2, "end_frame": 6, "subject": "dog", "subject_point": None}])
            write_json(stage3_video / "_SUCCESS.json", {"status": "success"})
            summary = run_stage4(root / "stage1", root / "stage3", root / "stage4", center_config(), strict=True)
            rows = [json.loads(line) for line in (root / "stage4/videos/0/crops.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(summary["success_count"], 1)
            self.assertEqual([row["frame"] for row in rows], [2, 3, 4, 5])
            self.assertTrue(all(len(row["bboxes"]) == 3 for row in rows))
            self.assertFalse((root / "stage2").exists())


if __name__ == "__main__":
    unittest.main()

"""Stage 4 构图几何、轨迹平滑和独立流水线测试。"""

from __future__ import annotations

import json
import cv2
import numpy as np
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from video_highlight.common.atomic_io import write_json, write_jsonl
from video_highlight.stage4_subject_crop.boundary_limiter import finalize_bbox, legal_crop_from_state, maximum_crop_width
from video_highlight.stage4_subject_crop.crop_candidates import generate_crop_candidates
from video_highlight.stage4_subject_crop.pipeline import run_stage4
from video_highlight.stage4_subject_crop.pipeline import _planning_spans
from video_highlight.stage4_subject_crop.propagation_visualizer import PropagationVisualizer
from video_highlight.stage4_subject_crop.sam2_adapter import anchor_windows
from video_highlight.stage4_subject_crop.subject_tracker import CenterSubjectTracker
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
    def test_visualization_sampling_and_forced_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            visualizer = PropagationVisualizer(
                {
                    "enabled": True,
                    "sample_fps": 2.0,
                    "always_save_anchor_frames": True,
                    "always_save_fallback_frames": True,
                    "save_images": True,
                    "write_video": False,
                },
                Path(temp),
                source_fps=30.0,
                span_start=100,
            )
            self.assertTrue(visualizer.should_save(100, is_anchor=False, is_fallback=False))
            self.assertFalse(visualizer.should_save(114, is_anchor=False, is_fallback=False))
            self.assertTrue(visualizer.should_save(115, is_anchor=False, is_fallback=False))
            self.assertTrue(visualizer.should_save(107, is_anchor=True, is_fallback=False))
            self.assertTrue(visualizer.should_save(108, is_anchor=False, is_fallback=True))

    def test_disabled_visualization_does_not_create_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "disabled"
            visualizer = PropagationVisualizer(
                {"enabled": False, "sample_fps": 2.0}, target, 30.0, 0
            )
            self.assertFalse(visualizer.active)
            self.assertFalse(target.exists())

    def test_visualization_writes_to_unicode_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "中文目录"
            source = root / "source.jpg"
            root.mkdir(parents=True)
            ok, encoded = cv2.imencode(".jpg", np.zeros((80, 120, 3), dtype=np.uint8))
            self.assertTrue(ok)
            encoded.tofile(str(source))
            output = root / "输出"
            visualizer = PropagationVisualizer(
                {"enabled": True, "sample_fps": 2.0, "save_images": True, "write_video": False},
                output,
                30.0,
                0,
            )
            from video_highlight.stage4_subject_crop.subject_tracker import TrackPoint
            visualizer.save(0, source, TrackPoint(0, [10.0, 10.0, 50.0, 60.0], 0.8, "sam2_anchor"), is_anchor=True)
            self.assertTrue((output / "frame_000000.jpg").is_file())

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

    def test_fixed_maximum_crop_has_only_one_max_width_candidate(self) -> None:
        candidates = generate_crop_candidates(
            [100.0, 100.0, 200.0, 300.0],
            (1920, 1080),
            (9, 16),
            (500.0, 0.0),
            {"fixed_maximum": True, "scales": [0.55, 0.7], "offsets": [-0.12, 0.12]},
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0][2], maximum_crop_width((1920, 1080), (9, 16)))

    def test_center_backend_interpolates_qwen_points_inside_each_scene(self) -> None:
        tracker = CenterSubjectTracker({"initial_width_ratio": 0.2, "initial_height_ratio": 0.2})
        interval = {
            "start_frame": 0,
            "end_frame": 12,
            "subject_points": [
                {"frame": 0, "subject_point": [0.2, 0.5]},
                {"frame": 4, "subject_point": [0.6, 0.5]},
                {"frame": 10, "subject_point": [0.8, 0.5]},
            ],
        }
        scenes = [{"start_frame": 0, "end_frame": 6}, {"start_frame": 6, "end_frame": 12}]
        points = tracker.track(Path(), interval, (1000, 500), scenes)
        centers = [(row.subject_box[0] + row.subject_box[2]) * 0.5 for row in points]
        self.assertAlmostEqual(centers[2], 400.0)
        # 不允许从第一镜头的末点跨硬切插值到第二镜头的首点。
        self.assertAlmostEqual(centers[5], 600.0)
        self.assertAlmostEqual(centers[6], 800.0)
        self.assertEqual(points[2].source, "qwen_linear")

    def test_sam2_windows_restart_at_every_qwen_anchor(self) -> None:
        interval = {
            "start_frame": 10,
            "end_frame": 30,
            "subject_points": [
                {"frame": 12, "subject_point": [0.2, 0.4]},
                {"frame": 20, "subject_point": [0.7, 0.4]},
                {"frame": 25, "subject_point": None},
            ],
        }
        windows = anchor_windows(interval, 10, 30)
        self.assertEqual([(row.start, row.end) for row in windows], [(10, 12), (12, 20), (20, 30)])
        self.assertEqual(windows[0].point, (0.2, 0.4))
        self.assertEqual(windows[2].point, (0.7, 0.4))


class CenterPipelineTests(unittest.TestCase):
    def test_stage4_consumes_stage1_and_stage3_5_without_stage3(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stage1_video = root / "stage1/videos/0"
            stage3_5_video = root / "stage3_5/videos/0"
            stage1_video.mkdir(parents=True)
            stage3_5_video.mkdir(parents=True)
            write_json(stage1_video / "metadata.json", {"video_id": "0", "source_path": "not-used.mp4", "width": 720, "height": 1280, "display_width": 720, "display_height": 1280, "frame_count": 20, "fps": 30.0, "targetRatioWH": [16, 9]})
            write_jsonl(stage1_video / "scenes.jsonl", [{"scene_id": 0, "start_frame": 0, "end_frame": 20}])
            write_json(stage1_video / "_SUCCESS.json", {"status": "success"})
            write_jsonl(stage3_5_video / "enriched_intervals.jsonl", [{"video_id": "0", "interval_id": "0_interval_0000", "start_frame": 2, "end_frame": 6, "subject": "dog"}])
            write_jsonl(stage3_5_video / "subject_points.jsonl", [{"video_id": "0", "interval_id": "0_interval_0000", "sample_index": 0, "frame": 2, "subject_point": [0.3, 0.4]}])
            write_json(stage3_5_video / "_SUCCESS.json", {"status": "success"})
            summary = run_stage4(
                root / "stage1",
                root / "stage3_5",
                root / "stage4",
                center_config(),
                {"video_root": str(root)},
                strict=True,
            )
            rows = [json.loads(line) for line in (root / "stage4/videos/0/crops.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(summary["success_count"], 1)
            self.assertEqual([row["frame"] for row in rows], [2, 3, 4, 5])
            self.assertTrue(all(len(row["bboxes"]) == 3 for row in rows))
            self.assertFalse((root / "stage2").exists())
            self.assertFalse((root / "stage3").exists())


if __name__ == "__main__":
    unittest.main()

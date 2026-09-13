"""Stage 3.5 采样、解析和跳过模式测试。"""

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
from video_highlight.stage3_5_subject_point.frame_sampler import _split_mjpeg_stream, plan_sample_frames
from video_highlight.stage3_5_subject_point.pipeline import run_stage3_5
from video_highlight.stage3_5_subject_point.response_parser import parse_predictions


def passthrough_config() -> dict:
    return {
        "api": {},
        "generation": {},
        "sampling": {"fps": 2.0, "jpeg_quality": 85, "max_side": 1024},
        "parsing": {"retries": 0},
        "runtime": {"mode": "passthrough", "backend": "openai"},
    }


class SamplingTests(unittest.TestCase):
    def test_plan_is_anchored_to_refined_start_and_end_is_exclusive(self) -> None:
        self.assertEqual(plan_sample_frames(7, 38, 30.0, 2.0), [7, 22, 37])

    def test_sample_rate_above_source_deduplicates_frames(self) -> None:
        self.assertEqual(plan_sample_frames(2, 6, 2.0, 10.0), [2, 3, 4, 5])

    def test_mjpeg_pipe_is_split_without_persistent_frame_files(self) -> None:
        first = b"\xff\xd8first\xff\xd9"
        second = b"\xff\xd8second\xff\xd9"
        self.assertEqual(_split_mjpeg_stream(first + second), [first, second])


class ParserTests(unittest.TestCase):
    def test_missing_sample_is_explicitly_filled(self) -> None:
        text = json.dumps({"predictions": [{"sample_index": 0, "subject_point": [0.25, 0.75], "confidence": 0.9, "visibility": "visible", "reason": "ok"}]})
        rows, missing,_errors = parse_predictions(text, 2) # 源码已修改，测试代码还未修改
        self.assertEqual(missing, [1])
        self.assertEqual(rows[0]["subject_point"], [0.25, 0.75])
        self.assertIsNone(rows[1]["subject_point"])


class PassthroughPipelineTests(unittest.TestCase):
    def test_skip_processing_neither_opens_source_video_nor_calls_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stage1_video = root / "stage1/videos/0"
            stage3_video = root / "stage3/videos/0"
            stage1_video.mkdir(parents=True)
            stage3_video.mkdir(parents=True)
            # 故意给不存在的视频路径：若旁路错误地打开视频，本测试会立即失败。
            write_json(stage1_video / "metadata.json", {"video_id": "0", "source_path": "missing.mp4", "fps": 10.0, "frame_count": 30, "width": 640, "height": 360})
            write_json(stage1_video / "_SUCCESS.json", {"status": "success"})
            write_jsonl(stage3_video / "refined_intervals.jsonl", [{"schema_version": "stage3.v2", "video_id": "0", "interval_id": "0_interval_0000", "start_frame": 3, "end_frame": 15, "subject": "dog"}])
            write_json(stage3_video / "_SUCCESS.json", {"status": "success"})
            summary = run_stage3_5(root / "stage1", root / "stage3", root / "stage3_5", passthrough_config(), strict=True)
            rows = [json.loads(line) for line in (root / "stage3_5/videos/0/subject_points.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(summary["success_count"], 1)
            self.assertEqual([row["frame"] for row in rows], [3, 8, 13])
            self.assertTrue(all(row["subject_point"] is None and row["status"] == "skipped" for row in rows))


if __name__ == "__main__":
    unittest.main()

"""Stage 3 帧映射、边界解码和旁路输出测试。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from video_highlight.common.atomic_io import write_json, write_jsonl
from video_highlight.stage3_boundary_refine.boundary_decoder import decode_boundaries
from video_highlight.stage3_boundary_refine.feature_extractor import FeatureSequence
from video_highlight.stage3_boundary_refine.frame_mapper import map_passthrough
from video_highlight.stage3_boundary_refine.pipeline import run_stage3
from video_highlight.stage3_boundary_refine.temporal_backend import TemporalScores


def passthrough_config() -> dict:
    return {
        "runtime": {"mode": "passthrough", "candidate_error_policy": "passthrough"},
        "decode": {},
        "temporal": {"backend": "rule", "rule": {}},
        "boundary": {},
        "empty_gate": {"enabled": True},
        "merging": {"enabled": True},
    }


class FrameMapperTests(unittest.TestCase):
    def test_passthrough_uses_half_open_frames(self) -> None:
        mapped = map_passthrough(
            {"start_sec": 1.01, "end_sec": 2.01},
            {"fps": 30.0, "frame_count": 100},
        )
        self.assertEqual(mapped["start_frame"], 30)
        self.assertEqual(mapped["end_frame"], 61)


class BoundaryDecoderTests(unittest.TestCase):
    def test_decoder_never_trims_more_than_configured(self) -> None:
        timestamps = np.arange(0.0, 5.0, 0.5)
        features = FeatureSequence(timestamps, np.arange(10), np.zeros((10, 8), dtype=np.float32))
        highlight = np.asarray([0.1, 0.1, 0.2, 0.8, 0.9, 0.9, 0.8, 0.2, 0.1, 0.1], dtype=np.float32)
        scores = TemporalScores(highlight, highlight, highlight[::-1].copy(), 0.1)
        decision = decode_boundaries(
            features,
            scores,
            {"start_sec": 0.0, "end_sec": 5.0},
            {"max_trim_sec": 1.0, "min_dynamic_range": 0.05, "core_threshold": 0.5},
        )
        self.assertLessEqual(timestamps[decision.start_index], 1.0)
        self.assertGreaterEqual(timestamps[decision.end_index], 4.0)


class PassthroughPipelineTests(unittest.TestCase):
    def test_skip_processing_persists_unmodified_stage2_times(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stage1_video = root / "stage1/videos/0"
            stage2_video = root / "stage2/videos/0"
            stage1_video.mkdir(parents=True)
            stage2_video.mkdir(parents=True)
            write_json(stage1_video / "metadata.json", {
                "video_id": "0", "duration_sec": 10.0, "fps": 30.0,
                "frame_count": 300, "source_path": "missing-is-not-opened.mp4",
            })
            write_jsonl(stage1_video / "scenes.jsonl", [{"scene_id": 0, "start_sec": 0.0, "end_sec": 10.0}])
            write_json(stage1_video / "_SUCCESS.json", {"status": "success"})
            write_jsonl(stage2_video / "candidates.jsonl", [{
                "video_id": "0", "candidate_id": "0_candidate_0000",
                "start_sec": 1.01, "end_sec": 2.01, "coarse_score": 0.8,
                "source_segment_ids": [0], "category": "action_peak", "subject": "dog",
                # 兼容读取历史 Stage 2，但 Stage 3 v2 必须丢弃这个旧字段。
                "subject_point": [0.5, 0.5],
            }])
            write_json(stage2_video / "_SUCCESS.json", {"status": "success"})
            summary = run_stage3(
                root / "stage1", root / "stage2", root / "stage3",
                passthrough_config(), strict=True,
            )
            rows = [json.loads(line) for line in (root / "stage3/videos/0/refined_intervals.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(summary["success_count"], 1)
            self.assertEqual(rows[0]["start_sec"], 1.01)
            self.assertEqual(rows[0]["end_sec"], 2.01)
            self.assertEqual(rows[0]["refine_mode"], "passthrough")
            self.assertEqual(rows[0]["source_candidate_ids"], ["0_candidate_0000"])
            self.assertNotIn("subject_point", rows[0])


if __name__ == "__main__":
    unittest.main()

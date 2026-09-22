"""MHS-1 多高光拆分器单元测试。"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import yaml

from video_highlight.stage2_5_multi_highlight_split.eventness import (
    build_eventness_bins,
    normalize_eventness,
    smooth_eventness,
)
from video_highlight.stage2_5_multi_highlight_split.schema import make_mhs1_candidate
from video_highlight.stage2_5_multi_highlight_split.splitter import (
    apply_split_guards,
    find_event_peaks,
    find_valleys_between_peaks,
    propose_split,
    propose_subsegments_from_peaks,
)


def formal_config() -> dict:
    return yaml.safe_load(
        (PROJECT_ROOT / "configs/stage2_5/mhs1_formal_splitter.yaml").read_text(encoding="utf-8")
    )


def make_bins(scores: list[float], *, bin_sec: float = 1.0) -> list[dict]:
    return [
        {
            "video_id": "v1",
            "candidate_id": "c1",
            "bin_index": index,
            "start_sec": index * bin_sec,
            "end_sec": (index + 1) * bin_sec,
            "frame_difference": score / 2,
            "shot_boundary": 0.0,
            "audio_energy": score / 3,
            "eventness_score": score,
        }
        for index, score in enumerate(scores)
    ]


# 双峰 + 明显低谷；谷在 idx 7（score 0.10）
DOUBLE_PEAK = [0.1, 0.2, 0.9, 0.8, 0.2, 0.1, 0.15, 0.1, 0.2, 0.15, 0.85, 0.9, 0.8, 0.2, 0.1, 0.15, 0.1, 0.2, 0.1, 0.05]
SINGLE_PEAK = [0.1, 0.15, 0.9, 0.8, 0.5, 0.3, 0.2, 0.15, 0.12, 0.1, 0.12, 0.1, 0.1, 0.08, 0.1, 0.09, 0.1, 0.08, 0.1, 0.05]


class EventnessTests(unittest.TestCase):
    def test_eventness_bin_schema(self) -> None:
        bins = [
            {"bin_index": 0, "start_sec": 0.0, "end_sec": 1.0, "frame_difference": 0.2, "shot_boundary": 0.0},
            {"bin_index": 1, "start_sec": 1.0, "end_sec": 2.0, "frame_difference": 0.8, "shot_boundary": 1.0},
        ]
        rows = build_eventness_bins(
            "v1",
            "c1",
            bins=bins,
            audio_energies=[0.1, 0.5],
            audio_available=True,
            config=formal_config(),
        )
        for row in rows:
            for key in (
                "video_id",
                "candidate_id",
                "bin_index",
                "start_sec",
                "end_sec",
                "frame_difference",
                "shot_boundary",
                "audio_energy",
                "eventness_score",
            ):
                self.assertIn(key, row)
            self.assertGreaterEqual(row["eventness_score"], 0.0)
            self.assertLessEqual(row["eventness_score"], 1.0)
            json.dumps(row)

    def test_eventness_finite_and_normalize(self) -> None:
        smoothed = smooth_eventness(DOUBLE_PEAK, 3.0, 1.0)
        normalized = normalize_eventness(smoothed)
        self.assertEqual(len(normalized), len(DOUBLE_PEAK))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in normalized))
        self.assertEqual(normalize_eventness([0.5, 0.5, 0.5]), [0.0, 0.0, 0.0])


class SplitterTests(unittest.TestCase):
    def test_single_peak_falls_back(self) -> None:
        config = formal_config()
        candidate = {"candidate_id": "c1", "start_sec": 0.0, "end_sec": 20.0}
        decision = propose_split(candidate, make_bins(SINGLE_PEAK), config, config_name="MHS-1-C1", video_duration=60.0)
        self.assertEqual(decision["action"], "identity")
        self.assertEqual(decision["fallback_reason"], "insufficient_peaks")
        self.assertEqual(len(decision["segments"]), 1)

    def test_multi_peak_splits(self) -> None:
        config = formal_config()
        candidate = {"candidate_id": "c1", "start_sec": 0.0, "end_sec": 20.0}
        decision = propose_split(candidate, make_bins(DOUBLE_PEAK), config, config_name="MHS-1-C1", video_duration=60.0)
        self.assertEqual(decision["action"], "split")
        self.assertGreaterEqual(len(decision["segments"]), 2)
        self.assertLess(decision["coverage_reduction_ratio"], 0.35)

    def test_flat_valley_falls_back(self) -> None:
        config = formal_config()
        # 两峰但谷不够低（谷 ~0.55，分位阈值无法通过）
        flat_valley = [0.9, 0.85, 0.8, 0.55, 0.6, 0.58, 0.62, 0.6, 0.8, 0.85, 0.9] + [0.5] * 9
        candidate = {"candidate_id": "c1", "start_sec": 0.0, "end_sec": 20.0}
        decision = propose_split(candidate, make_bins(flat_valley), config, config_name="MHS-1-C1", video_duration=60.0)
        self.assertIn(decision["action"], {"identity", "fallback_identity"})
        self.assertIn(
            decision["fallback_reason"],
            {"valley_not_prominent", "insufficient_peaks"},
        )

    def test_min_subsegment_duration_effect(self) -> None:
        config = formal_config()
        # 双峰但一段短（cut 贴近边界）
        scores = [0.1, 0.9, 0.8, 0.2, 0.1, 0.9, 0.85, 0.8, 0.7, 0.5]
        candidate = {"candidate_id": "c1", "start_sec": 0.0, "end_sec": 10.0}
        decision = propose_split(candidate, make_bins(scores), config, config_name="MHS-1-C1", video_duration=60.0)
        if decision["action"] == "split":
            for segment in decision["segments"]:
                self.assertGreaterEqual(
                    segment["end_sec"] - segment["start_sec"],
                    float(config["splitter"]["configs"]["MHS-1-C1"]["min_subsegment_duration_sec"]) - 1e-9,
                )
        else:
            self.assertEqual(decision["action"], "identity")

    def test_max_coverage_reduction_cap(self) -> None:
        # 构造极大收缩场景（两个短峰间长谷 + 严格 cap）→ fallback
        segments = [
            {"start_sec": 0.0, "end_sec": 4.0},
            {"start_sec": 16.0, "end_sec": 20.0},
        ]
        decision = apply_split_guards(
            0.0,
            20.0,
            segments,
            pad_sec=0.0,
            merge_gap_sec=2.0,
            min_subsegment_duration_sec=3.0,
            max_subsegments_per_candidate=4,
            max_coverage_reduction_ratio=0.10,
            video_duration=60.0,
        )
        self.assertEqual(decision["action"], "fallback_identity")
        self.assertEqual(decision["fallback_reason"], "coverage_reduction_exceeds_cap")

    def test_never_drop_all_subsegments(self) -> None:
        segments = [
            {"start_sec": 0.0, "end_sec": 1.0},
            {"start_sec": 10.0, "end_sec": 11.0},
        ]
        decision = apply_split_guards(
            0.0,
            20.0,
            segments,
            pad_sec=0.0,
            merge_gap_sec=2.0,
            min_subsegment_duration_sec=4.0,
            max_subsegments_per_candidate=4,
            max_coverage_reduction_ratio=1.0,
            video_duration=60.0,
        )
        self.assertEqual(decision["action"], "fallback_identity")
        self.assertEqual(decision["fallback_reason"], "all_subsegments_below_min_duration")

    def test_subsegment_outside_video_falls_back(self) -> None:
        segments = [{"start_sec": 0.0, "end_sec": 8.0}, {"start_sec": 10.0, "end_sec": 40.0}]
        decision = apply_split_guards(
            0.0,
            40.0,
            segments,
            pad_sec=2.0,
            merge_gap_sec=1.0,
            min_subsegment_duration_sec=3.0,
            max_subsegments_per_candidate=4,
            max_coverage_reduction_ratio=0.5,
            video_duration=30.0,
        )
        self.assertEqual(decision["action"], "fallback_identity")

    def test_peak_valley_detection(self) -> None:
        bins = make_bins(DOUBLE_PEAK)
        peaks = find_event_peaks(bins, peak_min_distance_sec=4.0)
        self.assertGreaterEqual(len(peaks), 2)
        valleys = find_valleys_between_peaks(bins, peaks)
        self.assertGreaterEqual(len(valleys), 1)
        segments = propose_subsegments_from_peaks(
            0.0,
            20.0,
            peaks,
            valleys,
            valley_quantile_max=0.40,
            all_scores=[float(item["eventness_score"]) for item in bins],
            merge_gap_sec=2.0,
        )
        self.assertGreaterEqual(len(segments), 2)


class SchemaTests(unittest.TestCase):
    def test_split_output_schema(self) -> None:
        candidate = make_mhs1_candidate(
            video_id="v1",
            source_candidate_id="c1",
            segment_index=0,
            start_sec=1.0,
            end_sec=6.0,
            parent_start_sec=0.0,
            parent_end_sec=10.0,
            score=0.5,
            reason="r",
            method="MHS-1-C1",
            split_decision={
                "action": "split",
                "num_peaks": 2,
                "num_subsegments": 2,
                "coverage_reduction_ratio": 0.2,
                "fallback_reason": None,
            },
        )
        row = candidate.to_row()
        for key in (
            "video_id",
            "source_candidate_id",
            "mhs1_candidate_id",
            "start_sec",
            "end_sec",
            "score",
            "reason",
            "method",
            "parent_start_sec",
            "parent_end_sec",
            "split_decision",
            "provenance",
        ):
            self.assertIn(key, row)
        self.assertFalse(row["provenance"]["uses_weak_reference_for_decision"])
        self.assertFalse(row["provenance"]["heldout_access"])
        json.dumps(row)


class ComplianceTests(unittest.TestCase):
    def test_splitter_does_not_use_weak_reference(self) -> None:
        import video_highlight.stage2_5_multi_highlight_split.splitter as splitter_module

        source = Path(splitter_module.__file__).read_text(encoding="utf-8")
        # 拆分决策模块不得读取任何 reference 数据源（docstring 合规声明允许）。
        self.assertNotIn("weak_reference_segments", source)
        self.assertNotIn("load_frozen", source)
        self.assertNotIn("frozen_predictions", source)
        self.assertNotIn("references_by_video", source)

    def test_heldout_split_rejected(self) -> None:
        import tempfile

        from video_highlight.stage2_5_multi_highlight_split.io import load_stage2_candidates

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "cache_manifest.json").write_text(
                json.dumps({"records": [{"path": "records/v1.json", "video_id": "v1"}]}),
                encoding="utf-8",
            )
            (root / "records").mkdir()
            (root / "records" / "v1.json").write_text(
                json.dumps(
                    {
                        "video_id": "v1",
                        "split": "heldout",
                        "duration_sec": 60.0,
                        "merged_candidates": [],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(Exception):
                load_stage2_candidates(root)

    def test_cli_help_runs(self) -> None:
        import subprocess

        script = PROJECT_ROOT / "scripts/run_mhs1_splitter.py"
        completed = subprocess.run(
            [sys.executable, str(script), "--help"], capture_output=True, text=True, check=False
        )
        self.assertEqual(completed.returncode, 0)
        self.assertIn("run-all", completed.stdout)

    def test_no_mainline_writes(self) -> None:
        import video_highlight.stage2_5_multi_highlight_split.pipeline as pipeline_module

        source = Path(pipeline_module.__file__).read_text(encoding="utf-8")
        self.assertNotIn("stage2_dir\", \"w", source)
        self.assertNotIn("submission", source.lower())
        self.assertIn('"heldout_accessed": False', source)


if __name__ == "__main__":
    unittest.main()

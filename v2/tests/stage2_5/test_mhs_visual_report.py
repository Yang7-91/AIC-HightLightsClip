"""MHS-VIS-0 可视化诊断工具单元测试。"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from video_highlight.stage2_5_visual_report.html_report import write_visual_html_report
from video_highlight.stage2_5_visual_report.io import (
    VisualReportIOError,
    load_stage2_candidates,
)
from video_highlight.stage2_5_visual_report.selector import select_visual_examples
from video_highlight.stage2_5_visual_report.summarizer import (
    summarize_candidate_rule_based,
    summarize_subsegment_rule_based,
)
from video_highlight.stage2_5_visual_report.timeline import (
    build_timeline_data,
    find_motion_peaks,
    render_timeline_svg,
)


def vis_config() -> dict:
    return {
        "schema_version": "stage2_5.mhs_vis0_visual_report.v1",
        "selection": {
            "mode": "auto",
            "max_videos": 3,
            "max_candidates_per_video": 2,
            "prefer_long_candidates": True,
            "min_candidate_duration_sec": 18.0,
            "prefer_multi_peak_candidates": True,
            "prefer_high_coverage_candidates": True,
            "seed": 20260922,
            "allow_manual_video_ids": [],
        },
        "visualization": {
            "timeline_width_px": 1400,
            "thumbnail_fps": 1.0,
            "max_thumbnails_per_candidate": 40,
            "contact_sheet_thumb_width": 180,
        },
        "highlight_summary": {"max_summary_chars": 80},
        "output": {
            "write_html": True,
            "write_contact_sheet_png": True,
            "write_json_summary": True,
            "write_markdown_report": True,
        },
        "safety": {"no_heldout": True, "does_not_modify_predictions": True},
    }


def make_candidate(video_id: str, index: int, start: float, duration: float) -> dict:
    return {
        "video_id": video_id,
        "candidate_id": f"{video_id}:c{index}",
        "start_sec": start,
        "end_sec": start + duration,
        "duration_sec": duration,
        "score": 0.5,
        "reason": None,
        "source_chunk": 0,
    }


class SelectorTests(unittest.TestCase):
    def test_selector_deterministic(self) -> None:
        candidates = {
            "v1": [make_candidate("v1", 1, 0.0, 30.0), make_candidate("v1", 2, 40.0, 12.0)],
            "v2": [make_candidate("v2", 1, 5.0, 25.0)],
        }
        durations = {"v1": 60.0, "v2": 30.0}
        peaks = {"v1:c1": [3.0, 12.0, 21.0], "v1:c2": [], "v2:c1": [4.0]}
        first = select_visual_examples(candidates, durations, vis_config(), peaks_by_candidate=peaks)
        second = select_visual_examples(candidates, durations, vis_config(), peaks_by_candidate=peaks)
        self.assertEqual([row["candidate_id"] for row in first], [row["candidate_id"] for row in second])

    def test_long_candidate_prioritized(self) -> None:
        candidates = {"v1": [make_candidate("v1", 1, 0.0, 8.0), make_candidate("v1", 2, 10.0, 30.0)]}
        durations = {"v1": 60.0}
        selected = select_visual_examples(candidates, durations, vis_config())
        self.assertEqual(selected[0]["candidate_id"], "v1:c2")
        self.assertIn("long_candidate", selected[0]["reason"])

    def test_multi_peak_prioritized(self) -> None:
        candidates = {"v1": [make_candidate("v1", 1, 0.0, 20.0), make_candidate("v1", 2, 30.0, 20.0)]}
        durations = {"v1": 60.0}
        peaks = {"v1:c1": [], "v1:c2": [2.0, 8.0, 15.0]}
        selected = select_visual_examples(candidates, durations, vis_config(), peaks_by_candidate=peaks)
        self.assertEqual(selected[0]["candidate_id"], "v1:c2")
        self.assertIn("multi_peak", selected[0]["reason"])

    def test_max_per_video_respected(self) -> None:
        candidates = {"v1": [make_candidate("v1", i, i * 15.0, 12.0) for i in range(4)]}
        durations = {"v1": 120.0}
        config = vis_config()
        config["selection"]["max_candidates_per_video"] = 2
        selected = select_visual_examples(candidates, durations, config)
        self.assertEqual(len(selected), 2)

    def test_manual_video_ids_filter(self) -> None:
        candidates = {
            "v1": [make_candidate("v1", 1, 0.0, 30.0)],
            "v2": [make_candidate("v2", 1, 0.0, 30.0)],
        }
        durations = {"v1": 60.0, "v2": 60.0}
        selected = select_visual_examples(candidates, durations, vis_config(), manual_video_ids=["v2"])
        self.assertTrue(all(row["video_id"] == "v2" for row in selected))


class IOTests(unittest.TestCase):
    def test_heldout_split_rejected(self) -> None:
        import tempfile

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
            with self.assertRaises(VisualReportIOError):
                load_stage2_candidates(root)

    def test_cache_candidates_loaded(self) -> None:
        import tempfile

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
                        "split": "dev",
                        "duration_sec": 60.0,
                        "merged_candidates": [
                            {"merged_candidate_id": "c1", "start_sec": 1.0, "end_sec": 21.0, "score": 0.7}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            candidates, durations = load_stage2_candidates(root)
            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0]["candidate_id"], "c1")
            self.assertEqual(durations["v1"], 60.0)


class TimelineTests(unittest.TestCase):
    def _timeline(self) -> dict:
        candidate = make_candidate("v1", 1, 10.0, 30.0)
        motion = [{"t_sec": 10.0 + i, "motion": 0.01 * (i % 3)} for i in range(30)]
        return build_timeline_data(
            candidate,
            60.0,
            motion_samples=motion,
            shot_boundaries=[14.0, 22.0],
            motion_peaks=[12.0, 25.0],
            subsegments=[{"start_sec": 11.0, "end_sec": 18.0, "label": "s1"}],
        )

    def test_timeline_schema(self) -> None:
        timeline = self._timeline()
        self.assertEqual(timeline["candidate"]["candidate_id"], "v1:c1")
        self.assertEqual(len(timeline["shot_boundaries_sec"]), 2)
        self.assertEqual(len(timeline["motion_peaks_sec"]), 2)
        self.assertEqual(len(timeline["subsegments"]), 1)
        self.assertFalse(timeline["eventness_available"])
        self.assertTrue(timeline["diagnostic_only"])
        json.dumps(timeline)

    def test_svg_render_non_empty(self) -> None:
        svg = render_timeline_svg(self._timeline(), width_px=800, height_px=150)
        self.assertTrue(svg.startswith("<svg"))
        self.assertIn("</svg>", svg)
        self.assertIn("candidate", svg)
        self.assertIn("motion proxy", svg)

    def test_motion_peaks(self) -> None:
        samples = [
            {"t_sec": float(i), "motion": value}
            for i, value in enumerate([0.0, 0.9, 0.1, 0.2, 0.95, 0.0])
        ]
        peaks = find_motion_peaks(samples, min_gap_sec=2.0)
        self.assertIn(1.0, peaks)
        self.assertIn(4.0, peaks)


class SummarizerTests(unittest.TestCase):
    def test_candidate_summary_non_empty(self) -> None:
        candidate = make_candidate("v1", 1, 0.0, 34.0)
        text = summarize_candidate_rule_based(
            candidate, motion_peaks=[3.0, 12.0, 25.0], shot_count=3
        )
        self.assertIn("34", text)
        self.assertIn("多个高光", text)

    def test_subsegment_summary_non_empty(self) -> None:
        motion = [{"t_sec": 1.0 + 0.5 * i, "motion": 0.01 * (i % 4)} for i in range(20)]
        text = summarize_subsegment_rule_based(
            1, {"start_sec": 1.0, "end_sec": 8.0}, motion
        )
        self.assertTrue(text.startswith("子片段 1"))
        self.assertGreater(len(text), 8)


class HTMLTests(unittest.TestCase):
    def test_html_contains_required_markers(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            entry = {
                "video_id": "v1",
                "selection_reason": "long_candidate",
                "candidate": {
                    "candidate_id": "v1:c1",
                    "start_sec": 0.0,
                    "end_sec": 30.0,
                    "duration_sec": 30.0,
                    "score": 0.5,
                },
                "subsegments": [
                    {"start_sec": 1.0, "end_sec": 8.0, "label": "1", "summary": "第一段"}
                ],
                "num_motion_peaks": 3,
                "candidate_summary": "疑似包含多个高光事件",
                "timeline_svg": "<svg xmlns='http://www.w3.org/2000/svg'></svg>",
                "contact_sheet": None,
            }
            path = write_visual_html_report(
                tmp,
                run_info={"num_selected_examples": 1},
                entries=[entry],
                summary_text="诊断摘要",
                eventness_available=False,
            )
            html = Path(path).read_text(encoding="utf-8")
            self.assertIn("v1:c1"[:4], html)
            self.assertIn("diagnostic only", html)
            self.assertIn("第一段", html)
            self.assertIn("motion proxy", html)
            self.assertNotIn("weak_reference", html)
            self.assertNotIn("oracle_label", html)
            # 合规声明必须存在（"不访问 Heldout" 是声明，不是数据泄漏）
            self.assertIn("Heldout", html)

    def test_cli_help_runs(self) -> None:
        import subprocess

        script = PROJECT_ROOT / "scripts/run_mhs_visual_report.py"
        completed = subprocess.run(
            [sys.executable, str(script), "--help"], capture_output=True, text=True, check=False
        )
        self.assertEqual(completed.returncode, 0)
        self.assertIn("smoke", completed.stdout)

    def test_module_has_no_heldout_or_prediction_writes(self) -> None:
        import video_highlight.stage2_5_visual_report.pipeline as pipeline_module

        source = Path(pipeline_module.__file__).read_text(encoding="utf-8")
        # 显式声明不访问 Heldout；且不存在写回正式预测/submission 的路径。
        self.assertIn('"heldout_accessed": False', source)
        self.assertNotIn("submission", source.lower())
        self.assertNotIn("predictions.jsonl", source)
        self.assertNotIn("refined_intervals.jsonl\", \"w", source)


if __name__ == "__main__":
    unittest.main()

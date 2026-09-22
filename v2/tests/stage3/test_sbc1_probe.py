"""SBC-1 语义边界判别探针的单元测试（diagnostic-only）。"""

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

from video_highlight.stage3_boundary_refine.sbc1_probe import (
    P1_VARIANT,
    P2_VARIANT,
    P3_VARIANT,
    Sbc1ParseError,
    Sbc1Prediction,
    Sbc1Sample,
    build_sbc1_samples,
    compute_clip_windows,
    compute_sbc1_metrics,
    load_oracle_boundary_labels,
    parse_sbc1_response,
    render_sbc1_prompt,
    write_sbc1_summary,
)


def probe_config() -> dict:
    return {
        "schema_version": "stage3.sbc1_semantic_boundary_probe.v1",
        "sampling": {
            "seed": 20260922,
            "max_samples_total": 240,
            "max_per_label_per_side": 40,
            "labels": ["TRIM", "KEEP", "EXPAND"],
            "sides": ["left", "right"],
            "stratified": True,
        },
        "video_context": {
            "boundary_window_sec": 4.0,
            "core_window_sec": 4.0,
            "fps": 2.0,
            "max_clip_sec": 8.0,
            "variant_shift_sec": 1.5,
            "variant_window_sec": 2.0,
        },
        "model": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 512,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
        "decision_rules": {
            "schema_success_rate_min": 0.95,
            "macro_f1_min": 0.45,
            "auc_like_min": 0.65,
            "p2_priority_if_tie": True,
        },
    }


def make_labels(count: int) -> list[dict]:
    labels = []
    for index in range(count):
        labels.append(
            {
                "video_id": f"v{index:03d}",
                "candidate_id": f"c{index}",
                "left_action": ["TRIM", "KEEP", "EXPAND"][index % 3],
                "right_action": ["KEEP", "EXPAND", "TRIM"][index % 3],
            }
        )
    return labels


def make_records(count: int) -> dict:
    return {
        f"v{index:03d}": {
            "video_id": f"v{index:03d}",
            "split": "dev",
            "duration_sec": 20.0 + index,
            "merged_candidates": [
                {"merged_candidate_id": f"c{index}", "start_sec": 5.0, "end_sec": 15.0}
            ],
        }
        for index in range(count)
    }


class SamplingTests(unittest.TestCase):
    def test_oracle_labels_loading(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.jsonl"
            path.write_text(
                json.dumps({"video_id": "v1", "left_action": "KEEP", "right_action": "TRIM"}) + "\n",
                encoding="utf-8",
            )
            rows = load_oracle_boundary_labels(path)
            self.assertEqual(rows[0]["left_action"], "KEEP")

    def test_sampling_deterministic_and_expands_variants(self) -> None:
        labels = make_labels(30)
        records = make_records(30)
        first, summary1 = build_sbc1_samples(labels, records, probe_config())
        second, summary2 = build_sbc1_samples(labels, records, probe_config())
        self.assertEqual([s.sample_id for s in first], [s.sample_id for s in second])
        self.assertEqual(summary1["base_sample_count"], summary2["base_sample_count"])
        self.assertEqual(summary1["total_samples"], summary1["base_sample_count"] * 3)
        self.assertLessEqual(summary1["total_samples"], 240 * 3)
        self.assertEqual(summary1["heldout_accessed"], False)

    def test_sampling_excludes_non_dev(self) -> None:
        labels = make_labels(2)
        records = make_records(2)
        records["v001"]["split"] = "hard"
        samples, summary = build_sbc1_samples(labels, records, probe_config())
        self.assertTrue(all(s.video_id != "v001" for s in samples))
        self.assertEqual(summary["skipped"]["non_dev"], 1)

    def test_cap_per_label_per_side(self) -> None:
        config = probe_config()
        config["sampling"]["max_per_label_per_side"] = 3
        labels = make_labels(30)
        records = make_records(30)
        samples, summary = build_sbc1_samples(labels, records, config)
        base = summary["base_sample_count"]
        self.assertLessEqual(base, 3 * 3 * 2)
        self.assertEqual(summary["groups"]["left:TRIM"]["selected"], 3)


class ClipWindowTests(unittest.TestCase):
    def _sample(self, variant: str, side: str = "left") -> Sbc1Sample:
        return Sbc1Sample(
            sample_id=f"s|{variant}",
            video_id="v000",
            candidate_id="c0",
            side=side,
            oracle_label="KEEP",
            original_start_sec=5.0,
            original_end_sec=15.0,
            boundary_sec=5.0 if side == "left" else 15.0,
            candidate_duration_sec=10.0,
            prompt_variant=variant,
        )

    def test_p1_single_boundary_window(self) -> None:
        windows = compute_clip_windows(self._sample(P1_VARIANT), probe_config(), 30.0)
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]["role"], "boundary")
        self.assertAlmostEqual(windows[0]["start_sec"], 1.0)
        self.assertAlmostEqual(windows[0]["end_sec"], 9.0)

    def test_p2_three_variants_ordered(self) -> None:
        windows = compute_clip_windows(self._sample(P2_VARIANT), probe_config(), 30.0)
        self.assertEqual([w["role"] for w in windows], ["KEEP", "TRIM", "EXPAND"])
        self.assertAlmostEqual(windows[1]["start_sec"], 5.0 - 0.5)  # TRIM shifts inward (+1.5)
        self.assertAlmostEqual(windows[2]["start_sec"], 5.0 - 3.5)  # EXPAND shifts outward (-1.5)

    def test_p3_core_then_boundary(self) -> None:
        windows = compute_clip_windows(self._sample(P3_VARIANT), probe_config(), 30.0)
        self.assertEqual([w["role"] for w in windows], ["core", "boundary"])
        self.assertAlmostEqual(windows[0]["start_sec"], 8.0)
        self.assertAlmostEqual(windows[0]["end_sec"], 12.0)

    def test_windows_clamped_to_video(self) -> None:
        sample = Sbc1Sample(
            sample_id="s",
            video_id="v",
            candidate_id="c",
            side="left",
            oracle_label="KEEP",
            original_start_sec=1.0,
            original_end_sec=5.0,
            boundary_sec=1.0,
            candidate_duration_sec=4.0,
            prompt_variant=P1_VARIANT,
        )
        windows = compute_clip_windows(sample, probe_config(), 3.0)
        self.assertGreaterEqual(windows[0]["start_sec"], 0.0)
        self.assertLessEqual(windows[0]["end_sec"], 3.0)

    def test_prompt_render_has_no_oracle_label(self) -> None:
        sample = self._sample(P1_VARIANT)
        template = "side={side} shift={variant_shift_sec} dir={side_direction} dur={candidate_duration_sec} off={boundary_offset_in_clip_sec} clip={clip_duration_sec}"
        windows = compute_clip_windows(sample, probe_config(), 30.0)
        rendered = render_sbc1_prompt(template, sample, windows, probe_config())
        self.assertIn("side=left", rendered)
        self.assertNotIn("TRIM", rendered)
        self.assertNotIn("EXPAND", rendered)
        self.assertNotIn("KEEP", rendered)


class ParseTests(unittest.TestCase):
    def test_p1_parse_ok(self) -> None:
        payload = parse_sbc1_response(
            P1_VARIANT,
            '{"action": "TRIM", "confidence": 0.7, "evidence": {"event_continues_outside": false}, "rationale_short": "setup only"}',
        )
        self.assertEqual(payload["action"], "TRIM")

    def test_p2_parse_ok_with_ranking(self) -> None:
        payload = parse_sbc1_response(
            P2_VARIANT,
            '{"best_variant": "EXPAND", "confidence": 0.6, "ranking": ["EXPAND", "KEEP", "TRIM"], "rationale_short": "onset"}',
        )
        self.assertEqual(payload["best_variant"], "EXPAND")

    def test_p3_parse_ok(self) -> None:
        payload = parse_sbc1_response(
            P3_VARIANT,
            '{"action": "KEEP", "confidence": 0.5, "same_event_outside": true, "boundary_context_is_redundant": false, "rationale_short": "mixed"}',
        )
        self.assertEqual(payload["action"], "KEEP")

    def test_invalid_json_raises(self) -> None:
        with self.assertRaises(Sbc1ParseError):
            parse_sbc1_response(P1_VARIANT, "not json")
        with self.assertRaises(Sbc1ParseError):
            parse_sbc1_response(P2_VARIANT, '{"best_variant": "MOVE", "confidence": 0.5}')
        with self.assertRaises(Sbc1ParseError):
            parse_sbc1_response(P3_VARIANT, '{"action": "KEEP", "confidence": 3.0}')


class MetricTests(unittest.TestCase):
    def _samples(self, count: int = 9, variant: str = P1_VARIANT) -> list[Sbc1Sample]:
        rows = []
        for index in range(count):
            rows.append(
                Sbc1Sample(
                    sample_id=f"s{index}",
                    video_id="v",
                    candidate_id="c",
                    side="left" if index % 2 == 0 else "right",
                    oracle_label=["TRIM", "KEEP", "EXPAND"][index % 3],
                    original_start_sec=0.0,
                    original_end_sec=10.0,
                    boundary_sec=0.0,
                    candidate_duration_sec=10.0,
                    prompt_variant=variant,
                )
            )
        return rows

    def test_confusion_and_macro_f1(self) -> None:
        samples = self._samples(9)
        predictions = [
            Sbc1Prediction(f"s{i}", P1_VARIANT, s.oracle_label, 0.9, True)
            for i, s in enumerate(samples)
        ]
        metrics = compute_sbc1_metrics(samples, predictions, P1_VARIANT, probe_config())
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["confusion_matrix"]["TRIM"]["TRIM"], 3)

    def test_parse_failures_lower_schema_rate(self) -> None:
        samples = self._samples(4)
        predictions = [
            Sbc1Prediction("s0", P1_VARIANT, "KEEP", 0.9, True),
            Sbc1Prediction("s1", P1_VARIANT, None, None, False, "parse_error"),
            Sbc1Prediction("s2", P1_VARIANT, None, None, False, "parse_error"),
            Sbc1Prediction("s3", P1_VARIANT, None, None, False, "parse_error"),
        ]
        metrics = compute_sbc1_metrics(samples, predictions, P1_VARIANT, probe_config())
        self.assertEqual(metrics["schema_success_rate"], 0.25)
        self.assertEqual(metrics["decision"], "NOT_ACTIONABLE")
        self.assertTrue(metrics["diagnostic_only"])
        self.assertFalse(metrics["deployable_method"])

    def test_metrics_serializable(self) -> None:
        samples = self._samples(6)
        predictions = [
            Sbc1Prediction(f"s{i}", P1_VARIANT, s.oracle_label, 0.5, True)
            for i, s in enumerate(samples)
        ]
        metrics = compute_sbc1_metrics(samples, predictions, P1_VARIANT, probe_config())
        json.dumps(metrics)

    def test_summary_best_variant_selection(self) -> None:
        evaluations = {}
        for variant in (P1_VARIANT, P2_VARIANT, P3_VARIANT):
            evaluations[variant] = {
                "accuracy": 0.4,
                "macro_f1": 0.4,
                "left_auc_like": 0.5,
                "right_auc_like": 0.5,
                "schema_success_rate": 1.0,
                "decision": "NOT_ACTIONABLE",
            }
        evaluations[P2_VARIANT]["macro_f1"] = 0.55
        evaluations[P2_VARIANT]["decision"] = "ACTIONABLE_SIGNAL"
        summary = write_sbc1_summary(evaluations, probe_config())
        self.assertEqual(summary["decision"], "ACTIONABLE_SIGNAL")
        self.assertEqual(summary["recommendation"], "DESIGN_BR3_DEV_ONLY")
        self.assertEqual(summary["best_variant"], P2_VARIANT)
        json.dumps(summary)


class ConfigSafetyTests(unittest.TestCase):
    def test_config_enable_thinking_false_and_safety(self) -> None:
        import yaml

        config_path = PROJECT_ROOT / "configs/stage3/sbc1_semantic_boundary_probe.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.assertFalse(
            config["model"]["extra_body"]["chat_template_kwargs"]["enable_thinking"]
        )
        self.assertEqual(config["model"]["temperature"], 0.0)
        self.assertTrue(config["diagnostic_only"])
        self.assertFalse(config["deployable_method"])
        self.assertTrue(config["safety"]["no_hard"])
        self.assertTrue(config["safety"]["no_heldout"])
        self.assertTrue(config["safety"]["no_training"])
        self.assertTrue(config["safety"]["does_not_modify_stage2_stage3_mainline"])

    def test_cli_help_runs(self) -> None:
        import subprocess

        script = PROJECT_ROOT / "scripts/run_sbc1_probe.py"
        completed = subprocess.run(
            [sys.executable, str(script), "--help"], capture_output=True, text=True, check=False
        )
        self.assertEqual(completed.returncode, 0)
        self.assertIn("build-samples", completed.stdout)

    def test_module_does_not_touch_frozen_outputs(self) -> None:
        # SBC-1 只读 cache；模块中不存在写入 Stage 2/3 正式产物的路径。
        import video_highlight.stage3_boundary_refine.sbc1_probe as module

        source = Path(module.__file__).read_text(encoding="utf-8")
        self.assertNotIn("refined_candidates", source)
        self.assertNotIn("stage2_dir", source)
        self.assertNotIn("write_json(\"stage3", source)


if __name__ == "__main__":
    unittest.main()

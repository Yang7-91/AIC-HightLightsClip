"""Stage 2 的响应解析、候选合并和 mock 流水线测试。"""

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
from video_highlight.stage2_coarse_highlight.candidate_merger import merge_candidates
from video_highlight.stage2_coarse_highlight.pipeline import run_stage2
from video_highlight.stage2_coarse_highlight.response_parser import parse_response
from video_highlight.stage2_coarse_highlight.segment_loader import LoadedSegment
from video_highlight.stage2_coarse_highlight.video_payload import prepare_video_payload


def make_segment() -> LoadedSegment:
    sample = {
        "sample_id": 0,
        "timestamp_sec": 10.0,
        "original_frame": 300,
        "scene_id": 0,
        "image_path": "coarse_frames/000000.jpg",
    }
    return LoadedSegment("0", 0, 10.0, 15.0, [0], [sample], [Path("frame.jpg")], [], [], [], {"source_path": "0.mp4"})


class ResponseParserTests(unittest.TestCase):
    def test_fenced_json_and_absolute_mapping(self) -> None:
        text = """```json
        {"has_highlight":true,"candidates":[
          {"summary":"goal","highlight_score":0.9,"completeness_score":0.8,
           "start_offset_sec":1.0,"end_offset_sec":3.0,"subject":"player",
           "category":"action_peak","reason":"result","evidence_sample_ids":[0]},
          {"summary":"reaction","highlight_score":0.7,"completeness_score":0.6,
           "start_offset_sec":3.2,"end_offset_sec":4.5,"subject":null,
           "category":"emotion_peak","reason":"reaction","evidence_sample_ids":[]}
        ]}
        ```"""
        result = parse_response(text, make_segment(), {"clamp_out_of_range_times": True})
        self.assertTrue(result["has_highlight"])
        self.assertEqual(len(result["candidates"]), 2)
        self.assertEqual(result["candidates"][0]["start_sec"], 11.0)
        self.assertEqual(result["candidates"][0]["end_sec"], 13.0)
        self.assertEqual(result["candidates"][0]["evidence_sample_ids"], [0])
        self.assertEqual(result["candidates"][1]["candidate_index"], 1)

    def test_no_highlight_has_empty_candidates(self) -> None:
        result = parse_response(
            '{"has_highlight":false,"candidates":[]}',
            make_segment(),
            {"clamp_out_of_range_times": True},
        )
        self.assertFalse(result["has_highlight"])
        self.assertEqual(result["candidates"], [])


class CandidateMergerTests(unittest.TestCase):
    def test_overlapping_candidates_merge(self) -> None:
        base = {
            "video_id": "0",
            "candidate_id": "raw",
            "coarse_score": 0.7,
            "semantic_score": 0.7,
            "event_score": 0.7,
            "audio_score": 0.2,
            "quality_score": 0.5,
            "stability_score": 0.5,
            "source_segment_ids": [0],
            "subject": "a",
            "subject_point": [0.5, 0.5],
            "category": "x",
            "reason": "r",
        }
        left = {**base, "start_sec": 1.0, "end_sec": 4.0}
        right = {**base, "start_sec": 4.5, "end_sec": 7.0, "source_segment_ids": [1]}
        result = merge_candidates("0", [left, right], 10.0, {
            "expand_before_sec": 0.0, "expand_after_sec": 0.0, "max_merge_gap_sec": 1.0, "min_duration_sec": 0.5
        })
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["source_segment_ids"], [0, 1])


class VideoPayloadTests(unittest.TestCase):
    def test_url_template(self) -> None:
        result = prepare_video_payload(make_segment(), {
            "mode": "url",
            "url_template": "http://media/{video_id}?start={start_ms}&end={end_ms}",
            "allowed_url_schemes": ["http"],
        }, Path("unused"))
        self.assertEqual(result.descriptor["url"], "http://media/0?start=10000&end=15000")


class MockPipelineTests(unittest.TestCase):
    def test_mock_pipeline_persists_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video_dir = root / "stage1/videos/0"
            (video_dir / "coarse_frames").mkdir(parents=True)
            (video_dir / "coarse_frames/000000.jpg").write_bytes(b"placeholder")
            write_json(video_dir / "metadata.json", {"video_id": "0", "duration_sec": 5.0, "source_path": "0.mp4"})
            write_jsonl(video_dir / "scenes.jsonl", [{"scene_id": 0, "start_sec": 0.0, "end_sec": 5.0}])
            write_jsonl(video_dir / "sample_map.jsonl", [{
                "sample_id": 0, "timestamp_sec": 0.0, "original_frame": 0, "scene_id": 0,
                "image_path": "coarse_frames/000000.jpg"
            }])
            write_jsonl(video_dir / "segments.jsonl", [{
                "segment_id": 0, "start_sec": 0.0, "end_sec": 5.0, "sample_ids": [0], "scene_ids": [0]
            }])
            for name in ("audio_events.jsonl", "audio_timeline.jsonl", "asr.jsonl"):
                write_jsonl(video_dir / name, [])
            write_json(video_dir / "_SUCCESS.json", {"status": "success"})
            config = {
                "api": {"healthcheck_on_start": False, "model": "mock", "base_url": "mock"},
                "runtime": {"backend": "mock", "persist_request_descriptors": True, "persist_raw_responses": True},
                "video_input": {"mode": "url", "url_template": "http://media/{video_id}/{segment_id}", "allowed_url_schemes": ["http"], "fps": 2.0},
                "context": {"max_frames": 64},
                "parsing": {"retries": 0, "clamp_out_of_range_times": True},
                "scoring": {"weights": {"semantic": 0.4, "event": 0.2, "audio": 0.15, "quality": 0.15, "stability": 0.1}, "neutral_quality_score": 0.5, "neutral_stability_score": 0.5, "min_candidate_score": 0.1},
                "merging": {"expand_before_sec": 0.0, "expand_after_sec": 0.0, "max_merge_gap_sec": 1.0, "min_duration_sec": 0.5},
            }
            prompts = {"system_prompt": "system", "user_template": "{video_id} {segment_id} {segment_start_sec} {segment_end_sec} {segment_duration_sec} {visual_fps} {frame_timeline} {audio_events} {audio_timeline} {asr_timeline}"}
            summary = run_stage2(root / "stage1", root / "stage2", config, prompts, strict=True)
            candidates = [json.loads(line) for line in (root / "stage2/videos/0/candidates.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(summary["success_count"], 1)
            self.assertEqual(len(candidates), 2)
            self.assertEqual(candidates[0]["start_sec"], 1.0)
            self.assertEqual(candidates[1]["start_sec"], 3.5)


if __name__ == "__main__":
    unittest.main()

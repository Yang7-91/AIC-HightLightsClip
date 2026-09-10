"""不依赖训练模型或外部数据的 Stage 1 单元测试。"""

from __future__ import annotations

import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from video_highlight.stage1_preprocess.audio.acoustic_features import extract_acoustic_features
from video_highlight.stage1_preprocess.segment_planner import plan_segments
from video_highlight.stage1_preprocess.timestamp_map import FrameClock, SamplingSchedule


class TimestampTests(unittest.TestCase):
    def test_sampling_schedule(self) -> None:
        schedule = SamplingSchedule(2.0)
        self.assertTrue(schedule.due(0.0))
        self.assertEqual(schedule.consume(0.0), 0.0)
        self.assertFalse(schedule.due(0.49))
        self.assertTrue(schedule.due(0.5))

    def test_frame_clock_falls_back_on_invalid_pts(self) -> None:
        clock = FrameClock(25.0)
        self.assertEqual(clock.resolve(0, 0.0), 0.0)
        self.assertAlmostEqual(clock.resolve(1, 0.0), 0.04)


class SegmentPlannerTests(unittest.TestCase):
    def test_segments_cover_video_and_overlap(self) -> None:
        scenes = [
            {"scene_id": 0, "start_sec": 0.0, "end_sec": 20.0},
            {"scene_id": 1, "start_sec": 20.0, "end_sec": 40.0},
            {"scene_id": 2, "start_sec": 40.0, "end_sec": 50.0},
        ]
        samples = [{"sample_id": i, "timestamp_sec": i * 0.5} for i in range(100)]
        segments = plan_segments(
            "x",
            50.0,
            scenes,
            samples,
            {"window_sec": 24.0, "overlap_ratio": 0.25, "boundary_snap_sec": 2.0, "min_window_sec": 6.0},
        )
        self.assertEqual(segments[0]["start_sec"], 0.0)
        self.assertEqual(segments[-1]["end_sec"], 50.0)
        for left, right in zip(segments, segments[1:]):
            self.assertLess(right["start_sec"], left["end_sec"])


class AudioFeatureTests(unittest.TestCase):
    def test_fixed_fft_shape_for_short_last_bin(self) -> None:
        sample_rate = 16000
        t = np.arange(int(sample_rate * 0.75), dtype=np.float32) / sample_rate
        signal = (0.2 * np.sin(2 * np.pi * 440 * t) * 32767).astype("<i2")
        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "audio.wav"
            with wave.open(str(wav_path), "wb") as writer:
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(sample_rate)
                writer.writeframes(signal.tobytes())
            features = extract_acoustic_features(wav_path, bin_sec=0.5)
        self.assertEqual(len(features["start_sec"]), 2)
        self.assertEqual(len(features["spectral_flux"]), 2)


if __name__ == "__main__":
    unittest.main()

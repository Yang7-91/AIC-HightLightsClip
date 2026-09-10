import importlib.util
import json
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np


def load_v1_module():
    cv2 = ModuleType("cv2")
    cv2.COLOR_BGR2RGB = 1
    cv2.cvtColor = lambda frame, code: frame
    openai = ModuleType("openai")
    openai.OpenAI = object
    path = Path(__file__).with_name("v1_qwen.py")
    spec = importlib.util.spec_from_file_location("v1_qwen_pipeline_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict("sys.modules", {"cv2": cv2, "openai": openai}):
        spec.loader.exec_module(module)
    return module


class SplitPipelineTests(unittest.TestCase):
    def setUp(self):
        self.module = load_v1_module()
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.args = SimpleNamespace(
            video_dir=str(root),
            stage1a_jsonl=str(root / "stage1a.jsonl"),
            stage1b_jsonl=str(root / "stage1b.jsonl"),
            out=str(root / "predictions.jsonl"),
            detect_fps=1.0, max_new_tokens=512, enable_thinking=True,
            scene_threshold=27.0, scene_min_frames=15,
            subject_max_tokens=256, tracking_fallback="error",
            min_mask_area_ratio=0.0005, max_mask_area_ratio=0.70,
            max_area_change=4.0, max_center_jump=0.20,
            crop_scales="0.55,0.70,0.85,1.00",
            margin_left=0.20, margin_right=0.20,
            margin_top=0.15, margin_bottom=0.30,
            center_alpha=0.25, width_alpha=0.15,
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_three_persisted_stages(self):
        video_path = Path(self.args.video_dir) / "0.mp4"
        video_path.touch()

        class HighlightModel:
            def detect_highlights(self, path, fps_sample, max_new_tokens):
                return [(0.0, 1.5)], {
                    "content": '{"segments":[[0.0,1.5]]}',
                    "reasoning": "thinking is enabled",
                }

        with patch.object(self.module, "video_meta", return_value=(4, 2.0, 100, 50)):
            stage1a = self.module.run_stage1a(
                [("0", (16, 9))], self.args, HighlightModel())
        self.assertEqual(stage1a[0]["schema_version"], self.module.STAGE1A_SCHEMA)
        self.assertEqual(stage1a[0]["segments_frame"], [[0, 3]])
        raw = json.loads(Path(self.args.stage1a_jsonl + ".raw.jsonl").read_text())
        self.assertTrue(raw["raw"]["reasoning"])

        class AnchorModel:
            def predict_subject_box(self, image, ratio, tokens):
                return {
                    "content": '{"subject_box":[100,100,900,900],"subject":"person"}',
                    "reasoning": None,
                }

        frame = np.zeros((50, 100, 3), dtype=np.uint8)
        with patch.object(self.module, "video_meta", return_value=(4, 2.0, 100, 50)), \
                patch.object(self.module, "detect_shots", return_value=[(0, 3)]), \
                patch.object(self.module, "extract_frames", return_value={0: frame}):
            stage1b = self.module.run_stage1b(stage1a, self.args, AnchorModel())
        self.assertEqual(stage1b[0]["schema_version"], self.module.STAGE1B_SCHEMA)
        self.assertEqual(stage1b[0]["shots"][0]["shot_frame"], [0, 3])
        self.assertEqual(stage1b[0]["shots"][0]["anchor_status"], "ok")

        class Tracker:
            def track_range(self, path, start, end, box, size):
                return {frame_id: list(box) for frame_id in range(start, end + 1)}

        with patch.object(self.module, "video_meta", return_value=(4, 2.0, 100, 50)):
            self.module.run_stage2(stage1b, self.args, Tracker())
        result = json.loads(Path(self.args.out).read_text())
        self.assertEqual(result["video_id"], "0")
        self.assertEqual([item["frame"] for item in result["predictions"]],
                         [0, 1, 2, 3])

    def test_wrong_stage_schema_is_rejected(self):
        path = Path(self.args.stage1a_jsonl)
        path.write_text(json.dumps({
            "schema_version": self.module.STAGE1B_SCHEMA,
            "video_id": "0",
        }) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "expected schema_version"):
            self.module.read_jsonl(str(path), self.module.STAGE1A_SCHEMA)

    def test_video_revision_mismatch_is_rejected(self):
        record = {
            "video_id": "0",
            "video_meta": {"num_frames": 10, "width": 100,
                           "height": 50, "fps": 2.0},
        }
        with self.assertRaisesRegex(ValueError, "num_frames mismatch"):
            self.module._validate_video_meta(record, (11, 2.0, 100, 50))


if __name__ == "__main__":
    unittest.main()

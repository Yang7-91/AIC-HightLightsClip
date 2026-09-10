"""独立校验单个 Stage 1 视频产物目录。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from video_highlight.stage1_preprocess.validators import validate_stage1_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description="校验一个 Stage 1 视频产物目录")
    parser.add_argument("video_artifact_dir", type=Path)
    args = parser.parse_args()
    metadata_path = args.video_artifact_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    audio_status_path = args.video_artifact_dir / "audio_status.json"
    audio_status = (
        json.loads(audio_status_path.read_text(encoding="utf-8-sig"))
        if audio_status_path.is_file()
        else {"enabled": False}
    )
    result = validate_stage1_artifacts(
        args.video_artifact_dir,
        has_audio=bool(metadata.get("has_audio")),
        audio_enabled=bool(audio_status.get("enabled", False)),
    )
    print(json.dumps({"status": "valid", **result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

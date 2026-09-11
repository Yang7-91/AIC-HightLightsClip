"""独立校验单个 Stage 2 视频产物目录。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from video_highlight.common.atomic_io import read_jsonl
from video_highlight.stage2_coarse_highlight.validators import validate_stage2_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description="校验一个 Stage 2 视频产物目录")
    parser.add_argument("video_artifact_dir", type=Path)
    args = parser.parse_args()
    result = validate_stage2_artifacts(args.video_artifact_dir)
    candidates = read_jsonl(args.video_artifact_dir / "candidates.jsonl")
    print(json.dumps({"status": "valid", "candidate_count": len(candidates), **result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

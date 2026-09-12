"""对已经生成的最终 JSONL 执行独立严格校验。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from video_highlight.stage5_export.submission_validator import (
    load_input_index,
    load_stage1_metadata,
    validate_submission_file,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="严格校验最终视频高光提交 JSONL")
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--input-index", type=Path, required=True)
    parser.add_argument("--stage1-dir", type=Path, required=True)
    args = parser.parse_args()
    index_rows = load_input_index(args.input_index)
    metadata = load_stage1_metadata(args.stage1_dir, index_rows)
    report = validate_submission_file(args.submission, index_rows, metadata, strict_fields=True)
    print(json.dumps({"status": "valid", **report}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

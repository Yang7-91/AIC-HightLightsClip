"""按输入索引顺序原子、紧凑地写出最终 JSONL。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from video_highlight.common.atomic_io import atomic_text_writer


def write_submission_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """一行一个 JSON 对象写出，不转义中文并拒绝 NaN/Infinity。"""

    with atomic_text_writer(path) as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
            )
            handle.write("\n")

"""Stage 5 运行摘要和最终验证报告持久化。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from video_highlight.common.atomic_io import write_json


def write_reports(
    output_dir: str | Path,
    summary: dict[str, Any],
    validation: dict[str, Any],
) -> None:
    root = Path(output_dir)
    write_json(root / "validation_report.json", validation)
    write_json(root / "run_manifest.json", summary)

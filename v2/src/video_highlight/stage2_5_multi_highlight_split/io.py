"""MHS-1 输入/输出：只读 Stage 1/2 产物，写新目录（不修改原始 candidates）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

# 复用 MHS-VIS-0 已验证的只读加载器（cache / JSONL 双格式 + Heldout 拒绝）。
from video_highlight.stage2_5_visual_report.io import (  # noqa: F401
    VisualReportIOError,
    group_candidates,
    load_video_manifest,
    resolve_video_path,
)

from .schema import Mhs1Candidate


class Mhs1IOError(RuntimeError):
    """MHS-1 输入产物缺失或写出失败。"""


def load_stage2_candidates(
    stage2_dir: str | Path,
    *,
    allowed_splits: tuple[str, ...] = ("dev",),
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """同 MHS-VIS-0：仅 dev split，Heldout 拒绝。"""
    from video_highlight.stage2_5_visual_report.io import load_stage2_candidates as _load

    return _load(stage2_dir, allowed_splits=allowed_splits)


def load_frozen_references(frozen_predictions: str | Path) -> dict[str, list[tuple[float, float]]]:
    """读取 weak-reference（仅用于事后评估；不得进入拆分决策）。"""
    references: dict[str, list[tuple[float, float]]] = {}
    text = Path(frozen_predictions).expanduser().resolve().read_text(encoding="utf-8-sig")
    for line in text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        video_id = str(row["video_id"])
        references[video_id] = [
            (float(segment["start_sec"]), float(segment["end_sec"]))
            for segment in row.get("weak_reference_segments", [])
            if float(segment.get("end_sec", 0.0)) > float(segment.get("start_sec", 0.0))
        ]
    return references


def load_frozen_baseline_segments(
    frozen_predictions: str | Path,
) -> dict[str, list[tuple[float, float]]]:
    """读取 frozen predictions 的 merged_prediction_segments（Stage 3 baseline 口径）。"""
    baseline: dict[str, list[tuple[float, float]]] = {}
    text = Path(frozen_predictions).expanduser().resolve().read_text(encoding="utf-8-sig")
    for line in text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        video_id = str(row["video_id"])
        baseline[video_id] = [
            (float(segment["start_sec"]), float(segment["end_sec"]))
            for segment in row.get("merged_prediction_segments", [])
            if float(segment.get("end_sec", 0.0)) > float(segment.get("start_sec", 0.0))
        ]
    return baseline


def write_json_lines(path: str | Path, rows: Iterable[dict[str, Any]]) -> str:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows)
    path.write_text(payload + ("\n" if payload else ""), encoding="utf-8")
    return str(path)


def write_mhs1_candidates(path: str | Path, candidates: Iterable[Mhs1Candidate]) -> str:
    return write_json_lines(path, [candidate.to_row() for candidate in candidates])


def read_json_lines(path: str | Path) -> list[dict[str, Any]]:
    text = Path(path).expanduser().resolve().read_text(encoding="utf-8-sig")
    return [json.loads(line) for line in text.splitlines() if line.strip()]

"""MHS-VIS-0 输入层：读取 Stage 1/2/3 / MHS-1 / frozen cache 产物（只读）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

# Heldout 相关标记：任何 artifact 中出现即拒绝加载（合规防线）。
_FORBIDDEN_SPLITS = {"heldout", "test_hidden"}


class VisualReportIOError(RuntimeError):
    """输入产物缺失或格式不支持。"""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    text = path.read_text(encoding="utf-8-sig")
    for line in text.splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_video_manifest(path: str | Path | None) -> dict[str, str]:
    """video_id -> relative path 映射（JSONL）。"""
    if path is None:
        return {}
    mapping: dict[str, str] = {}
    for row in _read_jsonl(Path(path)):
        video_id = row.get("video_id")
        relative = row.get("relative_video_path") or row.get("video_path") or row.get("filename")
        if isinstance(video_id, str) and isinstance(relative, str):
            mapping[video_id] = relative
    return mapping


def resolve_video_path(video_root: Path, video_id: str, manifest: dict[str, str]) -> Path | None:
    relative = manifest.get(video_id)
    if relative:
        for candidate in (video_root / relative, video_root / Path(relative).name):
            if candidate.is_file():
                return candidate
    candidate = video_root / f"{video_id}.mp4"
    return candidate if candidate.is_file() else None


def _normalize_candidate(raw: dict[str, Any], video_id: str) -> dict[str, Any] | None:
    start = raw.get("start_sec")
    end = raw.get("end_sec")
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        return None
    if float(end) <= float(start):
        return None
    candidate_id = (
        raw.get("candidate_id")
        or raw.get("merged_candidate_id")
        or raw.get("id")
        or raw.get("segment_id")
        or f"{video_id}:{float(start):.3f}-{float(end):.3f}"
    )
    score = raw.get("score")
    if score is None:
        score = raw.get("highlight_score")
    return {
        "video_id": video_id,
        "candidate_id": str(candidate_id),
        "start_sec": float(start),
        "end_sec": float(end),
        "duration_sec": float(end) - float(start),
        "score": float(score) if isinstance(score, (int, float)) else None,
        "reason": raw.get("reason"),
        "source_chunk": raw.get("source_chunk"),
    }


def load_stage2_candidates(
    stage2_dir: str | Path,
    *,
    allowed_splits: tuple[str, ...] = ("dev",),
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """读取 Stage 2 candidates；兼容 JSONL 与 frozen cache 两种目录。

    默认只加载 ``allowed_splits``（dev）中的记录；Heldout 永远拒绝。
    返回 (candidates, durations_by_video)。
    """
    root = Path(stage2_dir).expanduser().resolve()
    durations: dict[str, float] = {}
    candidates: list[dict[str, Any]] = []

    if (root / "cache_manifest.json").is_file():
        manifest = json.loads((root / "cache_manifest.json").read_text(encoding="utf-8"))
        for entry in manifest["records"]:
            record = json.loads((root / entry["path"]).read_text(encoding="utf-8"))
            split = record.get("split")
            if split in _FORBIDDEN_SPLITS:
                raise VisualReportIOError(
                    f"refusing to load heldout-like split: {split!r}"
                )
            if allowed_splits and split not in allowed_splits:
                continue
            video_id = str(record["video_id"])
            durations[video_id] = float(record["duration_sec"])
            for item in record.get("merged_candidates", []):
                normalized = _normalize_candidate(item, video_id)
                if normalized is not None:
                    candidates.append(normalized)
        return candidates, durations

    candidates_path = root / "candidates.jsonl"
    if not candidates_path.is_file():
        raise VisualReportIOError(
            f"candidates.jsonl or cache_manifest.json not found under: {root}"
        )
    for row in _read_jsonl(candidates_path):
        video_id = str(row.get("video_id"))
        nested = row.get("candidates")
        if isinstance(nested, list):
            for item in nested:
                normalized = _normalize_candidate(item, video_id)
                if normalized is not None:
                    candidates.append(normalized)
        else:
            normalized = _normalize_candidate(row, video_id)
            if normalized is not None:
                candidates.append(normalized)
    metadata_path = root / "video_metadata.jsonl"
    if metadata_path.is_file():
        for row in _read_jsonl(metadata_path):
            if isinstance(row.get("duration_sec"), (int, float)):
                durations[str(row["video_id"])] = float(row["duration_sec"])
    return candidates, durations


def load_stage3_intervals(stage3_dir: str | Path | None) -> dict[str, list[dict[str, Any]]]:
    """可选：读取 Stage 3 refined intervals（按 video_id 分组）。"""
    if stage3_dir is None:
        return {}
    root = Path(stage3_dir).expanduser().resolve()
    path = root / "refined_intervals.jsonl"
    if not path.is_file():
        return {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in _read_jsonl(path):
        grouped.setdefault(str(row.get("video_id")), []).append(row)
    return grouped


def load_mhs1_eventness(mhs1_dir: str | Path | None) -> dict[str, list[dict[str, Any]]]:
    """可选：读取 MHS-1 eventness bins（按 video_id 分组）。"""
    if mhs1_dir is None:
        return {}
    root = Path(mhs1_dir).expanduser().resolve()
    path = root / "mhs1_eventness_bins.jsonl"
    if not path.is_file():
        return {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in _read_jsonl(path):
        grouped.setdefault(str(row.get("video_id")), []).append(row)
    return grouped


def load_mhs1_subsegments(mhs1_dir: str | Path | None) -> dict[str, list[dict[str, Any]]]:
    """可选：读取 MHS-1 proposed subsegments（按 candidate_id 分组）。"""
    if mhs1_dir is None:
        return {}
    root = Path(mhs1_dir).expanduser().resolve()
    path = root / "mhs1_proposed_subsegments.jsonl"
    if not path.is_file():
        return {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in _read_jsonl(path):
        grouped.setdefault(str(row.get("candidate_id")), []).append(row)
    return grouped


def read_video_duration(video_path: Path) -> float:
    """用 OpenCV 读取视频时长（兜底，不修改源文件）。"""
    import cv2

    capture = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not capture.isOpened():
        raise VisualReportIOError(f"cannot open video: {video_path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if fps <= 0 or frames <= 0:
        raise VisualReportIOError(f"invalid video metadata: {video_path}")
    return frames / fps


def group_candidates(candidates: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate["video_id"], []).append(candidate)
    for rows in grouped.values():
        rows.sort(key=lambda item: (item["start_sec"], item["candidate_id"]))
    return grouped

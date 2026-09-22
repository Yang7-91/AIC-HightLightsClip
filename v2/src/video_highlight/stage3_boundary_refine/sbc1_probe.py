"""SBC-1 多提示词 / 多视角语义边界判别探针（diagnostic-only）。

本模块只做诊断：从已有 oracle boundary labels 构建边界侧样本，渲染三种
预注册 prompt 视角（P1 动作分类 / P2 候选片段 ranking / P3 core-vs-boundary
对比），解析模型严格 JSON 输出并计算可分性指标。

它不生成正式 refined candidates，不写回 Stage 2/3 产物，不访问 Hard/Heldout，
不训练模型。all outputs are diagnostic-only and non-deployable.
"""

from __future__ import annotations

import base64
import json
import math
import random
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

SBC1_SCHEMA_VERSION = "stage3.sbc1_semantic_boundary_probe.v1"
SBC1_ACTIONS = ("TRIM", "KEEP", "EXPAND")
SBC1_SIDES = ("left", "right")
P1_VARIANT = "P1_ACTION_V2"
P2_VARIANT = "P2_VARIANT_RANKING_V1"
P3_VARIANT = "P3_CORE_CONTRAST_V1"
SBC1_VARIANTS = (P1_VARIANT, P2_VARIANT, P3_VARIANT)


@dataclass(frozen=True, slots=True)
class Sbc1Sample:
    sample_id: str
    video_id: str
    candidate_id: str
    side: str
    oracle_label: str
    original_start_sec: float
    original_end_sec: float
    boundary_sec: float
    candidate_duration_sec: float
    prompt_variant: str
    clip_spec: dict[str, Any] = field(default_factory=dict)
    diagnostic_only: bool = True
    deployable_method: bool = False


@dataclass(frozen=True, slots=True)
class Sbc1Prediction:
    sample_id: str
    prompt_variant: str
    predicted_action: str | None
    confidence: float | None
    parse_ok: bool
    error: str | None = None
    raw_response: str | None = None
    payload: dict[str, Any] | None = None


class Sbc1ParseError(ValueError):
    """模型响应不符合该 prompt 变体的严格 JSON schema。"""


def load_oracle_boundary_labels(path: str | Path) -> list[dict[str, Any]]:
    """读取 BHD-0.1 风格 boundary_oracle_labels.jsonl。"""
    rows: list[dict[str, Any]] = []
    text = Path(path).read_text(encoding="utf-8-sig")
    for line in text.splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_cache_records(cache_dir: str | Path) -> dict[str, dict[str, Any]]:
    """读取 frozen candidate cache 的 records（只读）。"""
    root = Path(cache_dir).expanduser().resolve()
    manifest = json.loads((root / "cache_manifest.json").read_text(encoding="utf-8"))
    records: dict[str, dict[str, Any]] = {}
    for entry in manifest["records"]:
        record = json.loads((root / entry["path"]).read_text(encoding="utf-8"))
        records[record["video_id"]] = record
    return records


def load_video_manifest(path: str | Path) -> dict[str, str]:
    """video_id -> relative path 映射（JSONL）。"""
    mapping: dict[str, str] = {}
    text = Path(path).read_text(encoding="utf-8-sig")
    for line in text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        video_id = row.get("video_id")
        relative = row.get("relative_video_path") or row.get("video_path") or row.get("filename")
        if isinstance(video_id, str) and isinstance(relative, str):
            mapping[video_id] = relative
    return mapping


def build_sbc1_samples(
    labels: Iterable[Mapping[str, Any]],
    cache_records: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    video_ids_allowed: set[str] | None = None,
) -> tuple[list[Sbc1Sample], dict[str, Any]]:
    """分层采样边界侧基础样本（deterministic，含全部 3 个 prompt 视角）。"""
    sampling = config["sampling"]
    cap_per_group = int(sampling["max_per_label_per_side"])
    max_total = int(sampling["max_samples_total"])
    seed = int(sampling["seed"])
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    skipped = {"missing_video": 0, "missing_candidate": 0, "non_dev": 0, "unknown_action": 0}
    for row in labels:
        video_id = row.get("video_id")
        candidate_id = row.get("candidate_id")
        if video_ids_allowed is not None and str(video_id) not in video_ids_allowed:
            skipped["non_dev"] += 1
            continue
        record = cache_records.get(str(video_id))
        if record is None:
            skipped["missing_video"] += 1
            continue
        if record.get("split") != "dev":
            skipped["non_dev"] += 1
            continue
        candidate = next(
            (
                item
                for item in record.get("merged_candidates", [])
                if str(item.get("merged_candidate_id")) == str(candidate_id)
            ),
            None,
        )
        if candidate is None:
            skipped["missing_candidate"] += 1
            continue
        start = float(candidate["start_sec"])
        end = float(candidate["end_sec"])
        for side in SBC1_SIDES:
            action = row.get(f"{side}_action")
            if action not in SBC1_ACTIONS:
                skipped["unknown_action"] += 1
                continue
            grouped.setdefault((side, action), []).append(
                {
                    "video_id": str(video_id),
                    "candidate_id": str(candidate_id),
                    "start_sec": start,
                    "end_sec": end,
                }
            )
    rng = random.Random(seed)
    base_rows: list[dict[str, Any]] = []
    group_report: dict[str, Any] = {}
    for side in SBC1_SIDES:
        for action in SBC1_ACTIONS:
            members = sorted(
                grouped.get((side, action), []),
                key=lambda item: (item["video_id"], item["candidate_id"]),
            )
            cap = min(len(members), cap_per_group)
            chosen = (
                members
                if cap == len(members)
                else sorted(
                    rng.sample(members, cap),
                    key=lambda item: (item["video_id"], item["candidate_id"]),
                )
            )
            base_rows.extend({"side": side, "oracle_label": action, **item} for item in chosen)
            group_report[f"{side}:{action}"] = {"available": len(members), "selected": cap}
    if len(base_rows) > max_total:
        base_rows = base_rows[:max_total]
    samples: list[Sbc1Sample] = []
    for row in base_rows:
        record = cache_records[row["video_id"]]
        duration = float(record["duration_sec"])
        for variant in SBC1_VARIANTS:
            samples.append(
                Sbc1Sample(
                    sample_id=f"{row['video_id']}|{row['candidate_id']}|{row['side']}|{variant}",
                    video_id=row["video_id"],
                    candidate_id=row["candidate_id"],
                    side=row["side"],
                    oracle_label=row["oracle_label"],
                    original_start_sec=row["start_sec"],
                    original_end_sec=row["end_sec"],
                    boundary_sec=row["start_sec"] if row["side"] == "left" else row["end_sec"],
                    candidate_duration_sec=row["end_sec"] - row["start_sec"],
                    prompt_variant=variant,
                    clip_spec={},
                )
            )
    summary = {
        "schema_version": SBC1_SCHEMA_VERSION,
        "diagnostic_only": True,
        "deployable_method": False,
        "base_sample_count": len(base_rows),
        "total_samples": len(samples),
        "variant_counts": {variant: len(base_rows) for variant in SBC1_VARIANTS},
        "left_count": sum(1 for row in base_rows if row["side"] == "left"),
        "right_count": sum(1 for row in base_rows if row["side"] == "right"),
        "label_counts": {
            action: sum(1 for row in base_rows if row["oracle_label"] == action)
            for action in SBC1_ACTIONS
        },
        "groups": group_report,
        "skipped": skipped,
        "seed": seed,
        "heldout_accessed": False,
    }
    return samples, summary


def _clamp_window(start: float, end: float, duration: float) -> tuple[float, float]:
    start = max(0.0, min(start, duration))
    end = max(start + 0.5, min(end, duration))
    return start, end


def compute_clip_windows(
    sample: Sbc1Sample,
    config: Mapping[str, Any],
    duration_sec: float,
) -> list[dict[str, Any]]:
    """按 prompt 变体计算 clip 窗口（确定性，clamp 到视频范围）。"""
    ctx = config["video_context"]
    boundary_window = float(ctx["boundary_window_sec"])
    core_window = float(ctx["core_window_sec"])
    variant_window = float(ctx["variant_window_sec"])
    shift = float(ctx["variant_shift_sec"])
    b = sample.boundary_sec
    duration = max(duration_sec, 0.5)
    windows: list[dict[str, Any]] = []
    if sample.prompt_variant == P1_VARIANT:
        start, end = _clamp_window(b - boundary_window, b + boundary_window, duration)
        windows.append({"role": "boundary", "start_sec": start, "end_sec": end})
    elif sample.prompt_variant == P2_VARIANT:
        inward = 1.0 if sample.side == "left" else -1.0
        for label, position in (
            ("KEEP", b),
            ("TRIM", b + inward * shift),
            ("EXPAND", b - inward * shift),
        ):
            start, end = _clamp_window(position - variant_window, position + variant_window, duration)
            windows.append({"role": label, "start_sec": start, "end_sec": end})
    elif sample.prompt_variant == P3_VARIANT:
        core_center = (sample.original_start_sec + sample.original_end_sec) / 2.0
        half_core = min(core_window, sample.candidate_duration_sec) / 2.0
        core_start, core_end = _clamp_window(core_center - half_core, core_center + half_core, duration)
        windows.append({"role": "core", "start_sec": core_start, "end_sec": core_end})
        start, end = _clamp_window(b - boundary_window, b + boundary_window, duration)
        windows.append({"role": "boundary", "start_sec": start, "end_sec": end})
    else:
        raise ValueError(f"unknown SBC-1 prompt variant: {sample.prompt_variant}")
    return windows


def render_sbc1_prompt(
    template: str,
    sample: Sbc1Sample,
    windows: list[dict[str, Any]],
    config: Mapping[str, Any],
) -> str:
    """填充固定 prompt 模板（只注入确定性上下文，不注入 oracle label）。"""
    ctx = config["video_context"]
    boundary_window = float(ctx["boundary_window_sec"])
    shift = float(ctx["variant_shift_sec"])
    if sample.side == "left":
        direction = (
            "For this LEFT boundary, moving inward means moving later (to the right), "
            "and moving outward means moving earlier (to the left)."
        )
    else:
        direction = (
            "For this RIGHT boundary, moving inward means moving earlier (to the left), "
            "and moving outward means moving later (to the right)."
        )
    reference_window = windows[-1]
    if sample.prompt_variant == P3_VARIANT:
        offset = sample.boundary_sec - windows[1]["start_sec"]
        clip_duration = windows[1]["end_sec"] - windows[1]["start_sec"]
    else:
        offset = sample.boundary_sec - reference_window["start_sec"]
        clip_duration = reference_window["end_sec"] - reference_window["start_sec"]
    return (
        template.replace("{window_before_sec}", f"{boundary_window:.1f}")
        .replace("{window_after_sec}", f"{boundary_window:.1f}")
        .replace("{boundary_offset_in_clip_sec}", f"{offset:.2f}")
        .replace("{clip_duration_sec}", f"{clip_duration:.2f}")
        .replace("{side}", sample.side)
        .replace("{candidate_duration_sec}", f"{sample.candidate_duration_sec:.2f}")
        .replace("{variant_shift_sec}", f"{shift:.1f}")
        .replace("{side_direction}", direction)
    )


def extract_clip_data_url(
    video_path: str | Path,
    start_sec: float,
    end_sec: float,
    *,
    ffmpeg_bin: str = "ffmpeg",
    fps: float = 2.0,
    max_payload_mib: float = 64.0,
    work_dir: str | Path | None = None,
) -> str:
    """用 ffmpeg 截取片段并编码为 data:video/mp4;base64 URL。

    mp4 muxer 需要可 seek 的输出，因此先写入临时文件再读回编码。
    """
    import tempfile

    duration = end_sec - start_sec
    if duration <= 0:
        raise ValueError(f"invalid clip window: [{start_sec}, {end_sec}]")
    with tempfile.TemporaryDirectory(dir=str(work_dir) if work_dir else None) as tmp:
        clip_path = Path(tmp) / "clip.mp4"
        command = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start_sec:.3f}",
            "-i",
            str(video_path),
            "-t",
            f"{duration:.3f}",
            "-an",
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
            "-r",
            f"{fps}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "28",
            str(clip_path),
        ]
        completed = subprocess.run(command, capture_output=True, check=False)
        if completed.returncode != 0 or not clip_path.is_file() or clip_path.stat().st_size == 0:
            detail = (
                completed.stderr.decode("utf-8", errors="replace").strip()
                or "unknown ffmpeg error"
            )
            raise RuntimeError(f"ffmpeg clip extraction failed: {detail}")
        raw = clip_path.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    payload_mib = len(encoded) / (1024 * 1024)
    if payload_mib > max_payload_mib:
        raise RuntimeError(
            f"clip payload too large: {payload_mib:.1f} MiB > {max_payload_mib} MiB"
        )
    return f"data:video/mp4;base64,{encoded}"


def parse_sbc1_response(variant_id: str, content: str) -> dict[str, Any]:
    """按 prompt 变体解析严格 JSON 输出。"""
    if not isinstance(content, str) or not content.strip():
        raise Sbc1ParseError("empty response")
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise Sbc1ParseError(f"invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise Sbc1ParseError("response is not a JSON object")
    field_name = "best_variant" if variant_id == P2_VARIANT else "action"
    action = payload.get(field_name)
    if action not in SBC1_ACTIONS:
        raise Sbc1ParseError(f"invalid {field_name}: {action!r}")
    confidence = payload.get("confidence")
    if not isinstance(confidence, (int, float)) or not math.isfinite(float(confidence)):
        raise Sbc1ParseError(f"invalid confidence: {confidence!r}")
    if not 0.0 <= float(confidence) <= 1.0:
        raise Sbc1ParseError(f"confidence out of range: {confidence!r}")
    parsed: dict[str, Any] = {field_name: action, "confidence": float(confidence)}
    if variant_id == P2_VARIANT:
        ranking = payload.get("ranking")
        if ranking is not None and (
            not isinstance(ranking, list)
            or sorted(ranking) != sorted(SBC1_ACTIONS)
        ):
            raise Sbc1ParseError(f"invalid ranking: {ranking!r}")
        parsed["ranking"] = ranking
    rationale = payload.get("rationale_short", "")
    if rationale is not None and not isinstance(rationale, str):
        raise Sbc1ParseError("rationale_short must be a string")
    parsed["rationale_short"] = (rationale or "")[:200]
    for optional_key in ("evidence", "same_event_outside", "boundary_context_is_redundant"):
        if optional_key in payload:
            parsed[optional_key] = payload[optional_key]
    return parsed


def _one_vs_rest_auc(score_pos: list[float], score_neg: list[float]) -> float:
    if not score_pos or not score_neg:
        return 0.5
    wins = 0.0
    for p in score_pos:
        for n in score_neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(score_pos) * len(score_neg))


def _macro_f1(confusion: Mapping[str, Mapping[str, int]]) -> float:
    f1s = []
    for label in SBC1_ACTIONS:
        tp = confusion.get(label, {}).get(label, 0)
        fn = sum(confusion.get(label, {}).get(other, 0) for other in SBC1_ACTIONS if other != label)
        fp = sum(confusion.get(other, {}).get(label, 0) for other in SBC1_ACTIONS if other != label)
        if tp == 0 and fp == 0 and fn == 0:
            continue
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0.0


def compute_sbc1_metrics(
    samples: list[Sbc1Sample],
    predictions: list[Sbc1Prediction],
    variant_id: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """accuracy / macro-F1 / per-class P-R / per-side AUC-like / confusion。"""
    variant_samples = [s for s in samples if s.prompt_variant == variant_id]
    prediction_by_id = {p.sample_id: p for p in predictions if p.prompt_variant == variant_id}
    confusion = {label: {other: 0 for other in SBC1_ACTIONS} for label in SBC1_ACTIONS}
    per_class_total = {label: 0 for label in SBC1_ACTIONS}
    per_class_correct = {label: 0 for label in SBC1_ACTIONS}
    side_instances: dict[str, list[tuple[str, str, float]]] = {side: [] for side in SBC1_SIDES}
    parsed = 0
    for sample in variant_samples:
        prediction = prediction_by_id.get(sample.sample_id)
        if prediction is None or not prediction.parse_ok or prediction.predicted_action is None:
            continue
        parsed += 1
        confusion[sample.oracle_label][prediction.predicted_action] += 1
        per_class_total[sample.oracle_label] += 1
        if prediction.predicted_action == sample.oracle_label:
            per_class_correct[sample.oracle_label] += 1
        side_instances[sample.side].append(
            (sample.oracle_label, prediction.predicted_action, float(prediction.confidence or 0.0))
        )
    total = len(variant_samples)
    per_class = {}
    for label in SBC1_ACTIONS:
        tp = confusion[label][label]
        fn = sum(confusion[label][other] for other in SBC1_ACTIONS if other != label)
        fp = sum(confusion[other][label] for other in SBC1_ACTIONS if other != label)
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        per_class[label] = {
            "support": per_class_total[label],
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0,
        }
    side_metrics = {}
    for side in SBC1_SIDES:
        aucs = []
        instances = side_instances[side]
        for label in SBC1_ACTIONS:
            positives = [
                confidence if predicted == label else 0.0
                for oracle, predicted, confidence in instances
                if oracle == label
            ]
            negatives = [
                confidence if predicted == label else 0.0
                for oracle, predicted, confidence in instances
                if oracle != label
            ]
            if not positives or not negatives:
                continue
            auc = _one_vs_rest_auc(positives, negatives)
            aucs.append(max(auc, 1.0 - auc))
        correct = sum(1 for sample in variant_samples if sample.side == side and prediction_by_id.get(sample.sample_id, Sbc1Prediction("", "", None, None, False)).predicted_action == sample.oracle_label)
        side_total = sum(1 for sample in variant_samples if sample.side == side)
        side_metrics[side] = {
            "support": side_total,
            "accuracy": correct / side_total if side_total else 0.0,
            "auc_like": sum(aucs) / len(aucs) if aucs else 0.5,
        }
    rules = config["decision_rules"]
    schema_success_rate = parsed / total if total else 0.0
    macro_f1 = _macro_f1(confusion)
    max_auc = max(side_metrics["left"]["auc_like"], side_metrics["right"]["auc_like"])
    actionable = (
        schema_success_rate >= float(rules["schema_success_rate_min"])
        and macro_f1 >= float(rules["macro_f1_min"])
        and max_auc >= float(rules["auc_like_min"])
    )
    return {
        "stage": "Stage 3 diagnostic / SBC-1",
        "prompt_variant": variant_id,
        "diagnostic_only": True,
        "deployable_method": False,
        "num_samples": total,
        "num_parsed": parsed,
        "schema_success_rate": schema_success_rate,
        "accuracy": sum(per_class_correct.values()) / parsed if parsed else 0.0,
        "macro_f1": macro_f1,
        "left_auc_like": side_metrics["left"]["auc_like"],
        "right_auc_like": side_metrics["right"]["auc_like"],
        "per_class": per_class,
        "side_metrics": side_metrics,
        "confusion_matrix": confusion,
        "decision": "ACTIONABLE_SIGNAL" if actionable else "NOT_ACTIONABLE",
        "rule_checks": {
            "schema_success_rate_pass": schema_success_rate >= float(rules["schema_success_rate_min"]),
            "macro_f1_pass": macro_f1 >= float(rules["macro_f1_min"]),
            "auc_like_pass": max_auc >= float(rules["auc_like_min"]),
        },
        "hard_run": False,
        "heldout_accessed": False,
        "training": False,
    }


def write_sbc1_summary(
    evaluations: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """P1/P2/P3 对比表与 best variant 选择。"""
    comparison = {}
    for variant_id, evaluation in evaluations.items():
        comparison[variant_id] = {
            "accuracy": evaluation["accuracy"],
            "macro_f1": evaluation["macro_f1"],
            "left_auc_like": evaluation["left_auc_like"],
            "right_auc_like": evaluation["right_auc_like"],
            "schema_success_rate": evaluation["schema_success_rate"],
            "decision": evaluation["decision"],
        }
    actionable_variants = [
        variant for variant, row in comparison.items() if row["decision"] == "ACTIONABLE_SIGNAL"
    ]
    priority = {P2_VARIANT: 0, P1_VARIANT: 1, P3_VARIANT: 2}
    if actionable_variants:
        enforce_p2 = bool(config["decision_rules"].get("p2_priority_if_tie", True))
        best = (
            min(actionable_variants, key=lambda v: priority.get(v, 9))
            if enforce_p2
            else max(actionable_variants, key=lambda v: comparison[v]["macro_f1"])
        )
        decision = "ACTIONABLE_SIGNAL"
        recommendation = "DESIGN_BR3_DEV_ONLY"
    else:
        best = max(comparison, key=lambda v: comparison[v]["macro_f1"]) if comparison else None
        decision = "NOT_ACTIONABLE"
        recommendation = "FREEZE_TEMPORAL_OR_TRY_COARSE_RETRIEVAL_RESCORING"
    return {
        "stage": "Stage 3 diagnostic / SBC-1",
        "diagnostic_only": True,
        "deployable_method": False,
        "comparison": comparison,
        "best_variant": best,
        "decision": decision,
        "recommendation": recommendation,
        "hard_run": False,
        "heldout_accessed": False,
        "training": False,
    }


def sample_to_row(sample: Sbc1Sample, clip_spec: Mapping[str, Any] | None = None) -> dict[str, Any]:
    row = asdict(sample)
    if clip_spec is not None:
        row["clip_spec"] = dict(clip_spec)
    return row

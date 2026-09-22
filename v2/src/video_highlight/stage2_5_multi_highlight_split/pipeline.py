"""MHS-1 pipeline：编排 eventness 计算、拆分决策、写出与评估（不写回 Stage 2/3）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .eventness import (
    EventnessError,
    build_eventness_bins,
    compute_audio_energy_curve,
    compute_frame_difference_curve,
)
from .evaluate import (
    compare_aggregates,
    count_new_zero_recall,
    evaluate_segments,
)
from .io import (
    group_candidates,
    load_frozen_baseline_segments,
    load_frozen_references,
    load_stage2_candidates,
    load_video_manifest,
    resolve_video_path,
    write_json_lines,
    write_mhs1_candidates,
)
from .schema import make_mhs1_candidate
from .splitter import propose_split


def run_mhs1_config(
    *,
    stage2_dir: str | Path,
    video_root: str | Path,
    output_dir: str | Path,
    config: Mapping[str, Any],
    config_name: str,
    video_manifest: str | Path | None = None,
    frozen_predictions: str | Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """对单个配置运行 MHS-1（只读输入 + 新目录写出）。"""
    split_cfg = config["splitter"]["configs"].get(config_name)
    if not isinstance(split_cfg, dict):
        raise SystemExit(f"unknown MHS-1 config: {config_name}")

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    video_root = Path(video_root).expanduser().resolve()
    manifest = load_video_manifest(video_manifest)

    candidates, durations = load_stage2_candidates(stage2_dir)
    candidates_by_video = group_candidates(candidates)
    if limit is not None:
        limited: dict[str, list[dict[str, Any]]] = {}
        remaining = int(limit)
        for video_id in sorted(candidates_by_video):
            if remaining <= 0:
                break
            rows = candidates_by_video[video_id][:remaining]
            limited[video_id] = rows
            remaining -= len(rows)
        candidates_by_video = limited

    mhs1_rows = []
    eventness_rows = []
    decision_rows = []
    stats = {
        "num_parent_candidates": 0,
        "num_split": 0,
        "num_identity": 0,
        "num_fallback_identity": 0,
        "num_videos": len(candidates_by_video),
    }
    parent_segments: dict[str, list[tuple[float, float]]] = {}
    mhs1_segments: dict[str, list[tuple[float, float]]] = {}
    for video_id in sorted(candidates_by_video):
        video_path = resolve_video_path(video_root, video_id, manifest)
        if video_path is None:
            continue
        video_duration = float(durations.get(video_id, 0.0))
        for candidate in candidates_by_video[video_id]:
            parent_start = float(candidate["start_sec"])
            parent_end = float(candidate["end_sec"])
            parent_segments.setdefault(video_id, []).append((parent_start, parent_end))
            stats["num_parent_candidates"] += 1
            decision: dict[str, Any]
            bins: list[dict[str, Any]] = []
            try:
                frame_bins, _shots = compute_frame_difference_curve(
                    video_path,
                    parent_start,
                    parent_end,
                    sample_fps=float(config["eventness"]["sample_fps"]),
                    bin_sec=float(config["eventness"]["bin_sec"]),
                )
                audio_energies, audio_available = compute_audio_energy_curve(
                    video_path, parent_start, parent_end, bin_sec=float(config["eventness"]["bin_sec"])
                )
                bins = build_eventness_bins(
                    video_id,
                    candidate["candidate_id"],
                    bins=frame_bins,
                    audio_energies=audio_energies,
                    audio_available=audio_available,
                    config=config,
                )
                decision = propose_split(
                    candidate,
                    bins,
                    config,
                    config_name=config_name,
                    video_duration=video_duration,
                )
            except EventnessError as exc:
                decision = {
                    "action": "fallback_identity",
                    "segments": [{"start_sec": parent_start, "end_sec": parent_end}],
                    "fallback_reason": f"eventness_error: {exc}",
                    "coverage_reduction_ratio": 0.0,
                    "num_peaks": 0,
                    "num_valleys": 0,
                }
            eventness_rows.extend(bins)
            mhs1_segments.setdefault(video_id, [])
            if decision["action"] == "split":
                stats["num_split"] += 1
            elif decision["action"] == "identity":
                stats["num_identity"] += 1
            else:
                stats["num_fallback_identity"] += 1
            for index, segment in enumerate(decision["segments"]):
                mhs1 = make_mhs1_candidate(
                    video_id=video_id,
                    source_candidate_id=candidate["candidate_id"],
                    segment_index=index,
                    start_sec=segment["start_sec"],
                    end_sec=segment["end_sec"],
                    parent_start_sec=parent_start,
                    parent_end_sec=parent_end,
                    score=candidate.get("score"),
                    reason=candidate.get("reason"),
                    method=config_name,
                    split_decision={
                        "action": decision["action"],
                        "num_peaks": decision.get("num_peaks", 0),
                        "num_subsegments": len(decision["segments"]),
                        "coverage_reduction_ratio": decision.get(
                            "coverage_reduction_ratio", 0.0
                        ),
                        "fallback_reason": decision.get("fallback_reason"),
                    },
                )
                mhs1_rows.append(mhs1)
                mhs1_segments[video_id].append(
                    (float(segment["start_sec"]), float(segment["end_sec"]))
                )
            decision_rows.append(
                {
                    "video_id": video_id,
                    "candidate_id": candidate["candidate_id"],
                    "action": decision["action"],
                    "num_peaks": decision.get("num_peaks", 0),
                    "num_valleys": decision.get("num_valleys", 0),
                    "num_subsegments": len(decision["segments"]),
                    "coverage_reduction_ratio": decision.get("coverage_reduction_ratio", 0.0),
                    "fallback_reason": decision.get("fallback_reason"),
                    "method": config_name,
                }
            )

    summary = {
        "schema_version": "stage2_5.mhs1_summary.v1",
        "method": "MHS-1",
        "config": config_name,
        "diagnostic_only": False,
        "deployable_method": True,
        **stats,
        "num_mhs1_candidates": len(mhs1_rows),
        "split_rate": (
            stats["num_split"] / stats["num_parent_candidates"]
            if stats["num_parent_candidates"]
            else 0.0
        ),
        "fallback_rate": (
            stats["num_fallback_identity"] / stats["num_parent_candidates"]
            if stats["num_parent_candidates"]
            else 0.0
        ),
        "uses_weak_reference_for_decision": False,
        "heldout_accessed": False,
        "hard_used_for_tuning": False,
        "training": False,
    }

    write_mhs1_candidates(output_dir / "mhs1_candidates.jsonl", mhs1_rows)
    write_json_lines(output_dir / "mhs1_eventness_bins.jsonl", eventness_rows)
    write_json_lines(output_dir / "mhs1_split_decisions.jsonl", decision_rows)

    evaluation: dict[str, Any] | None = None
    if frozen_predictions is not None:
        references = load_frozen_references(frozen_predictions)
        baseline_segments_all = load_frozen_baseline_segments(frozen_predictions)
        # 评估必须限定在 MHS-1 实际覆盖的视频集合内（limit/smoke 与缺视频场景公平）。
        eval_video_ids = sorted(mhs1_segments)
        baseline_segments = {
            video_id: baseline_segments_all.get(video_id, []) for video_id in eval_video_ids
        }
        parent_agg, parent_per_video = evaluate_segments(parent_segments, references)
        baseline_agg, baseline_per_video = evaluate_segments(baseline_segments, references)
        mhs1_agg, mhs1_per_video = evaluate_segments(mhs1_segments, references)
        new_zero = count_new_zero_recall(baseline_per_video, mhs1_per_video)
        parent_count = sum(len(rows) for rows in parent_segments.values())
        mhs1_count = sum(len(rows) for rows in mhs1_segments.values())
        evaluation = {
            "schema_version": "stage2_5.mhs1_evaluation.v1",
            "config": config_name,
            "num_videos": len(mhs1_segments),
            "weak_reference_baseline": baseline_agg,
            "stage2_parent": parent_agg,
            "mhs1": mhs1_agg,
            "vs_stage3_baseline": compare_aggregates(
                baseline_agg,
                mhs1_agg,
                baseline_segments=sum(len(rows) for rows in baseline_segments.values()),
                candidate_segments=mhs1_count,
                num_videos=len(mhs1_segments),
                split_rate=summary["split_rate"],
                fallback_rate=summary["fallback_rate"],
                gate=config["promotion_gate"],
                new_zero_recall=new_zero,
            ),
            "vs_stage2_parent": compare_aggregates(
                parent_agg,
                mhs1_agg,
                baseline_segments=parent_count,
                candidate_segments=mhs1_count,
                num_videos=len(mhs1_segments),
                split_rate=summary["split_rate"],
                fallback_rate=summary["fallback_rate"],
                gate=config["promotion_gate"],
                new_zero_recall=new_zero,
            ),
            "heldout_accessed": False,
            "uses_weak_reference_for_decision": False,
        }
        (output_dir / "mhs1_evaluation.json").write_text(
            json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    (output_dir / "mhs1_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    result = {"summary": summary, "evaluation": evaluation, "output_dir": str(output_dir)}
    return result


def run_mhs1_all_configs(
    *,
    stage2_dir: str | Path,
    video_root: str | Path,
    output_dir: str | Path,
    config: Mapping[str, Any],
    video_manifest: str | Path | None = None,
    frozen_predictions: str | Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """运行全部预注册配置并汇总对比。"""
    output_dir = Path(output_dir).expanduser().resolve()
    results: dict[str, Any] = {}
    for config_name in config["splitter"]["configs"]:
        results[config_name] = run_mhs1_config(
            stage2_dir=stage2_dir,
            video_root=video_root,
            output_dir=output_dir / config_name,
            config=config,
            config_name=config_name,
            video_manifest=video_manifest,
            frozen_predictions=frozen_predictions,
            limit=limit,
        )
    comparison = {}
    for config_name, result in results.items():
        evaluation = result["evaluation"]
        if evaluation is None:
            comparison[config_name] = {"status": "NO_WEAK_REFERENCE_EVAL_AVAILABLE"}
            continue
        compare = evaluation["vs_stage3_baseline"]
        summary = result["summary"]
        comparison[config_name] = {
            "precision": evaluation["mhs1"]["precision"],
            "recall": evaluation["mhs1"]["recall"],
            "f1": evaluation["mhs1"]["f1"],
            "temporal_iou": evaluation["mhs1"]["temporal_iou"],
            "coverage_seconds": evaluation["mhs1"]["prediction_duration_sec"],
            "deltas": compare["deltas"],
            "num_candidates": summary["num_mhs1_candidates"],
            "segments_per_video": compare["segments_per_video"],
            "split_rate": summary["split_rate"],
            "fallback_rate": summary["fallback_rate"],
            "gate": compare["gate"],
            "gate_checks": compare["gate_checks"],
        }
    best = None
    if comparison:
        eligible = [
            name for name, row in comparison.items() if row.get("gate") == "PASS"
        ]
        best = eligible[0] if eligible else None
    summary_all = {
        "schema_version": "stage2_5.mhs1_final_summary.v1",
        "method": "MHS-1",
        "diagnostic_only": False,
        "comparison": comparison,
        "best_config": best,
        "result": (
            "DEV_CANDIDATE_ONLY / READY_FOR_FORMAL_INTEGRATION_DISCUSSION"
            if best
            else "NEGATIVE_RESULT / NO_PROMOTION"
        ),
        "heldout_accessed": False,
        "hard_used_for_tuning": False,
        "training": False,
        "uses_weak_reference_for_decision": False,
    }
    (output_dir / "mhs1_all_configs_summary.json").write_text(
        json.dumps(summary_all, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary_all

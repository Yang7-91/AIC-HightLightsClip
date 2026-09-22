"""MHS-VIS-0 pipeline：编排选择、抽帧、曲线、SVG 与 HTML 报告（只读诊断）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .html_report import write_visual_html_report
from .io import (
    VisualReportIOError,
    group_candidates,
    load_mhs1_eventness,
    load_mhs1_subsegments,
    load_stage2_candidates,
    load_video_manifest,
    resolve_video_path,
)
from .selector import select_visual_examples
from .summarizer import (
    summarize_candidate_rule_based,
    summarize_run_rule_based,
    summarize_subsegment_rule_based,
)
from .thumbnails import extract_candidate_thumbnails, make_contact_sheet
from .timeline import (
    build_timeline_data,
    compute_motion_and_shots,
    find_motion_peaks,
    render_timeline_svg,
)


def run_mhs_vis0(
    *,
    stage2_dir: str | Path,
    video_root: str | Path,
    output_dir: str | Path,
    config: Mapping[str, Any],
    video_manifest: str | Path | None = None,
    mhs1_dir: str | Path | None = None,
    stage3_dir: str | Path | None = None,
    limit: int | None = None,
    manual_video_ids: list[str] | None = None,
) -> dict[str, Any]:
    """运行 MHS-VIS-0，产出本地 HTML/PNG/JSON/MD 诊断产物（不写回任何正式预测）。"""
    output_dir = Path(output_dir).expanduser().resolve()
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    video_root = Path(video_root).expanduser().resolve()
    manifest = load_video_manifest(video_manifest)

    candidates, durations = load_stage2_candidates(stage2_dir)
    if not candidates:
        raise VisualReportIOError("no candidates loaded")
    candidates_by_video = group_candidates(candidates)
    if manual_video_ids:
        candidates_by_video = {
            video_id: rows
            for video_id, rows in candidates_by_video.items()
            if video_id in set(manual_video_ids)
        }
        if not candidates_by_video:
            raise VisualReportIOError("manual video ids match no candidates")

    eventness_by_video = load_mhs1_eventness(mhs1_dir)
    subsegments_by_candidate = load_mhs1_subsegments(mhs1_dir)

    # 预计算每个候选的 motion 峰，用于示例选择（只读视频）。
    peaks_by_candidate: dict[str, list[float]] = {}
    shots_by_candidate: dict[str, list[float]] = {}
    motion_by_candidate: dict[str, list[dict[str, float]]] = {}
    for video_id, rows in candidates_by_video.items():
        video_path = resolve_video_path(video_root, video_id, manifest)
        if video_path is None:
            continue
        for candidate in rows:
            try:
                computed = compute_motion_and_shots(
                    video_path, candidate["start_sec"], candidate["end_sec"]
                )
            except Exception:
                continue
            motion_by_candidate[candidate["candidate_id"]] = computed["samples"]
            shots_by_candidate[candidate["candidate_id"]] = computed["shot_boundaries_sec"]
            peaks_by_candidate[candidate["candidate_id"]] = find_motion_peaks(computed["samples"])

    selected = select_visual_examples(
        candidates_by_video,
        durations,
        config,
        peaks_by_candidate=peaks_by_candidate,
        subsegments_by_candidate=subsegments_by_candidate,
        manual_video_ids=manual_video_ids,
    )
    if limit is not None:
        selected = selected[: max(0, int(limit))]

    thumb_cfg = config["visualization"]
    summary_cfg = config["highlight_summary"]
    entries: list[dict[str, Any]] = []
    markdown_lines = [
        "# MHS-VIS-0 多高光候选可视化诊断（摘要）",
        "",
        "> diagnostic only / non-deployable；不访问 Heldout；不用于人工修正预测。",
        "",
    ]
    for selected_row in selected:
        video_id = selected_row["video_id"]
        candidate_id = selected_row["candidate_id"]
        video_path = resolve_video_path(video_root, video_id, manifest)
        if video_path is None:
            continue
        candidate = next(
            row for row in candidates_by_video[video_id] if row["candidate_id"] == candidate_id
        )
        video_duration = float(durations.get(video_id, candidate["end_sec"]))
        motion_samples = motion_by_candidate.get(candidate_id) or []
        shot_boundaries = shots_by_candidate.get(candidate_id) or []
        motion_peaks = peaks_by_candidate.get(candidate_id) or []
        subsegments = [
            {
                "start_sec": float(segment["start_sec"]),
                "end_sec": float(segment["end_sec"]),
                "label": str(segment.get("label", "")),
            }
            for segment in subsegments_by_candidate.get(candidate_id, [])
        ]
        eventness = eventness_by_video.get(video_id, [])
        timeline_data = build_timeline_data(
            candidate,
            video_duration,
            motion_samples=motion_samples,
            shot_boundaries=shot_boundaries,
            motion_peaks=motion_peaks,
            subsegments=subsegments,
            eventness_bins=eventness,
        )
        for index, segment in enumerate(subsegments):
            segment["summary"] = summarize_subsegment_rule_based(
                index + 1, segment, motion_samples, max_chars=int(summary_cfg["max_summary_chars"])
            )
        svg = render_timeline_svg(
            timeline_data, width_px=int(thumb_cfg["timeline_width_px"])
        )
        prefix = f"{video_id}_{candidate_id[:12]}"
        contact_sheet = None
        if bool(config["output"]["write_contact_sheet_png"]):
            thumbnails = extract_candidate_thumbnails(
                video_path,
                candidate["start_sec"],
                candidate["end_sec"],
                assets_dir,
                fps=float(thumb_cfg["thumbnail_fps"]),
                max_frames=int(thumb_cfg["max_thumbnails_per_candidate"]),
                prefix=prefix,
            )
            contact_sheet = make_contact_sheet(
                thumbnails,
                assets_dir / f"{prefix}_contact_sheet.png",
                thumb_width=int(thumb_cfg["contact_sheet_thumb_width"]),
                subsegments=subsegments,
            )
        candidate_summary = summarize_candidate_rule_based(
            candidate,
            motion_peaks=motion_peaks,
            shot_count=len(shot_boundaries),
            subsegments=subsegments,
            max_chars=int(summary_cfg["max_summary_chars"]),
        )
        entries.append(
            {
                "video_id": video_id,
                "selection_reason": selected_row["reason"],
                "candidate": timeline_data["candidate"],
                "subsegments": subsegments,
                "num_motion_peaks": len(motion_peaks),
                "candidate_summary": candidate_summary,
                "timeline_svg": svg,
                "contact_sheet": contact_sheet,
                "timeline": timeline_data,
            }
        )
        markdown_lines.extend(
            [
                f"## {video_id} · {candidate_id[:24]}…",
                "",
                f"- 原始区间：[{candidate['start_sec']:.2f}s, {candidate['end_sec']:.2f}s]"
                f"（{candidate['duration_sec']:.2f}s）",
                f"- 选择原因：{selected_row['reason']}",
                f"- 运动峰：{len(motion_peaks)} 个 @ "
                + (", ".join(f"{peak:.1f}s" for peak in motion_peaks[:6]) or "无"),
                f"- 镜头切换（候选内）：{len(shot_boundaries)} 个",
                f"- proposed subsegments：{len(subsegments)} 个",
                f"- 候选简介：{candidate_summary}",
                "",
            ]
        )
        if contact_sheet:
            markdown_lines.append(f"- contact sheet：`{contact_sheet}`")
            markdown_lines.append("")

    run_summary = summarize_run_rule_based(selected)
    summary = {
        "schema_version": "stage2_5.mhs_vis0.summary.v1",
        "stage": "Stage 2.5 diagnostic visualization",
        "method": "MHS-VIS-0",
        "diagnostic_only": True,
        "deployable_method": False,
        "num_selected_examples": len(entries),
        "num_videos": len({entry["video_id"] for entry in entries}),
        "eventness_available": bool(eventness_by_video),
        "mhs1_subsegments_available": bool(subsegments_by_candidate),
        "run_summary": run_summary,
        "entries": [
            {
                "video_id": entry["video_id"],
                "candidate_id": entry["candidate"]["candidate_id"],
                "duration_sec": entry["candidate"]["duration_sec"],
                "num_motion_peaks": entry["num_motion_peaks"],
                "num_subsegments": len(entry["subsegments"]),
                "candidate_summary": entry["candidate_summary"],
                "selection_reason": entry["selection_reason"],
            }
            for entry in entries
        ],
        "heldout_accessed": False,
        "hard_used_for_tuning": False,
    }
    output_json = output_dir / "visual_summary.json"
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    selected_path = output_dir / "selected_examples.jsonl"
    selected_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in selected) + "\n",
        encoding="utf-8",
    )
    html_path = None
    if bool(config["output"]["write_html"]):
        html_path = write_visual_html_report(
            output_dir,
            run_info={
                "num_selected_examples": len(entries),
                "num_videos": summary["num_videos"],
                "eventness_available": summary["eventness_available"],
            },
            entries=entries,
            summary_text=run_summary,
            eventness_available=summary["eventness_available"],
        )
    markdown_path = None
    if bool(config["output"]["write_markdown_report"]):
        markdown_lines.append(f"- Run summary：{run_summary}")
        markdown_path = output_dir / "mhs_vis0_report.md"
        markdown_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    summary["outputs"] = {
        "visual_summary_json": str(output_json),
        "selected_examples_jsonl": str(selected_path),
        "visual_report_html": html_path,
        "markdown_report": markdown_path,
        "assets_dir": str(assets_dir),
    }
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary

"""MHS-VIS-1：把 MHS-VIS-0 输出目录导出为 Markdown-first 图文案例报告。

full 报告嵌入 contact sheet 相对路径与内联 SVG 时间轴，只保存在本地 output_dir；
GitHub 脱敏摘要不含任何图片引用。只读上游产物，不写回任何正式预测。
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Any, Mapping

from .timeline import (
    build_timeline_data,
    compute_motion_and_shots,
    find_motion_peaks,
    render_timeline_svg,
)

CASE_REPORT_SCHEMA = "stage2_5.mhs_vis1.case_report.v1"

_SVG_PATTERN = re.compile(r"<svg[\s\S]*?</svg>")
_PEAK_LINE_PATTERN = re.compile(r"运动峰：(\d+) 个 @ (.+)")
_SHOT_LINE_PATTERN = re.compile(r"镜头切换（候选内）：(\d+) 个")


def load_visual_artifacts(vis_dir: str | Path) -> dict[str, Any]:
    """读取 MHS-VIS-0 输出目录的核心产物。"""
    vis_dir = Path(vis_dir).expanduser().resolve()
    summary_path = vis_dir / "visual_summary.json"
    selected_path = vis_dir / "selected_examples.jsonl"
    if not summary_path.is_file():
        raise FileNotFoundError(f"visual_summary.json not found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    selected: list[dict[str, Any]] = []
    if selected_path.is_file():
        for line in selected_path.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                selected.append(json.loads(line))
    html_svgs: list[str] = []
    html_path = vis_dir / "visual_report.html"
    if html_path.is_file():
        html_svgs = _SVG_PATTERN.findall(html_path.read_text(encoding="utf-8"))
    report_markdown = ""
    markdown_path = vis_dir / "mhs_vis0_report.md"
    if markdown_path.is_file():
        report_markdown = markdown_path.read_text(encoding="utf-8")
    return {
        "vis_dir": vis_dir,
        "summary": summary,
        "selected": selected,
        "html_svgs": html_svgs,
        "report_markdown": report_markdown,
    }


def parse_report_markdown_details(report_markdown: str) -> dict[str, dict[str, Any]]:
    """从 MHS-VIS-0 的运行 Markdown 中解析每候选的峰数与镜头数（只读兜底）。"""
    details: dict[str, dict[str, Any]] = {}
    current: str | None = None
    for line in report_markdown.splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            details.setdefault(current, {"peaks": [], "shot_count": 0})
        elif current is not None:
            peak_match = _PEAK_LINE_PATTERN.search(line)
            if peak_match:
                peaks_text = peak_match.group(2).strip()
                peaks = [
                    float(item.replace("s", "").strip())
                    for item in peaks_text.split(",")
                    if item.strip() and item.strip() != "无"
                ]
                details[current]["peaks"] = peaks
            shot_match = _SHOT_LINE_PATTERN.search(line)
            if shot_match:
                details[current]["shot_count"] = int(shot_match.group(1))
    return details


def _case_key(video_id: str, candidate_id: str) -> str:
    return f"{video_id} · {candidate_id[:24]}…"


def _relative_asset_path(vis_dir: Path, path: str | None) -> str | None:
    if not path:
        return None
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(vis_dir).as_posix()
    except ValueError:
        return candidate.name


def _visual_judgment(num_peaks: int) -> str:
    if num_peaks >= 3:
        return "强支持：一个大段包含多个高光事件"
    if num_peaks == 2:
        return "中等支持：疑似包含两个高光事件"
    return "不支持：未观察到明显多峰结构"


def _split_advice(peaks: list[float]) -> str:
    if len(peaks) < 2:
        return "单峰或无明显峰结构，不建议 MHS-1 split 尝试。"
    gaps = [round(peaks[i + 1] - peaks[i], 2) for i in range(len(peaks) - 1)]
    max_gap = max(gaps)
    if max_gap >= 3.0:
        return (
            f"峰之间存在明显低谷间隔（最大间隔 {max_gap:.1f}s ≥ 3s），"
            "形态上适合 MHS-1 split 尝试（需 MHS-1 产出 eventness 后复核）。"
        )
    return (
        f"峰间隔较密集（最大间隔 {max_gap:.1f}s < 3s），可能是同一事件的连续动作，"
        "暂不适合激进 split。"
    )


def _ascii_fallback(start: float, end: float, peaks: list[float]) -> str:
    if not peaks:
        return f"{start:.1f}s |----（无峰）----| {end:.1f}s"
    segments = []
    previous = start
    for peak in peaks:
        segments.append("----")
        segments.append(f"peak@{peak:.1f}s")
        previous = peak
    segments.append("----")
    return f"{start:.1f}s " + " | ".join(segments) + f"| {end:.1f}s"


def _compute_details_with_video(
    video_path: Path,
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    computed = compute_motion_and_shots(
        video_path, float(candidate["start_sec"]), float(candidate["end_sec"])
    )
    samples = computed["samples"]
    peaks = find_motion_peaks(samples)
    peak_rows = []
    for peak in peaks:
        local = min(
            samples, key=lambda item: abs(float(item["t_sec"]) - peak)
        )
        peak_rows.append({"t_sec": peak, "score": float(local["motion"])})
    return {
        "peaks": peak_rows,
        "shots": list(computed["shot_boundaries_sec"]),
        "samples": samples,
    }


def write_markdown_case_report(
    vis_dir: str | Path,
    output_path: str | Path,
    *,
    video_root: str | Path | None = None,
    video_manifest: str | Path | None = None,
    generated_date: str | None = None,
) -> str:
    """生成 full Markdown 报告（含相对路径图片与内联 SVG，仅本地保存）。"""
    from datetime import date

    from .io import load_video_manifest, resolve_video_path

    vis_dir = Path(vis_dir).expanduser().resolve()
    artifacts = load_visual_artifacts(vis_dir)
    summary = artifacts["summary"]
    entries = summary.get("entries", [])
    markdown_details = parse_report_markdown_details(artifacts["report_markdown"])
    svgs = artifacts["html_svgs"]
    manifest = load_video_manifest(video_manifest) if video_manifest else {}
    video_root_path = Path(video_root).expanduser().resolve() if video_root else None

    multi_peak = sum(1 for entry in entries if entry.get("num_motion_peaks", 0) >= 2)
    lines: list[str] = [
        "# MHS-VIS-1 多高光候选可视化案例报告",
        "",
        f"日期：{generated_date or date.today().isoformat()}",
        "输入目录：（本地 MHS-VIS-0 输出目录；本文件与 assets/ 必须整体一起移动）",
        f"结论：共 {len(entries)} 个典型候选（{summary.get('num_videos', 0)} 个视频），"
        f"其中 {multi_peak} 个存在 ≥2 个运动峰，"
        + ("支持" if multi_peak >= len(entries) / 2 else "部分支持")
        + "“一个大段包含多个高光事件”的观察。**本报告为 diagnostic-only，不用于人工修正预测。**",
        "",
        "## 1. 总体结论",
        "",
        f"- 选中候选数：{len(entries)}",
        f"- 多峰候选数（≥2 peaks）：{multi_peak}",
        f"- 典型现象：{summary.get('run_summary', '')}",
        "- 是否支持 MHS-1："
        + ("**支持继续推进 MHS-1 formal**（多峰段普遍存在）" if multi_peak >= len(entries) / 2 else "证据不足，建议先转 Stage 2 reranking"),
        "",
        "## 2. 阅读说明",
        "",
        "- 灰色长条 = 原始 Stage 2 candidate",
        "- 彩色短条 = proposed subsegments（本轮无 MHS-1 结果，before-only）",
        "- 折线 = motion proxy（帧差；MHS-1 eventness 不存在时使用）",
        "- 竖线 = shot boundary（直方图差异检测）",
        "- 黄点 = motion peak；底部刻度 = 时间（秒）",
        "- 缩略图墙绿框 = proposed subsegment 内帧（本轮无 subsegments，全部为灰框）",
        "",
        "## 3. 案例总览表",
        "",
        "| # | video_id | candidate | duration | peaks | shot cuts | visual judgment |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    case_payloads: list[dict[str, Any]] = []
    for index, entry in enumerate(entries, start=1):
        video_id = entry["video_id"]
        candidate_id = entry["candidate_id"]
        duration = float(entry.get("duration_sec", 0.0))
        num_peaks = int(entry.get("num_motion_peaks", 0))
        detail_key = _case_key(video_id, candidate_id)
        parsed = markdown_details.get(detail_key, {})
        shot_count_from_md = int(parsed.get("shot_count", 0))
        case_payloads.append(
            {
                "index": index,
                "video_id": video_id,
                "candidate_id": candidate_id,
                "duration_sec": duration,
                "num_peaks": num_peaks,
                "shot_count_md": shot_count_from_md,
                "svg": svgs[index - 1] if index - 1 < len(svgs) else None,
            }
        )
        lines.append(
            f"| {index} | {video_id} | `{candidate_id[:20]}…` | {duration:.1f}s |"
            f" {num_peaks} | {shot_count_from_md} | {_visual_judgment(num_peaks)} |"
        )
    lines.extend(["", "## 4. 典型案例", ""])

    for case in case_payloads:
        video_id = case["video_id"]
        candidate_id = case["candidate_id"]
        selected_row = next(
            (
                row
                for row in artifacts["selected"]
                if row.get("video_id") == video_id and row.get("candidate_id") == candidate_id
            ),
            {},
        )
        start = float(selected_row.get("candidate_start_sec", 0.0))
        end = float(selected_row.get("candidate_end_sec", start + case["duration_sec"]))
        detail_key = _case_key(video_id, candidate_id)
        parsed = markdown_details.get(detail_key, {})
        peaks: list[dict[str, Any]] = [
            {"t_sec": float(value), "score": None} for value in parsed.get("peaks", [])
        ]
        shots: list[float] = []
        samples: list[dict[str, Any]] = []
        video_path = None
        if video_root_path is not None:
            video_path = resolve_video_path(video_root_path, video_id, manifest)
        if video_path is not None:
            try:
                details = _compute_details_with_video(video_path, {"start_sec": start, "end_sec": end})
                peaks = details["peaks"]
                shots = details["shots"]
                samples = details["samples"]
            except Exception:
                pass
        contact_sheet = None
        assets_dir = Path(summary.get("outputs", {}).get("assets_dir", vis_dir / "assets"))
        for candidate_path in assets_dir.glob(f"{video_id}_{candidate_id[:12]}_contact_sheet.png"):
            contact_sheet = _relative_asset_path(vis_dir, str(candidate_path))
            break
        peak_rows = "\n".join(
            f"| {index + 1} | {row['t_sec']:.1f}s | "
            + (f"{row['score']:.3f}" if row.get("score") is not None else "-")
            + " | 运动强度局部峰 |"
            for index, row in enumerate(peaks)
        )
        shot_rows = "\n".join(f"| {index + 1} | {value:.1f}s |" for index, value in enumerate(shots))
        if shot_rows:
            shot_table = shot_rows
        elif samples:
            shot_table = "| - | （未检测到镜头切换） |"
        else:
            shot_table = f"| - | （明细需视频源；统计 {case['shot_count_md']} 次） |"
        case_lines: list[str] = [
            f"### Case {case['index']}: {video_id}",
            "",
            f"**原始 candidate：** [{start:.2f}s, {end:.2f}s]，duration {case['duration_sec']:.2f}s",
            f"**诊断判断：** {_visual_judgment(len(peaks) or case['num_peaks'])}",
            f"**一句话简介：** {selected_row.get('candidate_summary', entry_summary(artifacts, video_id, candidate_id))}",
            "",
            "#### 时间轴",
            "",
        ]
        if case["svg"]:
            case_lines.append(case["svg"])
        else:
            case_lines.append(f"```\n{_ascii_fallback(start, end, [row['t_sec'] for row in peaks])}\n```")
        case_lines.extend(["", "#### 缩略图墙", ""])
        if contact_sheet:
            case_lines.append(f"![contact sheet]({contact_sheet})")
        else:
            case_lines.append("（未找到 contact sheet）")
        case_lines.extend(
            [
                "",
                "#### 运动峰 / 事件峰",
                "",
                "| peak | time | local score | explanation |",
                "|---|---:|---:|---|",
                peak_rows or "| - | - | - | （无运动峰数据） |",
                "",
                "#### 镜头切换",
                "",
                "| cut | time |",
                "|---|---:|",
                shot_table,
                "",
                "#### 候选解释",
                "",
                f"- 为什么看起来像多个高光：候选内存在 {len(peaks) or case['num_peaks']} 个运动强度局部峰，"
                "峰间存在回落谷值。",
                f"- {_split_advice([row['t_sec'] for row in peaks])}",
                "- proposed subsegments：无 MHS-1 结果，当前为 before-only 诊断。",
                "",
                "---",
                "",
            ]
        )
        lines.extend(case_lines)
    lines.extend(
        [
            "## 5. 下一步建议",
            "",
            "1. 人工浏览本 Markdown（含 SVG 与 contact sheet）确认多峰段是否对应多个语义高光；",
            "2. 确认后推进 MHS-1 formal，本报告与 MHS-0 HTML 会自动获得 eventness/subsegments 对比；",
            "3. 若多峰只是同一事件运动起伏，转 Stage 2 coarse reranking（CRR-1），不做 MHS-1；",
            "4. 合规：本报告仅本地诊断使用，不含正式预测，不访问 Heldout。",
            "",
        ]
    )
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return str(output_path)


def entry_summary(artifacts: Mapping[str, Any], video_id: str, candidate_id: str) -> str:
    for entry in artifacts["summary"].get("entries", []):
        if entry.get("video_id") == video_id and entry.get("candidate_id") == candidate_id:
            return str(entry.get("candidate_summary", ""))
    return ""


def write_github_summary(
    vis_dir: str | Path,
    output_path: str | Path,
    *,
    generated_date: str | None = None,
) -> str:
    """生成脱敏摘要（不含任何图片引用），用于提交 GitHub。"""
    from datetime import date

    vis_dir = Path(vis_dir).expanduser().resolve()
    artifacts = load_visual_artifacts(vis_dir)
    summary = artifacts["summary"]
    entries = summary.get("entries", [])
    multi_peak = sum(1 for entry in entries if entry.get("num_motion_peaks", 0) >= 2)
    typical = [entry["video_id"] for entry in entries[:3]]
    lines = [
        "# MHS-VIS-1 — Markdown-first 可视化案例报告（脱敏摘要）",
        "",
        f"日期：{generated_date or date.today().isoformat()}",
        "类型：diagnostic-only / 本地报告摘要（**不含视频帧图片**）",
        "",
        "## Full 报告（仅本地）",
        "",
        f"- 本地 full Markdown：`{vis_dir / 'mhs_vis0_visual_case_report.md'}`（含内联 SVG 时间轴与 contact sheet 相对路径图片）",
        f"- 本地输出目录：`{vis_dir}`（assets 缩略图与 contact sheet 仅存本地，未上传）",
        "",
        "## 统计",
        "",
        f"- 选中候选数量：{len(entries)}；视频数量：{summary.get('num_videos', 0)}",
        f"- 多峰候选数量（≥2 运动峰）：{multi_peak}",
        f"- 典型 video_id：{', '.join(typical)}",
        "",
        "## 结论",
        "",
        f"- {summary.get('run_summary', '')}",
        "- 支持继续评估 MHS-1 formal（多峰结构在 Dev 候选中普遍存在），但需人工确认多峰是否对应多个语义高光。",
        "",
        "## 合规说明",
        "",
        "- 未访问 Heldout；未运行 Hard；未训练模型；未调用外部 API；",
        "- 未修改 Stage 2/3 主线；未生成正式 predictions/submission；",
        "- 视频、缩略图、contact sheet、HTML、zip 均未提交 GitHub（仅源码与本脱敏摘要）。",
        "",
        "## 下一步建议",
        "",
        "1. 下载本地 `$VIS_DIR`（或 zip bundle）查看图文案例；",
        "2. 确认多峰语义后推进 MHS-1 formal（eventness/subsegments 将自动叠加进本报告）；",
        "3. 否则转 Stage 2 coarse reranking（CRR-1），不再做边界/prompt 微调。",
        "",
    ]
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return str(output_path)


def make_bundle(vis_dir: str | Path, zip_path: str | Path | None = None) -> str:
    """可选：打包本地可视化报告（zip 只留本地）。"""
    vis_dir = Path(vis_dir).expanduser().resolve()
    zip_path = Path(zip_path) if zip_path else vis_dir / "mhs_vis0_visual_case_report_bundle.zip"
    plan = [vis_dir / "mhs_vis0_visual_case_report.md", vis_dir / "visual_summary.json", vis_dir / "selected_examples.jsonl"]
    assets = sorted((vis_dir / "assets").glob("*.png")) if (vis_dir / "assets").is_dir() else []
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in plan:
            if path.is_file():
                archive.write(path, path.relative_to(vis_dir).as_posix())
        for path in assets:
            archive.write(path, path.relative_to(vis_dir).as_posix())
    return str(zip_path)

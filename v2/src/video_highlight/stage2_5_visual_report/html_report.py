"""MHS-VIS-0 HTML 报告生成（内联 SVG + 本地 PNG 引用，不提交产物）。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


def _escape(text: Any) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _render_candidate_section(entry: Mapping[str, Any]) -> str:
    candidate = entry["candidate"]
    timeline_svg = entry["timeline_svg"]
    contact_sheet = entry.get("contact_sheet")
    subsegments = entry.get("subsegments", [])
    rows = "".join(
        f"<tr><td>{_escape(segment.get('label') or index + 1)}</td>"
        f"<td>{float(segment['start_sec']):.2f}s</td>"
        f"<td>{float(segment['end_sec']):.2f}s</td>"
        f"<td>{float(segment['end_sec']) - float(segment['start_sec']):.2f}s</td>"
        f"<td>{_escape(segment.get('summary', ''))}</td></tr>"
        for index, segment in enumerate(subsegments)
    )
    sheet_html = (
        f'<img class="sheet" src="{_escape(Path(contact_sheet).name) if contact_sheet else ""}" alt="contact sheet"/>'
        if contact_sheet
        else "<p>（无 contact sheet）</p>"
    )
    return f"""
    <section class="candidate">
      <h3>{_escape(entry['video_id'])} · {_escape(candidate['candidate_id'][:24])}…</h3>
      <p class="meta">原始时间段：[{candidate['start_sec']:.2f}s, {candidate['end_sec']:.2f}s]，
         时长 {candidate['duration_sec']:.2f}s，
         分数 {candidate.get('score') if candidate.get('score') is not None else 'N/A'}</p>
      <p class="summary">{_escape(entry.get('candidate_summary', ''))}</p>
      {timeline_svg}
      <details open><summary>contact sheet（1 fps 缩略图墙；绿框 = 落在 proposed subsegment 内的帧）</summary>
      {sheet_html}</details>
      <table>
        <thead><tr><th>#</th><th>start</th><th>end</th><th>duration</th><th>简介</th></tr></thead>
        <tbody>{rows or '<tr><td colspan="5">（无 proposed subsegments；before-only 模式）</td></tr>'}</tbody>
      </table>
      <p class="note">diagnostic only · 不得用于人工修正预测 · 未使用 Heldout/Hard 数据</p>
    </section>"""


def write_visual_html_report(
    output_dir: str | Path,
    *,
    run_info: Mapping[str, Any],
    entries: list[Mapping[str, Any]],
    summary_text: str,
    eventness_available: bool,
) -> str:
    """写出 visual_report.html（本地诊断产物，不入库）。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().isoformat(timespec="seconds")
    sections = "\n".join(_render_candidate_section(entry) for entry in entries)
    overview_rows = "".join(
        f"<tr><td>{_escape(entry['video_id'])}</td>"
        f"<td>{_escape(entry['candidate']['candidate_id'][:24])}…</td>"
        f"<td>{entry['candidate']['start_sec']:.1f}s - {entry['candidate']['end_sec']:.1f}s</td>"
        f"<td>{entry['candidate']['duration_sec']:.1f}s</td>"
        f"<td>{entry.get('num_motion_peaks', 0)}</td>"
        f"<td>{len(entry.get('subsegments', []))}</td>"
        f"<td>{_escape(entry.get('selection_reason', ''))}</td></tr>"
        for entry in entries
    )
    eventness_note = (
        "eventness 曲线来自 MHS-1 产物。"
        if eventness_available
        else "MHS-1 eventness 产物不存在：时间轴以视频自身 motion proxy（帧差）代替，报告已明确标注。"
    )
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <title>MHS-VIS-0 多高光候选可视化诊断报告</title>
  <style>
    body {{ background:#0f172a; color:#e2e8f0; font-family:system-ui,-apple-system,"Segoe UI",sans-serif;
           margin:0; padding:24px; }}
    h1,h2,h3 {{ color:#f8fafc; }}
    .banner {{ background:#7c2d12; padding:10px 14px; border-radius:8px; margin-bottom:16px; }}
    .banner b {{ color:#fdba74; }}
    section.candidate {{ background:#111c33; border:1px solid #1e293b; border-radius:10px;
                         padding:14px; margin:14px 0; }}
    .meta {{ color:#94a3b8; font-size:13px; }}
    .summary {{ color:#fbbf24; }}
    .note {{ color:#f87171; font-size:12px; }}
    table {{ width:100%; border-collapse:collapse; margin:10px 0; font-size:13px; }}
    th,td {{ border:1px solid #1e293b; padding:6px 8px; text-align:left; }}
    th {{ background:#1e293b; }}
    img.sheet {{ max-width:100%; border-radius:8px; border:1px solid #1e293b; }}
    details summary {{ cursor:pointer; color:#93c5fd; }}
    {json.dumps(
        {
            "run_info": dict(run_info),
            "generated_at": generated_at,
        },
        ensure_ascii=False,
    )}
  </style>
</head>
<body>
  <h1>MHS-VIS-0 多高光候选可视化诊断报告</h1>
  <div class="banner">
    <b>diagnostic only / non-deployable</b> · 本报告只用于观察"一个大候选是否包含多个高光事件"，
    不作为人工标注或修正依据；不访问 Heldout；{_escape(eventness_note)}
  </div>
  <h2>方法说明</h2>
  <p>对选中的 Stage 2 候选：抽取 1 fps 缩略图墙；在候选区间上计算逐点 motion（帧差）、镜头切换候选与运动峰；
     叠加 MHS-1 proposed subsegments（若存在）。选择规则：时长长 / 多运动峰 / 有多个 proposed subsegments / 覆盖率高。</p>
  <h2>示例总览</h2>
  <p>{_escape(summary_text)}</p>
  <table>
    <thead><tr><th>video</th><th>candidate</th><th>区间</th><th>时长</th><th>运动峰数</th><th>subsegments</th><th>选择原因</th></tr></thead>
    <tbody>{overview_rows}</tbody>
  </table>
  <h2>逐候选详情</h2>
  {sections}
</body>
</html>"""
    path = output_dir / "visual_report.html"
    path.write_text(html, encoding="utf-8")
    return str(path)

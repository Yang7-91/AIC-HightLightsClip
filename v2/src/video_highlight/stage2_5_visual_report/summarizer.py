"""MHS-VIS-0 规则式中文简介生成（可选 VLM 关闭）。"""

from __future__ import annotations

from typing import Any, Mapping


def summarize_candidate_rule_based(
    candidate: Mapping[str, Any],
    *,
    motion_peaks: list[float],
    shot_count: int,
    subsegments: list[Mapping[str, Any]] | None = None,
    max_chars: int = 80,
) -> str:
    """候选大段的一句话中文简介。"""
    duration = float(candidate["duration_sec"])
    peaks = len(motion_peaks)
    subsegments = list(subsegments or [])
    parts = [f"候选大段：约 {duration:.1f} 秒"]
    if peaks >= 2:
        parts.append(f"内部存在 {peaks} 个运动峰（motion proxy）")
        parts.append("疑似包含多个高光事件")
    elif peaks == 1:
        parts.append("内部存在 1 个运动峰，可能是单一高光事件")
    else:
        parts.append("内部运动变化平缓，可能是低动态段落")
    if shot_count >= 2:
        parts.append(f"含 {shot_count} 次镜头切换")
    if len(subsegments) > 1:
        parts.append(f"已有 {len(subsegments)} 个 proposed subsegments")
    text = "，".join(parts) + "。"
    return text[:max_chars]


def summarize_subsegment_rule_based(
    index: int,
    subsegment: Mapping[str, Any],
    motion_samples: list[Mapping[str, float]],
    *,
    max_chars: int = 80,
) -> str:
    """单个子片段的中文简介（基于片内运动统计）。"""
    start = float(subsegment["start_sec"])
    end = float(subsegment["end_sec"])
    values = [
        float(sample["motion"])
        for sample in motion_samples
        if start - 1e-9 <= float(sample["t_sec"]) <= end + 1e-9
    ]
    duration = end - start
    if not values:
        return f"子片段 {index}：约 {duration:.1f} 秒（无运动采样数据）。"[:max_chars]
    peak = max(values)
    mean = sum(values) / len(values)
    if peak >= 2.0 * mean and peak > 0.02:
        trend = "动作强度快速上升，可能是独立高光峰"
    elif mean > 0.02:
        trend = "运动持续活跃，可能属于同一高光事件"
    else:
        trend = "运动较弱，可能为过渡或上下文片段"
    text = f"子片段 {index}：约 {duration:.1f} 秒，{trend}（运动均值 {mean:.3f} / 峰值 {peak:.3f}）。"
    return text[:max_chars]


def summarize_run_rule_based(
    selected_examples: list[Mapping[str, Any]],
    *,
    max_chars: int = 200,
) -> str:
    total = len(selected_examples)
    multi_peak = sum(1 for row in selected_examples if row.get("num_motion_peaks", 0) >= 2)
    text = (
        f"本轮共选择 {total} 个候选用于可视化诊断，其中 {multi_peak} 个候选存在多个运动峰"
        f"（疑似多高光合并）。所有判断均为 diagnostic-only，不用于人工修正预测。"
    )
    return text[:max_chars]

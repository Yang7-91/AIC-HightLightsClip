"""用动态规划在逐帧候选构图之间寻找低抖动全局路径。"""

from __future__ import annotations

import math
from typing import Any


def _transition(left: tuple[float, ...], right: tuple[float, ...], config: dict[str, Any]) -> float:
    left_cx, left_cy = left[0] + left[2] * 0.5, left[1] + left[3] * 0.5
    right_cx, right_cy = right[0] + right[2] * 0.5, right[1] + right[3] * 0.5
    center = math.hypot(right_cx - left_cx, right_cy - left_cy) / max(left[2], right[2], 1.0)
    scale = abs(math.log(max(right[2], 1.0) / max(left[2], 1.0)))
    return float(config.get("center_weight", 0.20)) * center + float(config.get("scale_weight", 0.12)) * scale


def optimize_trajectory(
    candidates_by_frame: list[list[tuple[float, float, float, float]]],
    local_costs: list[list[float]],
    config: dict[str, Any],
) -> list[tuple[float, float, float, float]]:
    if not candidates_by_frame:
        return []
    costs = list(local_costs[0])
    parents: list[list[int]] = []
    for frame_index in range(1, len(candidates_by_frame)):
        previous, current = candidates_by_frame[frame_index - 1], candidates_by_frame[frame_index]
        next_costs: list[float] = []
        next_parents: list[int] = []
        for current_index, current_crop in enumerate(current):
            options = [costs[previous_index] + _transition(previous_crop, current_crop, config) for previous_index, previous_crop in enumerate(previous)]
            parent = min(range(len(options)), key=options.__getitem__)
            next_costs.append(options[parent] + local_costs[frame_index][current_index])
            next_parents.append(parent)
        parents.append(next_parents)
        costs = next_costs
    selected = [min(range(len(costs)), key=costs.__getitem__)]
    for frame_index in range(len(parents) - 1, -1, -1):
        selected.append(parents[frame_index][selected[-1]])
    selected.reverse()
    return [candidates_by_frame[index][state] for index, state in enumerate(selected)]

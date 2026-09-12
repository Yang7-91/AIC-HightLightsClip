"""对构图框中心和尺度做 EMA 平滑及逐帧运动限速。"""

from __future__ import annotations

import math
from typing import Any

from .boundary_limiter import legal_crop_from_state


def smooth_trajectory(
    crops: list[tuple[float, float, float, float]],
    frame_size: tuple[int, int],
    target_ratio: tuple[float, float],
    config: dict[str, Any],
) -> list[tuple[float, float, float, float]]:
    if not crops:
        return []
    center_alpha = float(config.get("center_alpha", 0.25))
    width_alpha = float(config.get("width_alpha", 0.15))
    max_step = float(config.get("max_center_step_ratio", 0.04)) * math.hypot(*frame_size)
    state: tuple[float, float, float] | None = None
    output: list[tuple[float, float, float, float]] = []
    for x, y, width, height in crops:
        desired = (x + width * 0.5, y + height * 0.5, math.log(max(width, 1.0)))
        if state is None:
            state = desired
        else:
            dx = center_alpha * (desired[0] - state[0])
            dy = center_alpha * (desired[1] - state[1])
            distance = math.hypot(dx, dy)
            if max_step > 0 and distance > max_step:
                scale = max_step / distance
                dx, dy = dx * scale, dy * scale
            state = (state[0] + dx, state[1] + dy, state[2] + width_alpha * (desired[2] - state[2]))
        output.append(legal_crop_from_state(state[0], state[1], math.exp(state[2]), frame_size, target_ratio))
    return output

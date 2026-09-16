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
    """对裁剪框轨迹做时域平滑。

        使用指数移动平均（EMA）平滑中心点与宽度（宽度在对数空间平滑），
        并限制中心点 EMA 每帧的最大移动步长，避免画面抖动或突然跳变。
        最后通过 ``legal_crop_from_state`` 将平滑后的状态重建为合法裁剪框。

        Args:
            crops: 原始裁剪框序列，每个元素为 ``(x, y, width, height)``。
            frame_size: 原始帧尺寸 ``(width, height)``。
            target_ratio: 目标宽高比 ``(ratio_w, ratio_h)``。
            config: 平滑参数配置，可包含：
                - ``center_alpha``: 中心 EMA 系数，默认 0.25，越小越平滑。
                - ``width_alpha``: 宽度 EMA 系数，默认 0.15。
                - ``max_center_step_ratio``: 每帧中心最大 EMA 步长占帧对角线比例，默认 0.04。

        Returns:
            平滑后的裁剪框列表，每个元素为 ``(x, y, width, height)``。
    """

    if not crops:
        return []
    # 平滑系数：中心与宽度分别使用不同的 EMA 系数
    center_alpha = float(config.get("center_alpha", 0.25))
    width_alpha = float(config.get("width_alpha", 0.15))
    # 中心点 EMA 每帧最大移动量（像素），按帧对角线长度的比例计算
    max_step = float(config.get("max_center_step_ratio", 0.04)) * math.hypot(*frame_size)
    # 平滑状态：(中心 x, 中心 y, log(宽度))
    # 宽度取对数是为了让放大/缩小在 EMA 中对称
    state: tuple[float, float, float] | None = None
    output: list[tuple[float, float, float, float]] = []
    for x, y, width, height in crops:
        # 当前帧的期望状态：中心点 + 对数宽度
        desired = (x + width * 0.5, y + height * 0.5, math.log(max(width, 1.0)))
        if state is None:
            state = desired  # 第一帧直接采用期望状态，不做平滑
        else:
            # 计算中心点的 EMA 步长
            dx = center_alpha * (desired[0] - state[0])
            dy = center_alpha * (desired[1] - state[1])
            distance = math.hypot(dx, dy)
            # 限制 EMA 单帧步长，避免突然跳变
            if max_step > 0 and distance > max_step:
                scale = max_step / distance
                dx, dy = dx * scale, dy * scale
            # 更新状态：中心点用 EMA，宽度在对数空间用 EMA
            state = (state[0] + dx, state[1] + dy, state[2] + width_alpha * (desired[2] - state[2]))
        # 从平滑状态重建合法裁剪框（修正宽高比、裁剪到帧边界内）
        output.append(legal_crop_from_state(state[0], state[1], math.exp(state[2]), frame_size, target_ratio))
    return output

"""无稳定主体或跟踪失败时使用的最大合法中心构图。"""

from __future__ import annotations

from .boundary_limiter import legal_crop_from_state, maximum_crop_width


def centered_crop(
    frame_size: tuple[int, int], target_ratio: tuple[float, float]
) -> tuple[float, float, float, float]:
    width, height = frame_size
    crop_width = maximum_crop_width(frame_size, target_ratio)
    return legal_crop_from_state(width * 0.5, height * 0.5, crop_width, frame_size, target_ratio)

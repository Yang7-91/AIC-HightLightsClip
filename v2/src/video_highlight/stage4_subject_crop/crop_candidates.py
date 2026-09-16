"""围绕逐帧主体框生成多尺度、多偏移的合法目标比例构图。"""

from __future__ import annotations

from typing import Any

from .boundary_limiter import legal_crop_from_state, maximum_crop_width


def generate_crop_candidates(
    subject_box: list[float] | tuple[float, float, float, float],
    frame_size: tuple[int, int],
    target_ratio: tuple[float, float],
    motion: tuple[float, float],
    config: dict[str, Any],
) -> list[tuple[float, float, float, float]]:
    frame_w, frame_h = frame_size
    target_w, target_h = target_ratio
    x1, y1, x2, y2 = map(float, subject_box)
    subject_w, subject_h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    margins = config.get("subject_margins", [0.20, 0.20, 0.15, 0.30])
    left, right, top, bottom = (float(value) for value in margins)
    # 同时满足边距与宽高比例约束
    required_width = max(
        subject_w * (1.0 + left + right),
        subject_h * (1.0 + top + bottom) * target_w / target_h,
    )
    max_width = maximum_crop_width(frame_size, target_ratio)
    # 固定最大框实验只替换主体中心轨迹，不再让掩码面积改变裁剪尺度，
    # 也不加入运动前视或离散偏移。这样可以公平比较 SAM2 中心与 Qwen 线性插值中心。
    if bool(config.get("fixed_maximum", False)):
        center_x, center_y = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        return [
            legal_crop_from_state(
                center_x,
                center_y,
                float(max_width),
                frame_size,
                target_ratio,
            )
        ]

    scales = sorted({float(value) for value in config.get("scales", [0.55, 0.70, 0.85, 1.0])})
    widths = sorted({max(1.0, min(float(max_width), max_width * scale)) for scale in scales if scale > 0.0})
    widths = [value for value in widths if value + 1e-6 >= required_width] or [float(max_width)]
    if float(max_width) not in widths:
        widths.append(float(max_width))

    center_x, center_y = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    offset_values = [float(value) for value in config.get("offsets", [-0.12, 0.0, 0.12])]
    look_ahead = float(config.get("motion_look_ahead", 2.0))
    candidates: list[tuple[float, float, float, float]] = []
    seen: set[tuple[int, int, int]] = set()
    for width in widths:
        height = width * target_h / target_w
        for offset_x in offset_values:
            for offset_y in offset_values:
                crop = legal_crop_from_state(
                    center_x + offset_x * width + motion[0] * look_ahead,
                    center_y + offset_y * height + motion[1] * look_ahead,
                    width,
                    frame_size,
                    target_ratio,
                )
                key = (round(crop[0]), round(crop[1]), round(crop[2]))
                if key not in seen:
                    candidates.append(crop)
                    seen.add(key)
    return candidates

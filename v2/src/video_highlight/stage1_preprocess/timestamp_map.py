"""解码帧时间戳和采样计划辅助工具。"""

from __future__ import annotations

import math


class FrameClock:
    """优先使用解码器 PTS，异常时退化为 frame/fps。"""

    def __init__(self, fps: float) -> None:
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps 必须为正数")
        self.fps = fps
        self.last_timestamp = -1.0

    def resolve(self, frame_index: int, position_msec: float) -> float:
        fallback = frame_index / self.fps
        candidate = position_msec / 1000.0 if math.isfinite(position_msec) else fallback
        if candidate < 0 or (frame_index > 0 and candidate <= self.last_timestamp):
            candidate = fallback
        if self.last_timestamp >= 0 and candidate <= self.last_timestamp:
            candidate = self.last_timestamp + (1.0 / self.fps)
        self.last_timestamp = candidate
        return candidate


class SamplingSchedule:
    """按绝对时间生成固定采样率的采样时刻。"""

    def __init__(self, fps: float) -> None:
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("采样 fps 必须为正数")
        self.step_sec = 1.0 / fps
        self._next_index = 0

    @property
    def next_sec(self) -> float:
        # 每次乘法，避免累加误差
        return self._next_index * self.step_sec

    def due(self, timestamp_sec: float, tolerance_sec: float = 1e-6) -> bool:
        return timestamp_sec + tolerance_sec >= self.next_sec

    def consume(self, timestamp_sec: float, tolerance_sec: float = 1e-6) -> float:
        scheduled = self.next_sec
        while self.next_sec <= timestamp_sec + tolerance_sec:
            self._next_index += 1
        return scheduled

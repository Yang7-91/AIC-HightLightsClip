"""简单耗时统计工具。"""

from __future__ import annotations

from time import perf_counter


class Timer:
    def __enter__(self) -> "Timer":
        self.started = perf_counter()
        self.elapsed_sec = 0.0
        return self

    def __exit__(self, *_: object) -> None:
        self.elapsed_sec = perf_counter() - self.started

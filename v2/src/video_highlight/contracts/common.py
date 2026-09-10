"""各阶段通用输入数据结构。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class VideoIndexEntry:
    video_id: str
    target_ratio_wh: tuple[int, int] | None = None
    video_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

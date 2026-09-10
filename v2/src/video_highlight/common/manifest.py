"""运行清单和阶段状态辅助函数。"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def success_record(video_id: str, **details: Any) -> dict[str, Any]:
    return {"video_id": video_id, "status": "success", "finished_at": utc_now_iso(), **details}


def failure_record(video_id: str, error: Exception) -> dict[str, Any]:
    return {
        "video_id": video_id,
        "status": "failed",
        "finished_at": utc_now_iso(),
        "error_type": type(error).__name__,
        "message": str(error),
    }

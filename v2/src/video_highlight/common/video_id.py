"""视频标识规范化。"""

from __future__ import annotations

import re

from .exceptions import ArtifactValidationError

_SAFE_VIDEO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def normalize_video_id(value: object) -> str:
    video_id = str(value).strip()
    if not video_id or not _SAFE_VIDEO_ID.fullmatch(video_id):
        raise ArtifactValidationError(f"非法 video_id: {value!r}")
    return video_id

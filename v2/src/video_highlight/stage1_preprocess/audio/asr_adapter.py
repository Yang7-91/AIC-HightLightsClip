"""Stage 1 的可替换 ASR 接口；当前默认禁用，不伪造转写。"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class ASRAdapter(Protocol):
    def transcribe(self, wav_path: str | Path) -> list[dict[str, object]]:
        """返回带 start_sec/end_sec/text 的转写片段。"""


class DisabledASRAdapter:
    def transcribe(self, wav_path: str | Path) -> list[dict[str, object]]:
        del wav_path
        return []


def build_asr_adapter(enabled: bool) -> ASRAdapter:
    if enabled:
        raise RuntimeError("尚未配置 ASR 后端；请保持 asr_enabled=false 或在此接入本地模型")
    return DisabledASRAdapter()

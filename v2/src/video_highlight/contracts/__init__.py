"""阶段间共享的版本化数据契约。"""

from .common import VideoIndexEntry
from .schema_versions import STAGE1_SCHEMA_VERSION
from .stage1 import AnalysisSegment, AudioEvent, SampledFrame, Scene

__all__ = [
    "AnalysisSegment",
    "AudioEvent",
    "SampledFrame",
    "Scene",
    "STAGE1_SCHEMA_VERSION",
    "VideoIndexEntry",
]

"""阶段间共享的版本化数据契约。"""

from .common import VideoIndexEntry
from .schema_versions import STAGE1_SCHEMA_VERSION, STAGE2_SCHEMA_VERSION
from .stage1 import AnalysisSegment, AudioEvent, SampledFrame, Scene
from .stage2 import HighlightCandidate, SegmentAnalysis, SubjectHint

__all__ = [
    "AnalysisSegment",
    "AudioEvent",
    "HighlightCandidate",
    "SampledFrame",
    "Scene",
    "SegmentAnalysis",
    "STAGE1_SCHEMA_VERSION",
    "STAGE2_SCHEMA_VERSION",
    "SubjectHint",
    "VideoIndexEntry",
]

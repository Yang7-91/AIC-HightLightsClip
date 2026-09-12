"""阶段间共享的版本化数据契约。"""

from .common import VideoIndexEntry
from .schema_versions import (
    STAGE1_SCHEMA_VERSION,
    STAGE2_SCHEMA_VERSION,
    STAGE3_SCHEMA_VERSION,
    STAGE4_SCHEMA_VERSION,
    STAGE5_SCHEMA_VERSION,
)
from .stage1 import AnalysisSegment, AudioEvent, SampledFrame, Scene
from .stage2 import HighlightCandidate, SegmentAnalysis, SubjectHint
from .submission import FinalPrediction, SubmissionRow

__all__ = [
    "AnalysisSegment",
    "AudioEvent",
    "HighlightCandidate",
    "FinalPrediction",
    "SampledFrame",
    "Scene",
    "SegmentAnalysis",
    "STAGE1_SCHEMA_VERSION",
    "STAGE2_SCHEMA_VERSION",
    "STAGE3_SCHEMA_VERSION",
    "STAGE4_SCHEMA_VERSION",
    "STAGE5_SCHEMA_VERSION",
    "SubjectHint",
    "SubmissionRow",
    "VideoIndexEntry",
]

"""项目自定义异常。"""


class VideoHighlightError(RuntimeError):
    """项目内可预期错误的基类。"""


class ConfigurationError(VideoHighlightError):
    """配置缺失或格式非法。"""


class ExternalToolError(VideoHighlightError):
    """FFmpeg、FFprobe 等外部工具执行失败。"""


class ArtifactValidationError(VideoHighlightError):
    """阶段输入或输出不满足数据契约。"""

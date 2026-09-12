"""按 Stage 3 最终帧边界即时解码原视频并在内存中编码 JPEG。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from video_highlight.common.exceptions import ArtifactValidationError


@dataclass(frozen=True, slots=True)
class SampledFrame:
    """一个将要发送给多模态模型的采样帧；JPEG 字节不会写入阶段产物。"""

    sample_index: int
    frame: int
    timestamp_sec: float
    jpeg_bytes: bytes
    width: int
    height: int


def plan_sample_frames(start_frame: int, end_frame: int, source_fps: float, sample_fps: float) -> list[int]:
    """生成锚定区间首帧的确定性采样计划，输入区间为左闭右开。"""

    if not (0 <= start_frame < end_frame):
        raise ArtifactValidationError(f"非法采样区间: [{start_frame},{end_frame})")
    if not (math.isfinite(source_fps) and source_fps > 0):
        raise ArtifactValidationError(f"非法源 FPS: {source_fps}")
    if not (math.isfinite(sample_fps) and sample_fps > 0):
        raise ArtifactValidationError(f"非法采样 FPS: {sample_fps}")
    # sample_fps 高于源 FPS 时，同一原始帧可能被多次命中；去重后等价于逐帧取样。
    step = source_fps / sample_fps
    frames: list[int] = []
    index = 0
    while True:
        frame = start_frame + int(round(index * step))
        if frame >= end_frame:
            break
        if not frames or frame != frames[-1]:
            frames.append(frame)
        index += 1
    return frames


def _resize(frame: Any, max_side: int) -> Any:
    height, width = frame.shape[:2]
    if max_side <= 0 or max(height, width) <= max_side:
        return frame
    scale = max_side / max(height, width)
    return cv2.resize(
        frame,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def sample_interval(
    source_path: str | Path,
    frame_numbers: list[int],
    source_fps: float,
    jpeg_quality: int = 85,
    max_side: int = 1024,
) -> list[SampledFrame]:
    """从源视频精确读取计划帧并即时 JPEG 编码；函数返回后立即关闭解码器。"""

    if not frame_numbers:
        return []
    if frame_numbers != sorted(set(frame_numbers)):
        raise ArtifactValidationError("frame_numbers 必须严格递增且无重复")
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise ArtifactValidationError(f"无法打开源视频: {source_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_numbers[0])
    targets = set(frame_numbers)
    last = frame_numbers[-1]
    current = frame_numbers[0]
    sampled: list[SampledFrame] = []
    try:
        while current <= last:
            ok, frame = capture.read()
            if not ok:
                raise ArtifactValidationError(f"视频在原始帧 {current} 前意外结束: {source_path}")
            if current in targets:
                encoded_frame = _resize(frame, max_side)
                height, width = encoded_frame.shape[:2]
                ok, encoded = cv2.imencode(
                    ".jpg", encoded_frame, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
                )
                if not ok:
                    raise ArtifactValidationError(f"JPEG 编码失败: frame={current}")
                sampled.append(SampledFrame(
                    sample_index=len(sampled),
                    frame=current,
                    timestamp_sec=current / source_fps,
                    jpeg_bytes=encoded.tobytes(),
                    width=width,
                    height=height,
                ))
            current += 1
    finally:
        capture.release()
    if [row.frame for row in sampled] != frame_numbers:
        raise ArtifactValidationError("实际解码帧与采样计划不一致")
    return sampled

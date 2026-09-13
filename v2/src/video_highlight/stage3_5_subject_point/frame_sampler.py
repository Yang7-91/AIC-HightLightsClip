"""按 Stage 3 最终帧边界即时解码原视频并在内存中编码 JPEG。"""

from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

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
    # 实际完成解码的后端；用于诊断 auto 是否因色彩兼容问题回退到 FFmpeg。
    decoder_backend: str = "opencv"


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


def _encode_sample(
    frame: Any,
    sample_index: int,
    frame_number: int,
    source_fps: float,
    jpeg_quality: int,
    max_side: int,
    decoder_backend: str,
) -> SampledFrame:
    """统一缩放和 JPEG 编码，保证两个解码后端产生完全相同的数据契约。"""

    encoded_frame = _resize(frame, max_side)
    height, width = encoded_frame.shape[:2]
    ok, encoded = cv2.imencode(
        ".jpg", encoded_frame, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
    )
    if not ok:
        raise ArtifactValidationError(f"JPEG 编码失败: frame={frame_number}")
    return SampledFrame(
        sample_index=sample_index,
        frame=frame_number,
        timestamp_sec=frame_number / source_fps,
        jpeg_bytes=encoded.tobytes(),
        width=width,
        height=height,
        decoder_backend=decoder_backend,
    )


def _sample_with_opencv(
    source_path: str | Path,
    frame_numbers: list[int],
    source_fps: float,
    jpeg_quality: int,
    max_side: int,
) -> list[SampledFrame]:
    """现有 OpenCV 快路径。部分新版 FFmpeg 构建会拒绝异常 Log316 色彩标记。"""

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
                sampled.append(_encode_sample(
                    frame, len(sampled), current, source_fps,
                    jpeg_quality, max_side, "opencv",
                ))
            current += 1
    finally:
        capture.release()
    if [row.frame for row in sampled] != frame_numbers:
        raise ArtifactValidationError("实际解码帧与采样计划不一致")
    return sampled


def _split_mjpeg_stream(payload: bytes) -> list[bytes]:
    """拆分 FFmpeg image2pipe 输出的连续 JPEG；不创建任何临时帧文件。"""

    images: list[bytes] = []
    cursor = 0
    while True:
        start = payload.find(b"\xff\xd8", cursor)
        if start < 0:
            break
        end = payload.find(b"\xff\xd9", start + 2)
        if end < 0:
            raise ArtifactValidationError("FFmpeg MJPEG 输出包含不完整 JPEG")
        images.append(payload[start:end + 2])
        cursor = end + 2
    return images


def _sample_with_ffmpeg(
    source_path: str | Path,
    frame_numbers: list[int],
    source_fps: float,
    jpeg_quality: int,
    max_side: int,
    ffmpeg_bin: str,
) -> list[SampledFrame]:
    """兼容解码回退：显式覆盖不完整/不支持的输入色彩描述后在内存中取帧。

    视频 97 被标记为 ``trc=log316``，但 ``colorspace`` 与 ``primaries`` 均缺失。
    新版 swscale 因而拒绝直接转 BGR。这里先保留原 YUV 解码结果，再显式指定
    BT.709 矩阵和传递特性完成缩放，最后经 image2pipe 返回 JPEG 字节流。
    """

    first, last = frame_numbers[0], frame_numbers[-1]
    offsets = [frame - first for frame in frame_numbers]
    # 反斜杠用于转义 FFmpeg filter graph 中作为分隔符的逗号。
    select_expression = "+".join(f"eq(n\\,{offset})" for offset in offsets)
    video_filter = (
        f"trim=start_frame={first}:end_frame={last + 1},"
        "setpts=PTS-STARTPTS,"
        f"select='{select_expression}',"
        "setparams=range=limited:color_primaries=bt709:color_trc=bt709:colorspace=bt709,"
        "scale=in_color_matrix=bt709:out_color_matrix=bt709,format=yuvj420p"
    )
    command = [
        ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-i", str(source_path),
        "-an", "-vf", video_filter, "-fps_mode", "passthrough",
        "-c:v", "mjpeg", "-q:v", "2", "-f", "image2pipe", "pipe:1",
    ]
    try:
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except OSError as error:
        raise ArtifactValidationError(f"无法启动 FFmpeg 兼容解码器 {ffmpeg_bin!r}: {error}") from error
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ArtifactValidationError(f"FFmpeg 兼容解码失败: {message[-2000:]}")
    encoded_frames = _split_mjpeg_stream(completed.stdout)
    if len(encoded_frames) != len(frame_numbers):
        raise ArtifactValidationError(
            f"FFmpeg 兼容解码帧数不一致: expected={len(frame_numbers)}, actual={len(encoded_frames)}"
        )
    sampled: list[SampledFrame] = []
    for sample_index, (frame_number, encoded) in enumerate(zip(frame_numbers, encoded_frames, strict=True)):
        frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ArtifactValidationError(f"FFmpeg JPEG 回读失败: frame={frame_number}")
        sampled.append(_encode_sample(
            frame, sample_index, frame_number, source_fps,
            jpeg_quality, max_side, "ffmpeg_bt709_fallback",
        ))
    return sampled


def sample_interval(
    source_path: str | Path,
    frame_numbers: list[int],
    source_fps: float,
    jpeg_quality: int = 85,
    max_side: int = 1024,
    decoder: str = "auto",
    ffmpeg_bin: str = "ffmpeg",
) -> list[SampledFrame]:
    """即时采样区间帧；auto 先用 OpenCV，失败后回退到色彩兼容 FFmpeg。"""

    if not frame_numbers:
        return []
    if frame_numbers != sorted(set(frame_numbers)):
        raise ArtifactValidationError("frame_numbers 必须严格递增且无重复")
    backend = str(decoder).lower()
    if backend not in {"auto", "opencv", "ffmpeg"}:
        raise ArtifactValidationError(f"未知采样解码器: {decoder}")
    if backend == "ffmpeg":
        return _sample_with_ffmpeg(
            source_path, frame_numbers, source_fps, jpeg_quality, max_side, ffmpeg_bin
        )
    try:
        return _sample_with_opencv(
            source_path, frame_numbers, source_fps, jpeg_quality, max_side
        )
    except ArtifactValidationError as opencv_error:
        if backend == "opencv":
            raise
        try:
            return _sample_with_ffmpeg(
                source_path, frame_numbers, source_fps, jpeg_quality, max_side, ffmpeg_bin
            )
        except ArtifactValidationError as ffmpeg_error:
            raise ArtifactValidationError(
                f"OpenCV 解码失败（{opencv_error}）；FFmpeg BT.709 回退也失败（{ffmpeg_error}）"
            ) from ffmpeg_error

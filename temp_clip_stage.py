#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""根据 temp_complete_stage.py 的 predictions.jsonl 生成指定视频的成片。

处理流程：

1. 用 ``--video-id`` 从 JSONL 中选择一个视频；
2. 把预测帧按连续帧号分成高光片段；
3. 顺序解码每个片段，对每帧应用其 ``[x, y, w]`` 动态裁剪框；
4. 把所有裁剪帧连续编码为 H.264 视频；
5. 按完全相同的帧区间裁切并拼接原音频，再封装为最终 MP4。

默认输入是同目录的 ``temp_complete_predictions.jsonl``，默认输出到
``temp_clips/<video_id>.mp4``。

示例：
    python temp_clip_stage.py --video-id 0 \
      --video-dir /root/autodl-tmp/video-clip-data/video \
      --overwrite
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PREDICTIONS = SCRIPT_DIR / "temp_complete_predictions.jsonl"
DEFAULT_VIDEO_DIR = Path("F:/datasets/video-clip/video")
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "temp_clips"


class ClipError(RuntimeError):
    """输入、视频解码或外部工具不满足成片生成要求。"""


@dataclass(frozen=True, slots=True)
class VideoMeta:
    frame_count: int
    fps: float
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class FrameCrop:
    frame: int
    x: int
    y: int
    width: int


@dataclass(frozen=True, slots=True)
class FrameSegment:
    start_frame: int
    end_frame: int
    crops: tuple[FrameCrop, ...]

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame + 1


def read_prediction_record(path: Path, video_id: str) -> dict[str, Any]:
    """从每行一个视频的 JSONL 中精确读取指定 video_id。"""

    matches: list[dict[str, Any]] = []
    try:
        handle = path.open("r", encoding="utf-8-sig")
    except FileNotFoundError as exc:
        raise ClipError(f"预测文件不存在: {path}") from exc
    with handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ClipError(f"JSONL 解析失败: {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ClipError(f"JSONL 每行必须是对象: {path}:{line_number}")
            if str(value.get("video_id")) == video_id:
                matches.append(value)
    if not matches:
        raise ClipError(f"预测文件中找不到 video_id={video_id}")
    if len(matches) > 1:
        raise ClipError(f"预测文件中 video_id={video_id} 出现了 {len(matches)} 次")
    return matches[0]


def parse_ratio(record: dict[str, Any]) -> tuple[float, float]:
    value = record.get("targetRatioWH")
    if not isinstance(value, list) or len(value) != 2:
        raise ClipError("targetRatioWH 必须是 [width, height]")
    try:
        width, height = float(value[0]), float(value[1])
    except (TypeError, ValueError) as exc:
        raise ClipError("targetRatioWH 必须包含两个数值") from exc
    if not all(math.isfinite(item) and item > 0 for item in (width, height)):
        raise ClipError("targetRatioWH 必须包含两个有限正数")
    return width, height


def parse_crops(record: dict[str, Any]) -> list[FrameCrop]:
    """校验预测、按帧排序，并拒绝重复帧。"""

    values = record.get("predictions")
    if not isinstance(values, list):
        raise ClipError("predictions 必须是数组")
    crops: list[FrameCrop] = []
    seen_frames: set[int] = set()
    for position, value in enumerate(values, start=1):
        if not isinstance(value, dict):
            raise ClipError(f"第 {position} 条 prediction 必须是对象")
        bbox = value.get("bboxes")
        if not isinstance(bbox, list) or len(bbox) != 3:
            raise ClipError(f"第 {position} 条 bboxes 必须是 [x, y, w]")
        try:
            frame = int(value["frame"])
            x, y, width = (int(item) for item in bbox)
        except (KeyError, TypeError, ValueError) as exc:
            raise ClipError(f"第 {position} 条 prediction 含无效整数") from exc
        if frame < 0 or min(x, y) < 0 or width <= 0:
            raise ClipError(f"第 {position} 条 prediction 含负坐标或非正宽度")
        if frame in seen_frames:
            raise ClipError(f"predictions 存在重复帧: {frame}")
        seen_frames.add(frame)
        crops.append(FrameCrop(frame, x, y, width))
    crops.sort(key=lambda item: item.frame)
    if not crops:
        raise ClipError("指定视频没有高光预测帧，无法生成成片")
    return crops


def group_segments(crops: Iterable[FrameCrop]) -> list[FrameSegment]:
    """把严格连续的预测帧分成独立高光片段。"""

    ordered = list(crops)
    segments: list[FrameSegment] = []
    current: list[FrameCrop] = []
    for crop in ordered:
        if current and crop.frame != current[-1].frame + 1:
            segments.append(
                FrameSegment(current[0].frame, current[-1].frame, tuple(current))
            )
            current = []
        current.append(crop)
    if current:
        segments.append(
            FrameSegment(current[0].frame, current[-1].frame, tuple(current))
        )
    return segments


def read_video_meta(video_path: Path) -> VideoMeta:
    try:
        import cv2
    except ImportError as exc:
        raise ClipError("缺少 opencv-python，请先在 video-clip 环境中安装") from exc
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ClipError(f"无法打开源视频: {video_path}")
        meta = VideoMeta(
            frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            fps=float(capture.get(cv2.CAP_PROP_FPS)),
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        capture.release()
    if (
        meta.frame_count <= 0
        or not math.isfinite(meta.fps)
        or meta.fps <= 0
        or min(meta.width, meta.height) <= 0
    ):
        raise ClipError(
            f"源视频元数据无效: frames={meta.frame_count}, fps={meta.fps}, "
            f"size={meta.width}x{meta.height}"
        )
    return meta


def derived_crop_height(width: int, ratio: tuple[float, float]) -> int:
    """根据比赛的 h=w*target_h/target_w 规则得到实际像素占用高度。"""

    target_width, target_height = ratio
    return max(1, int(math.ceil(width * target_height / target_width - 1e-9)))


def validate_crops(
    crops: Iterable[FrameCrop],
    ratio: tuple[float, float],
    meta: VideoMeta,
) -> None:
    for crop in crops:
        height = derived_crop_height(crop.width, ratio)
        if crop.frame >= meta.frame_count:
            raise ClipError(
                f"预测帧 {crop.frame} 超出源视频末帧 {meta.frame_count - 1}"
            )
        if crop.x + crop.width > meta.width or crop.y + height > meta.height:
            raise ClipError(
                f"frame={crop.frame} 裁剪框越界: "
                f"[{crop.x},{crop.y},{crop.width},{height}] / "
                f"{meta.width}x{meta.height}"
            )


def choose_output_size(
    first_crop: FrameCrop,
    ratio: tuple[float, float],
    requested_width: int,
) -> tuple[int, int]:
    """选择接近目标比例的偶数尺寸，兼容常见 yuv420p/H.264 播放器。"""

    width = requested_width or first_crop.width
    if width < 2:
        raise ClipError("输出宽度必须至少为 2")
    width -= width % 2
    ideal_height = width * ratio[1] / ratio[0]
    height = max(2, int(round(ideal_height / 2.0)) * 2)
    return width, height


def check_executable(executable: str, version_flag: str) -> None:
    try:
        completed = subprocess.run(
            [executable, version_flag],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except FileNotFoundError as exc:
        raise ClipError(f"找不到外部工具: {executable}") from exc
    if completed.returncode != 0:
        raise ClipError(f"外部工具不可用: {executable}")


def encode_dynamic_crops(
    video_path: Path,
    segments: list[FrameSegment],
    ratio: tuple[float, float],
    meta: VideoMeta,
    output_path: Path,
    output_size: tuple[int, int],
    args: argparse.Namespace,
) -> int:
    """将动态裁剪后的 BGR 帧通过管道交给 FFmpeg 编码。"""

    try:
        import cv2
    except ImportError as exc:
        raise ClipError("缺少 opencv-python，请先在 video-clip 环境中安装") from exc
    output_width, output_height = output_size
    command = [
        args.ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s:v",
        f"{output_width}x{output_height}",
        "-r",
        f"{meta.fps:.12g}",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        args.video_codec,
        "-preset",
        args.preset,
        "-crf",
        str(args.crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    written_frames = 0
    try:
        assert process.stdin is not None
        for segment in segments:
            capture = cv2.VideoCapture(str(video_path))
            try:
                if not capture.isOpened():
                    raise ClipError(f"无法打开源视频: {video_path}")
                capture.set(cv2.CAP_PROP_POS_FRAMES, segment.start_frame)
                for crop in segment.crops:
                    ok, frame = capture.read()
                    if not ok:
                        raise ClipError(f"源视频在 frame={crop.frame} 解码失败")
                    crop_height = derived_crop_height(crop.width, ratio)
                    image = frame[
                        crop.y : crop.y + crop_height,
                        crop.x : crop.x + crop.width,
                    ]
                    if image.size == 0:
                        raise ClipError(f"frame={crop.frame} 得到空裁剪图像")
                    if image.shape[1] != output_width or image.shape[0] != output_height:
                        interpolation = (
                            cv2.INTER_AREA
                            if image.shape[1] >= output_width and image.shape[0] >= output_height
                            else cv2.INTER_LINEAR
                        )
                        image = cv2.resize(
                            image, (output_width, output_height), interpolation=interpolation
                        )
                    process.stdin.write(image.tobytes())
                    written_frames += 1
            finally:
                capture.release()
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        return_code = process.wait()
    except BaseException:
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
        process.terminate()
        process.wait()
        raise
    if return_code != 0 or not output_path.is_file():
        raise ClipError(f"动态裁剪视频编码失败: {stderr.strip()}")
    return written_frames


def source_has_audio(video_path: Path, ffprobe_bin: str) -> bool:
    completed = subprocess.run(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise ClipError(f"FFprobe 检查音频失败: {completed.stderr.strip()}")
    return bool(completed.stdout.strip())


def encode_concatenated_audio(
    video_path: Path,
    segments: list[FrameSegment],
    fps: float,
    audio_path: Path,
    args: argparse.Namespace,
) -> None:
    filters: list[str] = []
    labels: list[str] = []
    for index, segment in enumerate(segments):
        start_sec = segment.start_frame / fps
        end_sec = (segment.end_frame + 1) / fps
        label = f"a{index}"
        filters.append(
            f"[0:a:0]atrim=start={start_sec:.9f}:end={end_sec:.9f},"
            f"asetpts=PTS-STARTPTS[{label}]"
        )
        labels.append(f"[{label}]")
    filters.append("".join(labels) + f"concat=n={len(labels)}:v=0:a=1[aout]")
    command = [
        args.ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[aout]",
        "-c:a",
        args.audio_codec,
        "-b:a",
        args.audio_bitrate,
        str(audio_path),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0 or not audio_path.is_file():
        raise ClipError(f"高光音频裁切/拼接失败: {completed.stderr.strip()}")


def mux_video_audio(
    silent_video: Path,
    audio_path: Path,
    output_path: Path,
    ffmpeg_bin: str,
) -> None:
    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(silent_video),
        "-i",
        str(audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c",
        "copy",
        "-shortest",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0 or not output_path.is_file():
        raise ClipError(f"视频与音频封装失败: {completed.stderr.strip()}")


def resolve_output_path(args: argparse.Namespace) -> Path:
    if args.out is not None:
        return args.out.resolve()
    return (DEFAULT_OUTPUT_DIR / f"{args.video_id}.mp4").resolve()


def run(args: argparse.Namespace) -> Path:
    video_id = str(args.video_id)
    if args.output_width < 0:
        raise ClipError("output_width 不能小于 0")
    if not 0 <= args.crf <= 51:
        raise ClipError("crf 必须在 0~51")
    predictions_path = args.predictions.resolve()
    record = read_prediction_record(predictions_path, video_id)
    ratio = parse_ratio(record)
    crops = parse_crops(record)
    segments = group_segments(crops)

    extension = args.video_extension
    if extension and not extension.startswith("."):
        extension = f".{extension}"
    video_path = args.video_dir.resolve() / f"{video_id}{extension}"
    if not video_path.is_file():
        raise ClipError(f"源视频不存在: {video_path}")
    meta = read_video_meta(video_path)
    validate_crops(crops, ratio, meta)
    output_size = choose_output_size(crops[0], ratio, args.output_width)
    output_path = resolve_output_path(args)
    if output_path == video_path.resolve():
        raise ClipError("输出路径不能与源视频相同")
    if output_path.exists() and not args.overwrite:
        raise ClipError(f"输出已存在；如需覆盖请增加 --overwrite: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    check_executable(args.ffmpeg_bin, "-version")
    if not args.no_audio:
        check_executable(args.ffprobe_bin, "-version")

    with tempfile.TemporaryDirectory(
        prefix=f".{output_path.stem}.clip-", dir=output_path.parent
    ) as temp_name:
        temp_dir = Path(temp_name)
        silent_video = temp_dir / "cropped_video.mp4"
        staged_output = temp_dir / output_path.name
        written_frames = encode_dynamic_crops(
            video_path,
            segments,
            ratio,
            meta,
            silent_video,
            output_size,
            args,
        )
        if written_frames != len(crops):
            raise ClipError(
                f"编码帧数不一致: expected={len(crops)}, actual={written_frames}"
            )
        has_audio = False
        if not args.no_audio:
            has_audio = source_has_audio(video_path, args.ffprobe_bin)
        if has_audio:
            audio_path = temp_dir / "highlight_audio.m4a"
            encode_concatenated_audio(video_path, segments, meta.fps, audio_path, args)
            mux_video_audio(silent_video, audio_path, staged_output, args.ffmpeg_bin)
        else:
            os.replace(silent_video, staged_output)
        os.replace(staged_output, output_path)

    duration_sec = len(crops) / meta.fps
    print(
        f"完成: video_id={video_id}, segments={len(segments)}, frames={len(crops)}, "
        f"duration={duration_sec:.3f}s, size={output_size[0]}x{output_size[1]}, "
        f"audio={'yes' if has_audio else 'no'}"
    )
    print(f"输出成片 -> {output_path}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="根据 predictions.jsonl 的逐帧动态裁剪框生成指定视频成片",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video-id", required=True, help="要剪辑的 video_id")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_PREDICTIONS,
        help="temp_complete_stage.py 输出的 predictions.jsonl",
    )
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--video-extension", default=".mp4")
    parser.add_argument("--out", type=Path, default=None, help="输出 MP4 路径")
    parser.add_argument(
        "--output-width",
        type=int,
        default=0,
        help="统一输出宽度；0 表示按预测裁剪宽度自动选择偶数尺寸",
    )
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--ffprobe-bin", default="ffprobe")
    parser.add_argument("--video-codec", default="libx264")
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--audio-codec", default="aac")
    parser.add_argument("--audio-bitrate", default="192k")
    parser.add_argument("--no-audio", action="store_true", help="不保留源视频音频")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有输出")
    return parser


def main() -> int:
    try:
        run(build_parser().parse_args())
    except (ClipError, OSError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

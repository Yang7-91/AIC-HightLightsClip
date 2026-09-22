"""MHS-VIS-0 缩略图抽取与 contact sheet 合成（OpenCV，输出仅本地）。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import cv2


class ThumbnailError(RuntimeError):
    """视频无法读取或缩略图写入失败。"""


def extract_candidate_thumbnails(
    video_path: str | Path,
    start_sec: float,
    end_sec: float,
    output_dir: str | Path,
    *,
    fps: float = 1.0,
    max_frames: int = 40,
    prefix: str = "thumb",
) -> list[dict[str, Any]]:
    """按固定 fps 抽取候选区间缩略图（每秒最多 1 帧），返回时间戳与文件路径。"""
    source = Path(video_path)
    if not source.is_file():
        raise ThumbnailError(f"video not found: {source}")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG)
    if not capture.isOpened():
        raise ThumbnailError(f"OpenCV could not open video: {source}")
    fps = max(fps, 0.1)
    step = 1.0 / fps
    times: list[float] = []
    t = start_sec
    while t < end_sec - 1e-9 and len(times) < max_frames:
        times.append(round(t, 3))
        t += step
    thumbnails: list[dict[str, Any]] = []
    try:
        for index, t_sec in enumerate(times):
            capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_sec) * 1000.0)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            height, width = frame.shape[:2]
            target_width = 320
            scale = target_width / width if width > target_width else 1.0
            if scale < 1.0:
                frame = cv2.resize(
                    frame,
                    (int(width * scale), int(height * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            path = out_dir / f"{prefix}_{index:03d}_{t_sec:.1f}s.jpg"
            cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            thumbnails.append({"t_sec": t_sec, "path": str(path)})
    finally:
        capture.release()
    if not thumbnails:
        raise ThumbnailError(f"no decodable frames in [{start_sec}, {end_sec}] of {source}")
    return thumbnails


def make_contact_sheet(
    thumbnails: list[Mapping[str, Any]],
    output_png: str | Path,
    *,
    thumb_width: int = 180,
    subsegments: list[Mapping[str, float]] | None = None,
    columns: int = 6,
) -> str:
    """合成 contact sheet PNG，帧上叠加时间戳；落在子片段内的帧加彩色边框。"""
    if not thumbnails:
        raise ThumbnailError("no thumbnails to compose")
    loaded: list[tuple[float, Any]] = []
    for item in thumbnails:
        image = cv2.imread(str(item["path"]))
        if image is not None:
            loaded.append((float(item["t_sec"]), image))
    if not loaded:
        raise ThumbnailError("all thumbnails unreadable")
    ratio = thumb_width / loaded[0][1].shape[1]
    thumb_height = max(1, int(loaded[0][1].shape[0] * ratio))
    label_height = 18
    cell_width = thumb_width
    cell_height = thumb_height + label_height
    columns = max(1, columns)
    rows = math.ceil(len(loaded) / columns)
    import numpy as np

    sheet = np.full((rows * cell_height, columns * cell_width, 3), 20, dtype=np.uint8)
    subsegments = list(subsegments or [])
    for index, (t_sec, image) in enumerate(loaded):
        row, col = divmod(index, columns)
        resized = cv2.resize(image, (cell_width, thumb_height), interpolation=cv2.INTER_AREA)
        in_subsegment = any(
            float(segment["start_sec"]) - 1e-9 <= t_sec <= float(segment["end_sec"]) + 1e-9
            for segment in subsegments
        )
        if in_subsegment:
            color = (0, 200, 0)
            thickness = 3
        else:
            color = (90, 90, 90)
            thickness = 1
        cv2.rectangle(resized, (0, 0), (cell_width - 1, thumb_height - 1), color, thickness)
        cv2.putText(
            resized,
            f"{t_sec:.1f}s",
            (6, thumb_height - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )
        y0 = row * cell_height
        x0 = col * cell_width
        sheet[y0 : y0 + thumb_height, x0 : x0 + cell_width] = resized
        cv2.putText(
            sheet,
            f"{t_sec:.1f}s",
            (x0 + 6, y0 + cell_height - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_png), sheet):
        raise ThumbnailError(f"cannot write contact sheet: {output_png}")
    return str(output_png)

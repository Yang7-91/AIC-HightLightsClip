#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 2 之后的快速闭环验证脚本。

本脚本参考 ``baseline_qwen.py`` 的确定性后处理方式，但跳过边界精定位、
主体定位、跟踪和平滑，只完成以下流程：

1. 读取 Stage 2 每个视频的 ``candidates.jsonl``；
2. 合并重叠、相邻或间隔不超过阈值的粗高光区间；
3. 把秒级区间转换为原视频帧区间；
4. 计算符合 ``targetRatioWH`` 的中心最大内接裁剪框；
5. 为高光区间内每一帧输出比赛要求的 ``[x, y, w]``；
6. 生成最终 ``predictions.jsonl`` 和便于人工检查的区间诊断 JSONL。

这里的“裁剪”指生成比赛提交所需的逐帧裁剪框，不会重新编码或导出 MP4。
该版本主要用于快速验证 Stage 2 高光召回结果能否走通完整提交链路。

示例：
    python temp_complete_stage.py `
      --stage2-dir v2/runs/somke-stage2/stage2 `
      --video-id 0

处理索引中的全部视频：
    python temp_complete_stage.py `
      --stage2-dir v2/runs/my-stage2/stage2 `
      --out temp_complete_predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, TextIO


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INDEX = SCRIPT_DIR / "test_index.json"
DEFAULT_VIDEO_DIR = Path("F:/datasets/video-clip/video")
DEFAULT_OUTPUT = SCRIPT_DIR / "temp_complete_predictions.jsonl"


class ValidationError(RuntimeError):
    """输入数据不满足快速闭环所需契约。"""


@dataclass(frozen=True, slots=True)
class VideoMeta:
    """生成逐帧裁剪结果所需的最小视频元数据。"""

    frame_count: int
    fps: float
    width: int
    height: int

    @property
    def duration_sec(self) -> float:
        return self.frame_count / self.fps


@dataclass(frozen=True, slots=True)
class Interval:
    """由一个或多个 Stage 2 候选合并得到的秒级高光区间。"""

    start_sec: float
    end_sec: float
    candidate_ids: tuple[str, ...]
    max_score: float


def load_index(path: Path) -> list[tuple[str, tuple[float, float]]]:
    """读取测试索引并返回 ``[(video_id, (target_w, target_h)), ...]``。"""

    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise ValidationError(f"测试索引不存在: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValidationError(f"测试索引不是合法 JSON: {path}: {exc}") from exc
    if not isinstance(value, list):
        raise ValidationError(f"测试索引根节点必须是数组: {path}")

    result: list[tuple[str, tuple[float, float]]] = []
    seen: set[str] = set()
    for position, item in enumerate(value, start=1):
        if not isinstance(item, dict) or "video_id" not in item:
            raise ValidationError(f"测试索引第 {position} 项缺少 video_id")
        video_id = str(item["video_id"])
        if video_id in seen:
            raise ValidationError(f"测试索引存在重复 video_id: {video_id}")
        ratio = item.get("targetRatioWH", [16, 9])
        if not isinstance(ratio, list) or len(ratio) != 2:
            raise ValidationError(f"{video_id}.targetRatioWH 必须是 [w, h]")
        try:
            target_w, target_h = float(ratio[0]), float(ratio[1])
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"{video_id}.targetRatioWH 必须是数值") from exc
        if not all(math.isfinite(v) and v > 0 for v in (target_w, target_h)):
            raise ValidationError(f"{video_id}.targetRatioWH 必须是有限正数")
        seen.add(video_id)
        result.append((video_id, (target_w, target_h)))
    return result


def resolve_stage2_videos_dir(stage2_dir: Path) -> Path:
    """兼容传入 ``stage2`` 根目录或直接传入其 ``videos`` 子目录。"""

    root = stage2_dir.resolve()
    if root.name == "videos" and root.is_dir():
        return root
    videos_dir = root / "videos"
    if videos_dir.is_dir():
        return videos_dir
    raise ValidationError(
        f"找不到 Stage 2 videos 目录: {root}；请传入 stage2 或 stage2/videos"
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL；异常信息包含具体文件和行号，便于定位坏产物。"""

    records: list[dict[str, Any]] = []
    try:
        handle = path.open("r", encoding="utf-8-sig")
    except FileNotFoundError as exc:
        raise ValidationError(f"Stage 2 候选文件不存在: {path}") from exc
    with handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"JSONL 解析失败: {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValidationError(f"JSONL 每行必须是对象: {path}:{line_number}")
            records.append(value)
    return records


def read_video_meta(video_path: Path) -> VideoMeta:
    """参考 baseline_qwen，使用 OpenCV 仅读取视频头部元信息。"""

    try:
        import cv2
    except ImportError as exc:
        raise ValidationError("缺少 opencv-python，请先在 video-clip 环境中安装") from exc

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValidationError(f"无法打开视频: {video_path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()

    # FPS=0 通常表示容器信息不可用。这里不静默使用 25 FPS，避免提交帧号错位。
    if frame_count <= 0 or not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
        raise ValidationError(
            f"视频元数据无效: {video_path} "
            f"frames={frame_count}, fps={fps}, size={width}x{height}"
        )
    return VideoMeta(frame_count=frame_count, fps=fps, width=width, height=height)


def candidate_intervals(
    video_id: str,
    candidates: Iterable[dict[str, Any]],
    duration_sec: float,
    min_score: float,
) -> list[Interval]:
    """校验、过滤并裁定 Stage 2 候选的合法时间范围。"""

    intervals: list[Interval] = []
    for index, row in enumerate(candidates):
        row_video_id = str(row.get("video_id", video_id))
        if row_video_id != video_id:
            raise ValidationError(
                f"{video_id} 的候选文件混入了其他视频: {row_video_id}"
            )
        try:
            start_sec = float(row["start_sec"])
            end_sec = float(row["end_sec"])
            score = float(row.get("coarse_score", 0.0))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError(f"{video_id} 第 {index + 1} 个候选字段无效") from exc
        if not all(math.isfinite(v) for v in (start_sec, end_sec, score)):
            raise ValidationError(f"{video_id} 第 {index + 1} 个候选包含非有限数值")
        if score < min_score:
            continue

        # 防御性处理模型或旧产物中颠倒的端点，并限制在真实视频时长内。
        start_sec, end_sec = sorted((start_sec, end_sec))
        start_sec = max(0.0, min(start_sec, duration_sec))
        end_sec = max(0.0, min(end_sec, duration_sec))
        if end_sec <= start_sec:
            continue
        candidate_id = str(row.get("candidate_id", f"{video_id}_raw_{index:04d}"))
        intervals.append(
            Interval(start_sec, end_sec, (candidate_id,), score)
        )
    return intervals


def merge_intervals(intervals: Iterable[Interval], max_gap_sec: float) -> list[Interval]:
    """合并重叠区间，以及间隔不超过 ``max_gap_sec`` 的相邻区间。"""

    if not math.isfinite(max_gap_sec) or max_gap_sec < 0:
        raise ValidationError("max_gap_sec 必须是有限非负数")
    ordered = sorted(intervals, key=lambda item: (item.start_sec, item.end_sec))
    merged: list[Interval] = []
    for current in ordered:
        if not merged or current.start_sec > merged[-1].end_sec + max_gap_sec:
            merged.append(current)
            continue
        previous = merged[-1]
        merged[-1] = Interval(
            start_sec=previous.start_sec,
            end_sec=max(previous.end_sec, current.end_sec),
            candidate_ids=tuple(dict.fromkeys(previous.candidate_ids + current.candidate_ids)),
            max_score=max(previous.max_score, current.max_score),
        )
    return merged


def seconds_to_frame_ranges(
    intervals: Iterable[Interval], fps: float, frame_count: int
) -> list[tuple[int, int]]:
    """按 baseline_qwen 的 round 规则将秒区间转换为闭区间帧号。"""

    ranges: list[tuple[int, int]] = []
    for interval in intervals:
        start_frame = int(round(interval.start_sec * fps))
        end_frame = int(round(interval.end_sec * fps))
        start_frame = max(0, min(start_frame, frame_count - 1))
        end_frame = max(0, min(end_frame, frame_count - 1))
        if end_frame >= start_frame:
            ranges.append((start_frame, end_frame))

    # 秒域合并后再做一次帧域相邻合并，消除浮点换算和取整带来的重复帧。
    merged: list[list[int]] = []
    for start_frame, end_frame in sorted(ranges):
        if merged and start_frame <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end_frame)
        else:
            merged.append([start_frame, end_frame])
    return [(start, end) for start, end in merged]


def compute_crop_size(
    source_width: int,
    source_height: int,
    target_width: float,
    target_height: float,
) -> tuple[int, int]:
    """计算完整落在源画面内、且符合目标比例的最大裁剪尺寸。"""

    if min(source_width, source_height) <= 0 or min(target_width, target_height) <= 0:
        raise ValidationError("源尺寸和目标画幅必须为正数")
    target_ratio = target_width / target_height
    source_ratio = source_width / source_height
    if source_ratio >= target_ratio:
        crop_height = source_height
        crop_width = min(int(round(crop_height * target_ratio)), source_width)
    else:
        crop_width = source_width
        crop_height = min(int(round(crop_width / target_ratio)), source_height)
    return max(1, crop_width), max(1, crop_height)


def centered_crop_box(meta: VideoMeta, target_ratio: tuple[float, float]) -> tuple[int, int, int, int]:
    """以画面中心 ``(0.5, 0.5)`` 放置最大内接裁剪框，并限制到边界内。"""

    crop_width, crop_height = compute_crop_size(
        meta.width, meta.height, target_ratio[0], target_ratio[1]
    )
    x = int(round((meta.width - crop_width) / 2.0))
    y = int(round((meta.height - crop_height) / 2.0))
    x = max(0, min(x, meta.width - crop_width))
    y = max(0, min(y, meta.height - crop_height))
    return x, y, crop_width, crop_height


def build_predictions(
    frame_ranges: Iterable[tuple[int, int]], crop_box: tuple[int, int, int, int]
) -> list[dict[str, Any]]:
    """为所有高光帧生成比赛格式记录；高度不写入 bboxes。"""

    x, y, width, _height = crop_box
    return [
        {"frame": frame, "bboxes": [x, y, width]}
        for start_frame, end_frame in frame_ranges
        for frame in range(start_frame, end_frame + 1)
    ]


def _open_atomic_target(path: Path) -> tuple[TextIO, Path]:
    """在目标目录创建临时文件，全部视频成功后再原子替换正式输出。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    return handle, Path(handle.name)


def _warn(message: str) -> None:
    print(f"[warning] {message}", file=sys.stderr)


def _target_ratio_for_output(target_ratio: tuple[float, float]) -> list[int | float]:
    """整数比例保持整数形式，非整数比例则保留浮点值。"""

    result: list[int | float] = []
    for value in target_ratio:
        result.append(int(value) if value.is_integer() else value)
    return result


def run(args: argparse.Namespace) -> tuple[int, int]:
    """执行快速闭环，返回 ``(视频数, 预测帧数)``。"""

    if not math.isfinite(args.min_score):
        raise ValidationError("min_score 必须是有限数值")
    if args.num_videos < 0:
        raise ValidationError("num_videos 不能小于 0")

    stage2_videos_dir = resolve_stage2_videos_dir(args.stage2_dir)
    index = load_index(args.index)
    if args.video_id:
        selected_ids = set(args.video_id)
        known_ids = {video_id for video_id, _ in index}
        unknown_ids = selected_ids - known_ids
        if unknown_ids:
            raise ValidationError(f"--video-id 不在测试索引中: {sorted(unknown_ids)}")
        index = [item for item in index if item[0] in selected_ids]
    if args.available_only:
        index = [
            item
            for item in index
            if (stage2_videos_dir / item[0] / "_SUCCESS.json").is_file()
        ]
    if args.num_videos > 0:
        index = index[: args.num_videos]
    if not index:
        raise ValidationError("没有需要处理的视频")

    output_path = args.out.resolve()
    intervals_path = (
        args.intervals_out.resolve()
        if args.intervals_out is not None
        else output_path.with_name(f"{output_path.stem}.intervals.jsonl")
    )
    if intervals_path == output_path:
        raise ValidationError("--intervals-out 不能与 --out 指向同一个文件")

    prediction_handle, prediction_tmp = _open_atomic_target(output_path)
    interval_handle, interval_tmp = _open_atomic_target(intervals_path)
    total_predictions = 0

    try:
        with prediction_handle, interval_handle:
            for position, (video_id, target_ratio) in enumerate(index, start=1):
                extension = args.video_extension
                if extension and not extension.startswith("."):
                    extension = f".{extension}"
                video_path = args.video_dir.resolve() / f"{video_id}{extension}"
                candidate_path = stage2_videos_dir / video_id / "candidates.jsonl"
                output_ratio = _target_ratio_for_output(target_ratio)
                predictions: list[dict[str, Any]] = []
                diagnostic: dict[str, Any] = {
                    "video_id": video_id,
                    "targetRatioWH": output_ratio,
                    "source_candidate_count": 0,
                    "merged_intervals": [],
                    "frame_ranges": [],
                    "crop_box_xywh": None,
                    "video_meta": None,
                }

                try:
                    success_path = candidate_path.parent / "_SUCCESS.json"
                    if not success_path.is_file():
                        raise ValidationError(f"Stage 2 没有成功标记: {success_path}")
                    candidates = read_jsonl(candidate_path)
                    diagnostic["source_candidate_count"] = len(candidates)
                    if candidates:
                        if not video_path.is_file():
                            raise ValidationError(f"视频文件不存在: {video_path}")
                        meta = read_video_meta(video_path)
                        raw_intervals = candidate_intervals(
                            video_id,
                            candidates,
                            meta.duration_sec,
                            args.min_score,
                        )
                        merged_intervals = merge_intervals(raw_intervals, args.merge_gap_sec)
                        frame_ranges = seconds_to_frame_ranges(
                            merged_intervals, meta.fps, meta.frame_count
                        )
                        crop_box = centered_crop_box(meta, target_ratio)
                        predictions = build_predictions(frame_ranges, crop_box)
                        diagnostic.update(
                            {
                                "merged_intervals": [
                                    {
                                        "start_sec": item.start_sec,
                                        "end_sec": item.end_sec,
                                        "candidate_ids": list(item.candidate_ids),
                                        "max_score": item.max_score,
                                    }
                                    for item in merged_intervals
                                ],
                                "frame_ranges": [list(item) for item in frame_ranges],
                                "crop_box_xywh": list(crop_box),
                                "video_meta": {
                                    "frame_count": meta.frame_count,
                                    "fps": meta.fps,
                                    "width": meta.width,
                                    "height": meta.height,
                                    "duration_sec": meta.duration_sec,
                                },
                            }
                        )
                except ValidationError as exc:
                    if args.strict:
                        raise
                    diagnostic["warning"] = str(exc)
                    _warn(str(exc))

                record = {
                    "video_id": video_id,
                    "targetRatioWH": output_ratio,
                    "predictions": predictions,
                }
                prediction_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                interval_handle.write(json.dumps(diagnostic, ensure_ascii=False) + "\n")
                total_predictions += len(predictions)
                print(
                    f"[{position}/{len(index)}] video={video_id} "
                    f"candidates={diagnostic['source_candidate_count']} "
                    f"merged={len(diagnostic['merged_intervals'])} "
                    f"frames={len(predictions)}"
                )
        os.replace(prediction_tmp, output_path)
        os.replace(interval_tmp, intervals_path)
    except BaseException:
        prediction_handle.close()
        interval_handle.close()
        prediction_tmp.unlink(missing_ok=True)
        interval_tmp.unlink(missing_ok=True)
        raise

    print(f"完成: {len(index)} 个视频，{total_predictions} 条逐帧预测 -> {output_path}")
    print(f"区间与裁剪框诊断 -> {intervals_path}")
    return len(index), total_predictions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="合并 Stage 2 粗高光区间并生成中心最大裁剪框提交结果",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stage2-dir",
        type=Path,
        required=True,
        help="Stage 2 根目录或其 videos 子目录",
    )
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX, help="测试索引 JSON")
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR, help="原始视频目录")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT, help="最终 predictions.jsonl")
    parser.add_argument(
        "--intervals-out",
        type=Path,
        default=None,
        help="区间诊断 JSONL；默认放在 --out 同目录",
    )
    parser.add_argument(
        "--video-id",
        action="append",
        default=[],
        help="只处理指定 video_id，可重复提供",
    )
    parser.add_argument("--num-videos", type=int, default=0, help="只处理索引前 N 项，0 表示不限")
    parser.add_argument(
        "--available-only",
        action="store_true",
        help="只处理存在 Stage 2 _SUCCESS.json 的视频，适合冒烟验证",
    )
    parser.add_argument(
        "--merge-gap-sec",
        type=float,
        default=1.0,
        help="间隔不超过该秒数的粗候选也合并",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="丢弃 coarse_score 低于该值的候选",
    )
    parser.add_argument("--video-extension", default=".mp4", help="源视频扩展名")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="遇到缺失或无效产物立即失败；默认记录空 predictions 后继续",
    )
    return parser


def main() -> int:
    try:
        run(build_parser().parse_args())
    except (ValidationError, OSError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

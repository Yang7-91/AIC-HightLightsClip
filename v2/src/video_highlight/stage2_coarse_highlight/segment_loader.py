"""从 Stage 1 持久化目录加载视频与分析片段。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from video_highlight.common.atomic_io import read_jsonl
from video_highlight.common.exceptions import ArtifactValidationError


@dataclass(frozen=True, slots=True)
class LoadedVideo:
    """Stage 1 单个视频的完整持久化产物。

    该对象只负责把 Stage 1 的多个文件聚合成便于 Stage 2 查询的内存视图，
    不会修改原始数据。``frozen=True`` 只能禁止属性被重新赋值；属性内部的
    ``dict`` 和 ``list`` 仍是可变对象，因此调用方应把它们当作只读数据使用。

    Attributes:
        video_id: 视频的稳定标识，同时也是 ``stage1/videos/<video_id>`` 的
            目录名。后续所有片段都通过它关联回同一个源视频。
        artifact_dir: 当前视频的 Stage 1 产物目录绝对路径。``sample_map.jsonl``
            中的 ``image_path`` 是相对此目录保存的，拼接后才能得到帧图片路径。
        metadata: 从 ``metadata.json`` 读取的视频级元数据，例如源文件路径、
            时长、帧率、分辨率、旋转信息、音频状态和目标画幅等。
        scenes: ``scenes.jsonl`` 的全部镜头记录。镜头时间使用原视频的绝对秒，
            一般按左闭右开区间 ``[start_sec, end_sec)`` 理解，并保留原始帧号。
        segments: ``segments.jsonl`` 的全部 Stage 2 分析窗口。窗口可以相互重叠，
            每条记录包含起止时间、关联镜头以及供模型使用的采样帧 ID。
        samples_by_id: 以整数 ``sample_id`` 为键的采样帧索引，值来自
            ``sample_map.jsonl``。该结构让片段加载器能以 O(1) 复杂度定位帧记录。
        audio_events: 全视频的显著音频事件，例如掌声、欢呼或能量峰值；时间戳
            仍是原视频绝对时间，在生成 ``LoadedSegment`` 时才按窗口过滤。
        audio_timeline: 全视频的分桶音频时间线，通常包含每个短时间窗的音量、
            能量等摘要，用来补充低帧率视觉输入缺失的声音变化。
        transcripts: 全视频 ASR 字幕记录，时间戳使用原视频绝对时间。若 Stage 1
            未启用语音识别或没有对应文件，该属性为空列表。

    Example:
        下面示例构造一个只包含一帧采样的最小视频产物视图::

            video = LoadedVideo(
                video_id="demo_001",
                artifact_dir=Path("F:/outputs/stage1/videos/demo_001"),
                metadata={"duration_sec": 12.0, "fps": 25.0},
                scenes=[{"scene_id": 0, "start_sec": 0.0, "end_sec": 12.0}],
                segments=[{"segment_id": 0, "start_sec": 0.0, "end_sec": 12.0}],
                samples_by_id={
                    0: {"sample_id": 0, "timestamp_sec": 0.5, "image_path": "frames/000000.jpg"}
                },
                audio_events=[{"start_sec": 4.0, "end_sec": 4.5, "label": "applause"}],
                audio_timeline=[{"start_sec": 0.0, "end_sec": 1.0, "rms": 0.08}],
                transcripts=[{"start_sec": 2.0, "end_sec": 3.0, "text": "开始了"}],
            )
            print(video.samples_by_id[0]["timestamp_sec"])  # 0.5
    """

    # Stage 1 分配的视频标识，也是该视频产物目录的名称。
    video_id: str
    # Stage 1 当前视频产物目录的绝对路径，采样帧相对路径以它为基准。
    artifact_dir: Path
    # 视频级 metadata.json 内容；所有时间与尺寸解释都应优先以它为准。
    metadata: dict[str, Any]
    # 全视频镜头切分记录，时间坐标基于原视频而非某个分析片段。
    scenes: list[dict[str, Any]]
    # 全部粗分析窗口；相邻窗口可能因滑窗策略而存在时间重叠。
    segments: list[dict[str, Any]]
    # sample_id 到采样帧记录的映射，用于快速解析片段引用的采样帧。
    samples_by_id: dict[int, dict[str, Any]]
    # 全视频显著音频事件，尚未按照某个片段的起止时间筛选。
    audio_events: list[dict[str, Any]]
    # 全视频分桶音频特征摘要，尚未按照某个片段的起止时间筛选。
    audio_timeline: list[dict[str, Any]]
    # 全视频带绝对时间戳的 ASR 文本；无 ASR 产物时为空列表。
    transcripts: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class LoadedSegment:
    """送入 Stage 2 多模态模型的单个粗分析片段。

    片段内的视觉、音频和文本信息共用原视频的绝对时间轴，模型可以据此对齐
    不同模态。``samples[i]`` 与 ``frame_paths[i]`` 始终指向同一张采样帧。

    Attributes:
        video_id: 所属源视频的稳定标识，用于关联 ``LoadedVideo`` 和最终输出。
        segment_id: 当前视频内部的分析窗口编号。它只在同一 ``video_id`` 下唯一，
            因而跨视频引用片段时应使用 ``(video_id, segment_id)`` 组合键。
        start_sec: 片段起点在原视频中的绝对秒数，包含该时刻。
        end_sec: 片段终点在原视频中的绝对秒数，通常不包含该时刻；片段区间按
            ``[start_sec, end_sec)`` 解释，不能把它当作片段内相对时间。
        scene_ids: 与当前时间窗口相交的 PySceneDetect 镜头 ID，便于模型结果回溯
            到镜头边界；一个分析片段可以覆盖一个或多个镜头。
        samples: 当前窗口选中的低帧率视觉采样记录，按时间升序排列。若原始帧数
            超过 ``max_frames``，加载阶段会均匀下采样，因此它不一定包含窗口内
            所有 2 FPS 粗采样帧。
        frame_paths: 与 ``samples`` 一一对应、顺序完全一致的采样 JPEG 绝对路径。
            ``data_url`` 模式会读取并编码这些文件；``url`` 模式通常只传视频 URL，
            但仍保留路径供调试、校验或切换传输模式使用。
        audio_events: 只保留与片段区间相交的显著音频事件。记录中的时间戳仍是
            原视频绝对时间，便于和画面及 ASR 精确对齐。
        audio_timeline: 只保留与片段区间相交的音频分桶摘要；它把音量、能量等
            声学变化以结构化文本交给模型，避免低视觉帧率造成“听觉信息丢失”。
        transcripts: 只保留与片段区间相交的 ASR 文本记录，时间戳仍为绝对秒。
            没有 ASR 产物或窗口内无人声时为空列表。
        metadata: 所属视频的完整 ``metadata.json`` 内容。多个片段可能共享同一个
            字典对象，调用方应只读使用，避免修改影响其他片段。

    Example:
        下面的片段覆盖原视频第 10 至 15 秒，并包含两张对齐的采样帧::

            segment = LoadedSegment(
                video_id="demo_001",
                segment_id=1,
                start_sec=10.0,
                end_sec=15.0,
                scene_ids=[2, 3],
                samples=[
                    {"sample_id": 20, "timestamp_sec": 10.0},
                    {"sample_id": 21, "timestamp_sec": 10.5},
                ],
                frame_paths=[Path("F:/frames/000020.jpg"), Path("F:/frames/000021.jpg")],
                audio_events=[{"start_sec": 12.1, "end_sec": 12.8, "label": "cheer"}],
                audio_timeline=[{"start_sec": 12.0, "end_sec": 13.0, "rms": 0.31}],
                transcripts=[{"start_sec": 11.0, "end_sec": 12.0, "text": "漂亮"}],
                metadata={"duration_sec": 120.0, "fps": 25.0},
            )
            assert segment.duration_sec == 5.0
            assert len(segment.samples) == len(segment.frame_paths)
    """

    # 所属源视频的标识；与 segment_id 一起构成片段的全局唯一身份。
    video_id: str
    # 当前视频内的分析窗口编号，不保证在不同视频之间唯一。
    segment_id: int
    # 分析窗口在原视频时间轴上的包含式起点，单位为秒。
    start_sec: float
    # 分析窗口在原视频时间轴上的排除式终点，单位为秒。
    end_sec: float
    # 与窗口相交的镜头 ID 列表，可用于恢复或约束镜头边界。
    scene_ids: list[int]
    # 按绝对时间升序排列的低帧率采样记录，可能已做均匀限帧。
    samples: list[dict[str, Any]]
    # 与 samples 等长且同序的采样图片绝对路径列表。
    frame_paths: list[Path]
    # 与窗口时间相交的显著音频事件，事件时间仍使用原视频绝对秒。
    audio_events: list[dict[str, Any]]
    # 与窗口时间相交的分桶声学摘要，时间仍使用原视频绝对秒。
    audio_timeline: list[dict[str, Any]]
    # 与窗口时间相交的 ASR 文本，时间仍使用原视频绝对秒。
    transcripts: list[dict[str, Any]]
    # 所属视频的完整元数据字典；通常由同一视频的多个片段共享。
    metadata: dict[str, Any]

    @property
    def duration_sec(self) -> float:
        """返回片段持续时间（秒），即 ``end_sec - start_sec``。"""

        return self.end_sec - self.start_sec


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ArtifactValidationError(f"缺少 Stage 1 文件: {path}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"JSON 根节点不是对象: {path}")
    return value


def _optional_jsonl(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.is_file() else []


def list_stage1_video_ids(stage1_dir: str | Path) -> list[str]:
    videos_dir = Path(stage1_dir).resolve() / "videos"
    if not videos_dir.is_dir():
        raise ArtifactValidationError(f"Stage 1 videos 目录不存在: {videos_dir}")
    return sorted(
        entry.name
        for entry in videos_dir.iterdir()
        if entry.is_dir() and not entry.name.startswith(".") and (entry / "_SUCCESS.json").is_file()
    )


def load_stage1_video(stage1_dir: str | Path, video_id: str) -> LoadedVideo:
    root = Path(stage1_dir).resolve() / "videos" / video_id
    if not (root / "_SUCCESS.json").is_file():
        raise ArtifactValidationError(f"Stage 1 视频没有成功标记: {root}")
    metadata = _read_json(root / "metadata.json")
    if str(metadata.get("video_id")) != video_id:
        raise ArtifactValidationError(f"metadata.video_id 与目录名不一致: {root}")
    samples = read_jsonl(root / "sample_map.jsonl")
    samples_by_id = {int(row["sample_id"]): row for row in samples}
    if len(samples_by_id) != len(samples):
        raise ArtifactValidationError(f"sample_map 存在重复 sample_id: {root}")
    return LoadedVideo(
        video_id=video_id,
        artifact_dir=root,
        metadata=metadata,
        scenes=read_jsonl(root / "scenes.jsonl"),
        segments=read_jsonl(root / "segments.jsonl"),
        samples_by_id=samples_by_id,
        audio_events=_optional_jsonl(root / "audio_events.jsonl"),
        audio_timeline=_optional_jsonl(root / "audio_timeline.jsonl"),
        transcripts=_optional_jsonl(root / "asr.jsonl"),
    )


def _overlaps(row: dict[str, Any], start_sec: float, end_sec: float) -> bool:
    return float(row.get("end_sec", 0.0)) > start_sec and float(row.get("start_sec", 0.0)) < end_sec


def _select_evenly(values: list[Any], maximum: int) -> list[Any]:
    if maximum <= 0 or len(values) <= maximum:
        return values
    if maximum == 1:
        return [values[len(values) // 2]]
    positions = [round(index * (len(values) - 1) / (maximum - 1)) for index in range(maximum)]
    return [values[position] for position in positions]


def load_segment(video: LoadedVideo, segment_row: dict[str, Any], max_frames: int) -> LoadedSegment:
    segment_id = int(segment_row["segment_id"])
    start_sec = float(segment_row["start_sec"])
    end_sec = float(segment_row["end_sec"])
    if end_sec <= start_sec:
        raise ArtifactValidationError(f"segment_id={segment_id} 时间区间非法")
    samples: list[dict[str, Any]] = []
    frame_paths: list[Path] = []
    for sample_id in segment_row.get("sample_ids", []):
        sample = video.samples_by_id.get(int(sample_id))
        if sample is None:
            raise ArtifactValidationError(f"segment_id={segment_id} 引用了未知 sample_id={sample_id}")
        frame_path = video.artifact_dir / str(sample["image_path"])
        if not frame_path.is_file():
            raise ArtifactValidationError(f"粗采样帧不存在: {frame_path}")
        samples.append(sample)
        frame_paths.append(frame_path)
    if not samples:
        raise ArtifactValidationError(f"segment_id={segment_id} 没有粗采样帧")
    pairs = _select_evenly(list(zip(samples, frame_paths)), max_frames)
    selected_samples = [pair[0] for pair in pairs]
    selected_paths = [pair[1] for pair in pairs]
    return LoadedSegment(
        video_id=video.video_id,
        segment_id=segment_id,
        start_sec=start_sec,
        end_sec=end_sec,
        scene_ids=[int(value) for value in segment_row.get("scene_ids", [])],
        samples=selected_samples,
        frame_paths=selected_paths,
        audio_events=[row for row in video.audio_events if _overlaps(row, start_sec, end_sec)],
        audio_timeline=[row for row in video.audio_timeline if _overlaps(row, start_sec, end_sec)],
        transcripts=[row for row in video.transcripts if _overlaps(row, start_sec, end_sec)],
        metadata=video.metadata,
    )

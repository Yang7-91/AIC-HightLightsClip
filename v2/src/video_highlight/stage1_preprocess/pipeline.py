"""Stage 1 可恢复、逐视频持久化的预处理流程。

本模块只负责“编排”，具体算法分别位于同目录下的独立模块中：

1. :mod:`video_probe` 使用 FFprobe 读取视频元信息；
2. :mod:`scene_detector` 使用 PySceneDetect 生成镜头区间；
3. :mod:`coarse_sampler` 顺序解码并保存约 2 FPS 的粗采样帧；
4. :mod:`segment_planner` 结合镜头边界生成 Stage 2 分析窗口；
5. :mod:`audio` 保留连续音轨并生成带时间戳的声学信息；
6. :mod:`validators` 在提交阶段产物前执行完整性检查。

原始视频可以位于代码目录之外（当前默认位于
``F:/datasets/video-clip/video``）。本模块对原始视频只读，所有生成内容均
写入 ``output_dir``。每个视频先写入隐藏的 ``.<video_id>.inprogress`` 目录，
只有所有步骤完成并通过校验后，才将其重命名为正式目录。这样中途异常不会
留下一个看似成功、实际不完整的阶段产物。
"""

from __future__ import annotations

import json
import os
import shutil
import traceback
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from video_highlight.common.atomic_io import write_json, write_jsonl
from video_highlight.common.exceptions import ArtifactValidationError
from video_highlight.common.hashing import mapping_sha256
from video_highlight.common.manifest import failure_record, success_record, utc_now_iso
from video_highlight.common.runtime import Timer
from video_highlight.common.video_id import normalize_video_id
from video_highlight.contracts.schema_versions import STAGE1_SCHEMA_VERSION

from .audio.acoustic_features import extract_acoustic_features, feature_rows
from .audio.asr_adapter import build_asr_adapter
from .audio.event_adapter import detect_audio_events
from .audio.extractor import extract_audio
from .audio.timeline_builder import build_audio_timeline
from .coarse_sampler import sample_video
from .scene_detector import detect_scenes
from .segment_planner import plan_segments
from .validators import validate_index_rows, validate_stage1_artifacts
from .video_probe import probe_video


def load_index(path: str | Path) -> list[dict[str, Any]]:
    """读取并校验视频索引。

    支持两种输入格式：

    - ``.jsonl``：每个非空行是一条视频记录；
    - ``.json``：根节点直接是记录数组，或使用 ``{"videos": [...]}``。

    每条记录至少需要 ``video_id``。比赛提供的 ``targetRatioWH`` 等其他字段
    会原样保留到后续处理过程中。函数只读取索引，不检查视频文件是否存在；
    文件解析由 :func:`resolve_video_path` 单独完成。

    Args:
        path: 索引文件路径，可位于代码目录之外。

    Returns:
        已完成基础结构校验的视频记录列表。

    Raises:
        ArtifactValidationError: 索引不存在、根结构错误、缺失或重复 video_id。
    """
    index_path = Path(path)
    if not index_path.is_file():
        raise ArtifactValidationError(f"输入索引不存在: {index_path}")
    if index_path.suffix.lower() == ".jsonl":
        # 忽略空行，便于手工拼接或增量生成 JSONL 索引。
        rows = [json.loads(line) for line in index_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    else:
        payload = json.loads(index_path.read_text(encoding="utf-8-sig"))
        # 同时兼容比赛常见的数组形式和带 videos 字段的封装形式。
        rows = payload if isinstance(payload, list) else payload.get("videos", [])
    if not all(isinstance(row, dict) for row in rows):
        raise ArtifactValidationError("输入索引必须是对象数组或逐行 JSON 对象")
    validate_index_rows(rows)
    return rows


def resolve_video_path(row: dict[str, Any], video_root: str | Path, extensions: Iterable[str]) -> Path:
    """将一条索引记录解析为真实视频文件的绝对路径。

    查找顺序是：

    1. 优先使用索引里的 ``video_path`` 或 ``path``；
    2. 若显式路径为相对路径，则以 ``video_root`` 为基准；
    3. 未提供显式路径时，依次尝试 ``<video_id><extension>``。

    这里只返回已经存在的普通文件，因此下游模块不需要再次猜测扩展名。解析
    过程不会移动、复制或写入源视频。
    """
    # resolve() 固定当前运行所使用的数据目录，避免工作目录变化影响后续查找。
    root = Path(video_root).resolve()
    explicit = row.get("video_path") or row.get("path")
    if explicit:
        # 显式路径的优先级最高，适合索引中视频名称不等于 video_id 的情况。
        candidate = Path(str(explicit))
        if not candidate.is_absolute():
            candidate = root / candidate
        if candidate.is_file():
            return candidate.resolve()
        raise ArtifactValidationError(f"索引指定的视频不存在: {candidate}")
    # 默认比赛数据采用 <video_id>.mp4；仍保留其他常用视频扩展名的兼容能力。
    video_id = normalize_video_id(row["video_id"])
    for extension in extensions:
        candidate = root / f"{video_id}{extension}"
        if candidate.is_file():
            return candidate.resolve()
    raise ArtifactValidationError(f"找不到 video_id={video_id} 对应的视频，目录: {root}")


def _single_scene(video_id: str, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """在关闭镜头检测时生成覆盖整个视频的退化镜头。

    即便用户通过配置关闭 PySceneDetect，下游仍然可以统一读取非空的
    ``scenes.jsonl``，无需针对“没有镜头信息”编写额外分支。
    """
    return [{
        "schema_version": STAGE1_SCHEMA_VERSION,
        "video_id": video_id,
        "scene_id": 0,
        "start_sec": 0.0,
        "end_sec": float(metadata["duration_sec"]),
        "start_frame": 0,
        "end_frame": int(metadata.get("frame_count") or 0),
        "frame_interval": "[start_frame,end_frame)",
        "transition_type": "scene_detection_disabled",
        "detectors": [],
    }]


def _write_npz_atomic(path: Path, features: dict[str, np.ndarray]) -> None:
    """原子写入 NumPy 压缩特征。

    ``numpy.savez_compressed`` 本身会直接写目标文件；若进程中断，可能留下损坏
    的 NPZ。因此先写同目录临时文件，再由 ``os.replace`` 一次性提交。
    """
    temporary = path.with_name(f".{path.stem}.tmp.npz")
    np.savez_compressed(temporary, **features)
    os.replace(temporary, path)


def _process_audio(
    video_id: str,
    video_path: Path,
    work_dir: Path,
    metadata: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """完成单个视频的音频预处理并持久化统一输出。

    音频不随视觉一起降采样到 2 FPS。本函数先通过 FFmpeg 提取连续的单声道
    PCM 音轨，再以较细时间箱计算能量、峰值和频谱变化。Stage 2 可以读取
    ``audio_timeline.jsonl`` 获得带时间戳的音频理解线索，Stage 3 则可以直接
    使用 ``audio_features.npz`` 做更细的边界定位。

    无论音频被禁用、源视频没有音轨，还是正常完成处理，都会写出事件、ASR、
    时间线和状态文件。空结果使用空 JSONL 表示，从而保持阶段数据契约稳定。

    Returns:
        音频处理状态，包括是否启用、是否有源音轨及产物数量。
    """
    audio_config = config.get("audio", {})
    enabled = bool(audio_config.get("enabled", True))
    status: dict[str, Any] = {"enabled": enabled, "source_has_audio": bool(metadata["has_audio"])}
    events: list[dict[str, Any]] = []
    transcripts: list[dict[str, object]] = []
    timeline: list[dict[str, Any]] = []
    if enabled and metadata["has_audio"]:
        # 1. 保留连续音轨。默认 16 kHz 单声道 PCM，时间精度不受视觉 2 FPS 影响。
        wav_path = work_dir / "audio.wav"
        status.update(
            extract_audio(
                video_path,
                wav_path,
                sample_rate=int(audio_config.get("sample_rate", 16000)),
                ffmpeg_bin=str(config.get("ffmpeg_bin", "ffmpeg")),
            )
        )
        # 2. 提取固定时间箱声学特征，并同时生成便于写入 JSONL 的行式表示。
        features = extract_acoustic_features(wav_path, bin_sec=float(audio_config.get("bin_sec", 0.5)))
        rows = feature_rows(features)
        # 3. 基于能量突增和频谱变化生成通用声音事件。这里不虚构具体声音类别。
        events = detect_audio_events(
            video_id,
            features,
            z_threshold=float(audio_config.get("event_z_threshold", 2.5)),
            onset_db_threshold=float(audio_config.get("onset_db_threshold", 6.0)),
        )
        # 4. ASR 使用可替换适配器。当前默认禁用，因此返回空列表而不是伪造字幕。
        adapter = build_asr_adapter(bool(audio_config.get("asr_enabled", False)))
        transcripts = adapter.transcribe(wav_path)
        # 5. 按同一时间轴合并数值特征、事件标签和可能存在的 ASR 文本。
        timeline = build_audio_timeline(video_id, rows, events, transcripts)
        # 高维/稠密数值保存为 NPZ；面向大模型的紧凑信息保存为 JSONL。
        _write_npz_atomic(work_dir / "audio_features.npz", features)
        status.update({"feature_bins": len(rows), "event_count": len(events), "asr_segments": len(transcripts)})
    else:
        # 区分用户主动禁用和源文件确实没有音频流，便于后续统计降级原因。
        status["status"] = "disabled" if not enabled else "no_audio_stream"
    # 始终生成下面四个契约文件；没有内容时 JSONL 文件合法地保持为空。
    write_jsonl(work_dir / "audio_events.jsonl", events)
    write_jsonl(work_dir / "asr.jsonl", transcripts)
    write_jsonl(work_dir / "audio_timeline.jsonl", timeline)
    write_json(work_dir / "audio_status.json", status)
    return status


def process_video(
    row: dict[str, Any],
    video_root: Path,
    videos_output_dir: Path,
    config: dict[str, Any],
    resume: bool,
    overwrite: bool,
) -> dict[str, Any]:
    """处理一条视频记录，并以目录为单位提交持久化结果。

    单视频执行顺序为：定位源视频 → FFprobe → PySceneDetect → 2 FPS 粗采样
    → 分析窗口规划 → 音频预处理 → 产物校验 → 写成功标记 → 原子目录提交。

    Args:
        row: 索引中的单条记录，必须含 ``video_id``。
        video_root: 外部原始视频根目录；本函数只读该目录。
        videos_output_dir: ``stage1/videos`` 输出目录。
        config: 已解析的 Stage 1 配置。
        resume: 正式目录已有 ``_SUCCESS.json`` 时是否直接跳过。
        overwrite: 正式目录存在时是否明确允许重算。

    Returns:
        可直接写入阶段 ``manifest.jsonl`` 的成功或跳过记录。

    Raises:
        ArtifactValidationError: 产物已存在但未指定策略，或输入/输出不合法。
        Exception: 解码、镜头检测、音频处理等下游错误会清理临时目录后上抛。
    """
    # video_id 同时用于文件查找和目录名称，必须先限制为安全字符集合。
    video_id = normalize_video_id(row["video_id"])
    final_dir = videos_output_dir / video_id
    success_path = final_dir / "_SUCCESS.json"
    # 断点续跑只信任显式成功标记，不能仅凭目录存在就认定处理完成。
    if resume and success_path.is_file():
        return {"video_id": video_id, "status": "skipped", "reason": "already_successful"}
    if final_dir.exists():
        # 避免默认覆盖一套可能仍有分析价值的旧产物。
        if not overwrite:
            raise ArtifactValidationError(f"输出已存在，请使用 --resume 或 --overwrite: {final_dir}")
        shutil.rmtree(final_dir)
    # 全部步骤写入同级隐藏目录。正式目录在函数成功返回前不会出现。
    work_dir = videos_output_dir / f".{video_id}.inprogress"
    if work_dir.exists():
        # 上一次中断遗留的临时目录不具备可信状态，可以安全重建。
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    try:
        with Timer() as timer:
            # 第一步：只解析路径，视频仍保留在外部数据目录中。
            video_path = resolve_video_path(row, video_root, config["video_extensions"])
            # 第二步：FFprobe 提供权威流信息；可选哈希用于缓存命中和来源追踪。
            metadata = probe_video(
                video_id,
                video_path,
                ffprobe_bin=str(config.get("ffprobe_bin", "ffprobe")),
                compute_sha256=bool(config.get("compute_sha256", True)),
            )
            # 目标画幅来自比赛索引，Stage 1 不使用它做裁剪，但需要向后传递。
            metadata["targetRatioWH"] = row.get("targetRatioWH")
            scene_config = config.get("scene_detection", {})
            # 第三步：正常模式调用 PySceneDetect；关闭时仍生成全视频单镜头契约。
            scenes = (
                detect_scenes(video_id, video_path, scene_config)
                if bool(scene_config.get("enabled", True))
                else _single_scene(video_id, metadata)
            )
            # 第四步：顺序解码保证 original_frame 与实际解码顺序一致，同时落盘图片。
            samples, decode_stats = sample_video(
                video_id,
                video_path,
                work_dir / "coarse_frames",
                scenes,
                metadata,
                config.get("sampling", {}),
            )
            metadata["decode_stats"] = decode_stats
            # 第五步：分析窗口可以跨短镜头，但会优先在邻近镜头边界处结束。
            segments = plan_segments(
                video_id,
                float(metadata["duration_sec"]),
                scenes,
                samples,
                config.get("segments", {}),
            )
            # 先写视觉侧结构化产物；所有写入仍位于 inprogress 临时目录。
            write_json(work_dir / "metadata.json", metadata)
            write_jsonl(work_dir / "scenes.jsonl", scenes)
            write_jsonl(work_dir / "sample_map.jsonl", samples)
            write_jsonl(work_dir / "segments.jsonl", segments)
            # 第六步：音频保持连续时间轴，独立于视觉粗采样率。
            audio_status = _process_audio(video_id, video_path, work_dir, metadata, config)
            # 第七步：只有必需文件和采样图片齐全，才允许生成成功标记。
            validation = validate_stage1_artifacts(
                work_dir,
                has_audio=bool(metadata["has_audio"]),
                audio_enabled=bool(config.get("audio", {}).get("enabled", True)),
            )
        # 计时器退出后记录完整统计，供批量运行、性能分析和断点恢复使用。
        success = success_record(
            video_id,
            elapsed_sec=timer.elapsed_sec,
            scene_count=len(scenes),
            segment_count=len(segments),
            sample_count=len(samples),
            audio_status=audio_status.get("status"),
            validation=validation,
        )
        # _SUCCESS 最后写入；随后把整个目录一次性提升为正式产物目录。
        write_json(work_dir / "_SUCCESS.json", success)
        work_dir.rename(final_dir)
        return success
    except Exception:
        # 失败目录绝不作为上游输入保留；具体异常由批处理层登记到 failures.jsonl。
        shutil.rmtree(work_dir, ignore_errors=True)
        raise


def run_stage1(
    index_path: str | Path,
    video_root: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
    video_ids: set[str] | None = None,
    limit: int | None = None,
    resume: bool = False,
    overwrite: bool = False,
    strict: bool = False,
    logger: Any = None,
) -> dict[str, Any]:
    """批量运行 Stage 1，并在每个视频完成后更新运行清单。

    本函数是 Stage 1 的公共 Python API，CLI 与未来的总流水线编排器都应调用它。
    单个视频失败默认不会终止整个数据集；使用 ``strict=True`` 时则在持久化失败
    记录后立即抛出异常。

    Args:
        index_path: JSON/JSONL 视频索引路径。
        video_root: 原始视频根目录，当前通常为 ``F:/datasets/video-clip/video``。
        output_dir: 本次 Stage 1 的独立输出目录。
        config: Stage 1 完整配置字典。
        video_ids: 可选 video_id 白名单，用于局部重跑和调试。
        limit: 可选处理数量上限，筛选 video_id 后再应用。
        resume: 跳过已有成功标记的视频。
        overwrite: 允许重建已存在的视频产物目录。
        strict: 任一视频失败时是否立即中止批处理。
        logger: 可选日志对象；传入 ``None`` 时保持静默。

    Returns:
        包含选择数、成功数、跳过数和失败数的运行摘要。
    """
    # 一个 run 的所有 Stage 1 文件都限制在 stage_root 下，便于独立删除或复用。
    stage_root = Path(output_dir).resolve()
    videos_output = stage_root / "videos"
    videos_output.mkdir(parents=True, exist_ok=True)
    # 先读取完整索引，再应用白名单和数量限制，保证 limit 的语义确定。
    rows = load_index(index_path)
    if video_ids:
        rows = [row for row in rows if str(row["video_id"]) in video_ids]
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ArtifactValidationError("筛选后没有待处理视频")
    # manifest 保存所有选中视频的最终状态，failures 只保存失败详情。
    manifest: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    # 固化解析后的配置，避免后续无法确认某批产物具体使用了哪些参数。
    write_json(stage_root / "resolved_config.json", config)
    run_info = {
        "schema_version": STAGE1_SCHEMA_VERSION,
        "started_at": utc_now_iso(),
        "index_path": str(Path(index_path).resolve()),
        "video_root": str(Path(video_root).resolve()),
        "config_sha256": mapping_sha256(config),
        "selected_video_count": len(rows),
    }
    # run_manifest 在运行开始时就落盘，即使中断也能追踪输入和配置哈希。
    write_json(stage_root / "run_manifest.json", run_info)
    for position, row in enumerate(rows, start=1):
        video_id = str(row.get("video_id", "unknown"))
        if logger:
            logger.info("[%d/%d] Stage 1 处理 video_id=%s", position, len(rows), video_id)
        try:
            # 单视频函数拥有自己的临时目录和异常边界，不会污染其他视频结果。
            record = process_video(row, Path(video_root), videos_output, config, resume, overwrite)
            manifest.append(record)
        except Exception as error:
            # 保留错误类型、错误文本及堆栈，便于之后只重跑失败视频。
            failure = failure_record(video_id, error)
            failure["traceback"] = traceback.format_exc()
            failures.append(failure)
            manifest.append(failure)
            if logger:
                logger.exception("video_id=%s 处理失败", video_id)
            if strict:
                # 严格模式中止前仍先持久化当前进度，避免错误信息随进程退出丢失。
                write_jsonl(stage_root / "manifest.jsonl", manifest)
                write_jsonl(stage_root / "failures.jsonl", failures)
                raise
        # 每处理完一个视频就原子重写清单，支持进程被打断后的人工检查和恢复。
        write_jsonl(stage_root / "manifest.jsonl", manifest)
        write_jsonl(stage_root / "failures.jsonl", failures)
    # 批次结束后用最终统计更新 run_manifest；没有失败时才生成阶段级成功标记。
    summary = {
        **run_info,
        "finished_at": utc_now_iso(),
        "success_count": sum(item["status"] == "success" for item in manifest),
        "skipped_count": sum(item["status"] == "skipped" for item in manifest),
        "failure_count": len(failures),
    }
    write_json(stage_root / "run_manifest.json", summary)
    if not failures:
        write_json(stage_root / "_SUCCESS.json", summary)
    return summary

"""Stage 3：候选高帧率解码、时序打分、边界恢复与持久化。

本模块只负责编排 Stage 3，不在这里实现具体的解码、特征计算或时序模型。
输入由两个上游阶段共同提供：Stage 1 提供源视频、FPS、总帧数、镜头和音频
特征；Stage 2 提供需要进一步定位的高光候选区间及其语义信息。

支持两种互斥运行模式：

``refine``
    对每个候选进行高帧率精解码，提取帧级特征，通过规则或 TCN 后端得到
    高光/起点/终点概率，再恢复成原视频的半开帧区间。候选级异常可按配置
    回退为旁路结果，防止单个坏候选导致整条视频丢失。

``passthrough``
    完全跳过视频解码、特征提取、时序推理、无高光门控和区间合并。该模式
    原样保留 Stage 2 秒边界，只执行下游数据契约要求的确定性秒到帧映射。

每个视频先写入 ``videos/.<video_id>.inprogress``。只有区间校验、JSONL 写入
和产物完整性校验全部成功，才写 ``_SUCCESS.json`` 并将临时目录重命名为
正式目录。这可以防止 Stage 4 读取到只完成一部分的 Stage 3 结果。
"""

from __future__ import annotations

import shutil
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any

from video_highlight.common.atomic_io import write_json, write_jsonl
from video_highlight.common.exceptions import ArtifactValidationError
from video_highlight.common.hashing import mapping_sha256
from video_highlight.common.manifest import failure_record, success_record, utc_now_iso
from video_highlight.common.runtime import Timer
from video_highlight.contracts.schema_versions import STAGE3_SCHEMA_VERSION

from .boundary_decoder import decode_boundaries
from .candidate_decoder import decode_candidate
from .empty_gate import apply_empty_gate
from .feature_extractor import extract_features, load_audio_features
from .frame_mapper import map_passthrough, map_refined
from .interval_merger import merge_intervals
from .temporal_backend import TemporalBackend, build_temporal_backend
from .validators import (
    list_stage2_video_ids,
    load_video_inputs,
    validate_config,
    validate_intervals,
    validate_stage3_artifacts,
)


def _base_interval(candidate: dict[str, Any], mapped: dict[str, Any], mode: str) -> dict[str, Any]:
    """组装所有执行路径共用的 Stage 3 区间字段。

    ``mapped`` 必须已经包含合法的 ``start/end_sec`` 和半开帧区间
    ``[start_frame, end_frame)``。本函数保留 Stage 2 的分数、类别和主体提示，
    使 Stage 4 无需回读 Stage 2；与细化有关的分数先填安全默认值，正常模式
    会在时序推理完成后覆盖。
    """

    return {
        # 契约与来源：interval_id 在合并或旁路编号阶段再确定。
        "schema_version": STAGE3_SCHEMA_VERSION,
        "video_id": str(candidate["video_id"]),
        "interval_id": "",
        "source_candidate_ids": [str(candidate["candidate_id"])],
        # mapped 中包含秒区间和原视频帧区间；后写入不会覆盖下方字段。
        **mapped,
        # end_frame 为排除式终点，所以两者相减就是区间内原始帧数量。
        "frame_count": int(mapped["end_frame"]) - int(mapped["start_frame"]),
        "frame_interval": "[start_frame,end_frame)",
        # refine_mode 描述采用哪条主路径；refine_status 描述该候选实际结果。
        "refine_mode": mode,
        "refine_status": "passthrough" if mode == "passthrough" else "refined",
        # 旁路时没有独立时序分数，暂用 coarse_score，边界置信度固定为 0。
        "coarse_score": float(candidate.get("coarse_score", 0.0)),
        "temporal_score": float(candidate.get("coarse_score", 0.0)),
        "boundary_confidence": 0.0,
        # 语义与主体信息透传给 Stage 4，用于选择需要跟踪和构图的对象。
        "source_segment_ids": [int(value) for value in candidate.get("source_segment_ids", [])],
        "subject": candidate.get("subject"),
        "subject_point": candidate.get("subject_point"),
        "category": candidate.get("category"),
        "reason": str(candidate.get("reason", "")),
    }


def _passthrough_intervals(
    video_id: str, candidates: list[dict[str, Any]], metadata: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """把 Stage 2 候选一对一封装成未经细化的 Stage 3 输出。

    这里不会调用任何会打开视频或改变候选集合的函数。``map_passthrough``
    只用 Stage 1 的 FPS 和总帧数把秒边界映射为合法帧号；候选顺序、数量、
    ``start_sec`` 和 ``end_sec`` 均保持不变。
    """

    intervals: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        # 使用独立旁路映射，避免意外进入高帧率解码或规则边界逻辑。
        row = _base_interval(candidate, map_passthrough(candidate, metadata), "passthrough")
        # 旁路模式禁止合并，因此按 Stage 2 候选顺序直接产生一对一稳定 ID。
        row["interval_id"] = f"{video_id}_interval_{index:04d}"
        intervals.append(row)
        # 即使明确跳过处理也写诊断记录，便于区分“主动旁路”和“处理失败回退”。
        diagnostics.append({
            "schema_version": STAGE3_SCHEMA_VERSION,
            "video_id": video_id,
            "candidate_id": candidate["candidate_id"],
            "mode": "passthrough",
            "status": "skipped_all_stage3_processing",
            "input_start_sec": float(candidate["start_sec"]),
            "input_end_sec": float(candidate["end_sec"]),
            "output_start_frame": row["start_frame"],
            "output_end_frame": row["end_frame"],
        })
    return intervals, diagnostics


def _refine_intervals(
    video_id: str,
    stage1_video_dir: Path,
    metadata: dict[str, Any],
    scenes: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    backend: TemporalBackend,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, dict[str, Any]]:
    """细化一个视频的全部 Stage 2 候选。

    Returns:
        四元组 ``(区间列表, 诊断列表, 回退次数, 无高光门控摘要)``。每个候选
        正常情况下产生一个区间；之后可因同类小间隔合并而减少，也可能被
        视频级无高光门控整体清空。

    Candidate error policy:
        ``runtime.candidate_error_policy=passthrough`` 时，仅失败候选回退到原始
        Stage 2 边界，其余候选继续细化；其他取值会重新抛出异常，由视频级
        异常边界记录失败。
    """

    # 音频 NPZ 每个视频只加载一次，在多个候选之间复用，避免重复磁盘读取。
    audio = load_audio_features(stage1_video_dir)
    intervals: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    fallback_count = 0
    error_policy = str(config.get("runtime", {}).get("candidate_error_policy", "passthrough"))
    for candidate in candidates:
        try:
            # 1. 只在候选秒区间内按 decode.fps 精解码，并保留原始帧号。
            decoded = decode_candidate(candidate, metadata, config.get("decode", {}))
            # 2. 生成规则后端与未来 TCN 共用的固定顺序帧级特征矩阵。
            features = extract_features(
                decoded, candidate, scenes, audio, config.get("features", {})
            )
            # 3. 后端输出逐采样帧高光/起点/终点概率及候选无高光概率。
            scores = backend.predict(features, candidate)
            # 4. 从概率序列寻找核心区间，并以 max_trim_sec 限制规则收缩幅度。
            decision = decode_boundaries(features, scores, candidate, config.get("boundary", {}))
            # 5. 使用解码获得的真实原始帧号恢复半开帧区间，而非重复粗略取整。
            mapped = map_refined(decoded, decision, candidate, metadata)
            # 6. 组装稳定输出，并用真实时序结果覆盖旁路默认分数。
            row = _base_interval(candidate, mapped, "refine")
            row["refine_status"] = decision.status
            row["temporal_score"] = decision.temporal_score
            row["boundary_confidence"] = decision.boundary_confidence
            intervals.append(row)
            diagnostics.append({
                "schema_version": STAGE3_SCHEMA_VERSION,
                "video_id": video_id,
                "candidate_id": candidate["candidate_id"],
                "mode": "refine",
                "status": decision.status,
                "sample_count": len(decoded.timestamps_sec),
                "input_start_sec": float(candidate["start_sec"]),
                "input_end_sec": float(candidate["end_sec"]),
                "output_start_frame": row["start_frame"],
                "output_end_frame": row["end_frame"],
                "temporal_score": decision.temporal_score,
                "boundary_confidence": decision.boundary_confidence,
                "empty_probability": scores.empty_probability,
            })
        except Exception as error:
            # candidate_error_policy=passthrough 是候选级降级，不等同于全局旁路模式。
            # 它仍会参与后面的合并和无高光门控，并明确记录失败原因。
            if error_policy != "passthrough":
                raise
            fallback_count += 1
            row = _base_interval(candidate, map_passthrough(candidate, metadata), "passthrough")
            row["refine_status"] = "fallback_after_error"
            intervals.append(row)
            diagnostics.append({
                "schema_version": STAGE3_SCHEMA_VERSION,
                "video_id": video_id,
                "candidate_id": candidate.get("candidate_id"),
                "mode": "refine",
                "status": "fallback_after_error",
                "error_type": type(error).__name__,
                "message": str(error),
            })
    # 所有候选处理完成后才统一合并，确保 interval_id 按最终时间顺序稳定生成。
    intervals = merge_intervals(video_id, intervals, config.get("merging", {}))
    # 无高光门控是视频级决策：只有粗分和时序分都不足时才可能清空全部区间。
    intervals, gate = apply_empty_gate(intervals, config.get("empty_gate", {}))
    # 门控可能清空结果；保留诊断使空输出可解释。
    diagnostics.append({
        "schema_version": STAGE3_SCHEMA_VERSION,
        "video_id": video_id,
        "candidate_id": None,
        "mode": "refine",
        "status": "empty_gate",
        **gate,
    })
    return intervals, diagnostics, fallback_count, gate


def process_video(
    stage1_dir: Path,
    stage2_dir: Path,
    video_id: str,
    videos_output_dir: Path,
    backend: TemporalBackend | None,
    config: dict[str, Any],
    resume: bool,
    overwrite: bool,
) -> dict[str, Any]:
    """处理并原子提交单个视频的 Stage 3 产物。

    Args:
        stage1_dir: Stage 1 输出根目录，用于读取视频元数据、镜头和音频特征。
        stage2_dir: Stage 2 输出根目录，用于读取最终粗候选。
        video_id: 当前视频标识，同时也是三个阶段的视频子目录名。
        videos_output_dir: Stage 3 的 ``videos`` 目录。
        backend: 已复用的时序推理后端；全局旁路模式下为 ``None``。
        config: 完整 Stage 3 配置。
        resume: 若正式目录带成功标记，是否直接跳过该视频。
        overwrite: 是否明确删除当前视频已有正式目录和未完成目录后重算。

    Returns:
        可直接写入批次 ``manifest.jsonl`` 的成功或跳过记录。

    目录状态约定：

    - ``videos/<id>/_SUCCESS.json``：该视频已经完整提交，可供 Stage 4 使用；
    - ``videos/.<id>.inprogress``：未提交的临时目录，不允许下游读取；
    - 已有正式目录但无成功标记：不自动覆盖，要求用户显式 ``--overwrite``。
    """

    final_dir = videos_output_dir / video_id
    # 已成功视频的恢复是视频级跳过，不会重新加载上游文件或初始化候选处理。
    if resume and (final_dir / "_SUCCESS.json").is_file():
        return {"video_id": video_id, "status": "skipped", "reason": "already_successful"}
    work_dir = videos_output_dir / f".{video_id}.inprogress"
    if overwrite:
        # overwrite 是唯一允许清除该视频既有 Stage 3 结果的入口。
        shutil.rmtree(final_dir, ignore_errors=True)
        shutil.rmtree(work_dir, ignore_errors=True)
    elif final_dir.exists():
        # 防止无意覆盖可能正在被 Stage 4 使用的正式结果。
        raise ArtifactValidationError(f"Stage 3 输出已存在，请使用 --resume 或 --overwrite: {final_dir}")
    elif work_dir.exists():
        # 当前实现不做候选级断点续跑；未完成目录必须显式覆盖，避免混用配置。
        raise ArtifactValidationError(f"Stage 3 存在未完成目录，请使用 --overwrite: {work_dir}")
    work_dir.mkdir(parents=True, exist_ok=True)

    # 同时核验 Stage 1/2 的视频级成功标记，并读取两个阶段的唯一共享契约。
    stage1_video, metadata, scenes, candidates = load_video_inputs(stage1_dir, stage2_dir, video_id)
    mode = str(config.get("runtime", {}).get("mode", "refine"))
    with Timer() as timer:
        if mode == "passthrough":
            # 全局旁路必须走独立短路径，确保没有解码、模型、门控或合并副作用。
            intervals, diagnostics = _passthrough_intervals(video_id, candidates, metadata)
            fallback_count = 0
            gate = {"applied": False, "is_empty": not intervals, "reason": "stage3_processing_skipped"}
        else:
            if backend is None:
                # 理论上会在 run_stage3 构建后端；此检查避免错误调用静默旁路。
                raise RuntimeError("refine 模式缺少 temporal backend")
            intervals, diagnostics, fallback_count, gate = _refine_intervals(
                video_id, stage1_video, metadata, scenes, candidates, backend, config
            )
        # 先验证内存结果，再写入临时目录，避免持久化已知越界或重叠区间。
        validate_intervals(intervals, metadata)
        # refined_intervals 是 Stage 4 的正式输入；diagnostics 只用于复现和排错。
        write_jsonl(work_dir / "refined_intervals.jsonl", intervals)
        write_jsonl(work_dir / "diagnostics.jsonl", diagnostics)
        # 检查必需文件是否齐全；空 JSONL 也是合法的无高光输出。
        validation = validate_stage3_artifacts(work_dir)
    # _SUCCESS 同时记录数量、耗时、回退和门控状态，便于批量质量审计。
    success = success_record(
        video_id,
        elapsed_sec=timer.elapsed_sec,
        mode=mode,
        input_candidate_count=len(candidates),
        output_interval_count=len(intervals),
        fallback_count=fallback_count,
        empty_gate=gate,
        validation=validation,
    )
    write_json(work_dir / "_SUCCESS.json", success)
    # 同文件系统目录重命名作为提交点；Stage 4 只读取带成功标记的正式目录。
    work_dir.rename(final_dir)
    return success


def run_stage3(
    stage1_dir: str | Path,
    stage2_dir: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
    video_ids: set[str] | None = None,
    limit: int | None = None,
    resume: bool = False,
    overwrite: bool = False,
    strict: bool = False,
    logger: Any = None,
) -> dict[str, Any]:
    """批量执行 Stage 3，是 CLI 与总流水线共用的 Python API。

    Args:
        stage1_dir: 与 Stage 2 同批次对应的 Stage 1 输出根目录。
        stage2_dir: 已完成的 Stage 2 输出根目录。
        output_dir: 本次 Stage 3 输出根目录。
        config: 解码、特征、后端、边界、门控、合并和运行配置。
        video_ids: 可选视频白名单；``None`` 表示全部成功的 Stage 2 视频。
        limit: 应用白名单后最多处理的视频数，主要用于冒烟测试。
        resume: 是否跳过已有视频级成功产物。
        overwrite: 是否重算所选视频并替换其旧产物。
        strict: 任一视频失败后是否立即终止批次；关闭时记录失败并继续。
        logger: 可选结构化日志对象。

    Returns:
        包含输入目录、执行模式、配置哈希及成功/跳过/失败数量的批次摘要。
    """

    # 在创建后端和输出文件前验证配置，尽早暴露缺失字段或非法模式。
    validate_config(config)
    # 全部路径固定为绝对路径并写入 run_manifest，保证实验可以复现。
    stage1_root = Path(stage1_dir).resolve()
    stage2_root = Path(stage2_dir).resolve()
    stage3_root = Path(output_dir).resolve()
    videos_output = stage3_root / "videos"
    videos_output.mkdir(parents=True, exist_ok=True)
    # 只枚举带 Stage 2 视频级 _SUCCESS 的目录，临时目录不会进入本阶段。
    available = list_stage2_video_ids(stage2_root)
    selected = [value for value in available if not video_ids or value in video_ids]
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ArtifactValidationError("筛选后没有可处理的 Stage 2 视频")
    mode = str(config["runtime"].get("mode", "refine"))
    # 旁路模式故意不实例化后端，因此即使缺少 PyTorch/TCN 权重也可以运行。
    # 正常模式只构建一次后端，在所有视频之间复用模型权重和设备资源。
    backend = None if mode == "passthrough" else build_temporal_backend(config["temporal"])
    # run_info 在开始和结束各写一次：开始时可供外部监控，结束时补充最终计数。
    run_info = {
        "schema_version": STAGE3_SCHEMA_VERSION,
        "started_at": utc_now_iso(),
        "stage1_dir": str(stage1_root),
        "stage2_dir": str(stage2_root),
        "mode": mode,
        "temporal_backend": None if mode == "passthrough" else config["temporal"].get("backend", "rule"),
        "config_sha256": mapping_sha256(config),
        "selected_video_count": len(selected),
    }
    # 保存实际生效配置快照，而不是只记录用户传入的配置文件路径。
    write_json(stage3_root / "resolved_config.json", deepcopy(config))
    write_json(stage3_root / "run_manifest.json", run_info)
    manifest: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for position, video_id in enumerate(selected, start=1):
        if logger:
            logger.info("[%d/%d] Stage 3 处理 video_id=%s mode=%s", position, len(selected), video_id, mode)
        try:
            # 每个视频拥有独立临时目录和异常边界，一个坏视频不会污染其他结果。
            record = process_video(
                stage1_root,
                stage2_root,
                video_id,
                videos_output,
                backend,
                config,
                resume,
                overwrite,
            )
        except Exception as error:
            # 保存完整堆栈；解码、配置、模型和契约错误均可在 failures.jsonl 定位。
            record = failure_record(video_id, error)
            record["traceback"] = traceback.format_exc()
            failures.append(record)
            if logger:
                logger.exception("Stage 3 video_id=%s 处理失败", video_id)
            if strict:
                # 严格模式退出前仍落盘当前状态，避免失败信息随异常丢失。
                manifest.append(record)
                write_jsonl(stage3_root / "manifest.jsonl", manifest)
                write_jsonl(stage3_root / "failures.jsonl", failures)
                raise
        manifest.append(record)
        # 每完成一个视频立即原子更新批次清单，长批次中断后仍可审计进度。
        write_jsonl(stage3_root / "manifest.jsonl", manifest)
        write_jsonl(stage3_root / "failures.jsonl", failures)
    # 阶段级成功只代表所选视频全部完成；各视频仍有独立的成功标记。
    summary = {
        **run_info,
        "finished_at": utc_now_iso(),
        "success_count": sum(row["status"] == "success" for row in manifest),
        "skipped_count": sum(row["status"] == "skipped" for row in manifest),
        "failure_count": len(failures),
    }
    write_json(stage3_root / "run_manifest.json", summary)
    if not failures:
        # 存在任何失败时不写阶段级 _SUCCESS，防止总流水线误判整批完成。
        write_json(stage3_root / "_SUCCESS.json", summary)
    return summary

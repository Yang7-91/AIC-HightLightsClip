"""Stage 2：调用 OpenAI 兼容 vLLM 服务并持久化粗高光候选。

本模块是 Stage 2 的流程编排层，不负责部署或加载 Qwen 模型。模型已经由
独立的 vLLM 服务提供，本阶段只通过 OpenAI 兼容 HTTP 接口发送请求。

单个分析片段的处理流程如下：

1. 从 Stage 1 目录读取低帧率帧序列、原始时间戳和音频时间线；
2. 将视觉时间、音频事件、声学强度和 ASR 组织成紧凑文本上下文；
3. 根据配置构建 ``url`` 或 ``data_url`` 视频载荷；
4. 调用 Qwen3.5-4B，并将原始响应保存为可追踪记录；
5. 把模型返回的相对片段时间转换成原视频绝对时间；
6. 融合语义、事件、音频、质量和稳定性分数；
7. 合并来自重叠窗口的候选，并生成 Stage 3 可消费的时间与语义结果。

Stage 2 只负责“何时高光”和“高光主体是什么”的粗语义判断，不输出空间坐标
``subject_point``。主体点将在后续 Stage 3.5 中基于精确高光区间单独生成。

Stage 2 采用两级持久化：每完成一个片段就更新视频临时目录中的 JSONL，
每完成一个视频再写入 ``_SUCCESS.json`` 并把 ``.<video_id>.inprogress``
重命名为正式目录。前者避免长视频中断后重复调用大模型，后者避免下游读取
未完成的视频结果。
"""

from __future__ import annotations

import shutil
import traceback
from pathlib import Path
from typing import Any

from video_highlight.common.atomic_io import read_jsonl, write_json, write_jsonl
from video_highlight.common.exceptions import ArtifactValidationError
from video_highlight.common.hashing import mapping_sha256
from video_highlight.common.manifest import failure_record, success_record, utc_now_iso
from video_highlight.common.runtime import Timer
from video_highlight.contracts.schema_versions import STAGE2_SCHEMA_VERSION

from .candidate_merger import merge_candidates
from .multimodal_context import build_multimodal_context
from .prompt_builder import build_prompts,OUTPUT_SCHEMA_OPENAI
from .qwen_adapter import QwenBackend, build_backend
from .response_parser import ResponseParseError, parse_response
from .score_fusion import build_candidate
from .segment_loader import LoadedSegment, list_stage1_video_ids, load_segment, load_stage1_video
from .subject_hint import build_subject_hints
from .validators import validate_stage2_artifacts
from .video_payload import prepare_video_payload


def _load_existing(path: Path) -> list[dict[str, Any]]:
    """读取恢复运行所需的 JSONL；文件尚未生成时返回空列表。

    该函数只用于 ``.inprogress`` 临时目录。正式目录是否可复用必须通过
    ``_SUCCESS.json`` 判断，不能仅根据某个 JSONL 是否存在来推断成功。
    """
    return read_jsonl(path) if path.is_file() else []


def _drop_deprecated_subject_points(analyses: list[dict[str, Any]]) -> None:
    """原地清理恢复目录中旧版 Stage 2 结构化结果的空间点字段。

    新响应解析器不会产生该字段；此兼容逻辑只避免 ``--resume`` 把历史
    ``analyses_segment_results.jsonl`` 中的旧字段再次写入 stage2.v2 产物。
    """

    for analysis in analyses:
        analysis.pop("subject_point", None)
        candidates = analysis.get("candidates", [])
        if isinstance(candidates, list):
            for candidate in candidates:
                if isinstance(candidate, dict):
                    candidate.pop("subject_point", None)


def _redacted_config(config: dict[str, Any]) -> dict[str, Any]:
    """生成可安全持久化的配置副本，将 API Key 替换为掩码。

    原始 ``config`` 不会被修改，因此实际 API 客户端仍可读取真实密钥。
    当前密钥虽然通常为 ``EMPTY``，仍统一执行脱敏，便于未来迁移到其他服务。
    """
    # 只复制顶层和 api 子对象即可，因为本函数仅修改 api.api_key。
    copy = {**config, "api": dict(config.get("api", {}))}
    if "api_key" in copy["api"]:
        copy["api"]["api_key"] = "***"
    return copy


def _request_segment(
    segment: LoadedSegment,
    work_dir: Path,
    backend: QwenBackend,
    config: dict[str, Any],
    prompt_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """完成一个分析片段的上下文构建、模型调用和响应解析。

    Args:
        segment: 从 Stage 1 加载的片段，含帧路径及绝对时间信息。
        work_dir: 当前视频的 Stage 2 临时目录，用于保存临时编码视频。
        backend: OpenAI vLLM 客户端或仅用于测试的 Mock 客户端。
        config: Stage 2 完整配置。
        prompt_config: 系统提示和用户提示模板。

    Returns:
        三元组 ``(请求描述, 结构化片段结果, 原始响应记录列表)``。请求描述
        不包含 Base64 正文，原始响应列表可能包含首次响应和格式修复重试响应。

    Raises:
        ResponseParseError: 所有格式修复重试均失败。
        ExternalToolError: 视频编码或 vLLM HTTP 调用失败。
    """
    # 文本上下文只包含经过限量的时间线，避免过长音频特征挤占模型上下文。
    context = build_multimodal_context(segment, config.get("context", {}))
    # visual_fps 同时用于提示模型理解帧间间隔，并用于 data_url 视频编码。
    visual_fps = float(config.get("video_input", {}).get("fps", 2.0))
    system_prompt, user_prompt = build_prompts(segment, context, prompt_config, visual_fps)
    # url 模式仅构造外部 URL；data_url 模式会把 Stage 1 JPEG 编为临时 MP4。
    data_payload = prepare_video_payload(segment, config.get("video_input", {}), work_dir / "encoded_segments")
    # descriptor 用于复现实验，但 payload.descriptor 有意排除了体积很大的 Base64 正文。
    requests_descriptor = {
        "schema_version": STAGE2_SCHEMA_VERSION,
        "video_id": segment.video_id,
        "segment_id": segment.segment_id,
        "segment_start_sec": segment.start_sec,
        "segment_end_sec": segment.end_sec,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "video": data_payload.descriptor,
        "context_counts": {
            "frames": context["selected_frame_count"],
            "audio_events": context["audio_event_count"],
            "audio_bins": context["audio_bin_count"],
            "asr_segments": context["asr_segment_count"],
        },
    }
    raw_records: list[dict[str, Any]] = []
    parse_retries = int(config.get("parsing", {}).get("retries", 1))
    for attempt_retry_times in range(parse_retries + 1):
        # 首次请求使用标准提示；后续请求追加最小格式修复指令，不改变视频和证据。
        prompt = user_prompt
        if attempt_retry_times:
            prompt += "\n\n上一次响应无法解析。请重新检查字段类型，并且只返回一个合法 JSON 对象。"
        # backend 只封装 OpenAI 请求，不涉及本地模型权重或 vLLM 生命周期。
        response = backend.analyze(system_prompt, prompt, data_payload.content_item,response_format=OUTPUT_SCHEMA_OPENAI)
        # 保存模型原文、finish_reason 和 token usage，便于定位解析失败或截断问题。
        raw_record = {
            "schema_version": STAGE2_SCHEMA_VERSION,
            "video_id": segment.video_id,
            "segment_id": segment.segment_id,
            "attempt_retry_times": attempt_retry_times,
            **response.to_dict(),
        }
        raw_records.append(raw_record)
        try:
            # 解析器负责清理思考标签/代码围栏、检查类型并完成相对时间映射。
            analysis = parse_response(response.text, segment, config.get("parsing", {}))
            return requests_descriptor, analysis, raw_records
        except ResponseParseError:
            # 仅响应格式问题允许重试；网络错误交给 OpenAI SDK 的 max_retries。
            if attempt_retry_times >= parse_retries:
                raise
    raise ResponseParseError("模型响应解析重试耗尽")


def process_video(
    stage1_dir: Path,
    video_id: str,
    videos_output_dir: Path,
    backend: QwenBackend,
    config: dict[str, Any],
    prompt_config: dict[str, Any],
    resume: bool,
    overwrite: bool,
) -> dict[str, Any]:
    """处理一个 Stage 1 视频目录，并提交完整的 Stage 2 视频产物。

    Args:
        stage1_dir: 某次 Stage 1 的输出根目录，不是原始视频目录。
        video_id: 当前视频标识。
        videos_output_dir: 本次 Stage 2 的 ``videos`` 输出目录。
        backend: 已初始化的 Qwen API 后端，在多个视频间复用连接池。
        config: Stage 2 模型、视频传输、解析、评分及合并配置。
        prompt_config: 外置提示词配置。
        resume: 是否从成功目录或 ``.inprogress`` 目录恢复。
        overwrite: 是否明确删除当前视频旧产物并重新请求全部片段。

    Returns:
        可直接写入批次 ``manifest.jsonl`` 的成功或跳过记录。

    目录状态约定：

    - ``videos/<id>/_SUCCESS.json``：整个视频已经完成，可由下游读取；
    - ``videos/.<id>.inprogress``：只包含部分片段，可使用 ``--resume`` 继续；
    - 正式目录存在但没有成功标记：不自动信任，需显式覆盖。
    """
    final_dir = videos_output_dir / video_id
    # 视频级恢复优先判断成功标记；命中后不会重新访问 vLLM。
    if resume and (final_dir / "_SUCCESS.json").is_file():
        return {"video_id": video_id, "status": "skipped", "reason": "already_successful"}
    work_dir = videos_output_dir / f".{video_id}.inprogress"
    if overwrite:
        # overwrite 是唯一会主动清除该视频历史正式结果和未完成结果的模式。
        shutil.rmtree(final_dir, ignore_errors=True)
        shutil.rmtree(work_dir, ignore_errors=True)
    elif final_dir.exists():
        # 防止误覆盖一份可能仍被 Stage 3 使用的正式结果。
        raise ArtifactValidationError(f"Stage 2 输出已存在，请使用 --resume 或 --overwrite: {final_dir}")
    elif work_dir.exists() and not resume:
        # 未明确恢复时不读取部分结果，避免混用不同配置下的大模型响应。
        raise ArtifactValidationError(f"发现未完成目录，请使用 --resume 或 --overwrite: {work_dir}")
    work_dir.mkdir(parents=True, exist_ok=True)

    # 恢复运行只加载临时目录；完整视频已经在上面的 _SUCCESS 分支直接跳过。
    requests = _load_existing(work_dir / "requests.jsonl") if resume else []
    raw_responses = _load_existing(work_dir / "raw_responses.jsonl") if resume else []
    analyses = _load_existing(work_dir / "analyses_segment_results.jsonl") if resume else []
    # 恢复运行允许读取旧临时结果，但后续落盘前必须升级为不含空间点的 v2 契约。
    _drop_deprecated_subject_points(analyses)
    # segment_results 是判断某个片段是否已完成的唯一依据。
    completed_ids = {int(row["segment_id"]) for row in analyses}
    # 加载 Stage 1 的 metadata、segments、sample_map、音频事件和 ASR 文件。
    video = load_stage1_video(stage1_dir, video_id)
    # 保存 segment_id 到 LoadedSegment 的映射，供候选评分阶段读取音频证据。
    segment_by_id: dict[int, LoadedSegment] = {}
    with Timer() as timer:
        for segment_row in video.segments:
            # max_frames 在这里限制单次模型请求的视觉规模；默认 24 秒×2 FPS 不超过 48 帧。
            segment = load_segment(video, segment_row, int(config.get("context", {}).get("max_frames", 64)))
            segment_by_id[segment.segment_id] = segment
            if segment.segment_id in completed_ids:
                # 已持久化的片段不会再次编码视频或调用模型。
                continue
            requests_descriptor, analysis, raw_response = _request_segment(
                segment, work_dir, backend, config, prompt_config
            )
            # 请求描述和原始响应可以按配置关闭，但结构化片段结果始终保存。
            if bool(config.get("runtime", {}).get("persist_request_descriptors", True)):
                requests.append(requests_descriptor)
            if bool(config.get("runtime", {}).get("persist_raw_responses", True)):
                raw_responses.extend(raw_response)
            analyses.append(analysis)
            completed_ids.add(segment.segment_id)
            # 每个模型调用后立即原子重写 JSONL，长视频中断时可从下一片段继续。
            write_jsonl(work_dir / "requests.jsonl", requests)
            write_jsonl(work_dir / "raw_responses.jsonl", raw_responses)
            write_jsonl(work_dir / "analyses_segment_results.jsonl", analyses)

        # 恢复运行时，已完成片段同样需要 LoadedSegment 才能重新计算融合分数。
        for segment_row in video.segments:
            segment_id = int(segment_row["segment_id"])
            if segment_id not in segment_by_id:
                segment_by_id[segment_id] = load_segment(
                    video, segment_row, int(config.get("context", {}).get("max_frames", 64))
                )
        # 一个模型判定现在可以包含多个高光；逐项展开后再做分数融合。
        raw_candidates = []
        for analysis in analyses:
            segment = segment_by_id[int(analysis["segment_id"])]
            candidate_analyses = analysis.get("candidates")
            if not isinstance(candidate_analyses, list):
                raise ArtifactValidationError("candidates格式错误：{}".format(candidate_analyses))
            for candidate_analysis in candidate_analyses:
                candidate = build_candidate(
                    candidate_analysis,
                    segment,
                    config.get("scoring", {}),
                )
                if candidate is not None:
                    raw_candidates.append(candidate)
        # 重叠分析窗口可能描述同一事件，统一扩展并合并为不重叠候选区间。
        candidates = merge_candidates(
            video_id,
            raw_candidates,
            float(video.metadata["duration_sec"]),
            config.get("merging", {}),
        )
        # subject_hints 只保留主体文本与来源，不包含不可靠的粗采样空间坐标；
        # 后续 Stage 3.5 可用这些语义提示在精确区间内生成 subject_point。
        subject_hints = build_subject_hints(candidates)
        # 即使没有高光，也必须写出合法的空 candidates.jsonl 和 subject_hints.jsonl。
        write_jsonl(work_dir / "requests.jsonl", requests)
        write_jsonl(work_dir / "raw_responses.jsonl", raw_responses)
        write_jsonl(work_dir / "analyses_segment_results.jsonl", analyses)
        write_jsonl(work_dir / "candidates.jsonl", candidates)
        write_jsonl(work_dir / "subject_hints.jsonl", subject_hints)
        # 完整性校验通过后才允许生成视频级 _SUCCESS.json。
        validation = validate_stage2_artifacts(work_dir)
    # 统计模型分析片段数、阈值前候选数和合并后候选数，用于后续误差分析。
    success = success_record(
        video_id,
        elapsed_sec=timer.elapsed_sec,
        segment_count=len(analyses),
        raw_candidate_count=len(raw_candidates),
        candidate_count=len(candidates),
        validation=validation,
    )
    # 成功标记写入临时目录后，才执行目录级提交。
    write_json(work_dir / "_SUCCESS.json", success)
    if bool(config.get("video_input", {}).get("keep_encoded_video", False)) is False:
        # Base64 模式默认不保留中间 MP4；请求描述和模型结果仍完整保留。
        shutil.rmtree(work_dir / "encoded_segments", ignore_errors=True)
    work_dir.rename(final_dir)
    return success


def run_stage2(
    stage1_dir: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
    prompt_config: dict[str, Any],
    video_ids: set[str] | None = None,
    limit: int | None = None,
    resume: bool = False,
    overwrite: bool = False,
    strict: bool = False,
    logger: Any = None,
) -> dict[str, Any]:
    """批量执行 Stage 2，是 CLI 和总流水线共用的 Python API。

    单视频异常默认记录后继续，使一个损坏片段或一次 API 错误不会阻断整个测试
    集；``strict=True`` 时会在写出当前失败记录后立即抛出异常。

    Args:
        stage1_dir: 上游 Stage 1 输出根目录，内部应包含 ``videos/<id>``。
        output_dir: 当前 Stage 2 输出根目录。
        config: API、生成、视频载荷、解析、评分、合并及运行配置。
        prompt_config: 提示词模板配置。
        video_ids: 可选视频白名单。
        limit: 应用白名单后的可选处理数量上限。
        resume: 是否复用已有视频成功结果或片段级临时结果。
        overwrite: 是否重算所选视频。
        strict: 单视频失败时是否终止批次。
        logger: 可选结构化日志对象。

    Returns:
        包含模型、接口地址、输入模式和成功/跳过/失败数量的批次摘要。
    """
    # 固化上下游绝对路径，避免运行期间工作目录变化导致读取不同产物。
    stage1_root = Path(stage1_dir).resolve()
    stage2_root = Path(output_dir).resolve()
    videos_output = stage2_root / "videos"
    videos_output.mkdir(parents=True, exist_ok=True)
    # 只选择 Stage 1 已带视频级成功标记的目录。
    available_ids = list_stage1_video_ids(stage1_root)
    selected_ids = [value for value in available_ids if not video_ids or value in video_ids]
    if limit is not None:
        selected_ids = selected_ids[:limit]
    if not selected_ids:
        raise ArtifactValidationError("筛选后没有可处理的 Stage 1 视频")
    # backend 在整个批次中只初始化一次，以复用 OpenAI 客户端连接池。
    backend = build_backend(config)
    if bool(config.get("api", {}).get("healthcheck_on_start", True)):
        # 健康检查同时确认服务可达且配置模型名确实由 vLLM 暴露。
        backend.healthcheck()
    manifest: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    # 保存配置和提示词快照；API Key 在落盘前被统一脱敏。
    write_json(stage2_root / "resolved_config.json", _redacted_config(config))
    write_json(stage2_root / "resolved_prompts.json", prompt_config)
    # run_manifest 记录上游目录、模型、服务地址和视频载荷方式，便于复现实验。
    run_info = {
        "schema_version": STAGE2_SCHEMA_VERSION,
        "started_at": utc_now_iso(),
        "stage1_dir": str(stage1_root),
        "config_sha256": mapping_sha256(config),
        "model": config.get("api", {}).get("model"),
        "base_url": config.get("api", {}).get("base_url"),
        "video_input_mode": config.get("video_input", {}).get("mode"),
        "selected_video_count": len(selected_ids),
    }
    write_json(stage2_root / "run_manifest.json", run_info)
    for position, video_id in enumerate(selected_ids, start=1):
        if logger:
            logger.info("[%d/%d] Stage 2 处理 video_id=%s", position, len(selected_ids), video_id)
        try:
            # 每个视频拥有独立临时目录和异常边界。
            record = process_video(
                stage1_root, video_id, videos_output, backend, config, prompt_config, resume, overwrite
            )
            manifest.append(record)
        except Exception as error:
            # 保留完整堆栈，网络、编码、解析和契约错误均可区分定位。
            failure = failure_record(video_id, error)
            failure["traceback"] = traceback.format_exc()
            failures.append(failure)
            manifest.append(failure)
            if logger:
                logger.exception("Stage 2 video_id=%s 处理失败", video_id)
            if strict:
                # 严格模式退出前仍持久化已经得到的批次状态。
                write_jsonl(stage2_root / "manifest.jsonl", manifest)
                write_jsonl(stage2_root / "failures.jsonl", failures)
                raise
        # 每完成一个视频就更新清单，外部监控无需等待整批结束。
        write_jsonl(stage2_root / "manifest.jsonl", manifest)
        write_jsonl(stage2_root / "failures.jsonl", failures)
    # 只有零失败批次才生成阶段级成功标记；单视频成功标记不受其他视频影响。
    summary = {
        **run_info,
        "finished_at": utc_now_iso(),
        "success_count": sum(row["status"] == "success" for row in manifest),
        "skipped_count": sum(row["status"] == "skipped" for row in manifest),
        "failure_count": len(failures),
    }
    write_json(stage2_root / "run_manifest.json", summary)
    if not failures:
        write_json(stage2_root / "_SUCCESS.json", summary)
    return summary

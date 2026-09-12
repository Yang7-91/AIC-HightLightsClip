"""Stage 4 主流水线：主体跟踪、目标画幅构图、轨迹优化与结果持久化。

本模块负责把前序阶段的“高光时间区间”转换成比赛所需的逐帧构图框。数据依赖被
刻意限制为以下两类，Stage 4 不读取 Stage 2：

1. Stage 3 ``refined_intervals.jsonl``：高光的左闭右开帧区间，以及主体描述、
   可选主体点等已经随候选向后传递的语义信息；
2. Stage 1 ``metadata.json`` 和 ``scenes.jsonl``：只有源视频路径、画面尺寸、
   ``targetRatioWH``、总帧数和镜头边界等空间处理不可替代的信息。

对每个 Stage 3 区间，处理链路为：

``主体逐帧跟踪 -> 按镜头切分 -> 构图候选生成 -> 构图代价计算``
``-> 动态规划选路 -> 时序平滑 -> 整数化与边界校验 -> 持久化``

正式结果写入 ``crops.jsonl``；``tracks.jsonl`` 和 ``diagnostics.jsonl`` 是可解释的
中间产物，便于定位主体漂移、降级和跨镜头问题。单视频先写隐藏的工作目录，只有在
全部文件通过校验并写入 ``_SUCCESS.json`` 后才重命名为正式目录，避免下游读取半成品。
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
from video_highlight.contracts.schema_versions import STAGE4_SCHEMA_VERSION

from .boundary_limiter import finalize_bbox
from .composition_scorer import composition_cost
from .crop_candidates import generate_crop_candidates
from .subject_tracker import CenterSubjectTracker, SubjectTracker, TrackPoint, build_subject_tracker
from .trajectory_optimizer import optimize_trajectory
from .trajectory_smoother import smooth_trajectory
from .validators import list_stage3_video_ids, load_video_inputs, validate_config, validate_crops, validate_stage4_artifacts


def _frame_size(metadata: dict[str, Any]) -> tuple[int, int]:
    """返回构图使用的 ``(宽, 高)``，优先采用 Stage 1 校正旋转后的显示尺寸。

    ``width/height`` 是视频流编码尺寸；带旋转元数据的视频，其真实观看方向可能由
    ``display_width/display_height`` 表示。Stage 1 已经统一探测了这些字段，因此
    Stage 4 只消费其结果，不重复解析视频流元数据。
    """

    return int(metadata.get("display_width", metadata["width"])), int(metadata.get("display_height", metadata["height"]))


def _target_ratio(metadata: dict[str, Any]) -> tuple[float, float]:
    """读取并校验目标画幅比例，返回 ``(target_w, target_h)``。

    比赛输出只保存 ``[x, y, w]``，框高由 ``w * target_h / target_w`` 推导，因此
    比例必须是两个正数。缺省值 ``16:9`` 仅用于兼容没有该字段的旧 Stage 1 产物。
    """

    value = metadata.get("targetRatioWH", [16, 9])
    # 显式拒绝标量、字典或长度异常的列表，避免后续产生难定位的除法/解包错误。
    if not isinstance(value, list) or len(value) != 2:
        raise ArtifactValidationError("metadata.targetRatioWH 必须是 [w,h]")
    ratio = float(value[0]), float(value[1])
    if ratio[0] <= 0 or ratio[1] <= 0:
        raise ArtifactValidationError("metadata.targetRatioWH 必须为正数")
    return ratio


def _motion(points: list[TrackPoint], index: int) -> tuple[float, float]:
    """估计当前主体相对上一帧的中心位移 ``(dx, dy)``。

    该位移不是新的跟踪结果，而是构图先验：候选框生成器会沿运动方向预留空间，
    使奔跑的人物或快速移动的物体不容易紧贴画面边缘。首帧没有历史信息，返回零。
    """

    if index == 0:
        return 0.0, 0.0
    previous, current = points[index - 1].subject_box, points[index].subject_box
    # xyxy 框中心为 ((x1+x2)/2, (y1+y2)/2)，下面直接计算两个中心之差。
    return ((current[0] + current[2] - previous[0] - previous[2]) * 0.5, (current[1] + current[3] - previous[1] - previous[3]) * 0.5)


def _plan_interval(
    interval: dict[str, Any],
    points: list[TrackPoint],
    frame_size: tuple[int, int],
    target_ratio: tuple[float, float],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """为一个不跨镜头的连续帧段生成构图轨迹和主体轨迹记录。

    Parameters
    ----------
    interval:
        Stage 3 区间或由 :func:`_plan_across_scenes` 派生出的镜头子区间。帧范围采用
        ``[start_frame, end_frame)`` 左闭右开语义。
    points:
        跟踪器输出的逐帧主体框，必须与区间中的每一帧严格一一对应。
    frame_size:
        原视频显示方向的 ``(宽, 高)``。
    target_ratio:
        目标构图比例 ``(宽比例, 高比例)``。
    config:
        完整 Stage 4 配置；本函数分别读取候选、构图、优化和平滑子配置。

    Returns
    -------
    tuple
        ``(crops, tracks)``。前者是 Stage 5 使用的逐帧构图，后者保留主体框与
        跟踪置信度，用于诊断构图错误究竟来自主体跟踪还是轨迹选择。
    """

    # 动态规划要求每一帧都有一个候选状态集合。先检查跟踪结果是否完整、顺序是否
    # 精确一致；如果在这里容忍缺帧，最终 frame 与 bbox 很容易发生静默错位。
    expected_frames = list(range(int(interval["start_frame"]), int(interval["end_frame"])))
    if [point.frame for point in points] != expected_frames:
        raise ArtifactValidationError(f"主体轨迹没有逐帧覆盖区间: {interval['interval_id']}")

    candidates_by_frame: list[list[tuple[float, float, float, float]]] = []
    local_costs: list[list[float]] = []
    for index, point in enumerate(points):
        # 每个候选均为浮点 xywh，且已经满足目标比例和画面边界。主体框、当前运动
        # 方向、多尺度及多偏移共同决定本帧可选的构图状态。
        candidates = generate_crop_candidates(point.subject_box, frame_size, target_ratio, _motion(points, index), config.get("crop_candidates", {}))
        candidates_by_frame.append(candidates)
        # local_cost 只衡量单帧构图质量，例如主体覆盖、中心性和裁剪尺度；相邻帧
        # 中心/尺度变化的代价由 optimize_trajectory 在状态转移时计算。
        local_costs.append([composition_cost(crop, point.subject_box, frame_size, config.get("composition", {})) for crop in candidates])

    # 动态规划从整段角度选择总代价最低的候选序列，避免逐帧独立取最优导致左右跳动。
    optimized = optimize_trajectory(candidates_by_frame, local_costs, config.get("optimizer", {}))
    # 动态规划输出仍是离散候选；EMA 和最大中心步长限制进一步消除细小抖动。
    smoothed = smooth_trajectory(optimized, frame_size, target_ratio, config.get("smoothing", {}))

    crops: list[dict[str, Any]] = []
    tracks: list[dict[str, Any]] = []
    for point, crop in zip(points, smoothed, strict=True):
        # finalize_bbox 在浮点限界后统一取整，并在取整后再次限界，输出比赛要求的
        # [x, y, w]。框高不重复保存，由 targetRatioWH 在 Stage 5/评测端推导。
        crops.append({
            "schema_version": STAGE4_SCHEMA_VERSION,
            "video_id": interval["video_id"],
            "interval_id": interval["interval_id"],
            "frame": point.frame,
            "bboxes": finalize_bbox(crop, frame_size, target_ratio),
            "track_confidence": point.confidence,
            "track_source": point.source,
        })
        # 同步保存原始主体 xyxy，而不是只保留最终构图框。若最终构图不理想，便可
        # 区分是“跟错主体”还是“主体正确但构图策略不合适”。
        tracks.append({
            "schema_version": STAGE4_SCHEMA_VERSION,
            "video_id": interval["video_id"],
            "interval_id": interval["interval_id"],
            "frame": point.frame,
            "subject_box_xyxy": [float(value) for value in point.subject_box],
            "confidence": point.confidence,
            "source": point.source,
        })
    return crops, tracks


def _planning_spans(interval: dict[str, Any], scenes: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """按 Stage 1 镜头边界切开区间，返回若干左闭右开的规划子区间。

    一个 Stage 3 高光区间可能覆盖多个镜头。硬切前后的主体位置通常没有连续关系，
    若放在同一次动态规划和平滑中处理，会把前一镜头的构图惯性错误带入后一镜头。
    因此仅把严格落在高光内部的 ``scene.start_frame`` 作为切点；区间端点无需重复。
    """

    start, end = int(interval["start_frame"]), int(interval["end_frame"])
    cuts = sorted(
        {
            int(scene["start_frame"])
            for scene in scenes
            if scene.get("start_frame") is not None and start < int(scene["start_frame"]) < end
        }
    )
    boundaries = [start, *cuts, end]
    # 两个序列本来就相差一个元素；相邻配对得到 [start, cut)、[cut, end)。
    return list(zip(boundaries[:-1], boundaries[1:], strict=True))


def _plan_across_scenes(
    interval: dict[str, Any],
    points: list[TrackPoint],
    scenes: list[dict[str, Any]],
    frame_size: tuple[int, int],
    target_ratio: tuple[float, float],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """逐镜头独立规划构图，再按原始帧顺序拼回一个 Stage 3 区间。

    跟踪器返回的 ``points`` 是相对整个高光区间的连续列表，而镜头边界是原视频绝对
    帧号。因此切片前先减去 ``interval_start`` 转为列表下标。子区间只临时覆盖起止帧，
    ``video_id``、``interval_id`` 等来源字段保持不变，保证输出仍可追溯到原区间。

    第三个返回值是实际规划的镜头子段数量，会写入诊断文件；大于 1 表示该高光曾在
    内部镜头边界处断开优化。
    """

    interval_start = int(interval["start_frame"])
    crops: list[dict[str, Any]] = []
    tracks: list[dict[str, Any]] = []
    spans = _planning_spans(interval, scenes)
    for span_start, span_end in spans:
        # 绝对帧号转换为 points 的零基切片下标。end 保持左闭右开约定。
        local_start = span_start - interval_start
        local_end = span_end - interval_start
        span_interval = {**interval, "start_frame": span_start, "end_frame": span_end}
        span_crops, span_tracks = _plan_interval(
            span_interval,
            points[local_start:local_end],
            frame_size,
            target_ratio,
            config,
        )
        crops.extend(span_crops)
        tracks.extend(span_tracks)
    return crops, tracks, len(spans)


def _center_points(interval: dict[str, Any], frame_size: tuple[int, int], tracking_config: dict[str, Any]) -> list[TrackPoint]:
    """构造覆盖整个区间的中心主体轨迹，作为确定性的最后降级结果。

    CenterSubjectTracker 不读取 ``video_path`` 或镜头列表，所以传入空 Path 和空列表。
    它仍逐帧生成 TrackPoint，确保后续候选生成、优化、校验和输出流程与正常路径一致，
    而不是在异常分支中绕开核心数据契约。
    """

    tracker = CenterSubjectTracker(tracking_config)
    return tracker.track(Path(), interval, frame_size, [])


def process_video(
    stage1_dir: Path,
    stage3_dir: Path,
    video_id: str,
    videos_output_dir: Path,
    tracker: SubjectTracker,
    config: dict[str, Any],
    resume: bool,
    overwrite: bool,
) -> dict[str, Any]:
    """处理单个视频并以目录为单位提交 Stage 4 产物。

    单视频的所有临时结果先写入 ``videos/.<video_id>.inprogress``。只有跟踪、构图、
    输出校验和成功标记全部完成，才把它重命名成 ``videos/<video_id>``。这样进程中断时
    不会留下看似完整的正式目录，``--resume`` 也只会跳过带 ``_SUCCESS.json`` 的视频。

    ``overwrite`` 会清除该视频以前的正式目录和未完成工作目录；未指定时，发现同名
    目录会报错，从而避免意外覆盖已有实验结果。
    """

    final_dir = videos_output_dir / video_id
    # resume 的判定以成功标记为准，而不是仅检查目录是否存在。
    if resume and (final_dir / "_SUCCESS.json").is_file():
        return {"video_id": video_id, "status": "skipped", "reason": "already_successful"}
    work_dir = videos_output_dir / f".{video_id}.inprogress"

    # overwrite 和“输出已存在时报错”是互斥语义。清理范围严格限制在当前 video_id。
    if overwrite:
        shutil.rmtree(final_dir, ignore_errors=True)
        shutil.rmtree(work_dir, ignore_errors=True)
    elif final_dir.exists():
        raise ArtifactValidationError(f"Stage 4 输出已存在，请使用 --resume 或 --overwrite: {final_dir}")
    elif work_dir.exists():
        raise ArtifactValidationError(f"Stage 4 存在未完成目录，请使用 --overwrite: {work_dir}")
    work_dir.mkdir(parents=True, exist_ok=True)

    # load_video_inputs 是 Stage 4 唯一的上游读取入口：
    # - Stage 1：metadata.json、scenes.jsonl；
    # - Stage 3：refined_intervals.jsonl；
    # 这里没有 Stage 2 路径，避免 Stage 4 绕过 Stage 3 的最终边界契约。
    metadata, scenes, intervals = load_video_inputs(stage1_dir, stage3_dir, video_id)
    frame_size = _frame_size(metadata)
    target_ratio = _target_ratio(metadata)
    # 源视频不复制到项目中，直接使用 Stage 1 已持久化的绝对 source_path。
    video_path = Path(str(metadata["source_path"]))

    # 三类逐视频内存结果最终分别落到 crops、tracks 和 diagnostics 文件。
    crops: list[dict[str, Any]] = []
    tracks: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    fallback_count = 0
    error_policy = str(config["runtime"].get("interval_error_policy", "center"))

    # Timer 覆盖实际区间处理和产物校验，不包含最终目录重命名后的批处理汇总。
    with Timer() as timer:
        for interval in intervals:
            try:
                # 正常路径：指定后端（OpenCV/SAM2/center）先产生逐帧主体框，再按
                # Stage 1 镜头边界独立完成构图优化，禁止跨硬切镜头平滑。
                points = tracker.track(video_path, interval, frame_size, scenes)
                interval_crops, interval_tracks, planning_span_count = _plan_across_scenes(
                    interval, points, scenes, frame_size, target_ratio, config
                )
                status = "tracked"
            except Exception as error:
                # interval_error_policy=error 时保留原异常，让 strict/批处理层决定是否
                # 终止整批；center 时只降级当前区间，其他高光区间仍使用正常跟踪结果。
                if error_policy != "center":
                    raise
                fallback_count += 1
                # 降级仍然经过同一套候选、DP、平滑、限界和校验流程，从而保证输出契约
                # 与正常跟踪路径完全一致。
                points = _center_points(interval, frame_size, config.get("tracking", {}))
                interval_crops, interval_tracks, planning_span_count = _plan_across_scenes(
                    interval, points, scenes, frame_size, target_ratio, config
                )
                status = "center_fallback"
                # 保存异常类型和消息但不中断本视频，方便后续统计哪些区间曾经降级。
                diagnostics.append({"schema_version": STAGE4_SCHEMA_VERSION, "video_id": video_id, "interval_id": interval["interval_id"], "status": status, "planning_span_count": planning_span_count, "error_type": type(error).__name__, "message": str(error)})

            # 每个 Stage 3 区间不会与其他区间重叠；此处先累积，循环结束后统一排序校验。
            crops.extend(interval_crops)
            tracks.extend(interval_tracks)
            if status == "tracked":
                # 平均置信度和镜头子段数量是区间级诊断值，不参与最终比赛输出。
                diagnostics.append({"schema_version": STAGE4_SCHEMA_VERSION, "video_id": video_id, "interval_id": interval["interval_id"], "status": status, "frame_count": len(interval_crops), "planning_span_count": planning_span_count, "mean_track_confidence": sum(point.confidence for point in points) / max(1, len(points))})
        # Stage 3 保证区间不重叠；这里仍按帧排序并显式拒绝任何重复帧。
        crops.sort(key=lambda row: int(row["frame"]))
        tracks.sort(key=lambda row: int(row["frame"]))
        # validate_crops 检查帧范围、严格升序、数值有限性、目标比例换算和画面边界。
        # 先校验内存结果，避免把明显非法的正式构图写给 Stage 5。
        validate_crops(crops, metadata)

        # 即使 intervals 为空，也会写出三个合法空 JSONL；无高光视频仍需形成完整产物。
        write_jsonl(work_dir / "crops.jsonl", crops)
        write_jsonl(work_dir / "tracks.jsonl", tracks)
        write_jsonl(work_dir / "diagnostics.jsonl", diagnostics)
        # 文件级校验确认三个必需产物都真实存在。
        validation = validate_stage4_artifacts(work_dir)

    # _SUCCESS.json 必须最后写入。它既是下游可消费标记，也是 --resume 的判定依据。
    success = success_record(video_id, elapsed_sec=timer.elapsed_sec, backend=config["tracking"].get("backend", "opencv"), interval_count=len(intervals), prediction_frame_count=len(crops), fallback_count=fallback_count, validation=validation)
    write_json(work_dir / "_SUCCESS.json", success)
    # 同一文件系统内目录重命名把已验证的工作目录一次性提交为正式视频产物。
    work_dir.rename(final_dir)
    return success


def run_stage4(
    stage1_dir: str | Path,
    stage3_dir: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
    video_ids: set[str] | None = None,
    limit: int | None = None,
    resume: bool = False,
    overwrite: bool = False,
    strict: bool = False,
    logger: Any = None,
) -> dict[str, Any]:
    """批量执行 Stage 4，并持续写入可恢复的运行清单。

    Parameters
    ----------
    stage1_dir, stage3_dir:
        同一数据批次的 Stage 1 与 Stage 3 输出根目录。Stage 2 不属于本函数契约。
    output_dir:
        本次 Stage 4 运行目录，其中每个视频写入 ``videos/<video_id>``。
    config:
        已加载的 Stage 4 配置映射。
    video_ids:
        可选视频 ID 白名单；``None`` 表示处理 Stage 3 中全部成功视频。
    limit:
        在 ID 筛选后最多处理多少个视频，便于冒烟测试。
    resume:
        跳过已经具有单视频 ``_SUCCESS.json`` 的结果。
    overwrite:
        允许覆盖当前视频已有的正式/未完成输出。
    strict:
        任一视频失败后立即抛出；关闭时记录失败并继续处理其他视频。
    logger:
        可选标准日志对象。

    Returns
    -------
    dict
        运行摘要，包含选中视频数、成功/跳过/失败数、后端、路径和配置哈希。
    """

    # 在创建任何结果之前校验配置结构，尽早暴露配置字段缺失或错误枚举值。
    validate_config(config)
    stage1_root, stage3_root, stage4_root = Path(stage1_dir).resolve(), Path(stage3_dir).resolve(), Path(output_dir).resolve()
    videos_output = stage4_root / "videos"
    videos_output.mkdir(parents=True, exist_ok=True)

    # Stage 3 是高光区间的权威来源，所以任务列表也从 Stage 3 成功目录枚举；Stage 1
    # 缺失会在处理对应视频时由 load_video_inputs 给出明确错误。
    available = list_stage3_video_ids(stage3_root)
    selected = [video_id for video_id in available if not video_ids or video_id in video_ids]
    # limit 在 video_ids 过滤后应用，使 --video-id 与 --limit 组合时含义稳定。
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ArtifactValidationError("筛选后没有可处理的 Stage 3 视频")

    # 跟踪器通常包含模型或后端状态，整批只构造一次，避免逐视频重复加载 SAM2 权重。
    tracker = build_subject_tracker(config["tracking"])
    # 配置哈希与解析后的配置一起持久化，便于确认不同实验是否真正使用同一参数。
    run_info = {"schema_version": STAGE4_SCHEMA_VERSION, "started_at": utc_now_iso(), "stage1_dir": str(stage1_root), "stage3_dir": str(stage3_root), "tracking_backend": config["tracking"].get("backend", "opencv"), "config_sha256": mapping_sha256(config), "selected_video_count": len(selected)}
    write_json(stage4_root / "resolved_config.json", deepcopy(config))
    write_json(stage4_root / "run_manifest.json", run_info)
    manifest: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for position, video_id in enumerate(selected, start=1):
        if logger:
            logger.info("[%d/%d] Stage 4 处理 video_id=%s", position, len(selected), video_id)
        try:
            record = process_video(stage1_root, stage3_root, video_id, videos_output, tracker, config, resume, overwrite)
        except Exception as error:
            # 非 strict 模式把异常转为结构化记录并继续下一视频；traceback 单独保留，
            # 既方便自动汇总，也能在无需复现的情况下定位具体代码路径。
            record = failure_record(video_id, error)
            record["traceback"] = traceback.format_exc()
            failures.append(record)
            if logger:
                logger.exception("Stage 4 video_id=%s 处理失败", video_id)
            if strict:
                # 立即退出前先刷新清单，保证失败现场不会因异常抛出而丢失。
                manifest.append(record)
                write_jsonl(stage4_root / "manifest.jsonl", manifest)
                write_jsonl(stage4_root / "failures.jsonl", failures)
                raise
        manifest.append(record)
        # 每处理完一个视频就重写当前快照。进程被中断时，已完成的视频状态仍可恢复。
        write_jsonl(stage4_root / "manifest.jsonl", manifest)
        write_jsonl(stage4_root / "failures.jsonl", failures)
    # 顶层摘要同时统计本次成功和 resume 跳过的数量，失败数以 failures 为准。
    summary = {**run_info, "finished_at": utc_now_iso(), "success_count": sum(row["status"] == "success" for row in manifest), "skipped_count": sum(row["status"] == "skipped" for row in manifest), "failure_count": len(failures)}
    write_json(stage4_root / "run_manifest.json", summary)
    # 只有整批无失败时才发布顶层成功标记；单视频成功标记仍各自保留。
    if not failures:
        write_json(stage4_root / "_SUCCESS.json", summary)
    return summary

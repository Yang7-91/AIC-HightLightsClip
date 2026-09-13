"""Stage 5：从 Stage 4 逐帧构图确定性生成最终比赛 JSONL。

Stage 4 的 ``crops.jsonl`` 是预测内容的唯一来源。Stage 5 不回读 Stage 2/3；
只用原始输入索引确定视频行顺序和目标比例，并用 Stage 1 元数据复核帧号与空间边界。
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from video_highlight.common.atomic_io import read_jsonl, write_json
from video_highlight.common.exceptions import ArtifactValidationError
from video_highlight.common.hashing import file_sha256, mapping_sha256
from video_highlight.common.manifest import utc_now_iso
from video_highlight.common.runtime import Timer
from video_highlight.contracts.schema_versions import STAGE5_SCHEMA_VERSION

from .jsonl_writer import write_submission_jsonl
from .prediction_assembler import assemble_submission_row
from .report_writer import write_reports
from .submission_validator import (
    load_input_index,
    load_stage1_metadata,
    validate_submission_file,
    validate_submission_rows,
)


def _validate_config(config: dict[str, Any]) -> None:
    for key in ("output", "validation", "quantization"):
        if not isinstance(config.get(key), dict):
            raise ArtifactValidationError(f"Stage 5 配置缺少对象字段: {key}")
    filename = str(config["output"].get("filename", "submission.jsonl"))
    if not filename or Path(filename).name != filename or not filename.lower().endswith(".jsonl"):
        raise ArtifactValidationError("output.filename 必须是不含目录的 .jsonl 文件名")
    if str(config["quantization"].get("mode", "round")) not in {"round", "floor"}:
        raise ArtifactValidationError("quantization.mode 只能是 round 或 floor")


def _require_stage_success(root: Path, stage_name: str, required: bool) -> None:
    if required and not (root / "_SUCCESS.json").is_file():
        raise ArtifactValidationError(f"{stage_name} 顶层缺少 _SUCCESS.json: {root}")


def _load_stage4_crops(stage4_dir: Path, video_id: str, require_success: bool) -> list[dict[str, Any]]:
    video_dir = stage4_dir / "videos" / video_id
    if require_success and not (video_dir / "_SUCCESS.json").is_file():
        raise ArtifactValidationError(f"Stage 4 视频没有成功标记: {video_dir}")
    crops_path = video_dir / "crops.jsonl"
    if not crops_path.is_file():
        raise ArtifactValidationError(f"Stage 4 缺少构图结果: {crops_path}")
    return read_jsonl(crops_path)


def run_stage5(
    input_index: str | Path,
    stage1_dir: str | Path,
    stage4_dir: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
    overwrite: bool = False,
    logger: Any = None,
    video_id: str | None = None,
) -> dict[str, Any]:
    """生成、复读并严格校验提交文件。

    ``video_id`` 为空时保持比赛正式提交行为，按完整输入索引逐行导出。指定
    ``video_id`` 时进入单视频测试模式：先从索引中选出唯一对应行，再加载该视频
    的 Stage 1/4 产物。因此其他视频尚未产生 Stage 4 结果也不会阻塞测试导出。
    """

    _validate_config(config)
    index_path = Path(input_index).resolve()
    stage1_root = Path(stage1_dir).resolve()
    stage4_root = Path(stage4_dir).resolve()
    stage5_root = Path(output_dir).resolve()
    stage5_root.mkdir(parents=True, exist_ok=True)

    output_config = config["output"]
    validation_config = config["validation"]
    quantization_config = config["quantization"]
    submission_path = stage5_root / str(output_config.get("filename", "submission.jsonl"))
    success_path = stage5_root / "_SUCCESS.json"
    if submission_path.exists() and not overwrite:
        raise ArtifactValidationError(f"Stage 5 提交文件已存在，请使用 --overwrite: {submission_path}")
    if overwrite:
        # 只清理 Stage 5 自己管理的固定文件，不删除日志或其他实验产物。
        for target in (
            submission_path,
            success_path,
            stage5_root / "validation_report.json",
            stage5_root / "run_manifest.json",
            stage5_root / "resolved_config.json",
        ):
            target.unlink(missing_ok=True)

    _require_stage_success(
        stage1_root,
        "Stage 1",
        video_id is None and bool(validation_config.get("require_stage1_run_success", True)),
    )
    _require_stage_success(
        stage4_root,
        "Stage 4",
        video_id is None and bool(validation_config.get("require_stage4_run_success", True)),
    )

    # 正式模式保持完整索引顺序；测试模式仍以索引为权威来源，只截取指定视频行。
    all_index_rows = load_input_index(index_path)
    if video_id is None:
        index_rows = all_index_rows
    else:
        requested_video_id = str(video_id)
        index_rows = [
            row for row in all_index_rows
            if str(row.get("video_id")) == requested_video_id
        ]
        if not index_rows:
            raise ArtifactValidationError(
                f"--video-id={requested_video_id} 不存在于输入索引: {index_path}"
            )
        if len(index_rows) != 1:
            raise ArtifactValidationError(
                f"输入索引包含重复 video_id={requested_video_id}，无法单视频导出"
            )
    metadata_by_id = load_stage1_metadata(
        stage1_root,
        index_rows,
        require_success=bool(validation_config.get("require_stage1_video_success", True)),
    )
    strict_fields = bool(validation_config.get("strict_fields", True))
    require_stage4_video_success = bool(
        validation_config.get("require_stage4_video_success", True)
    )
    quantization_mode = str(quantization_config.get("mode", "round"))

    rows: list[dict[str, Any]] = []
    repaired_bbox_count = 0
    started_at = utc_now_iso()
    with Timer() as timer:
        for position, index_row in enumerate(index_rows, start=1):
            video_id = str(index_row["video_id"])
            if logger and (position == 1 or position % 20 == 0 or position == len(index_rows)):
                logger.info("[%d/%d] Stage 5 组装 video_id=%s", position, len(index_rows), video_id)
            crop_rows = _load_stage4_crops(
                stage4_root, video_id, require_stage4_video_success
            )
            ratio_values = index_row["targetRatioWH"]
            target_ratio = (int(ratio_values[0]), int(ratio_values[1]))
            row, row_stats = assemble_submission_row(
                video_id,
                target_ratio,
                crop_rows,
                metadata_by_id[video_id],
                quantization_mode,
            )
            rows.append(row)
            repaired_bbox_count += row_stats["repaired_bbox_count"]

        # 先验证内存对象，再原子写文件；避免已知非法结果落盘。
        in_memory_validation = validate_submission_rows(
            rows, index_rows, metadata_by_id, strict_fields
        )
        write_submission_jsonl(submission_path, rows)
        # 重新从磁盘逐行解析，覆盖编码、换行和 JSON 序列化层面的错误。
        file_validation = validate_submission_file(
            submission_path, index_rows, metadata_by_id, strict_fields
        )
        if file_validation != in_memory_validation:
            raise ArtifactValidationError("提交文件复读校验结果与内存校验不一致")

    validation_report = {
        "schema_version": STAGE5_SCHEMA_VERSION,
        "status": "valid",
        "export_scope": "full_index" if video_id is None else "single_video_test",
        "video_id_filter": None if video_id is None else str(video_id),
        **file_validation,
        "repaired_bbox_count": repaired_bbox_count,
        "submission_sha256": file_sha256(submission_path),
        "submission_size_bytes": submission_path.stat().st_size,
    }
    summary = {
        "schema_version": STAGE5_SCHEMA_VERSION,
        "status": "success",
        "started_at": started_at,
        "finished_at": utc_now_iso(),
        "elapsed_sec": timer.elapsed_sec,
        "export_scope": "full_index" if video_id is None else "single_video_test",
        "input_index": str(index_path),
        "video_id_filter": None if video_id is None else str(video_id),
        "stage1_dir": str(stage1_root),
        "stage4_dir": str(stage4_root),
        "submission_path": str(submission_path),
        "config_sha256": mapping_sha256(config),
        "validation_status": validation_report["status"],
        "video_count": validation_report["video_count"],
        "prediction_count": validation_report["prediction_count"],
        "empty_video_count": validation_report["empty_video_count"],
        "repaired_bbox_count": validation_report["repaired_bbox_count"],
        "submission_sha256": validation_report["submission_sha256"],
        "submission_size_bytes": validation_report["submission_size_bytes"],
    }
    write_json(stage5_root / "resolved_config.json", deepcopy(config))
    write_reports(stage5_root, summary, validation_report)
    # 顶层成功标记最后写入，表示提交文件与报告均已通过并持久化。
    write_json(success_path, summary)
    return summary

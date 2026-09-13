"""Stage 5 独立命令行入口。"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Sequence

from video_highlight.common.config import load_mapping, resolve_path
from video_highlight.common.logging import configure_logging

from .pipeline import run_stage5


def build_parser(project_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 5：生成并严格校验最终提交 JSONL")
    parser.add_argument("--stage1-dir", type=Path, required=True, help="同批次 Stage 1 输出目录")
    parser.add_argument("--stage4-dir", type=Path, required=True, help="待导出的 Stage 4 输出目录")
    parser.add_argument("--input-index", type=Path, help="比赛视频索引；默认读取 configs/paths.yaml")
    parser.add_argument("--paths-config", type=Path, default=project_root / "configs/paths.yaml")
    parser.add_argument("--config", type=Path, default=project_root / "configs/stage5/submission.yaml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output-name", help="覆盖配置中的提交文件名，例如 submission.jsonl")
    parser.add_argument(
        "--video-id",
        help="测试模式：只导出索引中指定 video_id 的一行提交记录，例如 20",
    )
    parser.add_argument("--run-id", default=datetime.now().strftime("stage5_%Y%m%d_%H%M%S"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parents[3]
    args = build_parser(project_root).parse_args(argv)
    paths = load_mapping(args.paths_config)
    config = load_mapping(args.config)
    if args.output_name:
        config["output"]["filename"] = args.output_name
    input_index = args.input_index or resolve_path(paths["input_index"], project_root)
    runs_root = resolve_path(paths.get("runs_root", "runs"), project_root)
    output_dir = args.output_dir or (runs_root / args.run_id / "stage5")
    logger = configure_logging(Path(output_dir) / "stage5.log", verbose=args.verbose)
    summary = run_stage5(
        input_index=input_index,
        stage1_dir=args.stage1_dir,
        stage4_dir=args.stage4_dir,
        output_dir=output_dir,
        config=config,
        video_id=args.video_id,
        overwrite=args.overwrite,
        logger=logger,
    )
    logger.info("Stage 5 完成: %s", summary)
    return 0

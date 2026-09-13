"""Stage 4 独立命令行入口。"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Sequence

from video_highlight.common.config import load_mapping, resolve_path
from video_highlight.common.logging import configure_logging

from .pipeline import run_stage4


def build_parser(project_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 4：主体跟踪、构图框生成和轨迹平滑")
    parser.add_argument("--stage1-dir", type=Path, required=True, help="同批次 Stage 1 输出目录")
    parser.add_argument("--stage3-5-dir", type=Path, required=True, help="Stage 3.5 输出目录；Stage 4 不再读取 Stage 3")
    parser.add_argument("--paths-config", type=Path, default=project_root / "configs/paths.yaml")
    parser.add_argument("--config", type=Path, default=project_root / "configs/stage4/sam2_crop.yaml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-id", default=datetime.now().strftime("stage4_%Y%m%d_%H%M%S"))
    parser.add_argument("--video-id", action="append", dest="video_ids")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--backend", choices=["opencv", "center", "sam2"])
    parser.add_argument("--sam2-checkpoint", type=Path)
    parser.add_argument("--sam2-config")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parents[3]
    args = build_parser(project_root).parse_args(argv)
    paths = load_mapping(args.paths_config)
    config = load_mapping(args.config)
    if args.backend:
        config["tracking"]["backend"] = args.backend
    if args.sam2_checkpoint:
        config["tracking"]["checkpoint"] = str(args.sam2_checkpoint.resolve())
    if args.sam2_config:
        config["tracking"]["model_config"] = args.sam2_config
    runs_root = resolve_path(paths.get("runs_root", "runs"), project_root)
    output_dir = args.output_dir or (runs_root / args.run_id / "stage4")
    logger = configure_logging(Path(output_dir) / "stage4.log", verbose=args.verbose)
    summary = run_stage4(
        args.stage1_dir,
        args.stage3_5_dir,
        output_dir,
        config,
        paths,
        video_ids=set(args.video_ids) if args.video_ids else None,
        limit=args.limit,
        resume=args.resume,
        overwrite=args.overwrite,
        strict=args.strict,
        logger=logger,
    )
    logger.info("Stage 4 完成: %s", summary)
    return 1 if summary["failure_count"] else 0

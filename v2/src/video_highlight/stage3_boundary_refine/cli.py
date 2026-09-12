"""Stage 3 独立命令行入口。"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Sequence

from video_highlight.common.config import load_mapping, resolve_path
from video_highlight.common.logging import configure_logging

from .pipeline import run_stage3


def build_parser(project_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 3：高光候选帧级边界定位")
    parser.add_argument("--stage1-dir", type=Path, required=True, help="同批次 Stage 1 输出目录")
    parser.add_argument("--stage2-dir", type=Path, required=True, help="待细化的 Stage 2 输出目录")
    parser.add_argument("--paths-config", type=Path, default=project_root / "configs/paths.yaml")
    parser.add_argument("--config", type=Path, default=project_root / "configs/stage3/temporal_refine.yaml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-id", default=datetime.now().strftime("stage3_%Y%m%d_%H%M%S"))
    parser.add_argument("--video-id", action="append", dest="video_ids")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--backend", choices=["rule", "tcn"])
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--skip-processing",
        "--passthrough",
        action="store_true",
        help="跳过解码、特征、边界细化、门控和合并，直接封装 Stage 2 区间",
    )
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
    if args.skip_processing:
        config["runtime"]["mode"] = "passthrough"
    if args.backend:
        config["temporal"]["backend"] = args.backend
    if args.checkpoint:
        config["temporal"].setdefault("tcn", {})["checkpoint"] = str(args.checkpoint.resolve())
    runs_root = resolve_path(paths.get("runs_root", "runs"), project_root)
    output_dir = args.output_dir or (runs_root / args.run_id / "stage3")
    logger = configure_logging(Path(output_dir) / "stage3.log", verbose=args.verbose)
    summary = run_stage3(
        stage1_dir=args.stage1_dir,
        stage2_dir=args.stage2_dir,
        output_dir=output_dir,
        config=config,
        video_ids=set(args.video_ids) if args.video_ids else None,
        limit=args.limit,
        resume=args.resume,
        overwrite=args.overwrite,
        strict=args.strict,
        logger=logger,
    )
    logger.info("Stage 3 完成: %s", summary)
    return 1 if summary["failure_count"] else 0

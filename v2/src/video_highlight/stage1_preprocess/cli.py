"""Stage 1 命令行入口。"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Sequence

from video_highlight.common.config import load_mapping, resolve_path
from video_highlight.common.logging import configure_logging

from .pipeline import run_stage1


def build_parser(project_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 1：视频元信息、镜头切分、2 FPS 粗采样与音频预处理")
    parser.add_argument("--paths-config", type=Path, default=project_root / "configs/paths.yaml")
    parser.add_argument("--config", type=Path, default=project_root / "configs/stage1/default.yaml")
    parser.add_argument("--input-index", type=Path)
    parser.add_argument("--video-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-id", default=datetime.now().strftime("stage1_%Y%m%d_%H%M%S"))
    parser.add_argument("--video-id", action="append", dest="video_ids")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--skip-audio", action="store_true")
    parser.add_argument("--skip-hash", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parents[3]
    parser = build_parser(project_root)
    args = parser.parse_args(argv)
    paths = load_mapping(args.paths_config)
    config = load_mapping(args.config)
    if args.skip_audio:
        config.setdefault("audio", {})["enabled"] = False
    if args.skip_hash:
        config["compute_sha256"] = False
    index_path = args.input_index or resolve_path(paths["input_index"], project_root)
    video_root = args.video_root or resolve_path(paths["video_root"], project_root)
    runs_root = resolve_path(paths.get("runs_root", "runs"), project_root)
    output_dir = args.output_dir or (runs_root / args.run_id / "stage1")
    logger = configure_logging(Path(output_dir) / "stage1.log", verbose=args.verbose)
    summary = run_stage1(
        index_path=index_path,
        video_root=video_root,
        output_dir=output_dir,
        config=config,
        video_ids=set(args.video_ids) if args.video_ids else None,
        limit=args.limit,
        resume=args.resume,
        overwrite=args.overwrite,
        strict=args.strict,
        logger=logger,
    )
    logger.info("Stage 1 完成: %s", summary)
    return 1 if summary["failure_count"] else 0

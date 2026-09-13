"""Stage 3.5 独立命令行入口。"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Sequence

from video_highlight.common.config import load_mapping, resolve_path
from video_highlight.common.logging import configure_logging

from .pipeline import run_stage3_5


def build_parser(project_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 3.5：高光区间逐采样帧主体中心定位")
    parser.add_argument("--stage1-dir", type=Path, required=True, help="Stage 1 输出目录，仅读取源视频元数据")
    parser.add_argument("--stage3-dir", type=Path, required=True, help="Stage 3 最终边界输出目录")
    parser.add_argument("--paths-config", type=Path, default=project_root / "configs/paths.yaml")
    parser.add_argument("--config", type=Path, default=project_root / "configs/stage3_5/qwen_subject_point.yaml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-id", default=datetime.now().strftime("stage3_5_%Y%m%d_%H%M%S"))
    parser.add_argument("--video-id", action="append", dest="video_ids")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-fps", type=float)
    parser.add_argument(
        "--decoder",
        choices=["auto", "opencv", "ffmpeg"],
        help="采样解码器；video 97 等异常色彩元数据可指定 ffmpeg",
    )
    parser.add_argument("--backend", choices=["openai", "mock"])
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--api-key")
    parser.add_argument("--skip-processing", "--passthrough", action="store_true", help="不解码、不调用模型，输出同一采样时间轴上的 null 主体点")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parents[3]
    args = build_parser(project_root).parse_args(argv)
    paths, config = load_mapping(args.paths_config), load_mapping(args.config)
    if args.skip_processing:
        config["runtime"]["mode"] = "passthrough"
    if args.backend:
        config["runtime"]["backend"] = args.backend
    if args.sample_fps:
        config["sampling"]["fps"] = args.sample_fps
    if args.decoder:
        config["sampling"]["decoder"] = args.decoder
    for argument, key in ((args.base_url, "base_url"), (args.model, "model"), (args.api_key, "api_key")):
        if argument is not None:
            config["api"][key] = argument
    runs_root = resolve_path(paths.get("runs_root", "runs"), project_root)
    output_dir = args.output_dir or (runs_root / args.run_id / "stage3_5")
    logger = configure_logging(Path(output_dir) / "stage3_5.log", verbose=args.verbose)
    summary = run_stage3_5(
        args.stage1_dir, args.stage3_dir, output_dir, config, paths,
        video_ids=set(args.video_ids) if args.video_ids else None,
        limit=args.limit, resume=args.resume, overwrite=args.overwrite,
        strict=args.strict, logger=logger,
    )
    logger.info("Stage 3.5 完成: %s", summary)
    return 1 if summary["failure_count"] else 0

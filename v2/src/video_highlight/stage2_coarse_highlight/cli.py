"""Stage 2 独立命令行入口。"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Sequence

from video_highlight.common.config import load_mapping, resolve_path
from video_highlight.common.logging import configure_logging

from .pipeline import run_stage2


def build_parser(project_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 2：通过 vLLM OpenAI API 生成粗高光候选")
    parser.add_argument("--stage1-dir", type=Path, required=True, help="某次 Stage 1 输出目录")
    parser.add_argument("--paths-config", type=Path, default=project_root / "configs/paths.yaml")
    parser.add_argument("--config", type=Path, default=project_root / "configs/stage2/qwen3_5_4b.yaml")
    parser.add_argument("--prompts", type=Path, default=project_root / "configs/stage2/prompts.yaml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-id", default=datetime.now().strftime("stage2_%Y%m%d_%H%M%S"))
    parser.add_argument("--video-id", action="append", dest="video_ids")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--model")
    parser.add_argument("--backend", choices=["openai", "mock"])
    parser.add_argument("--video-mode", choices=["url", "data_url"])
    parser.add_argument("--url-template")
    parser.add_argument("--no-healthcheck", action="store_true")
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
    prompt_config = load_mapping(args.prompts)
    if args.base_url:
        config["api"]["base_url"] = args.base_url
    if args.api_key:
        config["api"]["api_key"] = args.api_key
    if args.model:
        config["api"]["model"] = args.model
    if args.backend:
        config["runtime"]["backend"] = args.backend
    if args.video_mode:
        config["video_input"]["mode"] = args.video_mode
    if args.url_template:
        config["video_input"]["url_template"] = args.url_template
    if args.no_healthcheck:
        config["api"]["healthcheck_on_start"] = False
    runs_root = resolve_path(paths.get("runs_root", "runs"), project_root)
    output_dir = args.output_dir or (runs_root / args.run_id / "stage2")
    logger = configure_logging(Path(output_dir) / "stage2.log", verbose=args.verbose)
    summary = run_stage2(
        stage1_dir=args.stage1_dir,
        output_dir=output_dir,
        config=config,
        prompt_config=prompt_config,
        video_ids=set(args.video_ids) if args.video_ids else None,
        limit=args.limit,
        resume=args.resume,
        overwrite=args.overwrite,
        strict=args.strict,
        logger=logger,
    )
    logger.info("Stage 2 完成: %s", summary)
    return 1 if summary["failure_count"] else 0

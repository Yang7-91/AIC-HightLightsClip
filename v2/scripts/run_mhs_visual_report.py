"""MHS-VIS-0 多高光候选可视化诊断 CLI（diagnostic-only）。

run / smoke 两个子命令；只读 Stage 1/2/3/MHS-1 产物与视频，输出本地诊断 HTML。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from video_highlight.common.config import load_mapping  # noqa: E402
from video_highlight.stage2_5_visual_report.pipeline import run_mhs_vis0  # noqa: E402

VIS0_CONFIG_SCHEMA = "stage2_5.mhs_vis0_visual_report.v1"


def _load_config(path: Path) -> dict:
    config = load_mapping(path)
    if config.get("schema_version") != VIS0_CONFIG_SCHEMA:
        raise SystemExit(
            f"config schema mismatch: expected {VIS0_CONFIG_SCHEMA}, "
            f"got {config.get('schema_version')}"
        )
    return config


def _run(args: argparse.Namespace, *, limit: int | None) -> int:
    config = _load_config(args.config)
    video_root = args.video_root
    if video_root is None:
        config_root = config.get("input", {}).get("video_root")
        if not config_root:
            raise SystemExit("--video-root is required (config input.video_root is null)")
        video_root = Path(config_root)
    summary = run_mhs_vis0(
        stage2_dir=args.stage2_dir,
        video_root=video_root,
        output_dir=args.output_dir,
        config=config,
        video_manifest=args.video_manifest,
        mhs1_dir=args.mhs1_dir,
        stage3_dir=args.stage3_dir,
        limit=limit,
        manual_video_ids=args.video_ids,
    )
    printable = {
        key: summary[key]
        for key in (
            "num_selected_examples",
            "num_videos",
            "eventness_available",
            "mhs1_subsegments_available",
        )
    }
    printable["outputs"] = summary["outputs"]
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    default_config = PROJECT_ROOT / "configs/stage2_5/mhs_vis0_visual_report.yaml"
    parser = argparse.ArgumentParser(description="MHS-VIS-0 多高光候选可视化诊断（diagnostic-only）")
    commands = parser.add_subparsers(dest="command", required=True)

    common_parent = argparse.ArgumentParser(add_help=False)
    common_parent.add_argument("--config", type=Path, default=default_config)
    common_parent.add_argument("--stage2-dir", type=Path, required=True, help="Stage 2 产物目录或 frozen cache 目录")
    common_parent.add_argument("--stage3-dir", type=Path)
    common_parent.add_argument("--mhs1-dir", type=Path)
    common_parent.add_argument("--video-root", type=Path, required=True)
    common_parent.add_argument("--video-manifest", type=Path)
    common_parent.add_argument("--video-id", action="append", dest="video_ids")

    run = commands.add_parser("run", parents=[common_parent], help="完整运行可视化报告")
    run.add_argument("--output-dir", type=Path, required=True)

    smoke = commands.add_parser("smoke", parents=[common_parent], help="小样本 smoke（默认 3 个）")
    smoke.add_argument("--limit", type=int, default=3)
    smoke.add_argument("--output-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        output_dir = args.output_dir
        if output_dir is None:
            output_dir = Path("runs") / f"stage2_5_mhs_vis0_{datetime.now():%Y%m%d_%H%M%S}"
        args.output_dir = output_dir
        return _run(args, limit=None)
    if args.command == "smoke":
        output_dir = args.output_dir
        if output_dir is None:
            output_dir = Path("runs") / f"stage2_5_mhs_vis0_smoke_{datetime.now():%Y%m%d_%H%M%S}"
        args.output_dir = output_dir
        return _run(args, limit=int(args.limit))
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

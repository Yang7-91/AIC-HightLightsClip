"""MHS-1 多高光候选拆分 CLI（run / run-all / smoke）。"""

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
from video_highlight.stage2_5_multi_highlight_split.pipeline import (  # noqa: E402
    run_mhs1_all_configs,
    run_mhs1_config,
)

MHS1_CONFIG_SCHEMA = "stage2_5.mhs1_formal_splitter.v1"


def _load_config(path: Path) -> dict:
    config = load_mapping(path)
    if config.get("schema_version") != MHS1_CONFIG_SCHEMA:
        raise SystemExit(
            f"config schema mismatch: expected {MHS1_CONFIG_SCHEMA}, "
            f"got {config.get('schema_version')}"
        )
    return config


def _print_result(payload: dict) -> None:
    summary = payload.get("summary") if "summary" in payload else payload
    evaluation = payload.get("evaluation") if isinstance(payload, dict) else None
    printable = {
        key: summary.get(key)
        for key in (
            "config",
            "num_parent_candidates",
            "num_mhs1_candidates",
            "num_split",
            "num_fallback_identity",
            "split_rate",
            "fallback_rate",
        )
        if isinstance(summary, dict)
    }
    if evaluation is not None:
        printable["vs_stage3_baseline"] = {
            "deltas": evaluation["vs_stage3_baseline"]["deltas"],
            "gate": evaluation["vs_stage3_baseline"]["gate"],
        }
    print(json.dumps(printable, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    default_config = PROJECT_ROOT / "configs/stage2_5/mhs1_formal_splitter.yaml"
    parser = argparse.ArgumentParser(description="MHS-1 多高光候选拆分（formal）")
    commands = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, default=default_config)
    common.add_argument("--stage2-dir", type=Path, required=True)
    common.add_argument("--stage1-dir", type=Path)
    common.add_argument("--video-root", type=Path, required=True)
    common.add_argument("--video-manifest", type=Path)
    common.add_argument("--frozen-predictions", type=Path, help="weak-reference 评估源（可选）")

    run = commands.add_parser("run", parents=[common], help="运行单个配置")
    run.add_argument("--config-name", required=True)
    run.add_argument("--output-dir", type=Path, required=True)

    run_all = commands.add_parser("run-all", parents=[common], help="运行全部配置")
    run_all.add_argument("--output-dir", type=Path, required=True)

    smoke = commands.add_parser("smoke", parents=[common], help="小样本 smoke（默认 5 个候选）")
    smoke.add_argument("--limit", type=int, default=5)
    smoke.add_argument("--output-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = _load_config(args.config)
    if args.command == "run":
        result = run_mhs1_config(
            stage2_dir=args.stage2_dir,
            video_root=args.video_root,
            output_dir=args.output_dir,
            config=config,
            config_name=args.config_name,
            video_manifest=args.video_manifest,
            frozen_predictions=args.frozen_predictions,
        )
        _print_result(result)
        return 0
    if args.command == "run-all":
        result = run_mhs1_all_configs(
            stage2_dir=args.stage2_dir,
            video_root=args.video_root,
            output_dir=args.output_dir,
            config=config,
            video_manifest=args.video_manifest,
            frozen_predictions=args.frozen_predictions,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "smoke":
        output_dir = args.output_dir or (
            Path("runs") / f"stage2_5_mhs1_smoke_{datetime.now():%Y%m%d_%H%M%S}"
        )
        result = run_mhs1_config(
            stage2_dir=args.stage2_dir,
            video_root=args.video_root,
            output_dir=output_dir,
            config=config,
            config_name="MHS-1-C1",
            video_manifest=args.video_manifest,
            frozen_predictions=args.frozen_predictions,
            limit=int(args.limit),
        )
        _print_result(result)
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

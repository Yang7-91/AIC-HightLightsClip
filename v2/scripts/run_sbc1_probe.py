"""SBC-1 探针 CLI（diagnostic-only）。

build-samples / classify / evaluate / run-all 四个子命令，严格只做诊断：
不生成 refined candidates、不写回 Stage 2/3 正式产物、不访问 Hard/Heldout。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
REPO_ROOT = PROJECT_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from video_highlight.common.config import load_mapping  # noqa: E402
from video_highlight.common.exceptions import ExternalToolError  # noqa: E402
from video_highlight.stage3_boundary_refine.sbc1_probe import (  # noqa: E402
    P1_VARIANT,
    P2_VARIANT,
    P3_VARIANT,
    SBC1_VARIANTS,
    Sbc1ParseError,
    Sbc1Prediction,
    Sbc1Sample,
    build_sbc1_samples,
    compute_clip_windows,
    compute_sbc1_metrics,
    extract_clip_data_url,
    load_cache_records,
    load_oracle_boundary_labels,
    load_video_manifest,
    parse_sbc1_response,
    render_sbc1_prompt,
    sample_to_row,
    write_sbc1_summary,
)

SBC1_CONFIG_SCHEMA = "stage3.sbc1_semantic_boundary_probe.v1"


def _load_sbc1_config(path: Path) -> dict[str, Any]:
    config = load_mapping(path)
    if config.get("schema_version") != SBC1_CONFIG_SCHEMA:
        raise SystemExit(
            f"config schema mismatch: expected {SBC1_CONFIG_SCHEMA}, "
            f"got {config.get('schema_version')}"
        )
    return config


def _prompt_path(config: dict[str, Any], variant_id: str) -> Path:
    for entry in config["prompt_variants"]:
        if entry["id"] == variant_id:
            raw = Path(entry["file"])
            for candidate in (raw, REPO_ROOT / raw, PROJECT_ROOT / raw.parent.name / raw.name):
                if candidate.is_file():
                    return candidate
            raise SystemExit(f"prompt file not found for {variant_id}: {raw}")
    raise SystemExit(f"unknown prompt variant: {variant_id}")


def _resolve_video(video_dir: Path, video_map: dict[str, str], video_id: str) -> Path | None:
    relative = video_map.get(video_id)
    if relative:
        for candidate in (video_dir / relative, video_dir / Path(relative).name):
            if candidate.is_file():
                return candidate
    candidate = video_dir / f"{video_id}.mp4"
    return candidate if candidate.is_file() else None


def _sample_from_row(row: dict[str, Any]) -> Sbc1Sample:
    return Sbc1Sample(
        sample_id=row["sample_id"],
        video_id=row["video_id"],
        candidate_id=row["candidate_id"],
        side=row["side"],
        oracle_label=row["oracle_label"],
        original_start_sec=float(row["original_start_sec"]),
        original_end_sec=float(row["original_end_sec"]),
        boundary_sec=float(row["boundary_sec"]),
        candidate_duration_sec=float(row["candidate_duration_sec"]),
        prompt_variant=row["prompt_variant"],
        clip_spec=row.get("clip_spec") or {},
    )


def _cmd_build_samples(args: argparse.Namespace) -> int:
    config = _load_sbc1_config(args.config)
    labels = load_oracle_boundary_labels(args.oracle_labels)
    cache_records = load_cache_records(args.cache_dir)
    video_map = load_video_manifest(args.video_manifest) if args.video_manifest else {}
    allowed = set(cache_records) if not video_map else set(video_map)
    samples, summary = build_sbc1_samples(labels, cache_records, config, video_ids_allowed=allowed)
    rows = []
    for sample in samples:
        duration = float(cache_records[sample.video_id]["duration_sec"])
        windows = compute_clip_windows(sample, config, duration)
        rows.append(sample_to_row(sample, {"windows": windows}))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = args.output_dir / "sbc1_samples.jsonl"
    samples_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    summary["samples_path"] = str(samples_path)
    summary["oracle_labels"] = str(args.oracle_labels)
    (args.output_dir / "sbc1_sample_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: summary[k] for k in ("base_sample_count", "total_samples", "label_counts")}))
    return 0


def _load_prompt_template(config: dict[str, Any], variant_id: str) -> str:
    return _prompt_path(config, variant_id).read_text(encoding="utf-8")


def _cmd_classify(args: argparse.Namespace) -> int:
    config = _load_sbc1_config(args.config)
    variant_id = args.prompt_variant
    if variant_id not in SBC1_VARIANTS:
        raise SystemExit(f"unknown prompt variant: {variant_id}")
    rows = [
        json.loads(line)
        for line in args.samples.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    samples = [_sample_from_row(row) for row in rows if row["prompt_variant"] == variant_id]
    template = _load_prompt_template(config, variant_id)
    video_dir = args.video_dir.expanduser().resolve()
    video_map = load_video_manifest(args.video_manifest) if args.video_manifest else {}

    model_cfg = config["model"]
    base_url = args.base_url or str(model_cfg["base_url"])
    model = args.model or str(model_cfg["model"])
    try:
        from openai import OpenAI
    except ImportError as error:
        raise SystemExit("missing dependency 'openai'") from error
    client = OpenAI(
        base_url=base_url.rstrip("/") + "/",
        api_key="EMPTY",
        timeout=float(model_cfg.get("timeout_sec", 300.0)),
        max_retries=0,
    )
    done_ids: set[str] = set()
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and output_path.is_file():
        for line in output_path.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                done_ids.add(json.loads(line)["sample_id"])
    mode = "a" if done_ids else "w"
    calls = 0
    failures = 0
    parse_failures = 0
    with output_path.open(mode, encoding="utf-8") as handle:
        for sample in samples:
            if sample.sample_id in done_ids:
                continue
            error: str | None = None
            parsed: dict[str, Any] | None = None
            raw_response: str | None = None
            video_path = _resolve_video(video_dir, video_map, sample.video_id)
            if video_path is None:
                error = "video_missing"
            else:
                try:
                    windows = sample.clip_spec.get("windows")
                    if not windows:
                        error = "clip_spec_missing: rebuild samples with clip windows"
                        windows = None
                    if windows is None:
                        pass
                    else:
                        data_urls = [
                            extract_clip_data_url(
                                video_path,
                                float(window["start_sec"]),
                                float(window["end_sec"]),
                                fps=float(config["video_context"]["fps"]),
                                max_payload_mib=64.0,
                            )
                            for window in windows
                        ]
                        prompt = render_sbc1_prompt(template, sample, windows, config)
                        content: list[dict[str, Any]] = [
                            {"type": "video_url", "video_url": {"url": url}} for url in data_urls
                        ]
                        content.append({"type": "text", "text": prompt})
                        calls += 1
                        response = client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": content}],
                            temperature=float(model_cfg["temperature"]),
                            top_p=float(model_cfg["top_p"]),
                            max_tokens=int(model_cfg["max_tokens"]),
                            response_format=model_cfg.get("response_format"),
                            extra_body=model_cfg.get("extra_body"),
                        )
                        raw_response = (response.choices[0].message.content or "")[:400]
                        parsed = parse_sbc1_response(variant_id, raw_response)
                except Sbc1ParseError as exc:
                    error = f"parse_error: {exc}"
                    parse_failures += 1
                except ExternalToolError as exc:
                    error = f"model_error: {exc}"
                    failures += 1
                except Exception as exc:  # clip or transport failure recorded, never silent
                    error = f"error: {type(exc).__name__}: {exc}"
                    failures += 1
            predicted_action = None
            confidence = None
            if parsed is not None:
                predicted_action = parsed.get("best_variant") or parsed.get("action")
                confidence = parsed.get("confidence")
            record = {
                "sample_id": sample.sample_id,
                "video_id": sample.video_id,
                "candidate_id": sample.candidate_id,
                "side": sample.side,
                "oracle_label": sample.oracle_label,
                "prompt_variant": variant_id,
                "predicted_action": predicted_action,
                "confidence": confidence,
                "parse_ok": parsed is not None,
                "error": error,
                "raw_response": raw_response,
                "payload": parsed,
                "diagnostic_only": True,
                "deployable_method": False,
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
    summary = {
        "prompt_variant": variant_id,
        "model": model,
        "base_url": base_url,
        "temperature": float(model_cfg["temperature"]),
        "enable_thinking": bool(
            (model_cfg.get("extra_body") or {})
            .get("chat_template_kwargs", {})
            .get("enable_thinking", False)
        ),
        "num_samples": len(samples),
        "qwen_calls": calls,
        "vllm_calls": calls,
        "parse_failures": parse_failures,
        "other_failures": failures,
        "diagnostic_only": True,
        "deployable_method": False,
        "heldout_accessed": False,
    }
    (output_path.parent / f"sbc1_classify_summary_{variant_id}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary))
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    config = _load_sbc1_config(args.config)
    variant_id = args.prompt_variant
    sample_rows = [
        json.loads(line)
        for line in args.samples.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    samples = [_sample_from_row(row) for row in sample_rows]
    prediction_rows = {}
    for line in args.predictions.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            row = json.loads(line)
            prediction_rows[row["sample_id"]] = row
    predictions = [
        Sbc1Prediction(
            sample_id=sample.sample_id,
            prompt_variant=variant_id,
            predicted_action=prediction_rows.get(sample.sample_id, {}).get("predicted_action"),
            confidence=prediction_rows.get(sample.sample_id, {}).get("confidence"),
            parse_ok=bool(prediction_rows.get(sample.sample_id, {}).get("parse_ok")),
            error=prediction_rows.get(sample.sample_id, {}).get("error"),
            raw_response=prediction_rows.get(sample.sample_id, {}).get("raw_response"),
            payload=prediction_rows.get(sample.sample_id, {}).get("payload"),
        )
        for sample in samples
    ]
    evaluation = compute_sbc1_metrics(samples, predictions, variant_id, config)
    evaluation["config_schema_version"] = config["schema_version"]
    evaluation["evaluation_time"] = datetime.now().isoformat(timespec="seconds")
    args.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.expanduser().resolve().write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "prompt_variant": variant_id,
                "schema_success_rate": evaluation["schema_success_rate"],
                "accuracy": evaluation["accuracy"],
                "macro_f1": evaluation["macro_f1"],
                "left_auc_like": evaluation["left_auc_like"],
                "right_auc_like": evaluation["right_auc_like"],
                "decision": evaluation["decision"],
            }
        )
    )
    return 0


def _cmd_run_all(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "logs").mkdir(parents=True, exist_ok=True)
    build_args = argparse.Namespace(
        oracle_labels=args.oracle_labels,
        cache_dir=args.cache_dir,
        video_dir=args.video_dir,
        video_manifest=args.video_manifest,
        config=args.config,
        output_dir=args.output_dir,
    )
    _cmd_build_samples(build_args)
    evaluations: dict[str, Any] = {}
    for variant_id in SBC1_VARIANTS:
        predictions_path = args.output_dir / f"predictions_{variant_id}.jsonl"
        classify_args = argparse.Namespace(
            samples=args.output_dir / "sbc1_samples.jsonl",
            config=args.config,
            prompt_variant=variant_id,
            output=predictions_path,
            video_dir=args.video_dir,
            video_manifest=args.video_manifest,
            model=args.model,
            base_url=args.base_url,
            resume=True,
        )
        _cmd_classify(classify_args)
        evaluation_path = args.output_dir / f"evaluation_{variant_id}.json"
        evaluate_args = argparse.Namespace(
            samples=args.output_dir / "sbc1_samples.jsonl",
            predictions=predictions_path,
            prompt_variant=variant_id,
            config=args.config,
            output=evaluation_path,
        )
        _cmd_evaluate(evaluate_args)
        evaluations[variant_id] = json.loads(evaluation_path.read_text(encoding="utf-8"))
    summary = write_sbc1_summary(evaluations, _load_sbc1_config(args.config))
    (args.output_dir / "sbc1_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    default_config = PROJECT_ROOT / "configs/stage3/sbc1_semantic_boundary_probe.yaml"
    parser = argparse.ArgumentParser(description="SBC-1 多提示词语义边界判别探针（diagnostic-only）")
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build-samples", help="构建分层边界侧样本")
    build.add_argument("--config", type=Path, default=default_config)
    build.add_argument("--oracle-labels", type=Path, required=True)
    build.add_argument("--cache-dir", type=Path, required=True)
    build.add_argument("--video-dir", type=Path, required=True)
    build.add_argument("--video-manifest", type=Path)
    build.add_argument("--output-dir", type=Path, required=True)

    classify = commands.add_parser("classify", help="对指定 prompt 变体调用 Qwen")
    classify.add_argument("--config", type=Path, default=default_config)
    classify.add_argument("--samples", type=Path, required=True)
    classify.add_argument("--prompt-variant", required=True)
    classify.add_argument("--output", type=Path, required=True)
    classify.add_argument("--video-dir", type=Path, required=True)
    classify.add_argument("--video-manifest", type=Path)
    classify.add_argument("--model")
    classify.add_argument("--base-url")
    classify.add_argument("--resume", action="store_true")

    evaluate = commands.add_parser("evaluate", help="计算可分性指标")
    evaluate.add_argument("--config", type=Path, default=default_config)
    evaluate.add_argument("--samples", type=Path, required=True)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--prompt-variant", required=True)
    evaluate.add_argument("--output", type=Path, required=True)

    run_all = commands.add_parser("run-all", help="build-samples + 3 变体 classify + evaluate")
    run_all.add_argument("--config", type=Path, default=default_config)
    run_all.add_argument("--oracle-labels", type=Path, required=True)
    run_all.add_argument("--cache-dir", type=Path, required=True)
    run_all.add_argument("--video-dir", type=Path, required=True)
    run_all.add_argument("--video-manifest", type=Path)
    run_all.add_argument("--output-dir", type=Path, required=True)
    run_all.add_argument("--model")
    run_all.add_argument("--base-url")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "build-samples":
        return _cmd_build_samples(args)
    if args.command == "classify":
        return _cmd_classify(args)
    if args.command == "evaluate":
        return _cmd_evaluate(args)
    if args.command == "run-all":
        return _cmd_run_all(args)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

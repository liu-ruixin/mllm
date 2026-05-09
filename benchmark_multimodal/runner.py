#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import requests
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_multimodal.scorer import postprocess_prediction
from benchmark_multimodal.scorer import score_prediction 
from benchmark_multimodal.report import print_result_card 
from benchmark_multimodal.report import build_output_record 
from benchmark_multimodal.report import resolve_path
from benchmark_multimodal.report import print_batch_summary
from benchmark_multimodal.report import summarize_records
from benchmark_multimodal.report import write_report_md


DEFAULT_CONFIG = PROJECT_ROOT / "benchmark_multimodal/config/sanity.yaml"
OUTPUT_PATH = PROJECT_ROOT / "benchmark_multimodal/output/output.json"

def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two config dictionaries without mutating inputs."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged

def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load a YAML config and resolve an optional `inherits` parent file."""
    path = resolve_path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Config root must be a mapping: {path}")

    inherits = config.pop("inherits", None)
    if inherits is None:
        return config

    parent_path = Path(inherits).expanduser()
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path

    return deep_merge(load_config(parent_path), config)


def load_samples(
    manifest: str | Path,
    limit: int | None = None,
    task_filter: List[str] | None = None,
) -> List[Dict[str, Any]]:
    """Load benchmark samples from a JSONL manifest."""
    path = resolve_path(manifest)
    if not path.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {path}")

    allowed_tasks = set(task_filter or [])
    samples: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc

            if not isinstance(sample, dict):
                raise ValueError(f"Sample must be a JSON object: {path}:{line_no}")
            if "id" not in sample:
                raise ValueError(f"Sample missing required field `id`: {path}:{line_no}")
            if "messages" not in sample:
                raise ValueError(
                    f"Sample missing required field `messages`: {path}:{line_no}"
                )
            if not isinstance(sample["messages"], list):
                raise ValueError(f"`messages` must be a list: {path}:{line_no}")

            task = sample.get("task")
            if allowed_tasks and task not in allowed_tasks:
                continue

            samples.append(sample)
            if limit is not None and len(samples) >= limit:
                break

    return samples


def build_payload(sample: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Build an OpenAI-compatible chat completions request."""
    generation = cfg.get("generation", {})
    request_cfg = cfg.get("request", {})
    model_cfg = cfg.get("model", {})

    payload: Dict[str, Any] = {
        "model": model_cfg.get("served_model_name") or model_cfg.get("name", ""),
        "messages": sample["messages"],
        "temperature": generation.get("temperature", 0.0),
        "top_p": generation.get("top_p"),
        "top_k": generation.get("top_k"),
        "max_tokens": generation.get("max_tokens", 64),
        "stream": request_cfg.get("stream", True),
        "stream_options": request_cfg.get("stream_options", {"include_usage": True}),
        "chat_template_kwargs": request_cfg.get("chat_template_kwargs", {}),
        "separate_reasoning": request_cfg.get("separate_reasoning", True),
        "stream_reasoning": request_cfg.get("stream_reasoning", False),
    }

    if generation.get("stop") is not None:
        payload["stop"] = generation["stop"]
    if generation.get("frequency_penalty") is not None:
        payload["frequency_penalty"] = generation["frequency_penalty"]
    if generation.get("presence_penalty") is not None:
        payload["presence_penalty"] = generation["presence_penalty"]
    if generation.get("repetition_penalty") is not None:
        payload["repetition_penalty"] = generation["repetition_penalty"]
    if generation.get("seed") is not None:
        payload["seed"] = generation["seed"]

    return {key: value for key, value in payload.items() if value is not None}


def extract_delta_text(data: Dict[str, Any]) -> str:
    """Extract content text from a streamed chat-completions chunk."""
    choices = data.get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}
    return delta.get("content") or ""


def run_one_sample(sample: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Run one streaming chat-completions request and collect prediction/metrics."""
    server_cfg = cfg["server"]
    base_url = server_cfg["base_url"].rstrip("/")
    endpoint = server_cfg.get("endpoint", "/v1/chat/completions")
    url = base_url + endpoint
    timeout_s = server_cfg.get("timeout_s", 300)
    payload = build_payload(sample, cfg)

    t_start = time.perf_counter()
    t_first_sse = None
    t_first_token = None
    t_usage = None
    pred_parts: List[str] = []
    usage: Dict[str, Any] = {}
    sse_chunk_count = 0
    chunk_count = 0
    empty_chunk_count = 0
    reasoning_chunk_count = 0
    role_chunk_count = 0
    finish_reason = None
    reasoning_closed = None
    visible_token_count = None
    think_end_token_id = None
    error = None

    try:
        resp = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            stream=True,
            timeout=timeout_s,
        )
        resp.raise_for_status()

        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue

            data_str = line[len("data: "):]
            if data_str.strip() == "[DONE]":
                break

            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            sse_chunk_count += 1
            if t_first_sse is None:
                t_first_sse = time.perf_counter()

            if data.get("usage"):
                if t_usage is None:
                    t_usage = time.perf_counter()
                usage = data["usage"]
                continue

            choices = data.get("choices") or []
            if choices:
                if choices[0].get("finish_reason") is not None:
                    finish_reason = choices[0].get("finish_reason")
                delta = choices[0].get("delta") or {}
                if delta.get("reasoning_closed") is not None:
                    reasoning_closed = delta.get("reasoning_closed")
                if delta.get("visible_token_count") is not None:
                    visible_token_count = delta.get("visible_token_count")
                if delta.get("think_end_token_id") is not None:
                    think_end_token_id = delta.get("think_end_token_id")
                if delta.get("role"):
                    role_chunk_count += 1
                if delta.get("reasoning_content"):
                    reasoning_chunk_count += 1

            text_delta = extract_delta_text(data)
            if not text_delta:
                empty_chunk_count += 1
                continue

            if t_first_token is None:
                t_first_token = time.perf_counter()
            pred_parts.append(text_delta)
            chunk_count += 1

        t_end = time.perf_counter()
    except Exception as exc:
        t_end = time.perf_counter()
        error = str(exc)

    if t_first_token is None:
        t_first_token = t_end
    if t_first_sse is None:
        t_first_sse = t_end

    prediction = "".join(pred_parts)
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    total_tokens = int(
        usage.get("total_tokens", prompt_tokens + completion_tokens) or 0
    )

    ttft_ms = (t_first_token - t_start) * 1000
    first_sse_ms = (t_first_sse - t_start) * 1000
    usage_ms = (t_usage - t_start) * 1000 if t_usage is not None else None
    e2e_ms = (t_end - t_start) * 1000
    decode_ms = max(0.0, e2e_ms - ttft_ms)
    prompt_tps = prompt_tokens / (ttft_ms / 1000) if ttft_ms and prompt_tokens else 0
    decode_tps = (
        completion_tokens / (decode_ms / 1000)
        if decode_ms and completion_tokens
        else 0
    )

    return {
        "id": sample.get("id"),
        "task": sample.get("task"),
        "answer_type": sample.get("answer_type"),
        "answer": sample.get("answer"),
        "prediction": prediction,
        "latency": {
            "first_sse_ms": first_sse_ms,
            "ttft_ms": ttft_ms,
            "e2e_ms": e2e_ms,
            "decode_ms": decode_ms,
            "usage_ms": usage_ms,
        },
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "sse_chunk_count": sse_chunk_count,
            "chunk_count": chunk_count,
            "empty_chunk_count": empty_chunk_count,
            "role_chunk_count": role_chunk_count,
            "reasoning_chunk_count": reasoning_chunk_count,
            "finish_reason": finish_reason,
            "reasoning_closed": reasoning_closed,
            "visible_token_count": visible_token_count,
            "think_end_token_id": think_end_token_id,
        },
        "throughput": {
            "prompt_tps": prompt_tps,
            "decode_tps": decode_tps,
        },
        "error": error,
    }


def write_jsonl(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    out_path = resolve_path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: Dict[str, Any]) -> None:
    out_path = resolve_path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, record: Dict[str, Any]) -> None:
    out_path = resolve_path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
        f.write("\n")


def check_health(cfg: Dict[str, Any]) -> Dict[str, Any]:
    server_cfg = cfg["server"]
    health_endpoint = server_cfg.get("health_endpoint", "/health")
    url = server_cfg["base_url"].rstrip("/") + health_endpoint
    try:
        resp = requests.get(url, timeout=5)
        return {"ok": 200 <= resp.status_code < 300, "status_code": resp.status_code, "error": None}
    except Exception as exc:
        return {"ok": False, "status_code": None, "error": str(exc)}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pymllm benchmark samples")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to benchmark YAML config.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Index of the filtered sample to run in single-sample mode.",
    )
    parser.add_argument("--sample-id", default=None, help="Run a specific sample id.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all filtered samples. Use this for MMLU/MMMU batch runs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Override dataset.limit from config.",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Disable configured warmup before the measured request.",
    )
    parser.add_argument("--output", default=None, help="Override output path.")
    return parser.parse_args()


def select_sample(samples: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    if args.sample_id:
        sample = next((s for s in samples if s.get("id") == args.sample_id), None)
        if sample is None:
            raise RuntimeError(f"Sample id not found: {args.sample_id}")
        return sample

    if args.sample_index < 0 or args.sample_index >= len(samples):
        raise RuntimeError(
            f"sample-index out of range: {args.sample_index}, samples={len(samples)}"
        )
    return samples[args.sample_index]


def run_warmup(
    sample: Dict[str, Any],
    cfg: Dict[str, Any],
    disabled: bool,
) -> List[Dict[str, Any]]:
    warmup_cfg = cfg.get("warmup", {})
    if disabled or not warmup_cfg.get("enabled", False):
        return []

    requests_num = int(warmup_cfg.get("requests", 0) or 0)
    if requests_num <= 0:
        return []

    warm_cfg = deep_merge(
        cfg,
        {"generation": {"max_tokens": warmup_cfg.get("max_tokens", 8)}},
    )
    print(f"\n--- Warmup ({requests_num} request(s), same sample) ---")
    rows = []
    for i in range(requests_num):
        row = run_one_sample(sample, warm_cfg)
        rows.append(row)
        status = "error" if row["error"] else "ok"
        message = (
            f"  warmup {i + 1}/{requests_num}: {status}, "
            f"TTFT={row['latency']['ttft_ms']:.1f}ms, "
            f"E2E={row['latency']['e2e_ms']:.1f}ms"
        )
        if row["error"]:
            message += f", error={row['error']}"
        print(message)
    return rows


def build_prediction_record(
    sample: Dict[str, Any],
    row: Dict[str, Any],
    score: Dict[str, Any],
) -> Dict[str, Any]:
    """Compact per-sample record for batch JSONL output."""
    return {
        "id": sample.get("id"),
        "task": sample.get("task"),
        "subject": sample.get("subject"),
        "answer_type": sample.get("answer_type"),
        "answer": sample.get("answer"),
        "prediction": row.get("prediction", ""),
        "raw_prediction": row.get("raw_prediction"),
        "status": "error" if row.get("error") else "ok",
        "error": row.get("error"),
        "score": score,
        "latency": row.get("latency", {}),
        "usage": {
            "prompt_tokens": row.get("usage", {}).get("prompt_tokens", 0),
            "completion_tokens": row.get("usage", {}).get("completion_tokens", 0),
            "total_tokens": row.get("usage", {}).get("total_tokens", 0),
            "finish_reason": row.get("usage", {}).get("finish_reason"),
            "reasoning_closed": row.get("usage", {}).get("reasoning_closed"),
            "visible_token_count": row.get("usage", {}).get("visible_token_count"),
            "think_end_token_id": row.get("usage", {}).get("think_end_token_id"),
        },
        "throughput": row.get("throughput", {}),
        "stream": {
            "sse_chunk_count": row.get("usage", {}).get("sse_chunk_count", 0),
            "content_chunk_count": row.get("usage", {}).get("chunk_count", 0),
            "empty_chunk_count": row.get("usage", {}).get("empty_chunk_count", 0),
            "role_chunk_count": row.get("usage", {}).get("role_chunk_count", 0),
            "reasoning_chunk_count": row.get("usage", {}).get(
                "reasoning_chunk_count", 0
            ),
        },
        "meta": sample.get("meta", {}),
    }


def run_batch(
    samples: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    args: argparse.Namespace,
    health: Dict[str, Any],
) -> int:
    output_path = args.output
    if output_path is None:
        output_dir = resolve_path(cfg["output"]["root_dir"])
        output_path = output_dir / cfg["output"].get(
            "predictions_jsonl", "predictions.jsonl"
        )

    # Truncate old output before appending this run.
    out_path = resolve_path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("", encoding="utf-8")

    warmup_sample = samples[0]
    warmup_rows = run_warmup(warmup_sample, cfg, args.no_warmup)
    if any(row.get("error") for row in warmup_rows):
        print("Warmup failed; aborting batch run. Use --no-warmup to skip warmup after diagnosing the server.")
        return 1

    print(f"\n--- Batch run ({len(samples)} samples) ---")
    records: List[Dict[str, Any]] = []
    fail_fast = cfg.get("runtime", {}).get("fail_fast", False)

    for idx, sample in enumerate(samples, start=1):
        row = run_one_sample(sample, cfg)
        raw_prediction = row.get("prediction", "")
        row = dict(row)
        row["raw_prediction"] = raw_prediction
        row["prediction"] = postprocess_prediction(sample, raw_prediction)
        score = score_prediction(sample, row.get("prediction", ""), cfg)
        record = build_prediction_record(sample, row, score)
        records.append(record)
        append_jsonl(out_path, record)

        status = record["status"]
        correct = score.get("correct")
        correct_text = "n/a" if correct is None else ("yes" if correct else "no")
        print(
            f"[{idx}/{len(samples)}] {sample.get('id')} "
            f"status={status} correct={correct_text} "
            f"ttft={row['latency']['ttft_ms']:.1f}ms "
            f"finish={row.get('usage', {}).get('finish_reason') or 'n/a'} "
            f"reasoning_closed={row.get('usage', {}).get('reasoning_closed')} "
            f"visible_tokens={row.get('usage', {}).get('visible_token_count')} "
            f"pred={row.get('prediction', '').strip()!r}"
        )

        if row.get("error") and fail_fast:
            break

    summary = summarize_records(records)
    summary["run"] = {
        "name": cfg.get("run", {}).get("name"),
        "model": cfg.get("model", {}).get("served_model_name")
        or cfg.get("model", {}).get("name"),
        "server": cfg.get("server", {}).get("base_url"),
        "endpoint": cfg.get("server", {}).get("endpoint"),
        "health": health,
    }
    summary_path = out_path.with_name("summary.json")
    report_path = out_path.with_name(cfg["output"].get("report_md", "report.md"))
    write_json(summary_path, summary)
    write_report_md(report_path, summary, records)
    print_batch_summary(summary, out_path)
    print(f"Summary     : {summary_path}")
    print(f"Report      : {report_path}")
    return 1 if summary["errors"] else 0


def run_single(
    samples: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    args: argparse.Namespace,
    health: Dict[str, Any],
) -> int:
    sample = select_sample(samples, args)
    warmup_rows = run_warmup(sample, cfg, args.no_warmup)
    print("\n--- Measured request ---")
    row = run_one_sample(sample, cfg)
    raw_prediction = row.get("prediction", "")
    row = dict(row)
    row["raw_prediction"] = raw_prediction
    row["prediction"] = postprocess_prediction(sample, raw_prediction)
    score = score_prediction(sample, row.get("prediction", ""), cfg)

    output_path = args.output or OUTPUT_PATH
    record = build_output_record(
        cfg=cfg,
        config_path=args.config,
        sample=sample,
        row=row,
        score=score,
        warmup_rows=warmup_rows,
        health=health,
    )
    write_json(output_path, record)
    print_result_card(record, output_path)
    return 1 if row["error"] else 0


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    dataset_cfg = cfg["dataset"]
    samples = load_samples(
        dataset_cfg["manifest"],
        limit=args.limit if args.limit is not None else dataset_cfg.get("limit"),
        task_filter=dataset_cfg.get("task_filter"),
    )
    if not samples:
        raise RuntimeError("No samples loaded after applying dataset filters.")

    health = check_health(cfg)

    if args.all:
        return run_batch(samples, cfg, args, health)
    return run_single(samples, cfg, args, health)


if __name__ == "__main__":
    raise SystemExit(main())

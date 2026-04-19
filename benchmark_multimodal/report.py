from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import requests
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def resolve_path(path: str | Path, base_dir: Path = PROJECT_ROOT) -> Path:
    """Resolve repo-relative paths while still accepting absolute paths."""
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    return (base_dir / p).resolve()

def fmt_ms(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.1f} ms"

def fmt_float(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}"


def clean_prediction(value: str) -> str:
    return value.strip().replace("\n", "\\n")

def status_from_row(row: Dict[str, Any]) -> str:
    return "error" if row.get("error") else "ok"


def build_output_record(
    cfg: Dict[str, Any],
    config_path: str | Path,
    sample: Dict[str, Any],
    row: Dict[str, Any],
    score: Dict[str, Any],
    warmup_rows: List[Dict[str, Any]],
    health: Dict[str, Any],
) -> Dict[str, Any]:
    server_cfg = cfg["server"]
    return {
        "run": {
            "name": cfg.get("run", {}).get("name"),
            "config": str(resolve_path(config_path)),
            "model": cfg.get("model", {}).get("served_model_name")
            or cfg.get("model", {}).get("name"),
            "server": server_cfg.get("base_url"),
            "endpoint": server_cfg.get("endpoint"),
            "health": health,
        },
        "sample": {
            "id": sample.get("id"),
            "task": sample.get("task"),
            "answer_type": sample.get("answer_type"),
            "answer": sample.get("answer"),
            "meta": sample.get("meta", {}),
        },
        "warmup": [
            {
                "status": status_from_row(warmup),
                "latency": warmup.get("latency", {}),
                "usage": warmup.get("usage", {}),
                "error": warmup.get("error"),
            }
            for warmup in warmup_rows
        ],
        "result": {
            "status": status_from_row(row),
            "prediction": row.get("prediction", ""),
            "error": row.get("error"),
        },
        "score": score,
        "latency": row.get("latency", {}),
        "usage": {
            "prompt_tokens": row.get("usage", {}).get("prompt_tokens", 0),
            "completion_tokens": row.get("usage", {}).get("completion_tokens", 0),
            "total_tokens": row.get("usage", {}).get("total_tokens", 0),
            "finish_reason": row.get("usage", {}).get("finish_reason"),
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
    }

def print_result_card(record: Dict[str, Any], output_path: str | Path) -> None:
    run = record["run"]
    sample = record["sample"]
    result = record["result"]
    score = record["score"]
    latency = record["latency"]
    usage = record["usage"]
    throughput = record["throughput"]
    stream = record["stream"]
    health = run["health"]

    print("\nBenchmark Run")
    print(f"Config      : {run['config']}")
    print(f"Server      : {run['server']}")
    print(f"Endpoint    : {run['endpoint']}")
    print(f"Health      : {health['status_code'] if health['status_code'] else 'failed'}")
    if health.get("error"):
        print(f"Health Error: {health['error']}")
    print(f"Model       : {run['model']}")
    print(f"Sample      : {sample['id']}")
    print(f"Task        : {sample.get('task')}")
    print(f"Answer      : {sample.get('answer')}")

    if record["warmup"]:
        print("\nWarmup")
        for idx, warmup in enumerate(record["warmup"], start=1):
            w_latency = warmup.get("latency", {})
            print(
                f"  #{idx:<2} {warmup['status']:<5} "
                f"TTFT {fmt_ms(w_latency.get('ttft_ms')):<12} "
                f"E2E {fmt_ms(w_latency.get('e2e_ms'))}"
            )

    print("\nMeasured")
    print(f"  Status     : {result['status']}")
    if result.get("error"):
        print(f"  Error      : {result['error']}")
    print(f"  Prediction : {clean_prediction(result.get('prediction', ''))}")
    print(
        "  Correct    : "
        + (
            "n/a"
            if score.get("correct") is None
            else ("yes" if score.get("correct") else "no")
        )
    )
    print(f"  Score      : {fmt_float(score.get('score'))}")
    print(f"  Scorer     : {score.get('scorer')}")
    print(f"  Reason     : {score.get('reason')}")

    print("\nLatency")
    print(f"  First SSE  : {fmt_ms(latency.get('first_sse_ms'))}")
    print(f"  TTFT       : {fmt_ms(latency.get('ttft_ms'))}")
    print(f"  Decode     : {fmt_ms(latency.get('decode_ms'))}")
    print(f"  E2E        : {fmt_ms(latency.get('e2e_ms'))}")

    print("\nTokens")
    print(f"  Prompt     : {usage.get('prompt_tokens', 0)}")
    print(f"  Completion : {usage.get('completion_tokens', 0)}")
    print(f"  Total      : {usage.get('total_tokens', 0)}")
    print(f"  Finish     : {usage.get('finish_reason') or 'n/a'}")

    print("\nThroughput")
    print(f"  Prompt TPS : {fmt_float(throughput.get('prompt_tps'))}")
    print(f"  Decode TPS : {fmt_float(throughput.get('decode_tps'))}")

    print("\nStream")
    print(f"  SSE chunks     : {stream.get('sse_chunk_count', 0)}")
    print(f"  Content chunks : {stream.get('content_chunk_count', 0)}")
    print(f"  Empty chunks   : {stream.get('empty_chunk_count', 0)}")
    print(f"  Role chunks    : {stream.get('role_chunk_count', 0)}")
    print(f"  Reason chunks  : {stream.get('reasoning_chunk_count', 0)}")

    print("\nOutput")
    print(f"  File       : {resolve_path(output_path)}")


def avg(values: List[float]) -> float | None:
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def summarize_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    errors = sum(1 for record in records if record.get("status") == "error")
    scored = [r for r in records if r.get("score", {}).get("correct") is not None]
    correct = sum(1 for record in scored if record["score"].get("correct"))

    summary = {
        "total": total,
        "errors": errors,
        "scored": len(scored),
        "correct": correct,
        "accuracy": correct / len(scored) if scored else None,
        "avg_ttft_ms": avg([r.get("latency", {}).get("ttft_ms") for r in records]),
        "avg_e2e_ms": avg([r.get("latency", {}).get("e2e_ms") for r in records]),
        "avg_prompt_tps": avg(
            [r.get("throughput", {}).get("prompt_tps") for r in records]
        ),
        "avg_decode_tps": avg(
            [r.get("throughput", {}).get("decode_tps") for r in records]
        ),
        "by_subject": {},
        "by_task": {},
    }

    for group_key, out_key in (("subject", "by_subject"), ("task", "by_task")):
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for record in records:
            name = str(record.get(group_key) or "unknown")
            groups.setdefault(name, []).append(record)

        for name, rows in groups.items():
            group_scored = [
                row for row in rows if row.get("score", {}).get("correct") is not None
            ]
            group_correct = sum(
                1 for row in group_scored if row["score"].get("correct")
            )
            summary[out_key][name] = {
                "total": len(rows),
                "errors": sum(1 for row in rows if row.get("status") == "error"),
                "scored": len(group_scored),
                "correct": group_correct,
                "accuracy": group_correct / len(group_scored)
                if group_scored
                else None,
                "avg_ttft_ms": avg(
                    [row.get("latency", {}).get("ttft_ms") for row in rows]
                ),
                "avg_e2e_ms": avg(
                    [row.get("latency", {}).get("e2e_ms") for row in rows]
                ),
                "avg_prompt_tps": avg(
                    [row.get("throughput", {}).get("prompt_tps") for row in rows]
                ),
                "avg_decode_tps": avg(
                    [row.get("throughput", {}).get("decode_tps") for row in rows]
                ),
            }

    return summary


def print_batch_summary(summary: Dict[str, Any], output_path: str | Path) -> None:
    print("\nBatch Summary")
    print(f"Samples     : {summary['total']}")
    print(f"Errors      : {summary['errors']}")
    if summary["accuracy"] is None:
        print("Accuracy    : n/a")
    else:
        print(
            f"Accuracy    : {summary['accuracy'] * 100:.2f}% "
            f"({summary['correct']}/{summary['scored']})"
        )

    if summary["avg_ttft_ms"] is None:
        print("Avg TTFT    : n/a")
    else:
        print(f"Avg TTFT    : {summary['avg_ttft_ms']:.1f} ms")

    if summary["avg_e2e_ms"] is None:
        print("Avg E2E     : n/a")
    else:
        print(f"Avg E2E     : {summary['avg_e2e_ms']:.1f} ms")

    if summary["avg_decode_tps"] is None:
        print("Avg Dec TPS : n/a")
    else:
        print(f"Avg Dec TPS : {summary['avg_decode_tps']:.2f}")

    if summary["avg_prompt_tps"] is None:
        print("Avg Prefill : n/a")
    else:
        print(f"Avg Prefill : {summary['avg_prompt_tps']:.2f} tok/s")

    print(f"Output      : {resolve_path(output_path)}")


def fmt_percent(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def md_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    return (
        text.replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\n", "\\n")
        .replace("\r", "")
    )


def md_num(value: Any, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def summary_table_rows(groups: Dict[str, Dict[str, Any]]) -> List[str]:
    rows = []
    for name, item in sorted(groups.items()):
        rows.append(
            "| "
            + " | ".join(
                [
                    md_escape(name),
                    str(item.get("total", 0)),
                    str(item.get("errors", 0)),
                    str(item.get("correct", 0)),
                    fmt_percent(item.get("accuracy")),
                    md_num(item.get("avg_ttft_ms")),
                    md_num(item.get("avg_e2e_ms")),
                    md_num(item.get("avg_prompt_tps"), 2),
                    md_num(item.get("avg_decode_tps"), 2),
                ]
            )
            + " |"
        )
    return rows


def error_example_rows(records: List[Dict[str, Any]], limit: int) -> List[str]:
    rows = []
    for record in records:
        if len(rows) >= limit:
            break
        if record.get("status") == "error" or record.get("score", {}).get("correct") is False:
            score = record.get("score", {})
            rows.append(
                "| "
                + " | ".join(
                    [
                        md_escape(record.get("id")),
                        md_escape(record.get("task")),
                        md_escape(record.get("subject")),
                        md_escape(record.get("answer")),
                        md_escape(clean_prediction(record.get("prediction", ""))[:120]),
                        md_escape(score.get("scorer")),
                        md_escape(score.get("reason") or record.get("error")),
                    ]
                )
                + " |"
            )
    return rows


def write_report_md(
    path: str | Path,
    summary: Dict[str, Any],
    records: List[Dict[str, Any]],
    *,
    max_error_examples: int = 20,
) -> None:
    """Write a human-readable Markdown report for a batch benchmark run."""
    out_path = resolve_path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    run = summary.get("run", {})
    health = run.get("health", {})
    lines = [
        "# Benchmark Report",
        "",
        "## Run",
        "",
        "| Key | Value |",
        "|---|---|",
        f"| Name | {md_escape(run.get('name'))} |",
        f"| Model | {md_escape(run.get('model'))} |",
        f"| Server | {md_escape(run.get('server'))} |",
        f"| Endpoint | {md_escape(run.get('endpoint'))} |",
        f"| Health | {md_escape(health.get('status_code') if health.get('status_code') else 'failed')} |",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Samples | {summary.get('total', 0)} |",
        f"| Errors | {summary.get('errors', 0)} |",
        f"| Scored | {summary.get('scored', 0)} |",
        f"| Correct | {summary.get('correct', 0)} |",
        f"| Accuracy | {fmt_percent(summary.get('accuracy'))} |",
        f"| Avg TTFT ms | {md_num(summary.get('avg_ttft_ms'))} |",
        f"| Avg E2E ms | {md_num(summary.get('avg_e2e_ms'))} |",
        f"| Avg Effective Prefill TPS | {md_num(summary.get('avg_prompt_tps'), 2)} |",
        f"| Avg Decode TPS | {md_num(summary.get('avg_decode_tps'), 2)} |",
    ]

    by_task = summary.get("by_task") or {}
    if by_task:
        lines.extend(
            [
                "",
                "## By Task",
                "",
                "| Task | Total | Errors | Correct | Accuracy | Avg TTFT ms | Avg E2E ms | Avg Prefill TPS | Avg Decode TPS |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
                *summary_table_rows(by_task),
            ]
        )

    by_subject = summary.get("by_subject") or {}
    if by_subject:
        lines.extend(
            [
                "",
                "## By Subject",
                "",
                "| Subject | Total | Errors | Correct | Accuracy | Avg TTFT ms | Avg E2E ms | Avg Prefill TPS | Avg Decode TPS |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
                *summary_table_rows(by_subject),
            ]
        )

    error_rows = error_example_rows(records, max_error_examples)
    lines.extend(
        [
            "",
            "## Error Examples",
            "",
        ]
    )
    if error_rows:
        lines.extend(
            [
                "| ID | Task | Subject | Answer | Prediction | Scorer | Reason |",
                "|---|---|---|---|---|---|---|",
                *error_rows,
            ]
        )
    else:
        lines.append("No errors found in the recorded samples.")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

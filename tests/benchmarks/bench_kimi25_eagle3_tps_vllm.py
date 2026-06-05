#!/usr/bin/env python3
"""Run the Kimi K2.5 Eagle3 TPS comparison against a vLLM OpenAI server."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import bench_eagle3_vllm_openai as openai_bench


DEFAULT_BENCHMARKS = [
    "speedbench_qualitative:80:coding",
    "speedbench_qualitative:80:math",
    "speedbench_qualitative:80:humanities",
    "speedbench_qualitative:80:stem",
    "speedbench_qualitative:80:writing",
    "speedbench_qualitative:80:summarization",
    "speedbench_qualitative:80:roleplay",
    "speedbench_qualitative:80:rag",
    "speedbench_qualitative:80:multilingual",
    "speedbench_qualitative:80:reasoning",
    "speedbench_qualitative:80:qa",
]
DEFAULT_SERVED_MODEL_NAME = "kimi25"


def temp_label(value: float) -> str:
    return str(value).replace(".", "p")


def benchmark_name(item: str) -> str:
    return item.split(":", 1)[0]


def display_benchmark_name(item: str) -> str:
    parts = item.split(":")
    if parts[0] == "speedbench_qualitative" and len(parts) >= 3:
        return parts[2]
    return benchmark_name(item)


def load_results(path: str) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def compare_results(args: argparse.Namespace) -> None:
    baseline = load_results(args.baseline_json)
    eagle3 = load_results(args.eagle3_json)
    baseline_results = baseline.get("benchmarks", {})
    eagle3_results = eagle3.get("benchmarks", {})

    print(
        "dataset\tn\tbaseline_tps\teagle3_tps\tspeedup\t"
        "accept_length\tdraft_acceptance_rate"
    )
    for item, eagle_metrics in eagle3_results.items():
        base_metrics = baseline_results.get(item)
        if base_metrics is None:
            base_metrics = baseline_results.get(benchmark_name(item))
        if base_metrics is None:
            continue

        baseline_tps = float(base_metrics.get("output_throughput") or 0.0)
        eagle_tps = float(eagle_metrics.get("output_throughput") or 0.0)
        speedup = eagle_tps / baseline_tps if baseline_tps > 0 else None
        acceptance_rate = eagle_metrics.get("draft_acceptance_rate")
        acceptance_rate_text = (
            "n/a" if acceptance_rate is None else f"{acceptance_rate:.2%}"
        )
        print(
            "\t".join(
                [
                    display_benchmark_name(item),
                    str(eagle_metrics.get("num_questions", "n/a")),
                    fmt(baseline_tps),
                    fmt(eagle_tps),
                    fmt(speedup, 3),
                    fmt(eagle_metrics.get("accept_length"), 3),
                    acceptance_rate_text,
                ]
            )
        )


def run_benchmarks(args: argparse.Namespace) -> None:
    name = args.name or (
        f"kimi25_vllm_concurrency{args.concurrency}_"
        f"temp{temp_label(args.temperature)}"
    )
    bench_args = SimpleNamespace(
        model_path=args.model_path,
        host=args.host,
        port=args.port,
        benchmark_list=args.benchmark_list,
        output_dir=args.output_dir,
        name=name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        num_threads=args.concurrency,
    )
    print(
        f"Running vLLM TPS: model={args.model_path} "
        f"port={args.port} concurrency={args.concurrency}"
    )
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    openai_bench.run_all_benchmarks(bench_args, timestamp)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Kimi-K2.5-MXFP4 TPS test on vLLM. "
            "SPEED-Bench Table 1 settings: BS=32."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a vLLM TPS benchmark")
    run_parser.add_argument("--model-path", default=DEFAULT_SERVED_MODEL_NAME)
    run_parser.add_argument("--host", default="localhost")
    run_parser.add_argument("--port", type=int, default=30000)
    run_parser.add_argument("--benchmark-list", nargs="+", default=DEFAULT_BENCHMARKS)
    run_parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent))
    run_parser.add_argument("--name", default=None)
    run_parser.add_argument("--temperature", type=float, choices=[0.0, 1.0], default=0.0)
    run_parser.add_argument("--concurrency", "--batch-size", type=int, default=32)
    run_parser.add_argument("--max-tokens", type=int, default=None)
    run_parser.add_argument("--timeout", type=float, default=600.0)
    run_parser.set_defaults(func=run_benchmarks)

    compare_parser = subparsers.add_parser(
        "compare", help="Compare baseline and Eagle3 JSON outputs"
    )
    compare_parser.add_argument("--baseline-json", required=True)
    compare_parser.add_argument("--eagle3-json", required=True)
    compare_parser.set_defaults(func=compare_results)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

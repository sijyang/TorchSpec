#!/usr/bin/env python3
"""Run the Kimi K2.5 Eagle3 TPS comparison against an ATOM OpenAI server."""

from __future__ import annotations

import argparse
import json
import os
import signal
import shlex
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import requests


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


def temp_label(value: float) -> str:
    return str(value).replace(".", "p")


def normalize_base_url(host: str, port: int) -> str:
    if not host.startswith(("http://", "https://")):
        host = f"http://{host}"
    return f"{host}:{port}"


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


def load_openai_bench() -> Any:
    import bench_eagle3_vllm_openai as openai_bench

    return openai_bench


def fetch_atom_spec_metrics(base_url: str) -> dict[str, float]:
    """Fetch ATOM speculative decoding counters from /debug/mtp_stats."""
    metrics = {
        "num_drafts": 0.0,
        "num_draft_tokens": 0.0,
        "num_accepted_tokens": 0.0,
    }
    try:
        response = requests.get(f"{base_url}/debug/mtp_stats", timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        print(f"Warning: failed to fetch ATOM /debug/mtp_stats: {exc}")
        return metrics

    if not data.get("enabled", False):
        return metrics

    distribution = data.get("distribution", {})
    metrics["num_drafts"] = float(sum(int(v) for v in distribution.values()))
    metrics["num_draft_tokens"] = float(data.get("total_draft_tokens") or 0.0)
    metrics["num_accepted_tokens"] = float(data.get("total_accepted_tokens") or 0.0)
    return metrics


def patch_openai_bench_for_atom(openai_bench: Any) -> None:
    openai_bench.fetch_spec_metrics = fetch_atom_spec_metrics


def run_benchmarks(args: argparse.Namespace) -> None:
    validate_args(args)
    openai_bench = load_openai_bench()
    patch_openai_bench_for_atom(openai_bench)
    name = args.name or (
        f"kimi25_{args.variant}_atom_concurrency{args.concurrency}_"
        f"dl{args.draft_length}_temp{temp_label(args.temperature)}"
    )
    bench_args = SimpleNamespace(
        model_path=args.target_model_path,
        host=args.host,
        port=args.server_port,
        benchmark_list=args.benchmark_list,
        output_dir=args.output_dir,
        name=name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        num_threads=args.concurrency,
    )
    print(
        f"Running ATOM {args.variant}: target_model={args.target_model_path} "
        f"draft_model={args.draft_model_path if args.variant == 'eagle3' else 'n/a'} "
        f"server_port={args.server_port} concurrency={args.concurrency} "
        f"draft_length={args.draft_length}"
    )
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    openai_bench.run_all_benchmarks(bench_args, timestamp)


def build_server_command(args: argparse.Namespace) -> list[str]:
    validate_args(args)
    command = [
        sys.executable,
        "-m",
        "atom.entrypoints.openai_server",
        "--model",
        args.target_model_path,
        "--host",
        args.server_host,
        "--server-port",
        str(args.server_port),
        "--port",
        str(args.internal_port),
        "--kv_cache_dtype",
        args.kv_cache_dtype,
        "-tp",
        str(args.tensor_parallel_size),
        "--level",
        str(args.level),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    if args.variant == "eagle3":
        command.extend(
            [
                "--method",
                "eagle3",
                "--draft-model",
                args.draft_model_path,
                "--num-speculative-tokens",
                str(args.draft_length),
            ]
        )
    command.extend(args.extra_server_args)
    return command


def print_server_command(args: argparse.Namespace) -> None:
    command = build_server_command(args)
    print("AITER_LOG_LEVEL=WARNING " + " ".join(shlex.quote(part) for part in command))


def wait_for_server(
    base_url: str,
    process: subprocess.Popen[Any],
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    health_url = f"{base_url}/health"
    last_error: str | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"ATOM server exited early with code {process.returncode}. "
                "Check the server log for details."
            )
        try:
            response = requests.get(health_url, timeout=5)
            if response.ok:
                print(f"ATOM server is ready: {health_url}")
                return
            last_error = f"HTTP {response.status_code}"
        except Exception as exc:
            last_error = repr(exc)
        time.sleep(5)
    raise TimeoutError(f"Timed out waiting for {health_url}; last error: {last_error}")


def terminate_server(process: subprocess.Popen[Any], timeout: float = 120.0) -> None:
    if process.poll() is not None:
        return
    print("Stopping ATOM server...")
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print("ATOM server did not stop after SIGINT; killing it.")
        process.kill()
        process.wait(timeout=30)


def run_with_server(args: argparse.Namespace) -> None:
    validate_args(args)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = Path(args.server_log) if args.server_log else (
        Path(args.output_dir) / f"kimi25_{args.variant}_atom_server_{timestamp}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)

    command = build_server_command(args)
    env = os.environ.copy()
    env.setdefault("AITER_LOG_LEVEL", "WARNING")
    atom_root = Path(args.atom_root).expanduser().resolve() if args.atom_root else Path.cwd()
    base_url = normalize_base_url(args.host, args.server_port)

    print("Starting ATOM server:")
    print("AITER_LOG_LEVEL=WARNING " + " ".join(shlex.quote(part) for part in command))
    print(f"Server log: {log_path}")
    with log_path.open("w") as log_file:
        process = subprocess.Popen(
            command,
            cwd=atom_root,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            wait_for_server(base_url, process, args.server_ready_timeout)
            run_benchmarks(args)
        finally:
            if args.keep_server_running:
                print(f"Leaving ATOM server running with pid {process.pid}")
            else:
                terminate_server(process)


def add_common_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--variant", choices=["baseline", "eagle3"], required=True)
    parser.add_argument("--target-model-path", required=True)
    parser.add_argument("--draft-model-path", default=None)
    parser.add_argument("--draft-length", type=int, default=3)


def add_server_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--atom-root",
        default=None,
        help="ATOM repository root. Defaults to the current working directory.",
    )
    parser.add_argument("--server-host", default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--internal-port", type=int, default=8006)
    parser.add_argument("--kv-cache-dtype", default="fp8")
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=8)
    parser.add_argument("--level", type=int, default=3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--extra-server-args", nargs="*", default=[])


def add_benchmark_args(
    parser: argparse.ArgumentParser,
    *,
    include_server_port: bool = True,
) -> None:
    parser.add_argument("--host", default="localhost")
    if include_server_port:
        parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--benchmark-list", nargs="+", default=DEFAULT_BENCHMARKS)
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--name", default=None)
    parser.add_argument("--temperature", type=float, choices=[0.0, 1.0], default=0.0)
    parser.add_argument("--concurrency", "--batch-size", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=600.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Kimi-K2.5-MXFP4 baseline vs Eagle3 TPS test on ATOM. "
            "SPEED-Bench Table 1 settings: BS=32, DL=3."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    server_parser = subparsers.add_parser(
        "server-command", help="Print an ATOM OpenAI server command"
    )
    add_common_model_args(server_parser)
    add_server_args(server_parser)
    server_parser.set_defaults(func=print_server_command)

    run_parser = subparsers.add_parser("run", help="Run one ATOM server variant")
    add_common_model_args(run_parser)
    add_benchmark_args(run_parser)
    run_parser.set_defaults(func=run_benchmarks)

    run_server_parser = subparsers.add_parser(
        "run-with-server", help="Start ATOM server, run benchmark, then stop it"
    )
    add_common_model_args(run_server_parser)
    add_server_args(run_server_parser)
    add_benchmark_args(run_server_parser, include_server_port=False)
    run_server_parser.add_argument("--server-ready-timeout", type=float, default=1800.0)
    run_server_parser.add_argument("--server-log", default=None)
    run_server_parser.add_argument("--keep-server-running", action="store_true")
    run_server_parser.set_defaults(func=run_with_server)

    compare_parser = subparsers.add_parser(
        "compare", help="Compare baseline and Eagle3 JSON outputs"
    )
    compare_parser.add_argument("--baseline-json", required=True)
    compare_parser.add_argument("--eagle3-json", required=True)
    compare_parser.set_defaults(func=compare_results)

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.variant == "eagle3" and not args.draft_model_path:
        raise ValueError("--draft-model-path is required when --variant=eagle3")


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

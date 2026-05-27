# OpenAI-Compatible Benchmark Scripts

This directory contains standalone benchmark helpers for servers that expose an
OpenAI-compatible API, such as vLLM, SGLang, or ATOM.

The scripts do not import TorchSpec or SpecForge internals. They can be run from
any working directory as long as their Python dependencies are installed and the
target server is already running.

By default, benchmark result files are written under `outputs/`.

## Generic Benchmarks

Use `bench_eagle3_vllm_openai.py` for text, code, math, and multimodal benchmark
datasets that are loaded through Hugging Face `datasets` or public JSONL files.

Install minimal dependencies:

```bash
pip install requests datasets
```

Example:

```bash
python tests/benchmarks/bench_eagle3_vllm_openai.py \
    --model-path /data/models/amd/Kimi-K2.5-MXFP4 \
    --host localhost \
    --port 30000 \
    --benchmark-list mtbench:80 gsm8k:200 humaneval:200 math500:200 ceval:200 aime \
    --name kimi25_vllm \
    --num-threads 8
```

Supported benchmark names are:

```text
aime ceval financeqa gpqa gsm8k humaneval livecodebench math500
mmlu mmstar mtbench simpleqa
```

The script writes one JSON file with aggregate metrics only. It does not save
per-sample prompts, outputs, predictions, or a separate log file. It also reads
vLLM speculative decoding counters from `/metrics` when available, and reports
accept length and draft acceptance rate.

Default output directory:

```text
outputs/benchmarks/
```

## BFCL Function Calling

Use `bfcl_eval_eagle3_vllm_openai.py` for BFCL function-calling benchmarks
against a local OpenAI-compatible Kimi EAGLE3 vLLM server. Defaults match
`http://localhost:30000/v1`, served model `kimi25`, and tokenizer path
`/data/models/amd/Kimi-K2.5-MXFP4`.

Install:

```bash
pip install bfcl-eval
```

Run against the already-started local server:

```bash
python tests/benchmarks/bfcl_eval_eagle3_vllm_openai.py \
    --num-threads 8
```

The script also collects EAGLE3 speculative decoding metrics from the vLLM
`/metrics` endpoint and writes a ready-to-paste summary table:

```bash
python tests/benchmarks/bfcl_eval_eagle3_vllm_openai.py \
    --num-threads 8
```

It runs categories one by one, snapshots the same vLLM spec counters used by
`bench_eagle3_vllm_openai.py` before and after each category, computes
`Accept Length = total_completion_tokens / num_drafts` and
`Draft Acceptance Rate = num_accepted_tokens / num_draft_tokens`, and writes:

Here `num_drafts`, `num_draft_tokens`, and `num_accepted_tokens` come from
vLLM's Prometheus counters. The benchmark table intentionally follows
SpecForge's `output_tokens / verify_steps` accept-length definition, not
vLLM's log-only `Mean acceptance length = 1 + accepted_tokens / num_drafts`.

```text
outputs/benchmarks/bfcl_eagle3_summary.csv
```

By default it runs the eight single-turn categories commonly shown in
function-calling tables:

```text
simple_python multiple parallel parallel_multiple
live_simple live_multiple live_parallel live_parallel_multiple
```

Run one category:

```bash
python tests/benchmarks/bfcl_eval_eagle3_vllm_openai.py \
    --category live_simple
```

You can also override local server settings:

```bash
python tests/benchmarks/bfcl_eval_eagle3_vllm_openai.py \
    --base-url http://localhost:30000/v1 \
    --served-model-name kimi25 \
    --tokenizer-path /data/models/amd/Kimi-K2.5-MXFP4 \
    --num-threads 8
```

Generated BFCL files are written under:

```text
outputs/benchmarks/result/<model-name>/
outputs/benchmarks/score/<model-name>/
```

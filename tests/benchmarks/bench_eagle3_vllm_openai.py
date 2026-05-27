#!/usr/bin/env python3
"""Run lightweight benchmarks against an OpenAI-compatible server.

Example:
    python3 bench_eagle3_vllm_openai.py \
        --model-path /data/models/amd/Kimi-K2.5-MXFP4 \
        --port 30000 \
        --benchmark-list mtbench:80 gsm8k:200 humaneval:200 math500:200 ceval:200 aime mmlu simpleqa financeqa livecodebench mmstar \
        --name kimi25_phase1_vllm_eagle3
"""

from __future__ import annotations

import argparse
import ast
import base64
import concurrent.futures as futures
import json
import math
import os
import random
import re
import tempfile
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests
from datasets import concatenate_datasets, load_dataset


INVALID = -9999999
TORCHSPEC_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = TORCHSPEC_ROOT / "outputs" / "benchmarks"


@dataclass
class BenchmarkMetrics:
    benchmark: str
    num_questions: int
    accuracy: float | None
    latency: float
    output_throughput: float
    total_completion_tokens: int
    accept_length: float | None
    draft_acceptance_rate: float | None
    spec_num_drafts: float | None
    spec_num_draft_tokens: float | None
    spec_num_accepted_tokens: float | None
    valid_predictions: int
    failed_requests: int


def parse_benchmark_item(item: str) -> tuple[str, int | None, list[str] | None]:
    parts = item.split(":")
    if len(parts) == 1:
        return parts[0], None, None
    if len(parts) == 2:
        return parts[0], int(parts[1]), None
    if len(parts) == 3:
        return parts[0], int(parts[1]), parts[2].split(",")
    raise ValueError(f"Invalid benchmark item: {item}")


def normalize_base_url(host: str, port: int) -> str:
    if not host.startswith(("http://", "https://")):
        host = f"http://{host}"
    return f"{host}:{port}"


def post_openai(
    base_url: str,
    path: str,
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    response = requests.post(
        f"{base_url}{path}",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def completion_tokens(data: dict[str, Any]) -> int:
    return int(data.get("usage", {}).get("completion_tokens") or 0)


def chat_completion(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
    stop: list[str] | None = None,
) -> tuple[str, int]:
    return chat_messages_completion(
        base_url=base_url,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
        stop=stop,
    )


def chat_messages_completion(
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    timeout: float,
    stop: list[str] | None = None,
) -> tuple[str, int]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if stop:
        payload["stop"] = stop
    data = post_openai(base_url, "/v1/chat/completions", payload, timeout)
    text = data["choices"][0]["message"].get("content") or ""
    return text, completion_tokens(data)


def text_completion(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
    stop: list[str] | None = None,
) -> tuple[str, int]:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if stop:
        payload["stop"] = stop
    data = post_openai(base_url, "/v1/completions", payload, timeout)
    text = data["choices"][0].get("text") or ""
    return text, completion_tokens(data)


def fetch_spec_metrics(base_url: str) -> dict[str, float]:
    """Fetch vLLM speculative decoding counters from Prometheus /metrics."""
    # vLLM definitions:
    # - spec_decode_num_drafts: number of speculative decoding draft/verify steps.
    # - spec_decode_num_draft_tokens: number of generated draft tokens.
    # - spec_decode_num_accepted_tokens: number of accepted draft tokens.
    wanted = {
        "vllm:spec_decode_num_drafts_total": "num_drafts",
        "vllm:spec_decode_num_draft_tokens_total": "num_draft_tokens",
        "vllm:spec_decode_num_accepted_tokens_total": "num_accepted_tokens",
    }
    metrics = {name: 0.0 for name in wanted.values()}
    try:
        response = requests.get(f"{base_url}/metrics", timeout=10)
        response.raise_for_status()
    except Exception as exc:
        print(f"Warning: failed to fetch /metrics for accept_length: {exc}")
        return metrics

    for line in response.text.splitlines():
        if not line or line.startswith("#"):
            continue
        metric_name = line.split("{", 1)[0].split(" ", 1)[0]
        key = wanted.get(metric_name)
        if key is None:
            continue
        try:
            value = float(line.rsplit(" ", 1)[-1])
        except ValueError:
            continue
        metrics[key] += value
    return metrics


def download_jsonl(url: str) -> list[dict[str, Any]]:
    cache_dir = Path(os.environ.get("HF_HOME", tempfile.gettempdir())) / "specforge_openai_bench"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / Path(url).name
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def gsm8k_answer_value(answer_str: str) -> int:
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID


def load_gsm8k(num_samples: int | None, subset: list[str] | None = None) -> dict[str, Any]:
    del subset
    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    rows = download_jsonl(url)

    def one_example(i: int, include_answer: bool) -> str:
        text = "Question: " + rows[i]["question"] + "\nAnswer:"
        if include_answer:
            text += " " + rows[i]["answer"]
        return text

    few_shot = "\n\n".join(one_example(i, True) for i in range(5)) + "\n\n"
    samples = []
    for i, row in enumerate(rows):
        if num_samples is not None and i >= num_samples:
            break
        samples.append(
            {
                "prompt": few_shot + one_example(i, False),
                "label": gsm8k_answer_value(row["answer"]),
            }
        )
    return {
        "samples": samples,
        "extract": gsm8k_answer_value,
        "max_tokens": 512,
        "stop": ["Question", "Assistant:", "<|separator|>"],
        # SpecForge's GSM8K few-shot function appends raw text and then sgl.gen,
        # rather than wrapping the prompt in a chat user role.
        "endpoint": "completion",
    }


def extract_math_answer(output: str) -> str | None:
    match = re.search(r"\\boxed\{([^}]+)\}", output)
    if match:
        return match.group(1).strip()
    match = re.search(r"\\boxed\s+([^\s]+)", output)
    if match:
        return match.group(1).strip()
    for pattern in [
        r"(?:answer|Answer|ANSWER)[\s:]+([-+]?\d*\.?\d+)",
        r"(?:is|equals?|=\s*)([-+]?\d*\.?\d+)\s*$",
    ]:
        matches = re.findall(pattern, output, re.IGNORECASE)
        if matches:
            return matches[-1].strip()
    numbers = re.findall(r"[-+]?\d*\.?\d+", output)
    return numbers[-1] if numbers else None


def load_math500(num_samples: int | None, subset: list[str] | None = None) -> dict[str, Any]:
    del subset
    dataset = load_dataset("HuggingFaceH4/MATH-500")["test"]
    samples = []
    for idx, row in enumerate(dataset):
        if num_samples is not None and idx >= num_samples:
            break
        answer = str(row["answer"]).strip() if "answer" in row else extract_math_answer(row["solution"])
        samples.append({"prompt": row["problem"], "label": answer})
    return {
        "samples": samples,
        "extract": extract_math_answer,
        "max_tokens": 2048,
        "stop": None,
        "endpoint": "chat",
    }


def extract_choice(output: str) -> str | None:
    output = output.strip().upper()
    match = re.search(r"\b([ABCD])\b", output)
    if match:
        return match.group(1)
    for pattern in [
        r"\(([ABCD])\)",
        r"\[([ABCD])\]",
        r"答案[：:]\s*([ABCD])",
        r"ANSWER[：:]\s*([ABCD])",
    ]:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    match = re.search(r"([ABCD])", output)
    return match.group(1) if match else None


def format_ceval_question(question: str, options: list[str]) -> str:
    prompt = question + "\n\n选项：\n"
    for i, option in enumerate(options):
        prompt += f"{chr(65 + i)}. {option}\n"
    prompt += "\n请从A、B、C、D中选择一个答案。"
    return prompt


def load_ceval(num_samples: int | None, subset: list[str] | None = None) -> dict[str, Any]:
    all_configs = [
        "accountant",
        "advanced_mathematics",
        "art_studies",
        "basic_medicine",
        "business_administration",
        "chinese_language_and_literature",
        "civil_servant",
        "clinical_medicine",
        "college_chemistry",
        "college_economics",
        "college_physics",
        "college_programming",
        "computer_architecture",
        "computer_network",
        "discrete_mathematics",
        "education_science",
        "electrical_engineer",
        "environmental_impact_assessment_engineer",
        "fire_engineer",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_chinese",
        "high_school_geography",
        "high_school_history",
        "high_school_mathematics",
        "high_school_physics",
        "high_school_politics",
        "ideological_and_moral_cultivation",
        "law",
        "legal_professional",
        "logic",
        "mao_zedong_thought",
        "marxism",
        "metrology_engineer",
        "middle_school_biology",
        "middle_school_chemistry",
        "middle_school_geography",
        "middle_school_history",
        "middle_school_mathematics",
        "middle_school_physics",
        "middle_school_politics",
        "modern_chinese_history",
        "operating_system",
        "physician",
        "plant_protection",
        "probability_and_statistics",
        "professional_tour_guide",
        "sports_science",
        "tax_accountant",
        "teacher_qualification",
        "urban_and_rural_planner",
        "veterinary_medicine",
    ]
    configs = all_configs if not subset else subset
    datasets = []
    for config in configs:
        try:
            datasets.append(load_dataset("ceval/ceval-exam", name=config, split="test"))
        except Exception as exc:
            print(f"Warning: failed to load C-Eval config {config}: {exc}")
    dataset = concatenate_datasets(datasets)

    samples = []
    for idx, item in enumerate(dataset):
        if num_samples is not None and idx >= num_samples:
            break
        question = item.get("question") or item.get("inputs") or item.get("problem") or item.get("content")
        options = item.get("options") or item.get("choices")
        if isinstance(options, dict):
            options = [options.get(k, "") for k in ["A", "B", "C", "D"]]
        if not options:
            options = [item.get(k, item.get(f"option_{k}", "")) for k in ["A", "B", "C", "D"]]
        options = [str(opt).strip() for opt in options if opt]
        answer = str(item.get("answer") or item.get("target") or item.get("label") or item.get("correct") or "").upper().strip()
        if question and len(options) >= 2 and answer in ["A", "B", "C", "D"]:
            samples.append({"prompt": format_ceval_question(str(question), options), "label": answer})
    return {
        "samples": samples,
        "extract": extract_choice,
        "max_tokens": 2048,
        "stop": None,
        "endpoint": "chat",
    }


def extract_code(output: str) -> str | None:
    match = re.search(r"```(?:python)?\n(.*?)```", output, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"(def\s+\w+\([^)]*\):.*?)(?=\n\ndef\s+|\Z)", output, re.DOTALL)
    if match:
        return match.group(1).strip()
    return output.strip() if output.strip() else None


def code_passes_tests(code: str, test_code: str) -> bool:
    namespace: dict[str, Any] = {}
    try:
        exec(code, namespace)
        exec(test_code, namespace)
        return True
    except Exception:
        return False


def load_humaneval(num_samples: int | None, subset: list[str] | None = None) -> dict[str, Any]:
    del subset
    dataset = load_dataset("openai/openai_humaneval")["test"]
    samples = []
    for idx, row in enumerate(dataset):
        if num_samples is not None and idx >= num_samples:
            break
        samples.append(
            {
                "prompt": row["prompt"],
                "label": {
                    "test": row.get("test", ""),
                    "entry_point": row.get("entry_point", ""),
                    "prompt": row["prompt"],
                },
            }
        )
    return {
        "samples": samples,
        "extract": extract_code,
        "max_tokens": 1024,
        "stop": None,
        "endpoint": "chat",
    }


def extract_aime_answer(output: str) -> str | None:
    match = re.search(r"\\boxed\{([^}]+)\}", output)
    if match:
        numbers = re.findall(r"\d+", match.group(1).strip())
        return numbers[-1] if numbers else match.group(1).strip()
    match = re.search(r"\\boxed\s+(\d+)", output)
    if match:
        return match.group(1).strip()
    for pattern in [
        r"(?:answer|Answer|ANSWER)[\s:]+(\d+)",
        r"(?:final\s+answer|Final\s+Answer)[\s:]+(\d+)",
        r"(?:is|equals?|=\s*)(\d+)\s*$",
    ]:
        matches = re.findall(pattern, output, re.IGNORECASE)
        if matches:
            return matches[-1].strip()
    numbers = re.findall(r"\b(\d+)\b", output)
    valid_numbers = [n for n in numbers if 0 <= int(n) <= 999]
    return valid_numbers[-1] if valid_numbers else None


def load_aime(num_samples: int | None, subset: list[str] | None = None) -> dict[str, Any]:
    del subset
    dataset = load_dataset("Maxwell-Jia/AIME_2024")["train"]
    samples = []
    for idx, row in enumerate(dataset):
        if num_samples is not None and idx >= num_samples:
            break
        answer = str(row.get("Answer", row.get("answer", ""))).strip() or None
        prompt = row["Problem"] + "\nPlease reason step by step, and put your final answer within \\boxed{}."
        samples.append({"prompt": prompt, "label": answer})
    return {"samples": samples, "extract": extract_aime_answer, "max_tokens": 32768, "stop": None, "endpoint": "chat"}


GPQA_QUERY_TEMPLATE = """Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}""".strip()


def extract_answer_colon(output: str) -> str | None:
    if "Answer: " not in output:
        return None
    return output.split("Answer: ", 1)[1].strip()


def load_gpqa(num_samples: int | None, subset: list[str] | None = None) -> dict[str, Any]:
    del subset
    dataset = load_dataset("Idavidrein/gpqa", "gpqa_main")["train"]
    samples = []
    for idx, row in enumerate(dataset):
        if num_samples is not None and idx >= num_samples:
            break
        gold_index = random.randint(0, 3)
        choices = [row["Incorrect Answer 1"], row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
        choices.insert(gold_index, row["Correct Answer"])
        prompt = GPQA_QUERY_TEMPLATE.format(
            Question=row["Question"].strip(),
            A=choices[0].strip(),
            B=choices[1].strip(),
            C=choices[2].strip(),
            D=choices[3].strip(),
        )
        samples.append({"prompt": prompt, "label": ["A", "B", "C", "D"][gold_index]})
    return {"samples": samples, "extract": extract_answer_colon, "max_tokens": 2048, "stop": None, "endpoint": "chat"}


def load_mmlu(num_samples: int | None, subset: list[str] | None = None) -> dict[str, Any]:
    subsets = subset or ["all"]
    samples = []
    for item in subsets:
        dataset = load_dataset("cais/mmlu", item)["test"]
        for idx, row in enumerate(dataset):
            if num_samples is not None and idx >= num_samples:
                break
            choices = row["choices"]
            prompt = GPQA_QUERY_TEMPLATE.format(
                Question=row["question"].strip(),
                A=choices[0].strip(),
                B=choices[1].strip(),
                C=choices[2].strip(),
                D=choices[3].strip(),
            )
            samples.append({"prompt": prompt, "label": ["A", "B", "C", "D"][row["answer"]]})
    return {"samples": samples, "extract": extract_answer_colon, "max_tokens": 2048, "stop": None, "endpoint": "chat"}


def load_financeqa(num_samples: int | None, subset: list[str] | None = None) -> dict[str, Any]:
    del subset
    dataset = load_dataset("AfterQuery/FinanceQA")["test"]
    samples = []
    for idx, row in enumerate(dataset):
        if num_samples is not None and idx >= num_samples:
            break
        if row["context"] is None:
            prompt = row["question"].strip()
        else:
            prompt = (
                "Given the following context:\n\n"
                f"{row['context'].strip()}\n\n"
                "Can you answer the following question?\n\n"
                f"{row['question'].strip()}"
            )
        samples.append({"prompt": prompt, "label": None})
    return {"samples": samples, "extract": lambda output: output, "max_tokens": 2048, "stop": None, "endpoint": "chat"}


def load_livecodebench(num_samples: int | None, subset: list[str] | None = None) -> dict[str, Any]:
    del subset
    dataset = load_dataset("livecodebench/code_generation")["test"]
    samples = []
    for idx, row in enumerate(dataset):
        if num_samples is not None and idx >= num_samples:
            break
        samples.append({"prompt": row["question_content"].strip(), "label": None})
    return {"samples": samples, "extract": lambda output: output, "max_tokens": 2048, "stop": None, "endpoint": "chat"}


def load_simpleqa(num_samples: int | None, subset: list[str] | None = None) -> dict[str, Any]:
    del subset
    dataset = load_dataset("basicv8vc/SimpleQA")["test"]
    samples = []
    for idx, row in enumerate(dataset):
        if num_samples is not None and idx >= num_samples:
            break
        samples.append({"prompt": row["problem"].strip(), "label": None})
    return {"samples": samples, "extract": lambda output: output, "max_tokens": 2048, "stop": None, "endpoint": "chat"}


MTBENCH_SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."


def load_mtbench(num_samples: int | None, subset: list[str] | None = None) -> dict[str, Any]:
    del subset
    rows = download_jsonl("https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl")
    samples = []
    for idx, row in enumerate(rows):
        if num_samples is not None and idx >= num_samples:
            break
        samples.append({"turns": row["turns"], "label": None})
    return {"samples": samples, "extract": lambda output: output, "max_tokens": 2048, "stop": None, "endpoint": "mtbench"}


def image_to_data_url(path: str) -> str:
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def load_mmstar(num_samples: int | None, subset: list[str] | None = None) -> dict[str, Any]:
    del subset
    dataset = load_dataset("Lin-Chen/MMStar")["val"]
    cache_dir = Path(".cache/mmstar_specforge")
    cache_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for idx, row in enumerate(dataset):
        if num_samples is not None and idx >= num_samples:
            break
        image_path = cache_dir / row["meta_info"]["image_path"]
        image_path.parent.mkdir(parents=True, exist_ok=True)
        row["image"].convert("RGB").save(image_path, "JPEG")
        question = row["question"].split("Options:", 1)[0].strip()
        answer = str(row.get("answer") or row.get("correct_answer") or row.get("ground_truth") or "").strip().upper()
        if not (len(answer) == 1 and "A" <= answer <= "Z"):
            answer = None
        samples.append({"prompt": question, "image_path": str(image_path), "label": answer})
    return {"samples": samples, "extract": extract_choice, "max_tokens": 2048, "stop": None, "endpoint": "image"}


LOADERS = {
    "aime": load_aime,
    "gsm8k": load_gsm8k,
    "math500": load_math500,
    "ceval": load_ceval,
    "humaneval": load_humaneval,
    "gpqa": load_gpqa,
    "mmlu": load_mmlu,
    "financeqa": load_financeqa,
    "livecodebench": load_livecodebench,
    "simpleqa": load_simpleqa,
    "mtbench": load_mtbench,
    "mmstar": load_mmstar,
}


def is_correct(benchmark: str, prediction: Any, label: Any) -> bool:
    if prediction is None:
        return False
    if benchmark == "gsm8k":
        return prediction == label
    if benchmark == "math500":
        pred = str(prediction).strip().lower()
        gold = str(label).strip().lower()
        if pred == gold:
            return True
        try:
            return math.isclose(float(pred), float(gold), rel_tol=0, abs_tol=1e-6)
        except ValueError:
            return False
    if benchmark == "ceval":
        return prediction == label
    if benchmark in ("gpqa", "mmlu"):
        return prediction == label
    if benchmark == "aime":
        pred = str(prediction).strip()
        gold = str(label).strip()
        if pred == gold:
            return True
        try:
            return int(pred) == int(gold)
        except ValueError:
            return False
    if benchmark == "mmstar":
        return str(prediction).strip().upper() == str(label).strip().upper()
    if benchmark in ("financeqa", "livecodebench", "simpleqa", "mtbench"):
        return False
    if benchmark == "humaneval":
        pred = str(prediction).strip()
        entry_point = label.get("entry_point", "")
        if pred.startswith("def ") and entry_point:
            func_name_match = re.match(r"def\s+(\w+)\s*\(", pred)
            if func_name_match and func_name_match.group(1) == entry_point:
                code = pred
            else:
                code = label["prompt"] + "\n" + pred
        elif pred.startswith("def "):
            code = pred
        else:
            code = label["prompt"] + "\n" + pred
        return code_passes_tests(code, label["test"])
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def compute_accuracy_like_specforge(
    benchmark: str, results: list[dict[str, Any]], samples: list[dict[str, Any]]
) -> tuple[float | None, int, int]:
    predictions = [result["prediction"] for result in results]
    labels = [sample["label"] for sample in samples]

    if benchmark == "gsm8k":
        correct = sum(1 for pred, label in zip(predictions, labels) if pred == label)
        return correct / len(labels) if labels else None, correct, sum(p is not None for p in predictions)

    if benchmark == "math500":
        valid_labels = 0
        correct = 0
        for pred, label in zip(predictions, labels):
            if label is None:
                continue
            valid_labels += 1
            if pred is not None and is_correct(benchmark, pred, label):
                correct += 1
        return (
            correct / valid_labels if valid_labels > 0 else None,
            correct,
            sum(p is not None for p in predictions),
        )

    if benchmark == "ceval":
        # Match SpecForge: C-Eval divides by successfully extracted predictions,
        # not by total questions.
        valid_predictions = 0
        correct = 0
        for pred, label in zip(predictions, labels):
            if pred is None:
                continue
            valid_predictions += 1
            if pred == label:
                correct += 1
        return (
            correct / valid_predictions if valid_predictions > 0 else 0.0,
            correct,
            valid_predictions,
        )

    if benchmark in ("gpqa", "mmlu"):
        correct = sum(1 for pred, label in zip(predictions, labels) if pred == label)
        return correct / len(labels) if labels else None, correct, sum(p is not None for p in predictions)

    if benchmark in ("aime", "mmstar"):
        valid_labels = 0
        correct = 0
        for pred, label in zip(predictions, labels):
            if label is None:
                continue
            valid_labels += 1
            if pred is not None and is_correct(benchmark, pred, label):
                correct += 1
        return (
            correct / valid_labels if valid_labels > 0 else None,
            correct,
            sum(p is not None for p in predictions),
        )

    if benchmark == "humaneval":
        correct = sum(
            1 for result, sample in zip(results, samples)
            if result["prediction"] is not None
            and is_correct(benchmark, result["prediction"], sample["label"])
        )
        return correct / len(labels) if labels else None, correct, sum(p is not None for p in predictions)

    if benchmark in ("financeqa", "livecodebench", "simpleqa", "mtbench"):
        return None, 0, sum(p is not None for p in predictions)

    raise ValueError(f"Unsupported benchmark: {benchmark}")


def run_sample(
    args: argparse.Namespace,
    base_url: str,
    endpoint: str,
    sample: dict[str, Any],
    max_tokens: int,
    stop: list[str] | None,
) -> tuple[Any, int]:
    if endpoint == "completion":
        return text_completion(
            base_url=base_url,
            model=args.model_path,
            prompt=sample["prompt"],
            max_tokens=max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
            stop=stop,
        )
    if endpoint == "mtbench":
        answer_1, tokens_1 = chat_messages_completion(
            base_url=base_url,
            model=args.model_path,
            messages=[
                {"role": "system", "content": MTBENCH_SYSTEM_PROMPT},
                {"role": "user", "content": sample["turns"][0]},
            ],
            max_tokens=max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
            stop=stop,
        )
        answer_2, tokens_2 = chat_messages_completion(
            base_url=base_url,
            model=args.model_path,
            messages=[
                {"role": "system", "content": MTBENCH_SYSTEM_PROMPT},
                {"role": "user", "content": sample["turns"][0]},
                {"role": "assistant", "content": answer_1},
                {"role": "user", "content": sample["turns"][1]},
            ],
            max_tokens=max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
            stop=stop,
        )
        return {"answer_1": answer_1, "answer_2": answer_2}, tokens_1 + tokens_2
    if endpoint == "image":
        return chat_messages_completion(
            base_url=base_url,
            model=args.model_path,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_to_data_url(sample["image_path"])}},
                        {"type": "text", "text": sample["prompt"]},
                    ],
                }
            ],
            max_tokens=max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
            stop=stop,
        )
    return chat_completion(
        base_url=base_url,
        model=args.model_path,
        prompt=sample["prompt"],
        max_tokens=max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
        stop=stop,
    )


def run_benchmark(args: argparse.Namespace, benchmark_item: str) -> dict[str, Any]:
    name, num_samples, subset = parse_benchmark_item(benchmark_item)
    if name not in LOADERS:
        raise ValueError(f"Unsupported benchmark {name}. Available: {', '.join(LOADERS)}")

    loaded = LOADERS[name](num_samples, subset)
    samples = loaded["samples"]
    extract = loaded["extract"]
    max_tokens = args.max_tokens or loaded["max_tokens"]
    stop = loaded["stop"]
    endpoint = loaded.get("endpoint", "chat")
    base_url = normalize_base_url(args.host, args.port)

    print(f"Running {name}: {len(samples)} samples, max_tokens={max_tokens}, threads={args.num_threads}")
    spec_before = fetch_spec_metrics(base_url)
    started = time.perf_counter()
    results: list[dict[str, Any] | None] = [None] * len(samples)

    def run_one(index: int, sample: dict[str, Any]) -> dict[str, Any]:
        try:
            output, completion_tokens = run_sample(args, base_url, endpoint, sample, max_tokens, stop)
            prediction = extract(output)
            correct = prediction is not None and is_correct(name, prediction, sample["label"])
            return {
                "index": index,
                "ok": True,
                "output": output,
                "prediction": prediction,
                "label": sample["label"],
                "correct": correct,
                "completion_tokens": completion_tokens,
            }
        except Exception as exc:
            return {
                "index": index,
                "ok": False,
                "error": repr(exc),
                "prediction": None,
                "label": sample["label"],
                "correct": False,
                "completion_tokens": 0,
            }

    with futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        pending = [executor.submit(run_one, i, sample) for i, sample in enumerate(samples)]
        for done_count, future in enumerate(futures.as_completed(pending), 1):
            item = future.result()
            results[item["index"]] = item
            if done_count % max(1, args.progress_interval) == 0 or done_count == len(samples):
                correct_so_far = sum(1 for r in results if r and r["correct"])
                print(f"  {done_count}/{len(samples)} done, correct_so_far={correct_so_far}")

    latency = time.perf_counter() - started
    spec_after = fetch_spec_metrics(base_url)
    spec_delta = {
        key: spec_after.get(key, 0.0) - spec_before.get(key, 0.0)
        for key in set(spec_before) | set(spec_after)
    }
    final_results = [r for r in results if r is not None]
    failed = sum(1 for r in final_results if not r["ok"])
    accuracy, correct, valid_predictions = compute_accuracy_like_specforge(
        name, final_results, samples
    )
    for result, sample in zip(final_results, samples):
        result["correct"] = (
            result["prediction"] is not None
            and is_correct(name, result["prediction"], sample["label"])
        )
    total_tokens = sum(int(r["completion_tokens"]) for r in final_results)
    num_drafts = spec_delta.get("num_drafts", 0.0)
    num_draft_tokens = spec_delta.get("num_draft_tokens", 0.0)
    num_accepted_tokens = spec_delta.get("num_accepted_tokens", 0.0)
    # Match SpecForge benchmark tables: output tokens per verification step.
    # This is intentionally different from vLLM's log-only
    # "Mean acceptance length" (1 + accepted draft tokens / drafts).
    accept_length = total_tokens / num_drafts if num_drafts > 0 else None
    draft_acceptance_rate = (
        num_accepted_tokens / num_draft_tokens if num_draft_tokens > 0 else None
    )
    metrics = BenchmarkMetrics(
        benchmark=name,
        num_questions=len(final_results),
        accuracy=accuracy,
        latency=latency,
        output_throughput=total_tokens / latency if latency > 0 else 0.0,
        total_completion_tokens=total_tokens,
        accept_length=accept_length,
        draft_acceptance_rate=draft_acceptance_rate,
        spec_num_drafts=num_drafts,
        spec_num_draft_tokens=num_draft_tokens,
        spec_num_accepted_tokens=num_accepted_tokens,
        valid_predictions=valid_predictions,
        failed_requests=failed,
    )

    accept_text = "n/a" if accept_length is None else f"{accept_length:.3f}"
    draft_accept_text = (
        "n/a" if draft_acceptance_rate is None else f"{draft_acceptance_rate:.2%}"
    )
    accuracy_text = "n/a" if metrics.accuracy is None else f"{metrics.accuracy:.4f}"
    print(
        f"{name}: accuracy={accuracy_text} "
        f"latency={metrics.latency:.2f}s throughput={metrics.output_throughput:.2f} tok/s "
        f"accept_length={accept_text} draft_acceptance_rate={draft_accept_text} "
        f"failed={failed}"
    )
    return asdict(metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmarks against a vLLM OpenAI-compatible server")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument(
        "--benchmark-list",
        nargs="+",
        default=[
            "mtbench:80",
            "gsm8k:200",
            "humaneval:200",
            "math500:200",
            "ceval:200",
            "aime",
            "mmlu",
            "simpleqa",
            "financeqa",
            "livecodebench",
            "mmstar",
        ],
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--name", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--progress-interval", type=int, default=10)
    return parser.parse_args()


def run_all_benchmarks(args: argparse.Namespace, timestamp: str) -> str:
    output: dict[str, Any] = {
        "model": args.model_path,
        "host": args.host,
        "port": args.port,
        "benchmarks": {},
    }
    for item in args.benchmark_list:
        result = run_benchmark(args, item)
        output["benchmarks"][item] = result

    prefix = f"{args.name}_" if args.name else ""
    output_path = os.path.join(args.output_dir, f"{prefix}openai_results_{timestamp}.json")
    with open(output_path, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {output_path}")
    return output_path


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_all_benchmarks(args, timestamp)


if __name__ == "__main__":
    main()

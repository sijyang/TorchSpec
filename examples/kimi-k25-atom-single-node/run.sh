#!/bin/bash
# Single-node Kimi-K2.5 Eagle3 training with ATOM inference.
#
# This launcher is ATOM-only. Backend selection is intentionally not configurable.
# The YAML config owns static model, training, inference, and decode defaults.
# This launcher only selects the phase, runtime paths, GPU layout, and resume mode.

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash examples/kimi-k25-atom-single-node/run.sh phase1 [config.yaml] [overrides...]
  bash examples/kimi-k25-atom-single-node/run.sh phase2 [config.yaml] [overrides...]
  bash examples/kimi-k25-atom-single-node/run.sh both   [config.yaml] [overrides...]

Common environment overrides:
  KIMI25_EAGLE3_DATA_ROOT  Data/output root (default: /data/kimi-k25-eagle3)
  MODEL_PATH               Target model path or Hugging Face model id
  DATASET_PATH             Use one train dataset for both phases
  PHASE1_DATASET           Phase 1 train dataset
  PHASE2_DATASET           Phase 2 train dataset
  PHASE1_EVAL_DATASET      Phase 1 evaluation JSONL path
  PHASE2_EVAL_DATASET      Phase 2 evaluation JSONL path
  TRAIN_GPUS               Training GPUs on this node (default: 4)
  INFERENCE_GPUS           ATOM inference GPUs and TP size (default: 4)
  NUM_STEPS_PHASE1         Optional fixed optimizer steps for phase 1
  NUM_STEPS_PHASE2         Optional fixed optimizer steps for phase 2
  AUTO_RESUME              Resume phase output checkpoints when present (default: 1)
USAGE
}

PHASE="${1:-phase1}"
case "$PHASE" in
  phase1|1) PHASE="phase1"; shift || true ;;
  phase2|2) PHASE="phase2"; shift || true ;;
  both) shift || true ;;
  -h|--help) usage; exit 0 ;;
  *)
    echo "ERROR: first argument must be phase1, phase2, or both"
    usage
    exit 1
    ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

DATA_ROOT="${KIMI25_EAGLE3_DATA_ROOT:-/data/kimi-k25-eagle3}"
CACHE_ROOT="${CACHE_ROOT:-$DATA_ROOT/cache}"
LOG_ROOT="${LOG_ROOT:-$DATA_ROOT/running_logs}"

CONFIG_FILE="${CONFIG_FILE:-$ROOT_DIR/configs/atom_kimi_k25_single_node.yaml}"
if [[ $# -gt 0 && "$1" != *=* && ( "$1" == *.yaml || "$1" == *.yml ) ]]; then
  CONFIG_FILE="$1"
  shift
fi
if [[ "$CONFIG_FILE" != /* && ! -f "$CONFIG_FILE" && -f "$ROOT_DIR/$CONFIG_FILE" ]]; then
  CONFIG_FILE="$ROOT_DIR/$CONFIG_FILE"
fi
DRAFT_MODEL_CONFIG="${DRAFT_MODEL_CONFIG:-$ROOT_DIR/configs/draft_models/kimi_k25_eagle3.json}"
MODEL_PATH="${MODEL_PATH:-/data/models/amd/Kimi-K2.5-MXFP4}"

PHASE1_DATASET="${PHASE1_DATASET:-${DATASET_PATH:-mlabonne/open-perfectblend}}"
PHASE2_DATASET="${PHASE2_DATASET:-${DATASET_PATH:-lightseekorg/kimi-mtp-dataset}}"
EVAL_SAMPLE_SIZE=256
PHASE1_EVAL_DATASET="${PHASE1_EVAL_DATASET:-$ROOT_DIR/examples/data/eval_phase1.jsonl}"
PHASE2_EVAL_DATASET="${PHASE2_EVAL_DATASET:-$ROOT_DIR/examples/data/eval_phase2.jsonl}"

TRAIN_GPUS="${TRAIN_GPUS:-4}"
INFERENCE_GPUS="${INFERENCE_GPUS:-4}"
INFERENCE_GPUS_PER_NODE="${INFERENCE_GPUS_PER_NODE:-$INFERENCE_GPUS}"
AUTO_RESUME="${AUTO_RESUME:-1}"

PHASE1_OUTPUT_DIR="${PHASE1_OUTPUT_DIR:-$DATA_ROOT/outputs/kimi25_atom_single_node_phase1}"
PHASE2_OUTPUT_DIR="${PHASE2_OUTPUT_DIR:-$DATA_ROOT/outputs/kimi25_atom_single_node_phase2}"
PHASE1_CACHE_DIR="${PHASE1_CACHE_DIR:-$CACHE_ROOT/kimi25_atom_single_node_phase1}"
PHASE2_CACHE_DIR="${PHASE2_CACHE_DIR:-$CACHE_ROOT/kimi25_atom_single_node_phase2}"
PHASE1_CHECKPOINT="${PHASE1_CHECKPOINT:-$PHASE1_OUTPUT_DIR/checkpoints}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export AITER_LOG_LEVEL="${AITER_LOG_LEVEL:-WARNING}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT/xdg}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_ROOT/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_ROOT/triton}"
export CUTE_DSL_CACHE_DIR="${CUTE_DSL_CACHE_DIR:-$CACHE_ROOT/cute_dsl}"
export TORCHSPEC_LOG_LEVEL="${TORCHSPEC_LOG_LEVEL:-INFO}"
export TORCHSPEC_EVAL_CACHE_IDLE_TIMEOUT="${TORCHSPEC_EVAL_CACHE_IDLE_TIMEOUT:-1200.0}"

require_file() {
  local path="$1"
  local description="$2"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: $description not found: $path"
    exit 1
  fi
}

require_positive_int() {
  local name="$1"
  local value="$2"
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: $name must be a positive integer, got: $value"
    exit 1
  fi
}

validate_model_path() {
  local path="$1"
  [[ "$path" = /* ]] || return 0

  if [[ ! -d "$path" ]]; then
    echo "ERROR: model path does not exist or is not visible: $path"
    exit 1
  fi
  require_file "$path/config.json" "model config.json"
  if [[ ! -f "$path/tokenizer.json" && ! -f "$path/tokenizer.model" && ! -f "$path/tiktoken.model" ]]; then
    echo "ERROR: model path is missing tokenizer.json, tokenizer.model, or tiktoken.model: $path"
    exit 1
  fi
}

append_env_override() {
  local -n target="$1"
  local env_name="$2"
  local config_key="$3"
  local value="${!env_name:-}"
  [[ -n "$value" ]] && target+=("$config_key=$value")
}

generate_eval_dataset() {
  local source_dataset="$1"
  local eval_dataset="$2"
  local seed="$3"

  if [[ -f "$eval_dataset" ]]; then
    return 0
  fi

  mkdir -p "$(dirname "$eval_dataset")"

  PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    python3 - "$source_dataset" "$eval_dataset" "$EVAL_SAMPLE_SIZE" "$seed" <<'PY'
import json
import os
import re
import sys

from torchspec.data.utils import load_hf_dataset

source_dataset, eval_dataset, sample_size_raw, seed_raw = sys.argv[1:5]
sample_size = int(sample_size_raw)
seed = int(seed_raw)
id_prefix = os.path.splitext(os.path.basename(eval_dataset))[0]
role_mapping = {
    "human": "user",
    "gpt": "assistant",
    "chatgpt": "assistant",
    "bing": "assistant",
    "bard": "assistant",
    "system": "system",
}
think_pattern = re.compile(r"<think>[\s\S]*?</think>\s*")


def normalize_message(message):
    if "role" in message and "content" in message:
        role = message["role"]
        content = message["content"]
    else:
        role = role_mapping.get(message["from"], message["from"])
        content = message["value"]

    if role == "assistant":
        content = think_pattern.sub("", content).lstrip()

    return {"role": role, "content": content}


def is_neat_eval_sample(sample):
    messages = sample["conversations"]
    if len(messages) != 2:
        return False
    if messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
        return False
    if not messages[0]["content"].strip() or not messages[1]["content"].strip():
        return False
    if len(messages[0]["content"]) > 2500 or len(messages[1]["content"]) > 2500:
        return False
    return len(json.dumps(sample, ensure_ascii=False)) <= 5500

samples = []
dataset = load_hf_dataset(source_dataset).shuffle(buffer_size=10000, seed=seed)

for row in dataset:
    prompt = row.get("conversations")
    if not isinstance(prompt, list):
        raise ValueError(
            "Expected 'conversations' to contain a conversation list "
            f"while sampling '{source_dataset}', got {type(prompt).__name__}"
        )

    sample = {
        "id": row.get("id", f"{id_prefix}_{len(samples):06d}"),
        "conversations": [normalize_message(message) for message in prompt],
    }

    if not is_neat_eval_sample(sample):
        continue

    samples.append(sample)
    if len(samples) >= sample_size:
        break

if len(samples) < sample_size:
    raise ValueError(
        f"Dataset '{source_dataset}' only has {len(samples)} valid samples; "
        f"need {sample_size} for eval"
    )

tmp_path = eval_dataset + ".tmp"
with open(tmp_path, "w", encoding="utf-8") as f:
    f.write("\n".join(json.dumps(sample, ensure_ascii=False) for sample in samples))
os.replace(tmp_path, eval_dataset)

print(
    f"Wrote {len(samples)} random eval samples from {source_dataset} "
    f"to {eval_dataset}"
)
PY
}

build_schedule_args() {
  local -n target="$1"
  local phase="$2"
  local phase_steps=""
  local phase_lr_total_steps=""
  local phase_epochs=""

  if [[ "$phase" == "phase1" ]]; then
    phase_steps="${NUM_STEPS_PHASE1:-}"
    phase_lr_total_steps="${LR_TOTAL_STEPS_PHASE1:-}"
    phase_epochs="${NUM_EPOCHS_PHASE1:-}"
  else
    phase_steps="${NUM_STEPS_PHASE2:-}"
    phase_lr_total_steps="${LR_TOTAL_STEPS_PHASE2:-}"
    phase_epochs="${NUM_EPOCHS_PHASE2:-}"
  fi

  if [[ -n "${phase_steps:-${NUM_STEPS:-}}" ]]; then
    local num_steps="${phase_steps:-$NUM_STEPS}"
    local lr_total_steps="${phase_lr_total_steps:-${LR_TOTAL_STEPS:-$num_steps}}"
    target+=(training.num_train_steps="$num_steps" training.lr_total_steps="$lr_total_steps")
    SCHEDULE_DESC="steps=$num_steps, lr_total_steps=$lr_total_steps"
  elif [[ -n "${phase_epochs:-${NUM_EPOCHS:-}}" ]]; then
    local num_epochs="${phase_epochs:-$NUM_EPOCHS}"
    target+=(training.num_train_steps=null training.num_epochs="$num_epochs")
    [[ -n "${LR_TOTAL_STEPS:-}" ]] && target+=(training.lr_total_steps="$LR_TOTAL_STEPS")
    SCHEDULE_DESC="epochs=$num_epochs"
  else
    [[ -n "${LR_TOTAL_STEPS:-}" ]] && target+=(training.lr_total_steps="$LR_TOTAL_STEPS")
    SCHEDULE_DESC="config default"
  fi
}

require_file "$CONFIG_FILE" "config file"
require_file "$DRAFT_MODEL_CONFIG" "draft model config"
validate_model_path "$MODEL_PATH"
require_positive_int TRAIN_GPUS "$TRAIN_GPUS"
require_positive_int INFERENCE_GPUS "$INFERENCE_GPUS"
require_positive_int INFERENCE_GPUS_PER_NODE "$INFERENCE_GPUS_PER_NODE"

IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
TOTAL_GPUS=${#GPU_ARRAY[@]}
REQUIRED_GPUS=$((TRAIN_GPUS + INFERENCE_GPUS_PER_NODE))
if (( TOTAL_GPUS < REQUIRED_GPUS )); then
  echo "ERROR: need at least $REQUIRED_GPUS visible GPUs, got $TOTAL_GPUS (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
  exit 1
fi

mkdir -p "$DATA_ROOT" "$CACHE_ROOT" "$LOG_ROOT"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_LOG_DIR="$LOG_ROOT/kimi25_atom_single_node_${PHASE}_${TIMESTAMP}"
LOG_FILE="$RUN_LOG_DIR/terminal.log"
LATEST_LOG_LINK="$LOG_ROOT/kimi25_atom_single_node_latest"

mkdir -p "$RUN_LOG_DIR"
ln -sfn "$RUN_LOG_DIR" "$LATEST_LOG_LINK"
export TORCHSPEC_LOG_DIR="${TORCHSPEC_LOG_DIR:-$RUN_LOG_DIR/actors}"
mkdir -p "$TORCHSPEC_LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

set -x

COMMON_ARGS=(
  model.target_model_path="$MODEL_PATH"
  model.draft_model_config="$DRAFT_MODEL_CONFIG"
  training.training_num_gpus_per_node="$TRAIN_GPUS"
  inference.inference_num_gpus="$INFERENCE_GPUS"
  inference.inference_num_gpus_per_engine="$INFERENCE_GPUS"
  inference.inference_num_gpus_per_node="$INFERENCE_GPUS_PER_NODE"
)

append_env_override COMMON_ARGS SAVE_INTERVAL training.save_interval
append_env_override COMMON_ARGS EVAL_INTERVAL dataset.eval_interval
append_env_override COMMON_ARGS MAX_SEQ_LENGTH training.max_seq_length
append_env_override COMMON_ARGS LEARNING_RATE training.learning_rate
append_env_override COMMON_ARGS WARMUP_RATIO training.warmup_ratio
append_env_override COMMON_ARGS MAX_SAMPLE_POOL_SIZE inference.max_sample_pool_size
append_env_override COMMON_ARGS INFERENCE_BUFFER_THRESHOLD inference.inference_buffer_threshold
append_env_override COMMON_ARGS INFERENCE_FETCH_BATCH inference.inference_fetch_batch
append_env_override COMMON_ARGS INFERENCE_BATCH_SIZE inference.inference_batch_size
append_env_override COMMON_ARGS MAX_NUM_BATCHED_TOKENS inference.atom.extra_args.max_num_batched_tokens
append_env_override COMMON_ARGS ATOM_MAX_MODEL_LEN inference.atom.extra_args.max_model_len

run_phase() {
  local phase="$1"
  shift

  local dataset eval_dataset output_dir cache_dir seed resume_mode
  local -a phase_args resume_args schedule_args
  phase_args=()
  resume_args=()
  schedule_args=()

  if [[ "$phase" == "phase1" ]]; then
    dataset="$PHASE1_DATASET"
    output_dir="$PHASE1_OUTPUT_DIR"
    cache_dir="$PHASE1_CACHE_DIR"
    seed="${SEED:-42}"
    eval_dataset="$PHASE1_EVAL_DATASET"
    resume_mode="fresh phase1"

    if [[ "$AUTO_RESUME" == "1" && -f "$output_dir/checkpoints/latest_checkpointed_iteration.txt" ]]; then
      resume_args=(training.load_path="$output_dir/checkpoints")
      resume_mode="resume phase1 from $output_dir/checkpoints"
    fi
  else
    dataset="$PHASE2_DATASET"
    output_dir="$PHASE2_OUTPUT_DIR"
    cache_dir="$PHASE2_CACHE_DIR"
    seed="${SEED:-43}"
    eval_dataset="$PHASE2_EVAL_DATASET"

    if [[ "$AUTO_RESUME" == "1" && -f "$output_dir/checkpoints/latest_checkpointed_iteration.txt" ]]; then
      resume_args=(training.load_path="$output_dir/checkpoints" training.continual_training=false)
      resume_mode="resume phase2 from $output_dir/checkpoints"
    else
      if [[ ! -d "$PHASE1_CHECKPOINT" ]]; then
        echo "ERROR: phase2 needs phase1 checkpoint directory: $PHASE1_CHECKPOINT"
        echo "Set PHASE1_CHECKPOINT=/path/to/phase1/checkpoints if different."
        exit 1
      fi
      require_file "$PHASE1_CHECKPOINT/latest_checkpointed_iteration.txt" "phase1 checkpoint tracker"
      resume_args=(training.load_path="$PHASE1_CHECKPOINT" training.continual_training=true)
      resume_mode="initialize phase2 from phase1 weights at $PHASE1_CHECKPOINT"
    fi
  fi

  SCHEDULE_DESC=""
  build_schedule_args schedule_args "$phase"
  generate_eval_dataset "$dataset" "$eval_dataset" "$seed"
  phase_args=(
    dataset.train_data_path="$dataset"
    dataset.eval_data_path="$eval_dataset"
    training.seed="$seed"
    output_dir="$output_dir"
    cache_dir="$cache_dir"
  )

  echo "=============================================="
  echo "Kimi-K2.5 Eagle3 ATOM Training ($phase)"
  echo "=============================================="
  echo "Config:              $CONFIG_FILE"
  echo "Draft config:        $DRAFT_MODEL_CONFIG"
  echo "Data root:           $DATA_ROOT"
  echo "HF cache:            $HF_HOME"
  echo "Model path:          $MODEL_PATH"
  echo "Train dataset:       $dataset"
  echo "Eval dataset:        $eval_dataset"
  echo "Eval samples:        $EVAL_SAMPLE_SIZE"
  echo "Output dir:          $output_dir"
  echo "Cache dir:           $cache_dir"
  echo "Visible GPUs:        $CUDA_VISIBLE_DEVICES"
  echo "Training GPUs:       $TRAIN_GPUS"
  echo "Inference GPUs:      $INFERENCE_GPUS (ATOM TP=$INFERENCE_GPUS)"
  echo "Schedule:            $SCHEDULE_DESC"
  echo "Resume mode:         $resume_mode"
  echo "Eval cache timeout:  ${TORCHSPEC_EVAL_CACHE_IDLE_TIMEOUT}s"
  echo "Log file:            $LOG_FILE"
  echo "Actor log dir:       $TORCHSPEC_LOG_DIR"
  echo "Extra overrides:     $*"
  echo "=============================================="

  python3 -m torchspec.train_entry \
    --config "$CONFIG_FILE" \
    "${COMMON_ARGS[@]}" \
    "${phase_args[@]}" \
    "${schedule_args[@]}" \
    "${resume_args[@]}" \
    "$@" \
    model.target_model_backend=atom \
    inference.inference_engine_type=atom \
    inference.atom.tp_size="$INFERENCE_GPUS"
}

if [[ "$PHASE" == "both" ]]; then
  run_phase phase1 "$@"
  PHASE1_CHECKPOINT="$PHASE1_OUTPUT_DIR/checkpoints"
  run_phase phase2 "$@"
else
  run_phase "$PHASE" "$@"
fi

echo "=============================================="
echo "Training completed!"
echo "Logs: $RUN_LOG_DIR"
echo "=============================================="

# Kimi-K2.5 ATOM Single-Node Training

Single-node ATOM-only launcher for training an Eagle3 draft model for Kimi-K2.5.

## GPU Layout

- 4 GPUs for Eagle3 training
- 4 GPUs for ATOM inference, using TP=4

The default target model path is `/data/models/amd/Kimi-K2.5-MXFP4`. Override it with `MODEL_PATH=/path/or/hf-id`.

By default, large run artifacts are written under `/data/kimi-k25-eagle3`:

- `outputs/`: training outputs and checkpoints.
- `cache/`: TorchSpec, Hugging Face, Torch, Triton, and generated-kernel caches.
- `running_logs/`: terminal and per-actor logs.

Override this root with `KIMI25_EAGLE3_DATA_ROOT=/path/to/data/root`.

## Config

Uses [`configs/atom_kimi_k25_single_node.yaml`](../../configs/atom_kimi_k25_single_node.yaml) with draft model config [`configs/draft_models/kimi_k25_eagle3.json`](../../configs/draft_models/kimi_k25_eagle3.json).

The launcher always passes `model.target_model_backend=atom`, `inference.inference_engine_type=atom`, and `inference.atom.tp_size=$INFERENCE_GPUS`.

The YAML config owns the static recipe: ATOM settings, decode settings, batch sizes, learning-rate defaults, and dataset formatting. The launcher owns runtime concerns: phase selection, train/eval datasets, output/cache/log paths, GPU topology, and resume behavior.

## How to Run

Single-node run:

- Phase 1: regenerated [`mlabonne/open-perfectblend`](https://huggingface.co/datasets/mlabonne/open-perfectblend) dataset.
- Phase 2: mixed `lightseekorg/kimi-mtp-dataset`, initialized from Phase 1 draft weights.
- By default, step count is derived by TorchSpec from the dataset size and `training.num_epochs` in the config.

```bash
# Phase 1
bash examples/kimi-k25-atom-single-node/run.sh phase1

# Phase 2, initialized from phase-1 draft checkpoint weights
PHASE1_CHECKPOINT=/data/kimi-k25-eagle3/outputs/kimi25_atom_single_node_phase1/checkpoints bash examples/kimi-k25-atom-single-node/run.sh phase2

# Run both phases sequentially
bash examples/kimi-k25-atom-single-node/run.sh both
```

Use a custom config file as the argument after the phase:

```bash
bash examples/kimi-k25-atom-single-node/run.sh phase1 configs/atom_kimi_k25_single_node.yaml
```

Override common settings:

```bash
MODEL_PATH=/data/models/amd/Kimi-K2.5-MXFP4 \
TRAIN_GPUS=4 \
INFERENCE_GPUS=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash examples/kimi-k25-atom-single-node/run.sh phase1 training.learning_rate=1e-5
```

Override dataset and schedule:

```bash
DATASET_PATH=lightseekorg/kimi-mtp-dataset \
NUM_STEPS_PHASE1=20000 \
EVAL_INTERVAL=1000 \
SAVE_INTERVAL=5000 \
bash examples/kimi-k25-atom-single-node/run.sh phase1
```

If you split the dataset into local phase-specific subsets:

```bash
PHASE1_DATASET=/data/datasets/kimi-mtp/phase1 \
PHASE2_DATASET=/data/datasets/kimi-mtp/phase2 \
EVAL_DATASET=/data/datasets/kimi-mtp/eval \
bash examples/kimi-k25-atom-single-node/run.sh both
```

Useful runtime environment variables:

- Custom config file: pass it after the phase, for example `bash examples/kimi-k25-atom-single-node/run.sh phase1 configs/your_config.yaml`.
- `DRAFT_MODEL_CONFIG`: alternate draft model JSON.
- `KIMI25_EAGLE3_DATA_ROOT`: shared root for outputs, caches, and logs.
- `CACHE_ROOT` / `LOG_ROOT`: override cache or log roots independently.
- `MODEL_PATH`: local model directory or Hugging Face model id.
- `TRAIN_GPUS` / `INFERENCE_GPUS`: training GPU count and ATOM TP size.
- `PHASE1_OUTPUT_DIR` / `PHASE2_OUTPUT_DIR`: phase output directories.
- `PHASE1_CHECKPOINT`: phase-1 checkpoint directory used to initialize phase 2.
- `AUTO_RESUME=0`: disable automatic resume from an existing phase output checkpoint.
- `NUM_STEPS_PHASE1` / `NUM_STEPS_PHASE2`: pin optimizer steps for a phase.
- `NUM_EPOCHS_PHASE1` / `NUM_EPOCHS_PHASE2`: override epoch count while keeping automatic step calculation.

## Logs

Each run writes logs under `/data/kimi-k25-eagle3/running_logs/kimi25_atom_single_node_<phase>_<timestamp>/` by default:

- `terminal.log`: full terminal output from the launch script.
- `actors/`: per-actor TorchSpec/Ray logs when `TORCHSPEC_LOG_DIR` is enabled.

`/data/kimi-k25-eagle3/running_logs/kimi25_atom_single_node_latest` points to the latest run.

## Outputs

The launcher overrides the YAML `output_dir` and `cache_dir` per phase:

- Phase 1 output directory: `/data/kimi-k25-eagle3/outputs/kimi25_atom_single_node_phase1/`
- Phase 2 output directory: `/data/kimi-k25-eagle3/outputs/kimi25_atom_single_node_phase2/`
- Phase 1 cache directory: `/data/kimi-k25-eagle3/cache/kimi25_atom_single_node_phase1/`
- Phase 2 cache directory: `/data/kimi-k25-eagle3/cache/kimi25_atom_single_node_phase2/`

Expected files and directories:

- `outputs/.../config.yaml`: resolved run config snapshot.
- `outputs/.../checkpoints/`: training checkpoints.
- `outputs/.../checkpoints/latest_checkpointed_iteration.txt`: tracker file used for auto-resume.
- `outputs/.../checkpoints/best_checkpointed_iteration.txt` and `best_meta.json`: best eval checkpoint tracker and metadata.
- `outputs/.../checkpoints/iter_0000000/`: per-iteration checkpoint directory containing `model/`, `optimizer/`, `lr_scheduler/`, `meta.json`, and `rng.pt`.
- `cache/.../tokenized_dataset/*.pt`: tokenized dataset cache files.
- `cache/.../eval_cache/<hash>/eval_rank_*.pt`: evaluation cache tensors.

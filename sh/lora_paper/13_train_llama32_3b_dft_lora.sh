#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR/verl"
TRAIN_CONDA_ENV="${TRAIN_CONDA_ENV:-DFT}"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29614}"

PROJECT_NAME="${PROJECT_NAME:-paper-lora-llama32-3b}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-llama32-3b-dft-lora-r8-a16}"
SAVE_PATH="${SAVE_PATH:-checkpoints/${EXPERIMENT_NAME}}"

export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export PYTHONNOUSERSITE=1

conda run -n "$TRAIN_CONDA_ENV" python -m torch.distributed.run --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  -m verl.trainer.fsdp_dft_trainer \
  data.train_files=data/numina_cot/train.parquet \
  data.val_files=data/math500/test.parquet \
  data.prompt_key=extra_info \
  data.response_key=extra_info \
  data.prompt_dict_keys=['question'] \
  data.response_dict_keys=['answer'] \
  data.train_batch_size="${TRAIN_BATCH_SIZE:-256}" \
  data.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU:-4}" \
  data.max_length="${MAX_LENGTH:-2048}" \
  optim.lr="${LR:-5e-5}" \
  model.partial_pretrain="${BASE_MODEL:-meta-llama/Llama-3.2-3B-Instruct}" \
  model.lora_rank="${LORA_RANK:-8}" \
  model.lora_alpha="${LORA_ALPHA:-16}" \
  model.target_modules="${TARGET_MODULES:-all-linear}" \
  model.use_liger="${USE_LIGER:-false}" \
  model.fsdp_config.model_dtype="${MODEL_DTYPE:-bf16}" \
  model.strategy="${MODEL_STRATEGY:-fsdp}" \
  trainer.default_local_dir="$SAVE_PATH" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.logger=['console','wandb'] \
  trainer.default_hdfs_dir=null \
  trainer.test_freq="${TEST_FREQ:-10}" \
  trainer.save_freq="${SAVE_FREQ:-50}" \
  trainer.total_epochs="${TOTAL_EPOCHS:-1}" \
  ulysses_sequence_parallel_size="${ULYSSES_SP:-1}" \
  use_remove_padding="${USE_REMOVE_PADDING:-true}"

echo "[OK] DFT+LoRA training finished: $SAVE_PATH"
#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR/math_evaluation"
EVAL_CONDA_ENV="${EVAL_CONDA_ENV:-DFT_EVAL}"

CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-$ROOT_DIR/merged/llama32-3b-dft-lora-r8-a16}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/math_evaluation/outputs/paper_llama32_3b_dft_lora}"
PROMPT_TYPE="${PROMPT_TYPE:-llama-base-boxed}"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export PYTHONNOUSERSITE=1

conda run --no-capture-output -n "$EVAL_CONDA_ENV" python -u math_eval.py \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --data_names "math_oai,minerva_math,olympiadbench,aime24,amc23" \
  --output_dir "$OUTPUT_DIR" \
  --split test \
  --prompt_type "$PROMPT_TYPE" \
  --num_test_sample -1 \
  --seed 0 \
  --temperature 1 \
  --n_sampling 16 \
  --top_p 1 \
  --start 0 \
  --end -1 \
  --use_safetensors

echo "[OK] Eval finished: $OUTPUT_DIR"
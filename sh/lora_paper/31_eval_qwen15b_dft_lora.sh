#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR/math_evaluation"
EVAL_CONDA_ENV="${EVAL_CONDA_ENV:-DFT_EVAL}"

CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-$ROOT_DIR/merged/qwen15b-dft-lora-r8-a16}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/math_evaluation/outputs/paper_qwen15b_dft_lora}"
PROMPT_TYPE="${PROMPT_TYPE:-qwen-boxed}"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export PYTHONNOUSERSITE=1

conda run -n "$EVAL_CONDA_ENV" python -u math_eval.py \
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
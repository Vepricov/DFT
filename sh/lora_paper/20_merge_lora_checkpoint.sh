#!/usr/bin/env bash
set -eo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <BASE_MODEL> <TRAIN_OUTPUT_DIR> <MERGED_OUTPUT_DIR>"
  echo "Example: $0 Qwen/Qwen2.5-Math-1.5B verl/checkpoints/qwen15b-dft-lora-r8-a16 merged/qwen15b-dft-lora-r8-a16"
  exit 1
fi

BASE_MODEL="$1"
TRAIN_OUTPUT_DIR="$2"
MERGED_OUTPUT_DIR="$3"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_CONDA_ENV="${TRAIN_CONDA_ENV:-DFT}"

export PYTHONNOUSERSITE=1

if [[ ! -d "$ROOT_DIR/$TRAIN_OUTPUT_DIR" ]]; then
  echo "Train output dir not found: $ROOT_DIR/$TRAIN_OUTPUT_DIR"
  exit 1
fi

LATEST_STEP_DIR="$(find "$ROOT_DIR/$TRAIN_OUTPUT_DIR" -maxdepth 1 -type d -name 'global_step_*' | sort -V | tail -n 1)"
if [[ -z "$LATEST_STEP_DIR" ]]; then
  echo "No global_step_* checkpoint found in $ROOT_DIR/$TRAIN_OUTPUT_DIR"
  exit 1
fi

ROOT_DIR="$ROOT_DIR" BASE_MODEL="$BASE_MODEL" ADAPTER_DIR="$LATEST_STEP_DIR" OUT_DIR="$ROOT_DIR/$MERGED_OUTPUT_DIR" conda run -n "$TRAIN_CONDA_ENV" python - <<'PY'
import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = os.environ["BASE_MODEL"]
adapter_dir = os.environ["ADAPTER_DIR"]
out_dir = os.environ["OUT_DIR"]
os.makedirs(out_dir, exist_ok=True)

base = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="cpu",
)
peft_model = PeftModel.from_pretrained(base, adapter_dir)
merged = peft_model.merge_and_unload()
merged.save_pretrained(out_dir, safe_serialization=True)

tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tok.save_pretrained(out_dir)
print(f"Merged model saved to: {out_dir}")
print(f"Adapter source: {adapter_dir}")
PY

echo "[OK] Merge completed: $ROOT_DIR/$MERGED_OUTPUT_DIR"
#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_CONDA_ENV="${TRAIN_CONDA_ENV:-DFT}"
EVAL_CONDA_ENV="${EVAL_CONDA_ENV:-DFT_EVAL}"
SETUP_MODE="${SETUP_MODE:-full}" # train | full
INSTALL_INFERENCE_STACK="${INSTALL_INFERENCE_STACK:-1}" # 1 follows original verl installation flow
USE_MEGATRON="${USE_MEGATRON:-0}" # original docs: USE_MEGATRON=0 for FSDP-only setup

if ! command -v conda >/dev/null 2>&1; then
	echo "conda not found. Please install conda/miniconda first."
	exit 1
fi

eval "$(conda shell.bash hook)"

echo "[INFO] Creating/updating train env: $TRAIN_CONDA_ENV (python=3.10)"
conda create -n "$TRAIN_CONDA_ENV" python=3.10 -y
echo "[INFO] Train env ready: $TRAIN_CONDA_ENV"

cd "$ROOT_DIR/verl"

if [[ "$INSTALL_INFERENCE_STACK" == "1" ]]; then
	echo "[INFO] Installing packages exactly like original verl docs..."
	echo "[INFO] Running: USE_MEGATRON=$USE_MEGATRON bash scripts/install_vllm_sglang_mcore.sh"
	conda run --no-capture-output -n "$TRAIN_CONDA_ENV" env USE_MEGATRON="$USE_MEGATRON" bash scripts/install_vllm_sglang_mcore.sh
else
	echo "[INFO] INSTALL_INFERENCE_STACK=0: installing only from verl/requirements.txt (no vllm/sglang stack)."
	conda run --no-capture-output -n "$TRAIN_CONDA_ENV" pip install -r requirements.txt
fi

echo "[INFO] Installing local verl package (-e . --no-deps)..."
conda run --no-capture-output -n "$TRAIN_CONDA_ENV" pip install --no-deps -e .

if [[ "$SETUP_MODE" == "full" ]]; then
	echo "[INFO] Creating/updating eval env: $EVAL_CONDA_ENV (python=3.10)"
	conda create -n "$EVAL_CONDA_ENV" python=3.10 -y

	cd "$ROOT_DIR/math_evaluation"
	conda run --no-capture-output -n "$EVAL_CONDA_ENV" pip install -r requirements.txt

	cd "$ROOT_DIR/math_evaluation/latex2sympy"
	conda run --no-capture-output -n "$EVAL_CONDA_ENV" pip install -e .

	echo "[INFO] Full mode: installed eval runtime deps into conda env $EVAL_CONDA_ENV."
else
	echo "[INFO] Train mode: skipped math_evaluation deps."
fi

echo "[OK] Environment setup finished."
echo "[OK] Train env: $TRAIN_CONDA_ENV"
if [[ "$SETUP_MODE" == "full" ]]; then
	echo "[OK] Eval env:  $EVAL_CONDA_ENV"
fi
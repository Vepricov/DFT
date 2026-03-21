#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_CONDA_ENV="${TRAIN_CONDA_ENV:-DFT}"
EVAL_CONDA_ENV="${EVAL_CONDA_ENV:-DFT_EVAL}"
SETUP_MODE="${SETUP_MODE:-full}" # train | full
INSTALL_INFERENCE_STACK="${INSTALL_INFERENCE_STACK:-1}" # 1 follows original README setup

if ! command -v conda >/dev/null 2>&1; then
	echo "conda not found. Please install conda/miniconda first."
	exit 1
fi

eval "$(conda shell.bash hook)"

echo "[INFO] Creating/updating train env: $TRAIN_CONDA_ENV (python=3.10)"
conda create -n "$TRAIN_CONDA_ENV" python=3.10 -y

cd "$ROOT_DIR/verl"

if [[ "$INSTALL_INFERENCE_STACK" == "1" ]]; then
	conda run -n "$TRAIN_CONDA_ENV" bash scripts/install_vllm_sglang_mcore.sh
else
	echo "[INFO] Skip vllm/sglang/flash-attn stack (INSTALL_INFERENCE_STACK=0)."
	conda run -n "$TRAIN_CONDA_ENV" pip install "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0"
fi

conda run -n "$TRAIN_CONDA_ENV" pip install --no-deps -e .

if [[ "$SETUP_MODE" == "full" ]]; then
	echo "[INFO] Creating/updating eval env: $EVAL_CONDA_ENV (python=3.10)"
	conda create -n "$EVAL_CONDA_ENV" python=3.10 -y

	cd "$ROOT_DIR/math_evaluation"
	conda run -n "$EVAL_CONDA_ENV" pip install -r requirements.txt

	cd "$ROOT_DIR/math_evaluation/latex2sympy"
	conda run -n "$EVAL_CONDA_ENV" pip install -e .

	echo "[INFO] Full mode: installed eval runtime deps into conda env $EVAL_CONDA_ENV."
else
	echo "[INFO] Train mode: skipped math_evaluation deps."
fi

echo "[OK] Environment setup finished."
echo "[OK] Train env: $TRAIN_CONDA_ENV"
if [[ "$SETUP_MODE" == "full" ]]; then
	echo "[OK] Eval env:  $EVAL_CONDA_ENV"
fi
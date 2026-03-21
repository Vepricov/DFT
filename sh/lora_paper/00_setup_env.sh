#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_CONDA_ENV="${TRAIN_CONDA_ENV:-DFT}"
EVAL_CONDA_ENV="${EVAL_CONDA_ENV:-DFT_EVAL}"
SETUP_MODE="${SETUP_MODE:-full}" # train | full

if ! command -v conda >/dev/null 2>&1; then
	echo "conda not found. Please install conda/miniconda first."
	exit 1
fi

eval "$(conda shell.bash hook)"

if conda run -n "$TRAIN_CONDA_ENV" python -V >/dev/null 2>&1; then
	echo "[INFO] Train env already exists: $TRAIN_CONDA_ENV (skip create)"
else
	echo "[INFO] Creating train env: $TRAIN_CONDA_ENV (python=3.10)"
	conda create -n "$TRAIN_CONDA_ENV" python=3.10 -y
	echo "[INFO] Train env created: $TRAIN_CONDA_ENV"
fi

cd "$ROOT_DIR/verl"

echo "[INFO] Installing train dependencies (single pip step)..."
conda run --no-capture-output -n "$TRAIN_CONDA_ENV" pip install -e .

if [[ "$SETUP_MODE" == "full" ]]; then
	if conda run -n "$EVAL_CONDA_ENV" python -V >/dev/null 2>&1; then
		echo "[INFO] Eval env already exists: $EVAL_CONDA_ENV (skip create)"
	else
		echo "[INFO] Creating eval env: $EVAL_CONDA_ENV (python=3.10)"
		conda create -n "$EVAL_CONDA_ENV" python=3.10 -y
		echo "[INFO] Eval env created: $EVAL_CONDA_ENV"
	fi

	cd "$ROOT_DIR/math_evaluation"
	echo "[INFO] Installing eval dependencies..."
	conda run --no-capture-output -n "$EVAL_CONDA_ENV" pip install torch
	tmp_req_file="$(mktemp)"
	grep -v '^flash_attn\s*$' requirements.txt > "$tmp_req_file"
	conda run --no-capture-output -n "$EVAL_CONDA_ENV" pip install -r "$tmp_req_file"
	rm -f "$tmp_req_file"

	cd "$ROOT_DIR/math_evaluation/latex2sympy"
	conda run --no-capture-output -n "$EVAL_CONDA_ENV" pip install -e .

	echo "[INFO] Full mode: eval dependencies installed into $EVAL_CONDA_ENV."
else
	echo "[INFO] Train mode: skipped math_evaluation deps."
fi

echo "[OK] Environment setup finished."
echo "[OK] Train env: $TRAIN_CONDA_ENV"
if [[ "$SETUP_MODE" == "full" ]]; then
	echo "[OK] Eval env:  $EVAL_CONDA_ENV"
fi
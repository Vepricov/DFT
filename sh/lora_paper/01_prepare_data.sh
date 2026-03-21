#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_END="${TRAIN_END:-100000}"
TRAIN_CONDA_ENV="${TRAIN_CONDA_ENV:-DFT}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH. Install conda/miniconda first."
  exit 1
fi

eval "$(conda shell.bash hook)"

cd "$ROOT_DIR/verl"
export PYTHONPATH="$PWD"

echo "[INFO] Running data preprocess in conda env: $TRAIN_CONDA_ENV"
conda run --no-capture-output -n "$TRAIN_CONDA_ENV" env PYTHONPATH="$PYTHONPATH" python examples/data_preprocess/numina_cot.py --train_end "$TRAIN_END"
conda run --no-capture-output -n "$TRAIN_CONDA_ENV" env PYTHONPATH="$PYTHONPATH" python examples/data_preprocess/math_dataset.py

echo "[OK] Data prepared: verl/data/numina_cot/train.parquet and verl/data/math500/test.parquet"
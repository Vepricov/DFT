#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_END="${TRAIN_END:-100000}"

if ! command -v python >/dev/null 2>&1; then
  echo "python not found in PATH. Activate your environment first (e.g., conda activate DFT)."
  exit 1
fi

if ! python - <<'PY' >/dev/null 2>&1
import datasets
PY
then
  echo "Missing package 'datasets' in current Python environment."
  echo "Run setup first: bash sh/lora_paper/00_setup_env.sh"
  echo "Or activate env with deps installed (e.g., conda activate DFT)."
  exit 1
fi

cd "$ROOT_DIR/verl"

python examples/data_preprocess/numina_cot.py --train_end "$TRAIN_END"
python examples/data_preprocess/math_dataset.py

echo "[OK] Data prepared: verl/data/numina_cot/train.parquet and verl/data/math500/test.parquet"
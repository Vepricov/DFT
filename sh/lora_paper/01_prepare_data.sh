#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_END="${TRAIN_END:-100000}"

cd "$ROOT_DIR/verl"

python examples/data_preprocess/numina_cot.py --train_end "$TRAIN_END"
python examples/data_preprocess/math_dataset.py

echo "[OK] Data prepared: verl/data/numina_cot/train.parquet and verl/data/math500/test.parquet"
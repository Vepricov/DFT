#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cd "$ROOT_DIR/verl"

bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .

cd "$ROOT_DIR/math_evaluation/latex2sympy"
pip install -e .

cd "$ROOT_DIR/math_evaluation"
pip install -r requirements.txt

echo "[OK] Environment setup finished for LoRA paper experiments."
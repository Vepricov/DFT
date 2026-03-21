#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# 1) Setup
bash sh/lora_paper/00_setup_env.sh

# 2) Data prep
bash sh/lora_paper/01_prepare_data.sh

# 3) Train Qwen LoRA baselines
bash sh/lora_paper/10_train_qwen15b_sft_lora.sh
bash sh/lora_paper/11_train_qwen15b_dft_lora.sh

# 4) Train Llama LoRA baselines
bash sh/lora_paper/12_train_llama32_3b_sft_lora.sh
bash sh/lora_paper/13_train_llama32_3b_dft_lora.sh

# 5) Merge adapters to full models for evaluation
bash sh/lora_paper/20_merge_lora_checkpoint.sh "Qwen/Qwen2.5-Math-1.5B" "verl/checkpoints/qwen15b-sft-lora-r8-a16" "merged/qwen15b-sft-lora-r8-a16"
bash sh/lora_paper/20_merge_lora_checkpoint.sh "Qwen/Qwen2.5-Math-1.5B" "verl/checkpoints/qwen15b-dft-lora-r8-a16" "merged/qwen15b-dft-lora-r8-a16"
bash sh/lora_paper/20_merge_lora_checkpoint.sh "meta-llama/Llama-3.2-3B-Instruct" "verl/checkpoints/llama32-3b-sft-lora-r8-a16" "merged/llama32-3b-sft-lora-r8-a16"
bash sh/lora_paper/20_merge_lora_checkpoint.sh "meta-llama/Llama-3.2-3B-Instruct" "verl/checkpoints/llama32-3b-dft-lora-r8-a16" "merged/llama32-3b-dft-lora-r8-a16"

# 6) Evaluate all merged models on paper benchmarks
bash sh/lora_paper/30_eval_qwen15b_sft_lora.sh
bash sh/lora_paper/31_eval_qwen15b_dft_lora.sh
bash sh/lora_paper/32_eval_llama32_3b_sft_lora.sh
bash sh/lora_paper/33_eval_llama32_3b_dft_lora.sh

echo "[OK] Full LoRA paper reproduction pipeline finished."
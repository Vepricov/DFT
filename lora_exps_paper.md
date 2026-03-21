# LoRA paper reproduction (полностью через sh-скрипты)

Ниже полный пайплайн, чтобы коллега клонировал этот репозиторий и запускал только `.sh`.

## Что покрыто

Репродьюснуты все LoRA-эксперименты из paper-сетапа:

1. Qwen2.5-Math-1.5B + SFT + LoRA (rank=8, alpha=16)
2. Qwen2.5-Math-1.5B + DFT + LoRA (rank=8, alpha=16)
3. Llama-3.2-3B-Instruct + SFT + LoRA (rank=8, alpha=16)
4. Llama-3.2-3B-Instruct + DFT + LoRA (rank=8, alpha=16)

С оценкой на paper-бенчмарках:
- `math_oai` (Math500 proxy)
- `minerva_math`
- `olympiadbench`
- `aime24`
- `amc23`

---

## Порядок запуска (по шагам)

### Шаг 0. Подготовка окружения

```bash
bash sh/lora_paper/00_setup_env.sh
```

### Шаг 1. Подготовка данных

```bash
bash sh/lora_paper/01_prepare_data.sh
```

### Шаг 2. Обучение 4 экспериментов

```bash
bash sh/lora_paper/10_train_qwen15b_sft_lora.sh
bash sh/lora_paper/11_train_qwen15b_dft_lora.sh
bash sh/lora_paper/12_train_llama32_3b_sft_lora.sh
bash sh/lora_paper/13_train_llama32_3b_dft_lora.sh
```

### Шаг 3. Merge LoRA adapter -> full model (для eval)

```bash
bash sh/lora_paper/20_merge_lora_checkpoint.sh "Qwen/Qwen2.5-Math-1.5B" "verl/checkpoints/qwen15b-sft-lora-r8-a16" "merged/qwen15b-sft-lora-r8-a16"
bash sh/lora_paper/20_merge_lora_checkpoint.sh "Qwen/Qwen2.5-Math-1.5B" "verl/checkpoints/qwen15b-dft-lora-r8-a16" "merged/qwen15b-dft-lora-r8-a16"
bash sh/lora_paper/20_merge_lora_checkpoint.sh "meta-llama/Llama-3.2-3B-Instruct" "verl/checkpoints/llama32-3b-sft-lora-r8-a16" "merged/llama32-3b-sft-lora-r8-a16"
bash sh/lora_paper/20_merge_lora_checkpoint.sh "meta-llama/Llama-3.2-3B-Instruct" "verl/checkpoints/llama32-3b-dft-lora-r8-a16" "merged/llama32-3b-dft-lora-r8-a16"
```

### Шаг 4. Eval 4 экспериментов

```bash
bash sh/lora_paper/30_eval_qwen15b_sft_lora.sh
bash sh/lora_paper/31_eval_qwen15b_dft_lora.sh
bash sh/lora_paper/32_eval_llama32_3b_sft_lora.sh
bash sh/lora_paper/33_eval_llama32_3b_dft_lora.sh
```

---

## Один командный запуск всего пайплайна

```bash
bash sh/lora_paper/40_run_all_lora_paper.sh
```

---

## Где лежат новые скрипты

- `sh/lora_paper/00_setup_env.sh`
- `sh/lora_paper/01_prepare_data.sh`
- `sh/lora_paper/10_train_qwen15b_sft_lora.sh`
- `sh/lora_paper/11_train_qwen15b_dft_lora.sh`
- `sh/lora_paper/12_train_llama32_3b_sft_lora.sh`
- `sh/lora_paper/13_train_llama32_3b_dft_lora.sh`
- `sh/lora_paper/20_merge_lora_checkpoint.sh`
- `sh/lora_paper/30_eval_qwen15b_sft_lora.sh`
- `sh/lora_paper/31_eval_qwen15b_dft_lora.sh`
- `sh/lora_paper/32_eval_llama32_3b_sft_lora.sh`
- `sh/lora_paper/33_eval_llama32_3b_dft_lora.sh`
- `sh/lora_paper/40_run_all_lora_paper.sh`

---

## Переменные окружения (опционально)

Во всех train/eval скриптах можно переопределить:

- `CUDA_VISIBLE_DEVICES`
- `NPROC_PER_NODE`
- `TRAIN_BATCH_SIZE`
- `MICRO_BATCH_SIZE_PER_GPU`
- `LR`
- `TOTAL_EPOCHS`
- `MODEL_NAME_OR_PATH` (в eval скриптах)
- `OUTPUT_DIR` (в eval скриптах)

По умолчанию выставлены paper-like параметры LoRA:

- `lora_rank=8`
- `lora_alpha=16`

---

## Техническое примечание

Для стабильного запуска в средах без `flash_attn` добавлен fallback на `sdpa` в:

- `verl/verl/trainer/fsdp_dft_trainer.py`
- `verl/verl/trainer/fsdp_sft_trainer.py`

Это не меняет логику DFT/LoRA, а только путь attention backend при отсутствии `flash_attn`.

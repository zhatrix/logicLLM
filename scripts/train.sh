#!/usr/bin/env bash
# LoRA 微调（MLX）。用法：bash scripts/train.sh [iters]
set -euo pipefail
cd "$(dirname "$0")/.."
ITERS=${1:-600}
BASE=${BASE_MODEL:-mlx-community/Qwen2.5-7B-Instruct-4bit}
ADAPTER=adapters/logistics-lora
mkdir -p "$ADAPTER"
uv run python -m mlx_lm lora \
  --model "$BASE" \
  --train \
  --data data/sft \
  --adapter-path "$ADAPTER" \
  --iters "$ITERS" \
  --batch-size 2 \
  --num-layers 16 \
  --learning-rate 1e-5 \
  --steps-per-eval 100 \
  --steps-per-report 20 \
  --max-seq-length 4096 \
  --mask-prompt \
  --seed 42
echo "训练完成 → $ADAPTER"

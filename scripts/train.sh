#!/usr/bin/env bash
# LoRA 微调（MLX）。用法：bash scripts/train.sh [iters]
# 环境变量：BASE_MODEL 基座；ADAPTER_PATH 输出目录（默认 adapters/logistics-lora，实验版可另设以免覆盖已发布权重）
set -euo pipefail
cd "$(dirname "$0")/.."
ITERS=${1:-800}
BASE=${BASE_MODEL:-mlx-community/Qwen3-14B-4bit}
ADAPTER=${ADAPTER_PATH:-adapters/logistics-lora}
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
  --grad-checkpoint \
  --mask-prompt \
  --seed 42
echo "训练完成 → $ADAPTER"

#!/usr/bin/env bash
# 把 LoRA 合并进基座（可选，用于分发）。
# ⚠️ 实测直接合并进 4bit 量化基座会让 LoRA 失效，必须 --de-quantize 得到 fp16 模型（约 15GB）。
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=${BASE_MODEL:-mlx-community/Qwen2.5-7B-Instruct-4bit}
uv run python -m mlx_lm fuse --model "$BASE" --adapter-path adapters/logistics-lora --save-path models/logistics-qwen-7b-fp16 --de-quantize
echo "已合并 → models/logistics-qwen-7b-fp16（fp16）"

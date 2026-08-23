#!/usr/bin/env bash
# 启动 mlx_lm OpenAI 兼容推理服务（端口 8080）：基座 + LoRA adapter。
# 说明：
# - 不要用 scripts/fuse.sh 合并到 4bit 基座再服务——实测合并后 LoRA 失效（行为等同基座）。
# - mlx_lm server 对 model=default_model 的请求不会应用 --adapter-path（bug），
#   所以应用侧在每个请求里显式带 "adapters": <绝对路径>（见 logicllm/config.py LLM_ADAPTER）。
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=${BASE_MODEL:-mlx-community/Qwen2.5-7B-Instruct-4bit}
ADAPTER="$(pwd)/adapters/logistics-lora"
if [ -f "$ADAPTER/adapters.safetensors" ]; then
  echo "基座 $BASE + LoRA $ADAPTER"
  exec uv run python -m mlx_lm server --model "$BASE" --adapter-path "$ADAPTER" --port 8080 --host 127.0.0.1
else
  echo "⚠️ 未找到 LoRA，使用原始基座"
  exec uv run python -m mlx_lm server --model "$BASE" --port 8080 --host 127.0.0.1
fi

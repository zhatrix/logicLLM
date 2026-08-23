"""全局配置，全部可通过环境变量覆盖（见 .env.example）。"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
KB_DIR = DATA_DIR / "kb"
KB_INDEX = DATA_DIR / "kb_index.npz"
ADAPTER_DIR = ROOT / "adapters" / "logistics-lora"

# 推理后端：mlx（mlx_lm.server）/ ollama（两者是 OpenAI 兼容 HTTP 接口）/ hf（进程内 transformers+PEFT，Linux GPU / ModelScope / HF Spaces）
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")
LLM_BASE_URL = os.getenv(
    "LLM_BASE_URL",
    "http://127.0.0.1:8080/v1" if LLM_BACKEND == "mlx" else "http://127.0.0.1:11434/v1",
)
LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "default_model" if LLM_BACKEND == "mlx" else "qwen2.5:3b",
)
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))

BASE_MODEL = os.getenv("BASE_MODEL", "mlx-community/Qwen2.5-7B-Instruct-4bit")
# mlx 后端：请求里显式带 adapters 路径（mlx_lm server 用 default_model 时不会应用 --adapter-path；
# 且 fuse 进 4bit 基座会让 LoRA 失效，所以不合并、直接挂 adapter）
_ADAPTER = ROOT / "adapters" / "logistics-lora"
LLM_ADAPTER = os.getenv("LLM_ADAPTER", str(_ADAPTER) if (LLM_BACKEND == "mlx" and (_ADAPTER / "adapters.safetensors").exists()) else "")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")

# hf 后端
HF_BASE_MODEL = os.getenv("HF_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
_PEFT = ROOT / "adapters" / "logistics-lora-peft"
HF_ADAPTER = os.getenv("HF_ADAPTER", str(_PEFT) if (_PEFT / "adapter_model.safetensors").exists() else "zh4trix/logistics-qwen-lora")
HF_4BIT = os.getenv("HF_4BIT", "1") == "1"  # 仅 CUDA 生效（bitsandbytes）
MODEL_SOURCE = os.getenv("MODEL_SOURCE", "modelscope")  # modelscope | hf：非本地路径的模型从哪里下载

RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.35"))
MAX_TOOL_ROUNDS = int(os.getenv("MAX_TOOL_ROUNDS", "5"))

SERVER_HOST = os.getenv("SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))

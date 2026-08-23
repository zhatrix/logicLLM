"""发布到 ModelScope：LoRA(PEFT) → 模型仓库；代码 → 创空间。
前置：uv run modelscope login --token <你的token>
用法：uv run python scripts/publish_modelscope.py [--owner zh4trix] [--model logistics-qwen-lora] [--studio logistics-llm]
"""
import argparse

from modelscope.hub.api import HubApi

ap = argparse.ArgumentParser()
ap.add_argument("--owner", default="zh4trix")
ap.add_argument("--model", default="logistics-qwen-lora")
ap.add_argument("--studio", default="logistics-llm")
ap.add_argument("--skip-model", action="store_true")
ap.add_argument("--skip-studio", action="store_true")
a = ap.parse_args()
api = HubApi()

if not a.skip_model:
    repo = f"{a.owner}/{a.model}"
    api.create_repo(repo, repo_type="model", visibility="public", license="apache-2.0",
                    chinese_name="物流通 LoRA（Qwen2.5-7B）", exist_ok=True)
    api.upload_folder(repo_id=repo, repo_type="model", folder_path="adapters/logistics-lora-peft",
                      commit_message="LoRA adapter (PEFT)")
    print("模型仓库:", f"https://www.modelscope.cn/models/{repo}")

if not a.skip_studio:
    repo = f"{a.owner}/{a.studio}"
    api.create_repo(repo, repo_type="studio", visibility="public", license="apache-2.0", sdk_type="gradio",
                    chinese_name="物流通 · 物流行业大模型", description="Qwen2.5-7B + 物流 LoRA + 知识库检索 + 业务工具调用",
                    exist_ok=True)
    api.upload_folder(repo_id=repo, repo_type="studio", folder_path=".",
                      allow_patterns=["app.py", "requirements.txt", "README.md", "Makefile", "pyproject.toml", ".env.example",
                                      "logicllm/**", "data/kb/**", "data/seed/**", "eval/**", "scripts/**"],
                      ignore_patterns=["**/__pycache__/**", "*.pyc", ".venv/**", "adapters/**", "models/**",
                                       "data/sft/**", "*.npz", "*.sqlite", ".git/**", "docs/**"],
                      commit_message="update space")
    print("创空间:", f"https://www.modelscope.cn/studios/{repo}")

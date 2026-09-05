"""发布到 Hugging Face：LoRA(PEFT) → 模型仓库；代码 → ZeroGPU Space。前置：hf auth login"""
import argparse, pathlib, shutil, tempfile
from huggingface_hub import HfApi

ap = argparse.ArgumentParser()
ap.add_argument("--owner", default="zhatrix"); ap.add_argument("--model", default="logistics-qwen3-lora"); ap.add_argument("--space", default="logistics-llm")
ap.add_argument("--skip-model", action="store_true"); ap.add_argument("--skip-space", action="store_true")
a = ap.parse_args(); api = HfApi(); root = pathlib.Path(".")

if not a.skip_model:
    repo = f"{a.owner}/{a.model}"
    api.create_repo(repo, repo_type="model", exist_ok=True)
    api.upload_folder(repo_id=repo, repo_type="model", folder_path="adapters/logistics-q3-14b-peft", commit_message="LoRA adapter (PEFT)")
    print("模型:", f"https://huggingface.co/{repo}")

if not a.skip_space:
    space = f"{a.owner}/{a.space}"
    api.create_repo(space, repo_type="space", space_sdk="gradio", space_hardware="zero-a10g", exist_ok=True)
    tmp = pathlib.Path(tempfile.mkdtemp())
    for rel in ["app.py", "requirements.txt", "Makefile", "pyproject.toml", ".env.example"]:
        shutil.copy(root / rel, tmp / rel)
    for d in ["logicllm", "data/kb", "data/seed", "eval", "scripts"]:
        shutil.copytree(root / d, tmp / d, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    front = ("---\ntitle: 物流通 · 物流行业大模型\nemoji: 🚚\ncolorFrom: yellow\ncolorTo: gray\nsdk: gradio\napp_file: app.py\n"
             "pinned: false\nlicense: apache-2.0\nshort_description: Qwen3-14B + 物流 LoRA + RAG + 工具调用\n"
             f"models:\n  - Qwen/Qwen3-14B\n  - {a.owner}/{a.model}\n---\n\n")
    (tmp / "README.md").write_text(front + (root / "README.md").read_text())
    api.upload_folder(repo_id=space, repo_type="space", folder_path=str(tmp), commit_message="update space")
    for k, v in {"HF_4BIT": "0", "MODEL_SOURCE": "hf", "HF_ADAPTER": f"{a.owner}/{a.model}", "HF_BASE_MODEL": "Qwen/Qwen3-14B", "LLM_TEMPERATURE": "0.1"}.items():
        api.add_space_variable(space, k, v)
    shutil.rmtree(tmp); print("Space:", f"https://huggingface.co/spaces/{space}")

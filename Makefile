PORT ?= 8010
.PHONY: install kb data train serve-mlx serve serve-ollama eval fuse
install:        ## 安装依赖
	uv sync
kb:             ## 构建知识库向量索引
	uv run python scripts/build_kb.py
data:           ## 生成 SFT 数据（知识问答复用 data/seed/teacher_kb_qa.json 缓存；题量与发布版一致）
	uv run python scripts/gen_sft.py --n-tool 400 --n-extract 240 --n-route 120
train:          ## LoRA 微调（默认 600 iters）
	bash scripts/train.sh $(ITERS)
serve-mlx:      ## 启动 mlx 推理服务（8080，带 LoRA）
	bash scripts/serve_mlx.sh
serve:          ## 启动应用服务（8000），LLM_BACKEND=mlx|ollama
	uv run uvicorn logicllm.server.app:app --host 127.0.0.1 --port $(PORT)
serve-ollama:   ## 用 Ollama 后端启动应用
	LLM_BACKEND=ollama uv run uvicorn logicllm.server.app:app --host 127.0.0.1 --port $(PORT)
eval:           ## 跑评测集
	uv run python scripts/eval.py
fuse:           ## 合并 LoRA 到基座
	bash scripts/fuse.sh

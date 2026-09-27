PORT ?= 8010
.PHONY: install kb data train serve-mlx serve serve-ollama eval eval-stress fuse
install:        ## 安装依赖
	uv sync
kb:             ## 构建知识库向量索引
	uv run python scripts/build_kb.py
data:           ## 生成 SFT 数据（知识问答复用 data/seed/teacher_kb_qa.json 缓存；题量与发布版一致）
	uv run python scripts/gen_sft.py --n-tool 480 --n-extract 240 --n-route 120 --n-numeric 320
train:          ## LoRA 微调（默认 800 iters ≈ 1 epoch）
	bash scripts/train.sh $(ITERS)
serve-mlx:      ## 启动 mlx 推理服务（8080，带 LoRA）
	bash scripts/serve_mlx.sh
serve:          ## 启动应用服务（8000），LLM_BACKEND=mlx|ollama
	uv run uvicorn logicllm.server.app:app --host 127.0.0.1 --port $(PORT)
serve-ollama:   ## 用 Ollama 后端启动应用
	LLM_BACKEND=ollama uv run uvicorn logicllm.server.app:app --host 127.0.0.1 --port $(PORT)
eval:           ## 跑评测集（55 题正式集）
	uv run python scripts/eval.py
eval-stress:    ## 跑压力集（89 题分布外变体，回答明细写到 /tmp/stress.jsonl）
	uv run python scripts/eval.py --cases eval/stress.jsonl --out /tmp/stress.jsonl
fuse:           ## 合并 LoRA 到基座
	bash scripts/fuse.sh

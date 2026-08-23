# logicLLM · 物流行业大模型「物流通」

一个可在 Apple Silicon 本机完整运行的物流行业大模型系统：**LoRA 微调（MLX）+ 知识库 RAG + 业务工具调用 + Web 服务**。

覆盖四类场景：

| 场景 | 能力 | 实现 |
|---|---|---|
| 客服问答 | 运单查询/轨迹、运费、时效、异常登记与理赔规则 | 工具调用 + RAG |
| 单据/信息抽取 | 收寄件人、电话、省市区、详细地址、物品 → JSON | 微调 + 规则解析器兜底 |
| 路径/调度 | 干线路径（Dijkstra）、多点配送顺序（最近邻+2-opt） | 工具调用 |
| 行业知识 | 计费、禁寄/危险品、理赔、仓储、报关、法规 | RAG + 微调 |

## 架构

```
用户 ─► Web UI / OpenAI 兼容 API (FastAPI :8010)
            │
            ▼
      LogisticsAgent ──► bge-m3 向量检索 (data/kb/*.md)
            │
            ├──► LLM 后端（OpenAI 协议，tools）
            │      ├ mlx_lm.server  : Qwen2.5-7B-Instruct ⊕ LoRA（合并） ← 微调后主力
            │      └ Ollama         : qwen2.5:3b                   ← 零配置回退
            │
            └──► 工具 (logicllm/tools)
                   query_waybill / track_waybill / search_waybills / create_waybill / report_exception
                   calc_freight / estimate_eta / plan_route / optimize_delivery_order / parse_address
```

工具调用统一走 OpenAI `tools`/`tool_calls`/`role=tool` 协议，两个后端都套用 Qwen 的 chat template。

**训练/推理分布一致**是微调见效的关键：SFT 样本与线上一样，system 里注入同一套检索上下文、请求里带同一份 `tools`
（`scripts/gen_sft.py` 复用 `LogisticsAgent` 的检索与提示词构造）。

## 快速开始

```bash
git clone https://github.com/zhatrix/logicLLM.git && cd logicLLM
uv sync                           # 安装依赖
make kb                           # 构建知识库索引（首次会下载 BAAI/bge-m3）

# 方式 A：零配置，用 Ollama 现成模型
ollama pull qwen2.5:3b
make serve-ollama                 # http://127.0.0.1:8010  (PORT=xxxx 可改)

# 方式 B-1：直接下载训练好的 LoRA（跳过训练，约 46MB）
mkdir -p adapters/logistics-lora
curl -L -o adapters/logistics-lora/adapters.safetensors https://github.com/zhatrix/logicLLM/releases/download/v0.1.0/adapters.safetensors
curl -L -o adapters/logistics-lora/adapter_config.json  https://github.com/zhatrix/logicLLM/releases/download/v0.1.0/adapter_config.json
make serve-mlx                    # 终端 1：mlx 推理服务 :8080（自动加载 adapter）
LLM_BACKEND=mlx make serve        # 终端 2：应用服务 :8010

# 方式 B-2：自己训练
make data                         # 生成 SFT 数据（知识问答部分需要 LLM 后端在线作教师）
make train ITERS=200              # LoRA 微调，M4 Max 约 1–2 小时（样本含工具定义+检索上下文，约 3k token/条）
make serve-mlx                    # 终端 1：mlx 推理服务 :8080（基座 + adapter，不合并）
LLM_BACKEND=mlx make serve        # 终端 2：应用服务 :8010
make eval                         # 跑评测集 eval/cases.jsonl
```

## 目录

```
logicllm/
  config.py          环境变量配置
  llm/client.py      OpenAI 兼容客户端（mlx / ollama），tool_calls 解析 + 文本兜底
  rag/kb.py          Markdown 切块 → bge-m3 → numpy 余弦检索
  tools/             业务工具（SQLite 运单库、计费、时效、路径、地址）
  agent/             系统提示词 + 工具循环（非流式 / SSE 流式）
  server/            FastAPI + 单页 Web UI
data/
  kb/*.md            物流知识库（8 篇：术语、计费、理赔、禁寄、仓储、报关、法规、FAQ）
  seed/*.jsonl       手写高质量种子样本
  sft/               生成的训练/验证集
scripts/
  build_kb.py        构建索引
  gen_sft.py         合成 SFT 数据（工具轨迹由真实工具执行保证正确；知识问答由教师模型生成）
  train.sh           mlx_lm lora
  serve_mlx.sh       mlx_lm server（自动加载 LoRA）
  fuse.sh            合并权重
  eval.py            规则评测（工具命中 / 关键词 / JSON 字段）
eval/cases.jsonl     评测用例
```

## API

- `POST /api/chat` `{"messages":[{"role":"user","content":"…"}], "use_tools":true, "use_rag":true}` → `{answer, steps[]}`
- `POST /api/chat/stream` 同上，SSE：`rag` / `tool_call` / `tool_result` / `delta` / `done`
- `POST /v1/chat/completions` OpenAI 兼容（可接任意前端）
- `GET /api/health` · `GET /api/tools`

## 配置（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `LLM_BACKEND` | `ollama` | `mlx` / `ollama` |
| `LLM_BASE_URL` | 随后端 | OpenAI 兼容地址 |
| `LLM_MODEL` | 随后端 | 模型名 |
| `BASE_MODEL` | `mlx-community/Qwen2.5-7B-Instruct-4bit` | 微调/推理基座 |
| `EMBED_MODEL` | `BAAI/bge-m3` | 向量模型 |
| `RAG_TOP_K` / `RAG_MIN_SCORE` | 4 / 0.35 | 检索参数 |

## 评测结果（eval/cases.jsonl，20 例，规则打分）

| 模型 | 总分 | 客服 | 知识 | 抽取 | 调度 |
|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct 基座 + RAG + 工具 | 0.825 | 0.833 | 0.938 | 0.000 | 1.000 |
| **+ LoRA（200 步，Val loss 0.104）** | **0.880** | 0.833 | 0.875 | 0.800 | 1.000 |

微调主要解决了：抽取任务按约定输出 JSON、运费计算不漏箱体尺寸、回答更精炼（全量评测 216s vs 基线 4934s）。
剩余失分：未指定服务类型时偶尔自行填 `经济`、"53 度 vs 70%"数值比较出错、省份输出全称（评测过严）。
加数据方向：不带服务类型的运费问法、数值比较类知识问答。

## 踩坑记录（mlx_lm 0.31）

1. `mlx_lm server` 对 `model=default_model` 的请求**不会应用** `--adapter-path`；应用侧在请求里显式带 `"adapters": <绝对路径>`（`config.LLM_ADAPTER`）。
2. `mlx_lm fuse` 直接合并进 4bit 量化基座后 LoRA **完全失效**；要合并请用 `make fuse`（带 `--de-quantize`，输出 fp16）。
3. `--mask-prompt` 只对**最后一条** assistant 消息计算 loss，多轮工具轨迹中间的 `tool_call` 不会被训练；`gen_sft.py` 会把每条轨迹额外拆出"到该次调用为止"的前缀样本。
4. 训练样本必须与线上 prompt 一致（同一份 `tools`、同样注入检索上下文），否则微调效果被分布偏移抵消。

## 扩展

- 加知识：往 `data/kb/` 放 Markdown，`make kb`。
- 加工具：在 `logicllm/tools/` 用 `@tool(name, description, json_schema)` 装饰函数即可自动注册。
- 加训练数据：`data/seed/*.jsonl`（mlx-lm chat 格式，可带 `tools`），或扩展 `scripts/gen_sft.py`。
- 接真实系统：把 `tools/waybill.py` 的 SQLite 换成你的 TMS/OMS 接口。

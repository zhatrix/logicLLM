"""Gradio 入口（ModelScope 创空间 / HF Spaces）。

环境变量：LLM_BACKEND=hf（默认）、HF_BASE_MODEL、HF_ADAPTER、HF_4BIT。
本地调试：LLM_BACKEND=mlx uv run python app.py  （复用本机 mlx 服务）
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("LLM_BACKEND", "hf")

import gradio as gr  # noqa: E402

from logicllm import config  # noqa: E402
from logicllm.agent.core import LogisticsAgent  # noqa: E402
from logicllm.tools.waybill import init_db  # noqa: E402

init_db()
AGENT = LogisticsAgent()

EXAMPLES = [
    "帮我查一下运单 LL2026080258",
    "从深圳寄一个 2.2kg、40×30×30cm 的箱子到拉萨，保价1000元，运费多少？几天能到？",
    "没保价的快递丢了能赔多少？",
    "充电宝能寄吗？",
    "从武汉出发送货到长沙、郑州、合肥、南昌，怎么排顺序最省路？",
    "提取信息：收件人王小明 13812345678 浙江省杭州市西湖区文三路123号5楼",
    "快递员没经过我同意把包裹放快递柜了，合规吗？",
    "FOB 和 CIF 有什么区别？",
]


def _fmt(obj, limit=600) -> str:
    s = json.dumps(obj, ensure_ascii=False)
    return s if len(s) <= limit else s[:limit] + "…"


async def respond(message: str, history: list[dict], use_tools: bool, use_rag: bool):
    agent = LogisticsAgent(client=AGENT.client, kb=AGENT.kb, use_tools=use_tools, use_rag=use_rag)
    msgs = [m for m in history if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)]
    msgs.append({"role": "user", "content": message})
    trace, answer = [], ""
    async for ev in agent.stream(msgs):
        t = ev["type"]
        if t == "rag":
            hits = "；".join(f"{h['title']}（{h['score']}）" for h in ev["data"]["hits"])
            trace.append(f"📚 知识库命中：{hits}")
        elif t == "tool_call":
            trace.append(f"🔧 调用 `{ev['data']['name']}` {_fmt(ev['data']['arguments'])}")
        elif t == "tool_result":
            trace.append(f"↩ 结果 {_fmt(ev['data'])}")
        elif t == "delta":
            answer += ev["data"]
        elif t == "done":
            answer = answer or ev["data"]
        elif t == "error":
            answer = f"⚠️ {ev['data']}"
        head = ("<details><summary>处理过程</summary>\n\n" + "\n\n".join(trace) + "\n\n</details>\n\n") if trace else ""
        yield head + answer


with gr.Blocks(title="物流通 · logicLLM") as demo:
    gr.Markdown(
        "## 🚚 物流通 · 物流行业大模型\n"
        f"基座 `{config.HF_BASE_MODEL if config.LLM_BACKEND == 'hf' else config.LLM_MODEL}` + 物流 LoRA · 知识库检索 · 10 个业务工具（运单 / 计费 / 时效 / 路径 / 地址）。"
        "运单数据为演示库，示例运单号 `LL2026080258`。"
    )
    with gr.Row():
        use_tools = gr.Checkbox(True, label="工具调用")
        use_rag = gr.Checkbox(True, label="知识库检索")
    gr.ChatInterface(
        respond,
        additional_inputs=[use_tools, use_rag],
        examples=[[e] for e in EXAMPLES],
        cache_examples=False,
        textbox=gr.Textbox(placeholder="输入物流相关问题…", scale=7),
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))

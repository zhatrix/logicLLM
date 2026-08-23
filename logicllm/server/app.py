"""FastAPI 服务：
- GET  /            Web 聊天界面
- POST /api/chat    非流式，返回答案 + 中间步骤
- POST /api/chat/stream  SSE 流式
- POST /v1/chat/completions  OpenAI 兼容（方便接入其他前端）
- GET  /api/health  后端模型/知识库状态
- GET  /api/tools   工具列表
"""
from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from logicllm import config
from logicllm.agent.core import LogisticsAgent
from logicllm.tools import all_specs
from logicllm.tools.waybill import init_db

STATIC = Path(__file__).parent / "static"
agent: LogisticsAgent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    init_db()
    agent = LogisticsAgent()
    yield
    await agent.client.aclose()


app = FastAPI(title="logicLLM 物流大模型", lifespan=lifespan)


class Msg(BaseModel):
    role: str
    content: str


class ChatReq(BaseModel):
    messages: list[Msg]
    use_tools: bool = True
    use_rag: bool = True


def _agent(req: ChatReq) -> LogisticsAgent:
    a = LogisticsAgent(client=agent.client, kb=agent.kb, use_tools=req.use_tools, use_rag=req.use_rag)
    return a


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
async def health():
    return {
        "backend": config.LLM_BACKEND, "base_url": config.LLM_BASE_URL, "model": config.LLM_MODEL,
        "llm_ok": await agent.client.healthy(), "kb_ok": agent.kb.available,
        "kb_chunks": int(agent.kb.emb.shape[0]) if agent.kb.available else 0,
        "tools": [t["name"] for t in all_specs()],
    }


@app.get("/api/tools")
async def tools():
    return all_specs()


@app.post("/api/chat")
async def chat(req: ChatReq):
    res = await _agent(req).run([m.model_dump() for m in req.messages])
    return {"answer": res.answer, "steps": [{"kind": s.kind, "content": s.content} for s in res.steps]}


@app.post("/api/chat/stream")
async def chat_stream(req: ChatReq):
    async def gen():
        try:
            async for ev in _agent(req).stream([m.model_dump() for m in req.messages]):
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except Exception as e:  # noqa: BLE001
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)}, ensure_ascii=False)}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")


class OpenAIReq(BaseModel):
    model: str | None = None
    messages: list[Msg]
    stream: bool = False
    temperature: float | None = None


@app.post("/v1/chat/completions")
async def openai_compat(req: OpenAIReq):
    msgs = [m.model_dump() for m in req.messages]
    rid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    if not req.stream:
        res = await agent.run(msgs)
        return {
            "id": rid, "object": "chat.completion", "created": int(time.time()), "model": "logicllm",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": res.answer}, "finish_reason": "stop"}],
        }

    async def gen():
        async for ev in agent.stream(msgs):
            if ev["type"] == "delta":
                chunk = {"id": rid, "object": "chat.completion.chunk", "model": "logicllm",
                         "choices": [{"index": 0, "delta": {"content": ev["data"]}, "finish_reason": None}]}
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/models")
async def models():
    return JSONResponse({"data": [{"id": "logicllm"}]})

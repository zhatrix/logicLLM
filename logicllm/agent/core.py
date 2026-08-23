"""Agent 主循环：RAG 注入 → LLM(tools) → 执行 tool_calls → role=tool 回填 → 直到最终回答。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import AsyncIterator

from logicllm import config
from logicllm.agent.prompts import build_system, openai_tools
from logicllm.llm import make_client
from logicllm.llm.client import ChatOut, tool_calls_to_openai
from logicllm.rag.kb import KnowledgeBase
from logicllm.tools import all_specs, call_tool


@dataclass
class Step:
    kind: str  # "rag" | "tool_call" | "tool_result" | "answer"
    content: dict | str


@dataclass
class AgentResult:
    answer: str
    steps: list[Step] = field(default_factory=list)


class LogisticsAgent:
    def __init__(self, client=None, kb: KnowledgeBase | None = None,
                 use_tools: bool = True, use_rag: bool = True):
        self.client = client or make_client()
        self.kb = kb or KnowledgeBase()
        self.use_tools = use_tools
        self.use_rag = use_rag

    def _prepare(self, messages: list[dict]) -> tuple[list[dict], list[dict] | None, list[Step]]:
        steps: list[Step] = []
        user_q = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        context = None
        if self.use_rag and self.kb.available:
            hits = self.kb.search(user_q)
            if hits:
                context = KnowledgeBase.format(hits)
                steps.append(Step("rag", {"hits": [{"title": h["title"], "score": round(h["score"], 3)} for h in hits]}))
        convo = [{"role": "system", "content": build_system(context)}] + [m for m in messages if m["role"] != "system"]
        tools = openai_tools(all_specs()) if self.use_tools else None
        return convo, tools, steps

    async def _apply(self, convo: list[dict], out: ChatOut) -> list[Step]:
        """把一轮工具调用写回对话，返回产生的步骤。"""
        steps = []
        convo.append({"role": "assistant", "content": out.content or "", "tool_calls": tool_calls_to_openai(out.tool_calls)})
        for c in out.tool_calls:
            steps.append(Step("tool_call", {"name": c.name, "arguments": c.arguments}))
            result = await call_tool(c.name, c.arguments)
            steps.append(Step("tool_result", result))
            convo.append({"role": "tool", "tool_call_id": c.id, "name": c.name,
                          "content": json.dumps(result, ensure_ascii=False)})
        return steps

    async def run(self, messages: list[dict]) -> AgentResult:
        convo, tools, steps = self._prepare(messages)
        for _ in range(config.MAX_TOOL_ROUNDS):
            out = await self.client.chat(convo, tools=tools)
            if not out.tool_calls:
                steps.append(Step("answer", out.content))
                return AgentResult(out.content, steps)
            steps.extend(await self._apply(convo, out))
        out = await self.client.chat(convo + [{"role": "user", "content": "请根据以上工具结果直接给出最终回答。"}])
        steps.append(Step("answer", out.content))
        return AgentResult(out.content, steps)

    async def stream(self, messages: list[dict]) -> AsyncIterator[dict]:
        """SSE 事件：{"type": "rag"|"tool_call"|"tool_result"|"delta"|"done", "data": ...}"""
        convo, tools, steps = self._prepare(messages)
        for s in steps:
            yield {"type": s.kind, "data": s.content}
        for _ in range(config.MAX_TOOL_ROUNDS):
            final: ChatOut | None = None
            async for kind, payload in self.client.stream(convo, tools=tools):
                if kind == "delta":
                    yield {"type": "delta", "data": payload}
                else:
                    final = payload
            if final is None or not final.tool_calls:
                yield {"type": "done", "data": (final.content if final else "")}
                return
            for s in await self._apply(convo, final):
                yield {"type": s.kind, "data": s.content}
        yield {"type": "done", "data": "已达到最大工具调用轮数。"}

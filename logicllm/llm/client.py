"""统一的 OpenAI 兼容聊天客户端（mlx_lm.server / Ollama）。

工具调用走标准 OpenAI `tools` / `tool_calls` / role=tool 协议：两个后端都会套用 Qwen 的 chat template，
把工具渲染进 system、把 <tool_call> 解析成 tool_calls，和微调数据格式完全一致。
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator

import httpx

from logicllm import config


@dataclass
class ToolCall:
    name: str
    arguments: dict
    id: str = field(default_factory=lambda: f"call_{uuid.uuid4().hex[:8]}")


@dataclass
class ChatOut:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


def _parse_tool_calls(raw) -> list[ToolCall]:
    out = []
    for tc in raw or []:
        fn = tc.get("function") or {}
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                args = {"_raw": args}
        out.append(ToolCall(fn.get("name") or "", args, tc.get("id") or f"call_{uuid.uuid4().hex[:8]}"))
    return out


XML_FN_RE = None


def _parse_xml_tool_call(block: str) -> ToolCall | None:
    """Qwen3/3.5 的 XML 参数风格：<function=name><parameter=key>value</parameter>…</function>"""
    import re
    m = re.search(r"<function=([\w-]+)>(.*?)(?:</function>|$)", block, re.S)
    if not m:
        return None
    args = {}
    for pm in re.finditer(r"<parameter=([\w-]+)>\s*(.*?)\s*(?:</parameter>|$)", m.group(2), re.S):
        v = pm.group(2)
        try:
            args[pm.group(1)] = json.loads(v)
        except json.JSONDecodeError:
            args[pm.group(1)] = v
    return ToolCall(m.group(1), args)


def _fallback_text_tool_calls(text: str) -> tuple[str, list[ToolCall]]:
    """兜底：后端没解析、模型直接输出了 <tool_call>…</tool_call> 文本（JSON 或 XML 参数风格）或裸 JSON。"""
    import re
    # 去掉思考段
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    calls = []
    for m in re.finditer(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", text, re.S):
        blk = m.group(1)
        if blk.lstrip().startswith("<function"):
            c = _parse_xml_tool_call(blk)
            if c:
                calls.append(c)
    if calls:
        return re.sub(r"<tool_call>.*?(?:</tool_call>|$)", "", text, flags=re.S).strip(), calls
    for m in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S):
        try:
            d = json.loads(m.group(1))
            calls.append(ToolCall(d["name"], d.get("arguments") or {}))
        except (json.JSONDecodeError, KeyError):
            continue
    if calls:
        return re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.S).strip(), calls
    s = text.strip()
    if s.startswith("{") and '"name"' in s:
        try:
            d = json.loads(s)
            if "name" in d and "arguments" in d:
                return "", [ToolCall(d["name"], d["arguments"] or {})]
        except json.JSONDecodeError:
            pass
    return text, []


def tool_calls_to_openai(calls: list[ToolCall]) -> list[dict]:
    return [{"id": c.id, "type": "function",
             "function": {"name": c.name, "arguments": json.dumps(c.arguments, ensure_ascii=False)}} for c in calls]


class ChatClient:
    def __init__(self, base_url: str | None = None, model: str | None = None):
        self.base_url = (base_url or config.LLM_BASE_URL).rstrip("/")
        self.model = model or config.LLM_MODEL
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))

    def _payload(self, messages, tools, temperature, max_tokens, stream):
        p = {"model": self.model, "messages": messages,
             "temperature": config.LLM_TEMPERATURE if temperature is None else temperature,
             "max_tokens": max_tokens or config.LLM_MAX_TOKENS, "stream": stream}
        if tools:
            p["tools"] = tools
        if config.LLM_ADAPTER:
            p["adapters"] = config.LLM_ADAPTER
        if config.LLM_BACKEND == "mlx" and not config.LLM_THINKING:
            p["chat_template_kwargs"] = {"enable_thinking": False}
        return p

    async def chat(self, messages: list[dict], tools: list[dict] | None = None,
                   temperature: float | None = None, max_tokens: int | None = None) -> ChatOut:
        payload = self._payload(messages, tools, temperature, max_tokens, False)
        try:
            r = await self._client.post(f"{self.base_url}/chat/completions", json=payload)
        except httpx.RemoteProtocolError:
            # 后端偶发崩溃断连（如 mlx_lm 解析异常工具调用），重试一次
            r = await self._client.post(f"{self.base_url}/chat/completions", json=payload)
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        content = msg.get("content") or ""
        calls = _parse_tool_calls(msg.get("tool_calls"))
        if not calls and tools:
            content, calls = _fallback_text_tool_calls(content)
        return ChatOut(content, calls)

    async def stream(self, messages: list[dict], tools: list[dict] | None = None,
                     temperature: float | None = None, max_tokens: int | None = None) -> AsyncIterator[tuple[str, object]]:
        """产出 ("delta", str) 与最终 ("final", ChatOut)。"""
        content, raw_calls = "", {}
        async with self._client.stream("POST", f"{self.base_url}/chat/completions",
                                       json=self._payload(messages, tools, temperature, max_tokens, True)) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    d = json.loads(data)["choices"][0]["delta"]
                except (KeyError, IndexError, ValueError):
                    continue
                if d.get("content"):
                    content += d["content"]
                    # 模型把工具调用当文本输出时先不往外吐
                    if "<tool_call>" not in content and not content.lstrip().startswith("{"):
                        yield ("delta", d["content"])
                for tc in d.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    slot = raw_calls.setdefault(idx, {"id": tc.get("id"), "function": {"name": "", "arguments": ""}})
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["function"]["name"] = fn["name"]
                    if fn.get("arguments"):
                        a = fn["arguments"]
                        slot["function"]["arguments"] += a if isinstance(a, str) else json.dumps(a, ensure_ascii=False)
        calls = _parse_tool_calls([raw_calls[k] for k in sorted(raw_calls)])
        if not calls and tools:
            content, calls = _fallback_text_tool_calls(content)
        yield ("final", ChatOut(content, calls))

    async def healthy(self) -> bool:
        try:
            r = await self._client.get(f"{self.base_url}/models", timeout=5)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def aclose(self):
        await self._client.aclose()

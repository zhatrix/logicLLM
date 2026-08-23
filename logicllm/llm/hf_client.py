"""进程内 transformers + PEFT 推理后端（Linux/NVIDIA、ModelScope 创空间、HF Spaces 用）。

与 ChatClient 接口一致：chat() / stream() / healthy() / aclose()。
工具调用走 Qwen chat template：tools 渲染进 system，模型输出 <tool_call>…</tool_call>，这里解析成 ToolCall。
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import threading
from typing import AsyncIterator

from logicllm import config
from logicllm.llm.client import ChatOut, ToolCall

TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def _parse(text: str) -> ChatOut:
    calls = []
    for m in TOOL_RE.finditer(text):
        try:
            d = json.loads(m.group(1))
            calls.append(ToolCall(d["name"], d.get("arguments") or {}))
        except (json.JSONDecodeError, KeyError):
            continue
    content = TOOL_RE.sub("", text).strip() if calls else text.strip()
    return ChatOut(content, calls)


class HFChatClient:
    def __init__(self, base: str | None = None, adapter: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.base = base or config.HF_BASE_MODEL
        self.adapter = adapter if adapter is not None else config.HF_ADAPTER
        self.model_name = f"{self.base}+{os.path.basename(self.adapter)}" if self.adapter else self.base
        self.model = self.base  # 兼容 health 输出
        self.tok = AutoTokenizer.from_pretrained(self.base)

        if torch.cuda.is_available():
            device, dtype = "cuda", torch.bfloat16
        elif torch.backends.mps.is_available():
            device, dtype = "mps", torch.bfloat16
        else:
            device, dtype = "cpu", torch.float32
        kw: dict = {"dtype": dtype}
        if config.HF_4BIT and device == "cuda":
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                                           bnb_4bit_quant_type="nf4")
            kw["device_map"] = "auto"
        self._model = AutoModelForCausalLM.from_pretrained(self.base, **kw)
        if "device_map" not in kw:
            self._model.to(device)
        if self.adapter:
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(self._model, self.adapter)
        self._model.eval()
        self.device = device
        self._lock = threading.Lock()

    # ---- 内部 ----
    def _inputs(self, messages, tools):
        import torch
        text = self.tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)
        enc = self.tok(text, return_tensors="pt")
        return {k: v.to(self._model.device) for k, v in enc.items()}

    def _gen_kwargs(self, temperature, max_tokens):
        t = config.LLM_TEMPERATURE if temperature is None else temperature
        kw = {"max_new_tokens": max_tokens or config.LLM_MAX_TOKENS, "pad_token_id": self.tok.eos_token_id}
        if t and t > 0:
            kw.update(do_sample=True, temperature=t, top_p=0.9)
        else:
            kw["do_sample"] = False
        return kw

    def _generate_sync(self, messages, tools, temperature, max_tokens) -> str:
        import torch
        with self._lock, torch.no_grad():
            inputs = self._inputs(messages, tools)
            out = self._model.generate(**inputs, **self._gen_kwargs(temperature, max_tokens))
        return self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # ---- 公共接口 ----
    async def chat(self, messages: list[dict], tools: list[dict] | None = None,
                   temperature: float | None = None, max_tokens: int | None = None) -> ChatOut:
        text = await asyncio.to_thread(self._generate_sync, messages, tools, temperature, max_tokens)
        return _parse(text)

    async def stream(self, messages: list[dict], tools: list[dict] | None = None,
                     temperature: float | None = None, max_tokens: int | None = None) -> AsyncIterator[tuple[str, object]]:
        from transformers import TextIteratorStreamer
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()

        def run():
            import torch
            with self._lock, torch.no_grad():
                inputs = self._inputs(messages, tools)
                self._model.generate(**inputs, streamer=streamer, **self._gen_kwargs(temperature, max_tokens))

        def pump():
            try:
                for piece in streamer:
                    loop.call_soon_threadsafe(q.put_nowait, piece)
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        threading.Thread(target=run, daemon=True).start()
        threading.Thread(target=pump, daemon=True).start()
        buf, yielded = "", 0
        while True:
            piece = await q.get()
            if piece is None:
                break
            buf += piece
            # 可能是工具调用时先不往外吐
            if "<tool_call>" in buf or buf.lstrip().startswith(("<", "{")) and len(buf) < 16:
                continue
            if "<tool" in buf:
                continue
            yield ("delta", buf[yielded:])
            yielded = len(buf)
        out = _parse(buf)
        if not out.tool_calls and yielded < len(buf):
            yield ("delta", buf[yielded:])
        yield ("final", out)

    async def healthy(self) -> bool:
        return True

    async def aclose(self):
        return None

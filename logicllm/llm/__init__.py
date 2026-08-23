"""推理客户端工厂：按 config.LLM_BACKEND 返回 HTTP 客户端或进程内 transformers 客户端。"""
from logicllm import config


def make_client():
    if config.LLM_BACKEND == "hf":
        from logicllm.llm.hf_client import HFChatClient
        return HFChatClient()
    from logicllm.llm.client import ChatClient
    return ChatClient()

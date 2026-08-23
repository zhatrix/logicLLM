"""工具注册表。每个工具：name / description / parameters(JSON Schema) / fn(**kwargs)->dict。"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., Any]

    def spec(self) -> dict:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


REGISTRY: dict[str, Tool] = {}


def tool(name: str, description: str, parameters: dict):
    def deco(fn):
        REGISTRY[name] = Tool(name, description, parameters, fn)
        return fn
    return deco


async def call_tool(name: str, args: dict) -> dict:
    t = REGISTRY.get(name)
    if t is None:
        return {"error": f"未知工具: {name}", "available": list(REGISTRY)}
    try:
        res = t.fn(**args)
        if inspect.isawaitable(res):
            res = await res
        return res
    except TypeError as e:
        return {"error": f"参数错误: {e}", "expected": t.parameters}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def all_specs() -> list[dict]:
    return [t.spec() for t in REGISTRY.values()]


# 导入即注册
from logicllm.tools import waybill, pricing, eta, route, address  # noqa: E402,F401

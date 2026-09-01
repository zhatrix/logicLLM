"""评测：对 eval/cases.jsonl 逐条跑 Agent，按规则打分（关键词命中 / 工具调用正确 / JSON 字段正确）。
用法：uv run python scripts/eval.py [--no-tools] [--no-rag] [--limit N]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import time

from logicllm import config
from logicllm.agent.core import LogisticsAgent

CASES = config.ROOT / "eval" / "cases.jsonl"


def score(case: dict, answer: str, steps: list) -> tuple[float, str]:
    tools_called = [s.content["name"] for s in steps if s.kind == "tool_call"]
    notes = []
    ok = 1.0
    if "expect_tool" in case:
        if case["expect_tool"] not in tools_called:
            ok = 0.0; notes.append(f"未调用 {case['expect_tool']} (实际 {tools_called})")
    if "expect_tools" in case:  # 多个工具都必须被调用
        miss_t = [t for t in case["expect_tools"] if t not in tools_called]
        if miss_t:
            ok = 0.0; notes.append(f"未调用 {miss_t} (实际 {tools_called})")
    if "expect_no_tool" in case and tools_called:
        ok = 0.0; notes.append(f"不应调用工具，实际 {tools_called}")
    norm = answer.replace(" ", "")
    if "keywords" in case:
        miss = [k for k in case["keywords"] if k.replace(" ", "") not in norm]
        if miss:
            ok = min(ok, 1 - len(miss) / len(case["keywords"])); notes.append(f"缺关键词 {miss}")
    if "keywords_any" in case:  # 命中其一即可
        if not any(k.replace(" ", "") in norm for k in case["keywords_any"]):
            ok = 0.0; notes.append(f"任一关键词均未命中 {case['keywords_any']}")
    if "forbid" in case:  # 出现即零分（用于臆测/编造检查）
        hit = [k for k in case["forbid"] if k.replace(" ", "") in norm]
        if hit:
            ok = 0.0; notes.append(f"出现禁用词 {hit}")
    if "expect_json" in case:
        m = re.search(r"\{.*\}", answer, re.S)
        try:
            got = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            got = {}
        exp = case["expect_json"]
        hit = sum(1 for k, v in exp.items() if str(got.get(k, "")).strip() == str(v))
        ok = min(ok, hit / len(exp)); notes.append(f"JSON 字段 {hit}/{len(exp)}")
    return ok, "; ".join(notes)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-tools", action="store_true")
    ap.add_argument("--no-rag", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    cases = [json.loads(l) for l in CASES.read_text(encoding="utf-8").splitlines()
             if l.strip() and not l.strip().startswith("//")]
    if a.limit:
        cases = cases[:a.limit]
    agent = LogisticsAgent(use_tools=not a.no_tools, use_rag=not a.no_rag)
    print(f"后端 {config.LLM_BACKEND}:{config.LLM_MODEL}  工具={not a.no_tools} RAG={not a.no_rag}  用例 {len(cases)}\n")
    total, by_cat = 0.0, {}
    t0 = time.time()
    for i, c in enumerate(cases, 1):
        try:
            turns = c.get("turns") or [c["q"]]
            history, steps_all = [], []
            for t in turns:
                history.append({"role": "user", "content": t})
                res = await agent.run(history)
                history.append({"role": "assistant", "content": res.answer})
                steps_all.extend(res.steps)
            s, note = score(c, res.answer, steps_all)
        except Exception as e:  # noqa: BLE001
            s, note, res = 0.0, f"异常 {e}", None
        total += s
        by_cat.setdefault(c["cat"], []).append(s)
        flag = "✅" if s >= 0.99 else ("🟡" if s > 0 else "❌")
        qshow = c.get("q") or " ▸ ".join(c.get("turns", []))
        print(f"{flag} [{c['cat']}] {qshow[:44]}  {s:.2f}  {note}")
    print(f"\n总分 {total / len(cases):.3f}  用时 {time.time() - t0:.0f}s")
    for k, v in by_cat.items():
        print(f"  {k}: {sum(v) / len(v):.3f} ({len(v)})")
    await agent.client.aclose()


if __name__ == "__main__":
    asyncio.run(main())

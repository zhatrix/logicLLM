"""生成 SFT 训练数据（mlx-lm chat 格式 jsonl）。

四类场景：
1. 行业知识问答  —— 由知识库切块 + 本地 LLM 生成 Q/A（教师模型：Ollama / mlx 后端任意）
2. 客服问答 + 工具调用 —— 程序化模板 + 真实工具执行结果（保证工具调用轨迹 100% 正确）
3. 单据/地址抽取 —— 程序化合成地址 → 规则解析器给标签
4. 路径/调度 —— 程序化模板 + 真实工具结果

输出：data/sft/train.jsonl, data/sft/valid.jsonl
用法：uv run python scripts/gen_sft.py --n-kb 200 --n-tool 300 --n-extract 150 --n-route 100
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path

from logicllm import config
from logicllm.agent.prompts import build_system, openai_tools
from logicllm.llm.client import ChatClient
from logicllm.rag.kb import split_markdown
from logicllm.tools import all_specs, call_tool
from logicllm.tools.geo import CITY_PROVINCE, CITY_DISTRICTS, province_full
from logicllm.tools.waybill import init_db, _conn

OUT = config.DATA_DIR / "sft"
rnd = random.Random(7)
SYSTEM = build_system()
TOOLS = openai_tools(all_specs())
SYSTEM_TOOLS, SYSTEM_PLAIN = "tools", "plain"  # 标记：是否附带 tools

NAMES = ["张伟", "王芳", "李娜", "刘强", "陈静", "杨洋", "赵磊", "黄敏", "周杰", "吴丹", "徐丽", "孙浩", "马超", "朱琳", "胡军"]
STREETS = ["人民路", "解放路", "中山路", "建设大道", "科技园路", "文三路", "南京东路", "天府大道", "金融街", "滨江路"]
ITEMS = [("笔记本电脑", 2.5), ("衣服", 1.2), ("书籍", 3.0), ("零食", 1.8), ("化妆品", 0.6), ("手机", 0.4), ("茶叶", 1.0), ("鞋子", 1.5), ("文件", 0.2), ("小家电", 5.0)]


_KB = None


def _context_for(question: str) -> str | None:
    """与线上 Agent 相同的检索逻辑，把知识库上下文注入 system，保证训练/推理分布一致。"""
    global _KB
    from logicllm.rag.kb import KnowledgeBase
    if _KB is None:
        _KB = KnowledgeBase()
    if not _KB.available:
        return None
    hits = _KB.search(question)
    return KnowledgeBase.format(hits) if hits else None


def sample(mode: str, turns: list[dict], rag_prob: float = 1.0) -> dict:
    """turns 里 assistant 的工具调用用 tc()、工具返回用 tr() 构造。
    线上推理几乎总会命中检索（min_score 0.35），所以所有样本默认都注入上下文——否则模型会学成"有上下文就不调工具"。"""
    q = turns[0]["content"]
    ctx = _context_for(q) if (mode == SYSTEM_PLAIN or rnd.random() < rag_prob) else None
    # 线上推理始终带工具，训练样本也一律带
    return {"messages": [{"role": "system", "content": build_system(ctx)}, *turns], "tools": TOOLS}


def tc(call: dict) -> dict:
    return {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": call["name"], "arguments": call["arguments"]}}]}


def tr(result: dict) -> dict:
    return {"role": "tool", "content": json.dumps(result, ensure_ascii=False)}


# ---------- 1. 知识问答（教师模型生成） ----------
async def gen_kb(n: int, client: ChatClient) -> list[dict]:
    chunks = []
    for p in sorted(config.KB_DIR.glob("*.md")):
        chunks.extend(split_markdown(p))
    out = []
    sem = asyncio.Semaphore(4)

    async def one(chunk):
        prompt = (
            "你是物流培训讲师。根据下面的资料，生成 3 个用户可能提出的问题及其专业、准确、简洁的回答。"
            "问题要多样（客户口吻、员工口吻、术语解释等），回答必须只依据资料，不要编造。"
            "严格输出 JSON 数组：[{\"q\":\"...\",\"a\":\"...\"}]，不要输出其他内容。\n\n"
            f"资料标题：{chunk.title}\n资料：\n{chunk.text}"
        )
        async with sem:
            try:
                txt = (await client.chat([{"role": "user", "content": prompt}], temperature=0.7, max_tokens=1500)).content
            except Exception as e:  # noqa: BLE001
                print("teacher error:", e)
                return []
        s, e = txt.find("["), txt.rfind("]")
        try:
            arr = json.loads(txt[s:e + 1])
        except Exception:  # noqa: BLE001
            return []
        return [sample(SYSTEM_PLAIN, [{"role": "user", "content": x["q"]}, {"role": "assistant", "content": x["a"]}])
                for x in arr if isinstance(x, dict) and x.get("q") and x.get("a")]

    while len(out) < n:
        batch = rnd.sample(chunks, min(len(chunks), max(1, (n - len(out)) // 3 + 1)))
        res = await asyncio.gather(*(one(c) for c in batch))
        for r in res:
            out.extend(r)
        print(f"  kb: {len(out)}/{n}")
        if not any(res):
            break
    return out[:n]


# ---------- 2. 客服 + 工具调用 ----------
NO_SVC_NOTE = "您未指明服务类型，按默认的标准快递计算（如需特快/经济请告诉我）。\n"


def _svc_prefix(svc_in_text: bool, svc: str) -> str:
    """用户说了服务类型就明确复述"按您指定的 X 计算"，没说才用默认说明——两种情形都给模型明确的开头，避免乱套模板。"""
    return f"按您指定的{svc}计算：\n" if svc_in_text else NO_SVC_NOTE
EXC_TIPS = {"破损": "请保留外包装并拍照（外包装+内件），如已保价按声明价值赔付，未保价按实际损失赔付，上限为运费 3 倍。",
            "丢失": "轨迹停更超过 7 天可认定丢失；未保价按运费 3 倍赔偿（最高 300 元）并退还运费，已保价按声明价值赔付。",
            "延误": "特快超承诺时效 1 天以上可退还特快与标准快递的差价。",
            "地址错误": "派送前可联系修改，同城免费；跨城需补差价。"}


def _query_answer(no: str, res: dict) -> str:
    exc = f"，当前标记有异常「{res['exception']}」，我们会尽快处理" if res.get("exception") else ""
    return (f"运单 {no} 的情况如下：\n- 路线：{res['origin']} → {res['destination']}\n- 服务：{res['service']}，重量 {res['weight_kg']}kg，运费 {res['fee']} 元\n"
            f"- 当前状态：**{res['status']}**{exc}\n- 下单时间：{res['created_at'].replace('T', ' ')}\n如需查看详细轨迹，我可以继续为您查询。")


def _freight_answer(o: str, d: str, w: float, res: dict) -> str:
    return (f"{o} → {d}（{res['zone']}），计费重量 {res['billable_weight_kg']}kg"
            + (f"（体积重 {res['volume_weight_kg']}kg 大于实重）" if res["volume_weight_kg"] > w else "")
            + f"：\n- {res['breakdown']}\n- **合计 {res['total_fee']} 元**")


def _exception_answer(no: str, typ: str, res: dict) -> str:
    return f"非常抱歉给您带来不便。已为运单 {no} 登记「{typ}」异常，工单号 **{res['ticket_no']}**，客服将在 48 小时内回访处理。\n{EXC_TIPS[typ]}"


async def gen_tool(n: int) -> list[dict]:
    init_db()
    c = _conn()
    wbs = [dict(r) for r in c.execute("SELECT * FROM waybill").fetchall()]
    c.close()
    out = []
    for _ in range(n):
        kind = rnd.choice(["query", "track", "freight", "eta", "search", "search", "exception", "ask_no", "create",
                           "combo", "followup_svc", "ask_then_no", "complain_then_no",
                           "eta_date", "exc_status", "phone_mask", "no_field"])
        if kind == "query":
            w = rnd.choice(wbs)
            u = rnd.choice([f"帮我查一下运单 {w['no']}", f"{w['no']} 这个单子现在什么情况", f"查询 {w['no']}", f"我的快递 {w['no']} 到哪了"])
            call = {"name": "query_waybill", "arguments": {"waybill_no": w["no"]}}
            res = await call_tool(**{"name": call["name"], "args": call["arguments"]})
            out.append(sample(SYSTEM_TOOLS, [{"role": "user", "content": u}, tc(call),
                                             tr(res), {"role": "assistant", "content": _query_answer(w["no"], res)}]))
        elif kind == "track":
            w = rnd.choice(wbs)
            u = rnd.choice([f"{w['no']} 的物流轨迹", f"看看 {w['no']} 的运输记录", f"运单{w['no']}现在在哪个环节"])
            call = {"name": "track_waybill", "arguments": {"waybill_no": w["no"]}}
            res = await call_tool(call["name"], call["arguments"])
            lines = "\n".join(f"- {e['ts'].replace('T', ' ')} {e['location']}：{e['event']}" for e in res["events"])
            a = f"运单 {w['no']} 的轨迹：\n{lines}\n\n最新状态：{res['current']['event']}（{res['current']['location']}）。"
            out.append(sample(SYSTEM_TOOLS, [{"role": "user", "content": u}, tc(call),
                                             tr(res), {"role": "assistant", "content": a}]))
        elif kind == "freight":
            o, d = rnd.sample(list(CITY_PROVINCE), 2)
            item, w = rnd.choice(ITEMS)
            w = round(w * rnd.uniform(0.6, 2.5), 1)
            dims = rnd.choice([None, (rnd.randint(20, 60), rnd.randint(15, 45), rnd.randint(10, 40))])
            svc = rnd.choice(["标准快递", "特快", "经济"])
            svc_in_text = rnd.random() > 0.3
            if not svc_in_text:
                svc = "标准快递"
            dv = rnd.choice([0, 0, 500, 1000, 3000])
            args = {"origin": o, "destination": d, "weight_kg": w, "service": svc}
            utxt = f"从{o}寄{item}到{d}，{w}kg"
            if dims:
                args.update(length_cm=dims[0], width_cm=dims[1], height_cm=dims[2])
                utxt += f"，箱子 {dims[0]}×{dims[1]}×{dims[2]}cm"
            if dv:
                args["declared_value"] = dv
                utxt += f"，保价{dv}元"
            utxt += (f"，{svc}" if svc_in_text else "") + rnd.choice(["多少钱？", "运费怎么算？", "要花多少运费"])
            call = {"name": "calc_freight", "arguments": args}
            res = await call_tool(call["name"], call["arguments"])
            a = _svc_prefix(svc_in_text, svc) + _freight_answer(o, d, w, res)
            out.append(sample(SYSTEM_TOOLS, [{"role": "user", "content": utxt}, tc(call),
                                             tr(res), {"role": "assistant", "content": a}]))
        elif kind == "eta":
            o, d = rnd.sample(list(CITY_PROVINCE), 2)
            svc = rnd.choice(["标准快递", "特快", "经济"])
            svc_in_text = rnd.random() > 0.35
            if not svc_in_text:
                svc = "标准快递"  # 用户未指明时不得臆测，按默认标准快递并说明
            u = rnd.choice([f"{o}到{d}{svc}几天能到？", f"今天从{o}发{svc}到{d}，什么时候到"]) if svc_in_text else rnd.choice([f"{o}寄{d}要多久", f"{o}到{d}几天能到"])
            call = {"name": "estimate_eta", "arguments": {"origin": o, "destination": d, "service": svc}}
            res = await call_tool(call["name"], call["arguments"])
            a = ("" if svc_in_text else "按默认的标准快递估算（特快可再快 1 天）：") + f"{o} → {d} 约 {res['distance_km']}km，{svc}预计 **{res['est_days']} 天**送达（约 {res['est_delivery_date']}）。" + (f"{res['note']}。" if res["note"] else "") + "实际以揽收后轨迹为准，恶劣天气可能延误。"
            out.append(sample(SYSTEM_TOOLS, [{"role": "user", "content": u}, tc(call),
                                             tr(res), {"role": "assistant", "content": a}]))
        elif kind == "search":
            w = rnd.choice(wbs)
            ph = rnd.choice([w["sender_phone"], w["receiver_phone"]])
            u = rnd.choice([f"我手机号 {ph}，帮我看看有哪些快递", f"用 {ph} 查一下我的运单"])
            call = {"name": "search_waybills", "arguments": {"phone": ph}}
            res = await call_tool(call["name"], call["arguments"])
            lines = "\n".join(f"- {x['no']}：{x['origin']} → {x['destination']}，{x['status']}" for x in res["waybills"])
            a = f"手机号 {ph} 名下共 {res['count']} 个运单：\n{lines}\n需要查看某一单的详细轨迹请告诉我运单号。"
            out.append(sample(SYSTEM_TOOLS, [{"role": "user", "content": u}, tc(call),
                                             tr(res), {"role": "assistant", "content": a}]))
        elif kind == "exception":
            w = rnd.choice(wbs)
            typ = rnd.choice(["破损", "丢失", "延误", "地址错误"])
            first = {"破损": f"运单 {w['no']} 收到时外箱破损了，要投诉", "丢失": f"运单 {w['no']} 好多天没更新，应该是丢了，要投诉",
                     "延误": f"运单 {w['no']} 超过承诺时效还没到，要投诉", "地址错误": f"运单 {w['no']} 地址填错了，要改"}[typ]
            u = rnd.choice([first, f"{w['no']} {typ}，怎么处理", f"我要给 {w['no']} 登记{typ}"])
            call = {"name": "report_exception", "arguments": {"waybill_no": w["no"], "type": typ, "description": f"用户反馈{typ}"}}
            res = await call_tool(call["name"], call["arguments"])
            out.append(sample(SYSTEM_TOOLS, [{"role": "user", "content": u}, tc(call),
                                             tr(res), {"role": "assistant", "content": _exception_answer(w["no"], typ, res)}]))
        elif kind == "ask_no":
            u = rnd.choice(["我的快递到哪了？", "帮我查下快递", "我的包裹怎么还没到", "查一下物流"])
            a = "好的，请提供您的运单号（LL 开头的 12 位编号），或者告诉我寄/收件人手机号，我来帮您查询。"
            out.append(sample(SYSTEM_TOOLS, [{"role": "user", "content": u}, {"role": "assistant", "content": a}]))
        elif kind == "combo":  # 一句话同时问运费和时效：两个工具都要调，先算钱再算时效
            o, d = rnd.sample(list(CITY_PROVINCE), 2)
            item, w = rnd.choice(ITEMS)
            w = round(w * rnd.uniform(0.6, 3.0), 1)
            svc = rnd.choice(["标准快递", "特快", "经济"])
            svc_in_text = rnd.random() > 0.4
            if not svc_in_text:
                svc = "标准快递"
            svc_txt = f"{svc}" if svc_in_text else ""
            u = rnd.choice([f"{o}寄{w}公斤{item}到{d}，{svc_txt}多少钱几天能到", f"从{o}发{item}到{d}，{w}kg，{svc_txt}运费和时效分别是多少",
                            f"{o}到{d} {w}kg {item}，{svc_txt}要多少钱、什么时候能到"])
            c1 = {"name": "calc_freight", "arguments": {"origin": o, "destination": d, "weight_kg": w, "service": svc}}
            r1 = await call_tool(c1["name"], c1["arguments"])
            c2 = {"name": "estimate_eta", "arguments": {"origin": o, "destination": d, "service": svc}}
            r2 = await call_tool(c2["name"], c2["arguments"])
            a = (_svc_prefix(svc_in_text, svc) + f"{o} → {d}（{r1['zone']}），计费重量 {r1['billable_weight_kg']}kg：\n"
                 f"- 运费：{r1['breakdown']}，**合计 {r1['total_fee']} 元**\n"
                 f"- 时效：约 {r2['distance_km']}km，{svc}预计 **{r2['est_days']} 天**送达（约 {r2['est_delivery_date']}）" + (f"，{r2['note']}" if r2["note"] else "")
                 + "\n实际以揽收后轨迹为准。")
            out.append(sample(SYSTEM_TOOLS, [{"role": "user", "content": u}, tc(c1), tr(r1), tc(c2), tr(r2), {"role": "assistant", "content": a}]))
        elif kind == "followup_svc":  # 多轮：先按默认标准快递报价，追问换服务类型要重新调工具
            o, d = rnd.sample(list(CITY_PROVINCE), 2)
            item, w = rnd.choice(ITEMS)
            w = round(w * rnd.uniform(0.6, 2.5), 1)
            u1 = rnd.choice([f"从{o}寄{w}公斤{item}到{d}多少钱？", f"{o}到{d}，{item} {w}kg，运费多少"])
            r1 = await call_tool("calc_freight", {"origin": o, "destination": d, "weight_kg": w, "service": "标准快递"})
            a1 = NO_SVC_NOTE + _freight_answer(o, d, w, r1)
            new_svc = rnd.choice(["特快", "经济"])
            u2 = rnd.choice([f"改成{new_svc}呢？", f"{new_svc}多少钱", f"换{new_svc}的话", f"那{new_svc}要多少"])
            c2 = {"name": "calc_freight", "arguments": {"origin": o, "destination": d, "weight_kg": w, "service": new_svc}}
            r2 = await call_tool(c2["name"], c2["arguments"])
            diff = round(r2["total_fee"] - r1["total_fee"], 2)
            a2 = (f"换成{new_svc}的话，{o} → {d}，计费重量 {r2['billable_weight_kg']}kg：\n- {r2['breakdown']}\n- **合计 {r2['total_fee']} 元**"
                  + (f"，比标准快递贵 {diff} 元" if diff > 0 else f"，比标准快递便宜 {-diff} 元") + "。")
            # 历史轮次只保留最终答复文本（与线上会话历史一致，不含工具轨迹）
            out.append(sample(SYSTEM_TOOLS, [{"role": "user", "content": u1}, {"role": "assistant", "content": a1},
                                             {"role": "user", "content": u2}, tc(c2), tr(r2), {"role": "assistant", "content": a2}]))
        elif kind == "ask_then_no":  # 多轮：先没给单号，追问后给了单号再查
            w = rnd.choice(wbs)
            u1 = rnd.choice(["帮我查个运单", "查一下我的快递", "我想查物流", "帮我看看包裹到哪了"])
            a1 = "好的，请提供您的运单号（LL 开头的 12 位编号），或者告诉我寄/收件人手机号，我来帮您查询。"
            u2 = rnd.choice([w["no"], f"运单号是 {w['no']}", f"{w['no']}，麻烦了", f"单号 {w['no']}"])
            call = {"name": "query_waybill", "arguments": {"waybill_no": w["no"]}}
            res = await call_tool(call["name"], call["arguments"])
            out.append(sample(SYSTEM_TOOLS, [{"role": "user", "content": u1}, {"role": "assistant", "content": a1},
                                             {"role": "user", "content": u2}, tc(call), tr(res), {"role": "assistant", "content": _query_answer(w["no"], res)}]))
        elif kind == "complain_then_no":  # 多轮：先描述问题，再给单号登记异常
            w = rnd.choice(wbs)
            typ = rnd.choice(["破损", "丢失", "延误"])
            u1 = {"破损": rnd.choice(["我的快递收到的时候摔坏了，怎么办", "包裹外箱破了，里面东西也坏了"]),
                  "丢失": rnd.choice(["我的快递好多天没动静了，是不是丢了", "包裹一直不更新物流，怎么办"]),
                  "延误": rnd.choice(["我寄的特快已经超时了还没到", "快递比承诺时间晚了两天了"])}[typ]
            a1 = f"{EXC_TIPS[typ]}\n请提供运单号，我帮您登记「{typ}」异常工单，客服会在 48 小时内回访。"
            u2 = rnd.choice([f"运单号是 {w['no']}，帮我登记投诉", f"{w['no']}，麻烦登记一下", f"单号 {w['no']}，要投诉"])
            call = {"name": "report_exception", "arguments": {"waybill_no": w["no"], "type": typ, "description": f"用户反馈{typ}"}}
            res = await call_tool(call["name"], call["arguments"])
            out.append(sample(SYSTEM_TOOLS, [{"role": "user", "content": u1}, {"role": "assistant", "content": a1},
                                             {"role": "user", "content": u2}, tc(call), tr(res), {"role": "assistant", "content": _exception_answer(w["no"], typ, res)}]))
        elif kind == "eta_date":  # 用户给了寄出日期：必须把 ship_date 传给工具，送达日按该日期推算，而不是按"今天"
            o, d = rnd.sample(list(CITY_PROVINCE), 2)
            svc = rnd.choice(["标准快递", "特快", "经济"])
            svc_in_text = rnd.random() > 0.4
            if not svc_in_text:
                svc = "标准快递"
            month = rnd.randint(9, 12)
            day = rnd.randint(1, 28)
            ship = f"2026-{month:02d}-{day:02d}"
            date_txt = rnd.choice([f"{month} 月 {day} 日", f"{month}月{day}号", ship, f"2026 年 {month} 月 {day} 日"])
            svc_txt = svc if svc_in_text else ""
            u = rnd.choice([f"{date_txt}从{o}寄{svc_txt}到{d}，哪天能到", f"我{date_txt}发{svc_txt}件，{o}到{d}，预计几号送达",
                            f"{o}寄{d}，{date_txt}寄出，{svc_txt}什么时候到"])
            call = {"name": "estimate_eta", "arguments": {"origin": o, "destination": d, "service": svc, "ship_date": ship}}
            res = await call_tool(call["name"], call["arguments"])
            a = (("" if svc_in_text else "按默认的标准快递估算（特快可再快 1 天）：")
                 + f"{date_txt}从{o}寄出，{o} → {d} 约 {res['distance_km']}km，{svc}预计 **{res['est_days']} 天**送达，即 **{res['est_delivery_date']}** 前后。"
                 + (f"{res['note']}。" if res["note"] else "") + "实际以揽收后轨迹为准，恶劣天气可能延误。")
            out.append(sample(SYSTEM_TOOLS, [{"role": "user", "content": u}, tc(call), tr(res), {"role": "assistant", "content": a}]))
        elif kind == "exc_status":  # 只是问"有没有异常"：查运单即可，绝不能登记新异常（report_exception 只在用户明确要投诉/登记时用）
            w = rnd.choice(wbs)
            u = rnd.choice([f"{w['no']} 这单有没有异常", f"运单 {w['no']} 是不是出问题了", f"{w['no']} 有异常记录吗", f"帮我看看 {w['no']} 有没有什么问题"])
            call = {"name": "query_waybill", "arguments": {"waybill_no": w["no"]}}
            res = await call_tool(call["name"], call["arguments"])
            if res.get("exception"):
                a = (f"运单 {w['no']} 当前标记有异常「{res['exception']}」，状态 {res['status']}（{res['origin']} → {res['destination']}，{res['service']}）。"
                     f"网点正在处理；如需登记投诉工单或了解处理进度，请告诉我。")
            else:
                a = (f"运单 {w['no']} 目前没有异常记录，状态正常：**{res['status']}**（{res['origin']} → {res['destination']}，{res['service']}）。"
                     f"如果您实际遇到了破损、丢失或延误等问题，告诉我可以帮您登记工单。")
            out.append(sample(SYSTEM_TOOLS, [{"role": "user", "content": u}, tc(call), tr(res), {"role": "assistant", "content": a}]))
        elif kind == "phone_mask":  # 问电话：查到后必须脱敏（隐藏中间四位、姓名部分字），并说明依据
            w = rnd.choice(wbs)
            who = rnd.choice(["收件人", "寄件人"])
            u = rnd.choice([f"{w['no']} {who}的电话是多少", f"查一下运单 {w['no']} 的{who}手机号", f"{w['no']} {who}联系方式", f"运单 {w['no']} 的{who}怎么联系"])
            call = {"name": "query_waybill", "arguments": {"waybill_no": w["no"]}}
            res = await call_tool(call["name"], call["arguments"])
            ph = res["receiver_phone"] if who == "收件人" else res["sender_phone"]
            nm = res["receiver"] if who == "收件人" else res["sender"]
            masked, nm_masked = ph[:3] + "****" + ph[7:], nm[0] + "*" * (len(nm) - 1)
            a = (f"运单 {w['no']} 的{who}是 {nm_masked}，联系电话 {masked}。"
                 f"按《个人信息保护法》和公司隐私规定，手机号中间四位需脱敏展示，完整号码请在有权限的内部系统中查看。"
                 f"该运单当前状态：{res['status']}。")
            out.append(sample(SYSTEM_TOOLS, [{"role": "user", "content": u}, tc(call), tr(res), {"role": "assistant", "content": a}]))
        elif kind == "no_field":  # 问系统里根本没有的字段：如实说明没有，不编造原因
            w = rnd.choice(wbs)
            ask, fld = rnd.choice([("司机是谁", "司机信息"), ("司机电话", "司机电话"), ("车牌号", "车牌号"), ("派送员的电话", "派送员电话"),
                                   ("快递员叫什么", "快递员姓名"), ("航班号", "航班/车次信息"), ("用的哪家航空公司", "承运航空公司")])
            u = rnd.choice([f"{w['no']} 这单的{ask}", f"运单 {w['no']} 的{ask}是多少", f"帮我查 {w['no']} 的{ask}"])
            call = {"name": "query_waybill", "arguments": {"waybill_no": w["no"]}}
            res = await call_tool(call["name"], call["arguments"])
            a = (f"抱歉，运单系统里不记录{fld}，我无法提供。能查到的是运单 {w['no']} 的当前状态：**{res['status']}**（{res['origin']} → {res['destination']}，{res['service']}）。"
                 f"如需联系派送环节，建议联系{res['destination']}的派送网点，或者我帮您登记一条咨询工单。")
            out.append(sample(SYSTEM_TOOLS, [{"role": "user", "content": u}, tc(call), tr(res), {"role": "assistant", "content": a}]))
        else:  # create
            o, d = rnd.sample(list(CITY_PROVINCE), 2)
            s, r = rnd.sample(NAMES, 2)
            sp, rp = f"13{rnd.randint(100000000, 999999999)}", f"15{rnd.randint(100000000, 999999999)}"
            item, w = rnd.choice(ITEMS)
            addr = f"{d}市{rnd.choice(['高新区', '中心区', '新华区'])}{rnd.choice(STREETS)}{rnd.randint(1, 300)}号"
            u = f"帮我下单：寄件人{s} {sp}，{o}；收件人{r} {rp}，{addr}；{item} {w}kg，标准快递"
            args = {"sender": s, "sender_phone": sp, "origin": o, "receiver": r, "receiver_phone": rp,
                    "destination": d, "address": addr, "weight_kg": w, "service": "标准快递"}
            call = {"name": "create_waybill", "arguments": args}
            # 不真的写库，构造一个合理结果
            fee = await call_tool("calc_freight", {"origin": o, "destination": d, "weight_kg": w})
            res = {"waybill_no": f"LL2026{rnd.randint(100000, 999999)}{rnd.randint(10, 99)}", "fee": fee["total_fee"], "status": "已揽收", "created_at": "2026-08-22T10:00"}
            a = f"下单成功！运单号 **{res['waybill_no']}**，{o} → {d}，运费 {res['fee']} 元，当前状态：已揽收。快递员会在揽收时段上门取件，请保持电话畅通。"
            out.append(sample(SYSTEM_TOOLS, [{"role": "user", "content": u}, tc(call),
                                             tr(res), {"role": "assistant", "content": a}]))
    return out


# ---------- 3. 抽取 ----------
def gen_extract(n: int) -> list[dict]:
    from logicllm.tools.address import parse_address
    out = []
    for _ in range(n):
        city = rnd.choice(list(CITY_DISTRICTS))
        prov = CITY_PROVINCE[city]
        name = rnd.choice(NAMES)
        phone = f"1{rnd.choice('3589')}{rnd.randint(100000000, 999999999)}"
        # 电话写法变体：+86 / 空格 / 连字符分组，标签一律归一为 11 位连续数字
        phone_txt = rnd.choice([phone, phone, phone, f"+86 {phone[:3]} {phone[3:7]} {phone[7:]}", f"{phone[:3]}-{phone[3:7]}-{phone[7:]}", f"{phone[:3]} {phone[3:7]} {phone[7:]}"])
        dist = rnd.choice(CITY_DISTRICTS[city])
        detail = f"{rnd.choice(STREETS)}{rnd.randint(1, 500)}号{rnd.choice(['', '3栋2单元501', 'A座1201', '科技园T3栋'])}"
        provtxt = "" if prov == city else rnd.choice([province_full(prov), prov, ""])
        citytxt = f"{city}市" if rnd.random() < 0.7 else city
        item, w = rnd.choice(ITEMS)
        layout = rnd.choice([
            f"收件人：{name} {phone_txt} {provtxt}{citytxt}{dist}{detail}",
            f"{provtxt}{citytxt}{dist}{detail}，{name}，{phone_txt}",
            f"{name} {phone_txt}\n{provtxt}{citytxt}{dist}{detail}\n物品：{item} {w}kg",
            f"请发到{provtxt}{citytxt}{dist}{detail}，收件人{name}，电话{phone_txt}，寄的是{item}",
        ])
        u = rnd.choice(["提取收件信息：", "把下面的地址解析成结构化字段：", "帮我抽取这段文本中的收件人信息，输出JSON：", ""]) + layout
        label = {"name": name, "phone": phone, "province": prov, "city": city, "district": dist, "detail": detail,
                 "item": item if item in layout else "", "weight_kg": w if f"{w}kg" in layout else ""}
        answer = "```json\n" + json.dumps(label, ensure_ascii=False, indent=2) + "\n```"
        if rnd.random() < 0.7:
            out.append(sample(SYSTEM_TOOLS, [{"role": "user", "content": u}, {"role": "assistant", "content": answer}]))
        else:  # 先调规则解析器再输出 JSON
            call = {"name": "parse_address", "arguments": {"text": layout}}
            res = parse_address(layout)
            out.append(sample(SYSTEM_TOOLS, [{"role": "user", "content": u}, tc(call), tr(res), {"role": "assistant", "content": answer}]))
    return out


# ---------- 4. 路径/调度 ----------
async def gen_route(n: int) -> list[dict]:
    out = []
    cities = list(CITY_PROVINCE)
    for _ in range(n):
        if rnd.random() < 0.6:
            o, d = rnd.sample(cities, 2)
            via = rnd.sample([c for c in cities if c not in (o, d)], 1) if rnd.random() < 0.3 else None
            u = f"{o}到{d}的干线怎么走" + (f"，要经过{via[0]}" if via else "") + rnd.choice(["？", "，大概多少公里", "，需要开多久"])
            args = {"origin": o, "destination": d}
            if via:
                args["via"] = via
            call = {"name": "plan_route", "arguments": args}
            res = await call_tool(call["name"], call["arguments"])
            if "error" in res:
                continue
            a = f"推荐路线：{' → '.join(res['path'])}\n- 总里程约 {res['distance_km']} km，{res['legs']} 段干线\n- 按 70km/h 均速估算行驶约 {res['est_drive_hours']} 小时，单司机需 {max(1, round(res['est_drive_hours'] / 8 + 0.49))} 天（每日驾驶 ≤ 8 小时）。"
        else:
            depot = rnd.choice(cities)
            stops = rnd.sample([c for c in cities if c != depot], rnd.randint(3, 6))
            no_return = rnd.random() < 0.3
            u = (f"从{depot}出发，要给{'、'.join(stops)}送货，怎么排顺序最省路？" if not no_return
                 else rnd.choice([f"从{depot}出发送{'、'.join(stops)}，送完不回{depot}，怎么排", f"{depot}发车给{'、'.join(stops)}送货，单程不返回，顺序怎么排最省"]))
            call = {"name": "optimize_delivery_order", "arguments": {"depot": depot, "stops": stops, **({"return_to_depot": False} if no_return else {})}}
            res = await call_tool(call["name"], call["arguments"])
            if "error" in res:
                continue
            a = (f"建议配送顺序：{' → '.join(res['sequence'])}" + ("（单程，不返回）" if no_return else "")
                 + f"\n- 总里程约 {res['total_km']} km，行驶约 {res['est_drive_hours']} 小时\n- 采用最近邻 + 2-opt 优化；若各点有收货时间窗，请告诉我以便进一步调整。")
        out.append(sample(SYSTEM_TOOLS, [{"role": "user", "content": u}, tc(call),
                                         tr(res), {"role": "assistant", "content": a}]))
    return out


# ---------- 5. 失败与边界轨迹 ----------
async def gen_failures(n: int) -> list[dict]:
    init_db()
    out = []
    unknown_cities = ["洛阳", "三亚", "桂林", "唐山", "襄阳", "赣州", "遵义", "绵阳"]
    foreign_cities = ["纽约", "伦敦", "东京", "洛杉矶", "新加坡", "巴黎", "首尔", "悉尼"]
    for _ in range(n):
        kind = rnd.choice(["bad_no", "bad_no_track", "bad_city_eta", "bad_city_freight", "bad_city_route", "no_result_phone", "intl_freight"])
        if kind in ("bad_no", "bad_no_track"):
            no = f"LL{rnd.randint(2027000000, 2029999999)}{rnd.randint(10, 99)}"
            tool_name = "query_waybill" if kind == "bad_no" else "track_waybill"
            u = rnd.choice([f"帮我查下 {no}", f"{no} 到哪了", f"查一下运单 {no} 的物流"])
            call = {"name": tool_name, "arguments": {"waybill_no": no}}
            res = await call_tool(call["name"], call["arguments"])
            a = f"抱歉，系统里没有找到运单 {no}。请核对运单号是否正确（本系统运单号为 LL 开头加 10 位数字），也可以告诉我寄/收件人手机号，我用手机号帮您查。"
        elif kind == "bad_city_eta":
            c1 = rnd.choice(unknown_cities); c2 = rnd.choice(list(CITY_PROVINCE))
            u = f"{c1}到{c2}几天能到？"
            call = {"name": "estimate_eta", "arguments": {"origin": c1, "destination": c2, "service": "标准快递"}}
            res = await call_tool(call["name"], call["arguments"])
            near = rnd.sample(list(CITY_PROVINCE), 2)
            a = f"抱歉，{c1} 暂不在系统收录的城市范围内，无法直接估算时效。您可以确认一下城市名，或者选择邻近的已开通城市（如 {near[0]}、{near[1]}）再帮您估算。"
        elif kind == "bad_city_freight":
            c1 = rnd.choice(list(CITY_PROVINCE)); c2 = rnd.choice(unknown_cities)
            item, w = rnd.choice(ITEMS)
            u = f"从{c1}寄{item}到{c2}，{w}kg 多少钱？"
            call = {"name": "calc_freight", "arguments": {"origin": c1, "destination": c2, "weight_kg": w}}
            res = await call_tool(call["name"], call["arguments"])
            a = f"抱歉，{c2} 暂不在已开通的计费城市范围内，算不了这条线路的运费。请确认城市名是否有误，或告诉我具体想寄到哪个已开通城市。"
        elif kind == "intl_freight":  # 境外目的地：工具返回未收录，不得编造总价
            c1 = rnd.choice(list(CITY_PROVINCE)); c2 = rnd.choice(foreign_cities)
            item, w = rnd.choice(ITEMS)
            u = rnd.choice([f"从{c1}寄{item}到{c2}多少钱？", f"{c1}发{w}kg{item}去{c2}，运费怎么算", f"寄东西到{c2}要多少运费"])
            args = {"origin": c1, "destination": c2, "weight_kg": w} if c1 in u else {"origin": "上海", "destination": c2, "weight_kg": w}
            call = {"name": "calc_freight", "arguments": args}
            res = await call_tool(call["name"], call["arguments"])
            a = (f"抱歉，{c2} 是境外城市，本系统的运费计算只覆盖国内已开通城市，无法直接报价。"
                 "国际件请走国际业务：国际快递（DHL/FedEx/UPS）约 3–7 天，适合文件和样品；空运 5–10 天适合批量货；海运最便宜但需 20–45 天。"
                 "告诉我货物类型、重量和时效要求，我可以帮您选合适的方式。")
        elif kind == "bad_city_route":
            c1 = rnd.choice(unknown_cities); c2 = rnd.choice(list(CITY_PROVINCE))
            u = f"{c1}到{c2}的干线怎么走？"
            call = {"name": "plan_route", "arguments": {"origin": c1, "destination": c2}}
            res = await call_tool(call["name"], call["arguments"])
            a = f"抱歉，路网数据里没有收录 {c1}，无法规划这条干线。如果您指的是其他城市请更正一下；也可以从收录的邻近枢纽城市出发规划。"
        else:  # no_result_phone
            ph = f"13{rnd.randint(100000000, 999999999)}"
            u = rnd.choice([f"手机号 {ph} 名下有哪些运单？", f"用 {ph} 帮我查下快递"])
            call = {"name": "search_waybills", "arguments": {"phone": ph}}
            res = await call_tool(call["name"], call["arguments"])
            a = f"手机号 {ph} 名下没有查到运单。请确认号码是否为下单时预留的手机号，或直接提供运单号（LL 开头 12 位）我来查询。"
        out.append(sample(SYSTEM_TOOLS, [{"role": "user", "content": u}, tc(call), tr(res), {"role": "assistant", "content": a}]))
    return out


# ---------- 6. 数值套档（无城市，纯规则问答） ----------
# 答案全部由与 estimate_eta / calc_freight 相同的规则算出；分界点附近加密采样，专治"1600km 答 3 天"这类套错档
ETA_TIERS = [(500, 2, "≤500km"), (1500, 3, "500–1500km"), (2500, 4, "1500–2500km"), (float("inf"), 5, ">2500km")]
ETA_ADJ = {"标准快递": 0, "特快": -1, "经济": 2, "零担": 3}


def _n(x: float) -> str:
    return f"{x:g}" if float(x).is_integer() else f"{round(x, 2):g}"


def _eta_days(d: int, svc: str, remote: bool) -> tuple[int, int, str]:
    base, label = next((days, lab) for ub, days, lab in ETA_TIERS if d <= ub)
    return max(1, base + (2 if remote else 0) + ETA_ADJ[svc]), base, label


def _tier_chain(d: int) -> str:
    """把里程与 500/1500/2500 逐个比较写出来，让模型先比大小再套档，而不是凭感觉猜档。"""
    return "，".join(f"{d} {'>' if d > b else '≤'} {b}" for b in (500, 1500, 2500))


def _near_boundary(bounds: list[int], lo: int, hi: int) -> int:
    if rnd.random() < 0.7:
        b = rnd.choice(bounds)
        return max(lo, b + rnd.choice([0, 0, -1, 1]) * rnd.choice([10, 20, 50, 80, 100, 150]))
    return rnd.randint(lo, hi) // 10 * 10


def gen_numeric(n: int) -> list[dict]:
    from logicllm.tools.pricing import ADD_KG, FIRST_KG, SERVICE_MULT, VOLUME_DIVISOR
    import math
    out = []
    for _ in range(n):
        kind = rnd.choice(["eta", "eta", "eta", "eta", "eta_cmp", "zone", "zone", "freight", "freight", "volume", "insurance", "oversize", "alcohol", "alcohol", "ltl", "ltl"])
        if kind == "eta":
            d = _near_boundary([500, 1500, 2500], 60, 4000)
            svc = rnd.choice(["", "", "标准快递", "特快", "经济", "零担"])
            remote = rnd.random() < 0.2
            days, base, label = _eta_days(d, svc or "标准快递", remote)
            where = rnd.choice(["新疆", "西藏", "青海", "内蒙古", "甘肃", "宁夏"]) if remote else ""
            q = rnd.choice([f"{d} 公里{svc}大概几天到？", f"{d}km 的线路{svc}要几天？", f"两地相距 {d} 公里，{svc or '快递'}几天能到",
                            f"寄到 {d} 公里外{svc}多久能收到？"])
            if remote:
                q = q.rstrip("？") + f"，收件地在{where}？"
            steps = [f"里程 {d}km 逐档比较：" + _tier_chain(d) + f" → {label} 档，标准快递 {base} 天"]
            if remote:
                steps.append(f"{where}属偏远六省区，+2 天")
            if svc and svc != "标准快递":
                steps.append({"特快": "特快 −1 天（最少 1 天）", "经济": "经济 +2 天", "零担": "零担 +3 天"}[svc])
            a = ("；".join(steps) + f" → **{svc or '标准快递'}约 {days} 天**。"
                 + ("" if svc else "（未指明服务类型，按标准快递计算；特快可快 1 天。）")
                 + ("边界值归低一档：500/1500/2500km 整分别是 2/3/4 天。" if any(abs(d - b) <= 150 for b in (500, 1500, 2500)) else ""))
        elif kind == "eta_cmp":
            b = rnd.choice([500, 1500, 2500])
            d1, d2 = b - rnd.choice([20, 50, 100, 150]), b + rnd.choice([20, 50, 100, 150])
            if rnd.random() < 0.5:
                d1, d2 = d2, d1
            (t1, _, l1), (t2, _, l2) = _eta_days(d1, "标准快递", False), _eta_days(d2, "标准快递", False)
            q = rnd.choice([f"{d1} 公里和 {d2} 公里时效一样吗？", f"{d1}km 跟 {d2}km 标准快递差几天？"])
            a = (f"分界线是 {b}km：{d1} {'>' if d1 > b else '≤'} {b} → {l1} 档 {t1} 天；{d2} {'>' if d2 > b else '≤'} {b} → {l2} 档 {t2} 天。"
                 f"**不一样，相差 {abs(t1 - t2)} 天**（{min(d1, d2)}km {min(t1, t2)} 天，{max(d1, d2)}km {max(t1, t2)} 天）。")
        elif kind == "zone":
            d = _near_boundary([800], 100, 3000)
            zone = "邻近" if d <= 800 else "跨区"
            q = rnd.choice([f"跨省寄件，两地 {d} 公里，按什么区域计费？", f"{d}km 跨省算邻近还是跨区？"])
            a = (f"跨省件看 800km 分界线：{d} {'>' if d > 800 else '≤'} 800 → **按{zone}计费**"
                 f"（{'首重 12 元、续重 6 元/kg' if zone == '跨区' else '首重 10 元、续重 4 元/kg'}）。"
                 f"规则：跨省且里程 ≤800km 为邻近，>800km 为跨区，800km 整算邻近；同省不论距离都按同省（首重 8 元、续重 2 元/kg）。")
        elif kind in ("freight", "volume"):
            zone = rnd.choice(list(FIRST_KG))
            svc = rnd.choice(["标准快递", "标准快递", "特快", "经济"])
            w = round(rnd.uniform(0.3, 12), 1)
            dims = (rnd.randint(20, 70), rnd.randint(15, 50), rnd.randint(10, 45)) if (kind == "volume" or rnd.random() < 0.4) else None
            vol = dims[0] * dims[1] * dims[2] / VOLUME_DIVISOR if dims else 0
            bill = math.ceil(max(w, vol) * 2) / 2
            base = (FIRST_KG[zone] + max(0, bill - 1) * ADD_KG[zone]) * SERVICE_MULT[svc]
            box = f"，箱子 {dims[0]}×{dims[1]}×{dims[2]}cm" if dims else ""
            if kind == "volume":
                q = rnd.choice([f"{w}kg 的包裹{box}，计费重量是多少？", f"实重 {w}kg{box}，按多少公斤收费？"])
                a = (f"计费重量 **{_n(bill)}kg**。体积重 = {dims[0]}×{dims[1]}×{dims[2]}÷6000 ≈ {round(vol, 2)}kg，"
                     f"取实重与体积重较大者 {round(max(w, vol), 2)}kg，再向上取整到 0.5kg。")
            else:
                q = rnd.choice([f"{zone}件 {w}kg{box}，{svc}运费多少？", f"按{zone}计费，{w}公斤{box}，{svc}要多少钱？"])
                steps = []
                if dims:
                    steps.append(f"体积重 {dims[0]}×{dims[1]}×{dims[2]}÷6000 ≈ {round(vol, 2)}kg")
                steps.append(f"计费重量 {_n(bill)}kg（取实重与体积重较大者，向上取整到 0.5kg）")
                steps.append(f"{zone}：首重 {_n(FIRST_KG[zone])} 元 + 续重 {_n(max(0, bill - 1))}kg×{_n(ADD_KG[zone])} 元"
                             + (f"，×{SERVICE_MULT[svc]}（{svc}）" if svc != "标准快递" else ""))
                a = "；".join(steps) + f"，**合计 {_n(round(base, 2))} 元**（不含偏远、保价等附加费）。"
        elif kind == "insurance":
            v = rnd.choice([50, 100, 150, 199, 200, 201, 300, 500, 800, 1000, 2000, 3000, 5000, 8000, 10000])
            fee = max(1.0, v * 0.005)
            q = rnd.choice([f"保价 {v} 元要交多少保价费？", f"声明价值 {v} 元，保价费怎么算？"])
            a = (f"规则是声明价值 × 0.5%，最低 1 元：{v}×0.5% = {_n(round(v * 0.005, 2))} 元"
                 + ("，不足 1 元按 1 元收" if v * 0.005 < 1 else "") + f" → **保价费 {_n(round(fee, 2))} 元**。")
        elif kind == "alcohol":  # 酒精度三档：≤24 不限 / 24–70 限寄 / >70 禁寄；酒精制品（非酒类饮品）≥24% 直接按易燃易爆禁寄
            deg = rnd.choice([10, 15, 20, 23, 24, 25, 30, 38, 42, 45, 50, 52, 53, 56, 60, 65, 68, 70, 71, 72, 75, 80, 90, 95, 99])
            is_drink = rnd.random() < 0.7
            thing = rnd.choice(["酒", "白酒", "米酒", "黄酒", "洋酒", "威士忌"]) if is_drink else rnd.choice(["酒精", "医用酒精", "酒精消毒液", "酒精喷雾"])
            q = rnd.choice([f"{deg} 度的{thing}能寄吗？", f"{deg}% 的{thing}可以走快递吗？", f"酒精浓度 {deg}% 的{thing}寄得了吗", f"{thing}，{deg} 度，能不能寄"])
            cmp24, cmp70 = (">" if deg > 24 else "≤"), (">" if deg > 70 else "≤")
            chain = f"{deg} {cmp24} 24，{deg} {cmp70} 70"
            if not is_drink and deg >= 24:
                a = (f"{thing}是酒精制品而不是酒类饮品，规则是酒精浓度 ≥24% 即按易燃易爆品禁寄：{deg} ≥ 24 → **不能寄**"
                     + ("；而且任何超过 70% 的酒精液体都禁寄" if deg > 70 else "") + "。")
            elif deg <= 24:
                a = f"酒精度分三档（≤24 不限；24–70 限寄；>70 禁寄）。{chain} → ≤24% 档 → **能寄，不限量**，密封防漏、正常防碎包装即可。"
            elif deg <= 70:
                a = (f"酒精度分三档（≤24 不限；24–70 限寄；>70 禁寄）。{chain} → 24%–70% 档 → **能寄，有条件**：每件不超过 5L、只能陆运、需防碎包装。"
                     + ("70 度整仍在区间内，超过 70 才禁寄。" if deg == 70 else ""))
            else:
                a = f"酒精度分三档（≤24 不限；24–70 限寄；>70 禁寄）。{chain} → >70% 档 → **不能寄**，属高浓度易燃液体，禁寄。"
        elif kind == "ltl":  # 零担门槛 ≥30kg
            w = rnd.choice([5, 10, 15, 20, 25, 28, 29, 29.5, 30, 31, 35, 40, 50, 80, 120, 200])
            q = rnd.choice([f"{_n(w)} 公斤的货能走零担吗？", f"{_n(w)}kg 可以发零担吗", f"{_n(w)} 公斤走零担还是快递？"])
            if w >= 30:
                a = f"零担（不足整车、多票拼车）在本公司的起运门槛是单票 ≥30kg：{_n(w)} ≥ 30 → **能走零担**，按标准快递运费 ×0.5 计收，时效比标准快递慢 3 天，适合不急的大件。"
            else:
                a = f"零担（不足整车、多票拼车）在本公司的起运门槛是单票 ≥30kg：{_n(w)} < 30 → **不能走零担**，请走快递：标准快递，或特快（×1.6，快 1 天）、经济（×0.8，慢 2 天）。"
        else:  # oversize
            if rnd.random() < 0.5:
                w = rnd.choice([45, 48, 49.5, 50, 50.5, 51, 52, 55, 60])
                hit = w > 50
                q = rnd.choice([f"单件 {_n(w)} 公斤要加超重操作费吗？", f"{_n(w)}kg 的货算超重吗？"])
                a = (f"超重线是单件 > 50kg（另有单边 > 150cm 的超长线）：{_n(w)} {'>' if hit else '≤'} 50 → "
                     + ("**要加收 30 元操作费**。" if hit else "**不收超重费**（恰好 50kg 不算超）。"))
            else:
                L = rnd.choice([120, 140, 149, 150, 151, 155, 160, 180])
                hit = L > 150
                q = rnd.choice([f"箱子最长边 {L}cm 要加钱吗？", f"单边 {L} 厘米算超长吗？"])
                a = (f"超长线是单边 > 150cm（另有单件 > 50kg 的超重线）：{L} {'>' if hit else '≤'} 150 → "
                     + ("**要加收 30 元操作费**。" if hit else "**不收超长费**（恰好 150cm 不算超）。"))
        out.append(sample(SYSTEM_PLAIN, [{"role": "user", "content": q}, {"role": "assistant", "content": a}]))
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-kb", type=int, default=200)
    ap.add_argument("--n-tool", type=int, default=300)
    ap.add_argument("--n-extract", type=int, default=150)
    ap.add_argument("--n-route", type=int, default=100)
    ap.add_argument("--n-fail", type=int, default=60)
    ap.add_argument("--n-numeric", type=int, default=0, help="数值套档问答（时效/计费区域/运费/体积重/保价/超重）")
    ap.add_argument("--valid-ratio", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=7, help="随机种子；换种子即得到一批新样本")
    ap.add_argument("--out", default=str(OUT), help="输出目录")
    ap.add_argument("--skip-kb", action="store_true", help="不用教师模型生成知识问答")
    ap.add_argument("--kb-cache", default=str(config.DATA_DIR / "seed" / "teacher_kb_qa.json"), help="已生成的知识问答缓存，存在则直接复用")
    a = ap.parse_args()
    rnd.seed(a.seed)
    # report_exception 会写库：在临时的全新演示库上生成，既不污染 data/waybills.sqlite，也避免历次生成的异常轨迹累积进样本
    import tempfile
    from logicllm.tools import waybill
    waybill.DB_PATH = Path(tempfile.mkdtemp()) / "waybills.sqlite"
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = []
    print("生成工具调用样本…"); data += await gen_tool(a.n_tool)
    print("生成抽取样本…"); data += gen_extract(a.n_extract)
    print("生成路径样本…"); data += await gen_route(a.n_route)
    print("生成失败/边界样本…"); data += await gen_failures(a.n_fail)
    if a.n_numeric:
        print("生成数值套档样本…"); data += gen_numeric(a.n_numeric)
    cache = Path(a.kb_cache)
    if not a.skip_kb and a.n_kb > 0 and cache.exists():
        pairs = json.load(open(cache, encoding="utf-8"))
        print(f"复用缓存知识问答 {len(pairs)} 条：{cache}")
        data += [sample(SYSTEM_PLAIN, [{"role": "user", "content": x["q"]}, {"role": "assistant", "content": x["a"]}]) for x in pairs]
    elif not a.skip_kb and a.n_kb > 0:
        client = ChatClient()
        if await client.healthy():
            print(f"用教师模型 {client.model} 生成知识问答…")
            kb_samples = await gen_kb(a.n_kb, client)
            data += kb_samples
            json.dump([{"q": x["messages"][1]["content"], "a": x["messages"][2]["content"]} for x in kb_samples],
                      open(cache, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
        else:
            print("⚠️ LLM 后端不可用，跳过知识问答生成")
        await client.aclose()
    # 手写高质量种子
    for p in (config.DATA_DIR / "seed").glob("*.jsonl"):
        for l in p.read_text(encoding="utf-8").splitlines():
            if l.strip():
                d = json.loads(l)
                data.append(sample(SYSTEM_PLAIN, d["messages"][1:]))  # 手写种子也注入检索上下文

    # 会话级去重（避免同一轨迹既进训练又进验证）
    seen, uniq = set(), []
    for d in data:
        key = json.dumps(d["messages"], ensure_ascii=False, sort_keys=True)
        if key not in seen:
            seen.add(key)
            uniq.append(d)
    print(f"会话去重：{len(data)} → {len(uniq)}")

    # 按用户问题分组后切分训练/验证，再在各自分区内展开工具调用前缀
    # （mlx_lm --mask-prompt 只训最后一条 assistant，前缀样本让模型学会发出调用；
    #  同一问题的会话/前缀必须落在同一分区，否则验证损失虚低）
    groups: dict = {}
    for d in uniq:
        groups.setdefault(d["messages"][1]["content"], []).append(d)
    gkeys = list(groups)
    rnd.shuffle(gkeys)
    valid_sessions, train_sessions, nv = [], [], max(10, int(len(uniq) * a.valid_ratio))
    for k in gkeys:
        (valid_sessions if len(valid_sessions) < nv else train_sessions).extend(groups[k])

    def expand(sessions):
        out_rows = list(sessions)
        for d in sessions:
            msgs = d["messages"]
            for i, m in enumerate(msgs):
                if m["role"] == "assistant" and m.get("tool_calls") and i < len(msgs) - 1:
                    out_rows.append({**d, "messages": msgs[:i + 1]})
        return out_rows

    train_rows, valid_rows = expand(train_sessions), expand(valid_sessions)
    rnd.shuffle(train_rows)
    with open(out_dir / "valid.jsonl", "w", encoding="utf-8") as f:
        for x in valid_rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    with open(out_dir / "train.jsonl", "w", encoding="utf-8") as f:
        for x in train_rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"完成：train {len(train_rows)} 条（会话 {len(train_sessions)}），valid {len(valid_rows)} 条（会话 {len(valid_sessions)}）→ {out_dir}")


if __name__ == "__main__":
    asyncio.run(main())

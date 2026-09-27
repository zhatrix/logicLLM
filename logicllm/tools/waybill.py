"""运单系统（SQLite 模拟）：查询、创建、轨迹、异常登记。"""
from __future__ import annotations

import json
import random
import sqlite3
from datetime import datetime, timedelta

from logicllm import config
from logicllm.tools import tool
from logicllm.tools.geo import CITY_PROVINCE

DB_PATH = config.DATA_DIR / "waybills.sqlite"

STATUS_FLOW = ["已揽收", "运输中", "到达转运中心", "派送中", "已签收"]


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db(seed: int = 42, n: int = 60):
    from logicllm.tools.pricing import calc_freight as _calc
    config.DATA_DIR.mkdir(exist_ok=True)
    c = _conn()
    c.executescript("""
    CREATE TABLE IF NOT EXISTS waybill(
        no TEXT PRIMARY KEY, sender TEXT, sender_phone TEXT, origin TEXT,
        receiver TEXT, receiver_phone TEXT, destination TEXT, address TEXT,
        weight_kg REAL, service TEXT, fee REAL, status TEXT, created_at TEXT,
        declared_value REAL DEFAULT 0, exception TEXT DEFAULT ''
    );
    CREATE TABLE IF NOT EXISTS track(
        no TEXT, ts TEXT, location TEXT, event TEXT
    );
    """)
    if c.execute("SELECT COUNT(*) FROM waybill").fetchone()[0] > 0:
        c.close()
        return
    rnd = random.Random(seed)
    names = ["张伟", "王芳", "李娜", "刘强", "陈静", "杨洋", "赵磊", "黄敏", "周杰", "吴丹"]
    cities = list(CITY_PROVINCE)
    for i in range(n):
        # 注意保持随机数取用顺序与最初版本一致，确保演示运单号（如 LL2026080258）稳定
        no = f"LL{20260800 + i:08d}{rnd.randint(10, 99)}"
        o, d = rnd.sample(cities, 2)
        stage = rnd.randint(0, 4)
        created = datetime(2026, 8, rnd.randint(10, 21), rnd.randint(8, 20))
        w = round(rnd.uniform(0.3, 25), 1)
        sender, sender_phone = rnd.choice(names), f"13{rnd.randint(100000000, 999999999)}"
        receiver, receiver_phone = rnd.choice(names), f"15{rnd.randint(100000000, 999999999)}"
        addr = f"{d}市某某区某某路{rnd.randint(1, 500)}号"
        svc = rnd.choice(["标准快递", "特快", "经济"])
        dv = rnd.choice([0, 0, 500, 2000])
        exc = "" if rnd.random() > 0.12 else rnd.choice(["地址不详", "电话无人接听", "包裹破损"])
        fee = _calc(o, d, w, service=svc, declared_value=dv)["total_fee"]  # 统一计费引擎，与 calc_freight 一致
        c.execute("INSERT INTO waybill VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            no, sender, sender_phone, o, receiver, receiver_phone, d, addr, w, svc,
            fee, STATUS_FLOW[stage], created.isoformat(timespec="minutes"), dv, exc,
        ))
        ts = created
        locs = [o, o + "转运中心", "干线运输", d + "转运中心", d]
        for s in range(stage + 1):
            ts += timedelta(hours=rnd.randint(4, 20))
            c.execute("INSERT INTO track VALUES(?,?,?,?)", (no, ts.isoformat(timespec="minutes"), locs[s], STATUS_FLOW[s]))
    c.commit()
    c.close()


def _row(r) -> dict:
    return dict(r) if r else {}


@tool(
    "query_waybill",
    "按运单号查询运单详情（寄收件人、起止地、重量、费用、当前状态、异常标记）。",
    {"type": "object", "properties": {"waybill_no": {"type": "string", "description": "运单号，如 LL2026080512"}},
     "required": ["waybill_no"]},
)
def query_waybill(waybill_no: str):
    init_db()
    c = _conn()
    r = c.execute("SELECT * FROM waybill WHERE no=?", (waybill_no.strip().upper(),)).fetchone()
    c.close()
    return _row(r) or {"error": f"未找到运单 {waybill_no}"}


@tool(
    "track_waybill",
    "查询运单的物流轨迹（时间、地点、事件）。",
    {"type": "object", "properties": {"waybill_no": {"type": "string"}}, "required": ["waybill_no"]},
)
def track_waybill(waybill_no: str):
    init_db()
    c = _conn()
    rows = c.execute("SELECT ts, location, event FROM track WHERE no=? ORDER BY ts", (waybill_no.strip().upper(),)).fetchall()
    c.close()
    if not rows:
        return {"error": f"未找到运单 {waybill_no} 的轨迹"}
    return {"waybill_no": waybill_no, "events": [dict(r) for r in rows], "current": dict(rows[-1])}


@tool(
    "search_waybills",
    "按收/寄件人电话或姓名查找运单列表。",
    {"type": "object", "properties": {
        "phone": {"type": "string"}, "name": {"type": "string"},
        "status": {"type": "string", "enum": STATUS_FLOW},
    }},
)
def search_waybills(phone: str = "", name: str = "", status: str = ""):
    init_db()
    import re
    phone = re.sub(r"^\+?86|[\s-]", "", (phone or "").strip())
    if phone and not re.fullmatch(r"1[3-9]\d{9}", phone):  # 参数校验：模型偶尔会把号码抄漏几位，返回明确错误让它重试，而不是静默查空
        return {"error": f"手机号格式不正确（需 11 位数字），收到「{phone}」，请核对后重新查询"}
    if name and name.strip() in {"我", "本人", "用户", "客户", "我的"}:  # 代词不是姓名
        name = ""
    if not phone and not name and not status:
        return {"error": "请提供手机号或姓名"}
    q, args = "SELECT no, origin, destination, status, created_at, exception FROM waybill WHERE 1=1", []
    if phone:
        q += " AND (sender_phone=? OR receiver_phone=?)"; args += [phone, phone]
    if name:
        q += " AND (sender=? OR receiver=?)"; args += [name, name]
    if status:
        q += " AND status=?"; args.append(status)
    c = _conn()
    rows = c.execute(q + " LIMIT 20", args).fetchall()
    c.close()
    return {"count": len(rows), "waybills": [dict(r) for r in rows]}


@tool(
    "create_waybill",
    "创建新运单。需要寄/收件人姓名电话、起止城市、收件详细地址、重量；自动计算运费并返回运单号。",
    {"type": "object", "properties": {
        "sender": {"type": "string"}, "sender_phone": {"type": "string"}, "origin": {"type": "string"},
        "receiver": {"type": "string"}, "receiver_phone": {"type": "string"}, "destination": {"type": "string"},
        "address": {"type": "string"}, "weight_kg": {"type": "number"},
        "service": {"type": "string", "enum": ["标准快递", "特快", "经济", "零担"]},
        "declared_value": {"type": "number"},
    }, "required": ["sender", "sender_phone", "origin", "receiver", "receiver_phone", "destination", "address", "weight_kg"]},
)
def create_waybill(sender, sender_phone, origin, receiver, receiver_phone, destination, address,
                   weight_kg, service="标准快递", declared_value=0):
    from logicllm.tools.pricing import calc_freight
    init_db()
    fee = calc_freight(origin, destination, weight_kg, service=service, declared_value=declared_value)
    if "error" in fee:
        return fee
    c = _conn()
    n = c.execute("SELECT COUNT(*) FROM waybill").fetchone()[0]
    no = f"LL{20260800 + n:08d}{random.randint(10, 99)}"
    now = datetime.now().isoformat(timespec="minutes")
    c.execute("INSERT INTO waybill VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
        no, sender, sender_phone, origin, receiver, receiver_phone, destination, address,
        weight_kg, service, fee["total_fee"], "已揽收", now, declared_value, ""))
    c.execute("INSERT INTO track VALUES(?,?,?,?)", (no, now, origin, "已揽收"))
    c.commit(); c.close()
    return {"waybill_no": no, "fee": fee["total_fee"], "status": "已揽收", "created_at": now}


@tool(
    "report_exception",
    "为运单登记异常（如 破损/丢失/地址错误/延误），并生成工单号。",
    {"type": "object", "properties": {
        "waybill_no": {"type": "string"},
        "type": {"type": "string", "enum": ["破损", "丢失", "地址错误", "延误", "其他"]},
        "description": {"type": "string"},
    }, "required": ["waybill_no", "type"]},
)
def report_exception(waybill_no: str, type: str, description: str = ""):
    init_db()
    c = _conn()
    r = c.execute("SELECT no FROM waybill WHERE no=?", (waybill_no.strip().upper(),)).fetchone()
    if not r:
        c.close()
        return {"error": f"未找到运单 {waybill_no}"}
    c.execute("UPDATE waybill SET exception=? WHERE no=?", (type, waybill_no))
    ticket = f"TK{datetime.now():%Y%m%d%H%M%S}"
    c.execute("INSERT INTO track VALUES(?,?,?,?)", (waybill_no, datetime.now().isoformat(timespec="minutes"), "客服中心", f"异常登记:{type} {description}"))
    c.commit(); c.close()
    return {"ticket_no": ticket, "waybill_no": waybill_no, "type": type, "sla": "48小时内客服回访处理"}

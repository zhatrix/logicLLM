"""零担通 TMS 真实后端工具（TOOLS_BACKEND=tms 时启用，覆盖同名模拟工具）。

对接本地「零担通 LTL GO Backend」（FastAPI，默认 http://127.0.0.1:8000）：
- query_waybill / track_waybill / search_waybills / report_exception / waybill_status_counts
- calc_freight → /pricing/trial-quote（真实报价引擎，按网点对计价）

鉴权：手机号+密码登录 → 选租户 → Bearer token；401 时自动重登。
环境变量：TMS_BASE_URL、TMS_PHONE、TMS_PASSWORD、TMS_TENANT（租户名）。
"""
from __future__ import annotations

import os
import threading
import uuid

import httpx

from logicllm.tools import tool

BASE = os.getenv("TMS_BASE_URL", "http://127.0.0.1:8000/api/v1").rstrip("/")
PHONE = os.getenv("TMS_PHONE", "13800000001")
PASSWORD = os.getenv("TMS_PASSWORD", "Dev@12345")
TENANT = os.getenv("TMS_TENANT", "Dev Tenant")

_lock = threading.Lock()
_token: str | None = None

STATUS_ZH = {
    "draft": "草稿", "pending_confirmation": "待确认", "routed": "已排路由", "in_progress": "运输中",
    "delivered": "已送达", "completed": "已完成", "archived": "已归档", "canceled": "已取消",
    "rerouting": "改道中", "split": "已拆单", "returning": "退回中", "returned": "已退回",
}
EXC_CODE = {
    "地址错误": "ADDRESS_ISSUE", "派送失败": "DELIVERY_FAILED", "拒收": "REFUSED", "破损": "DAMAGED",
    "丢失": "LOST", "扣货": "DETAINED", "拦截": "INTERCEPTED", "揽收被拒": "PICKUP_REJECTED",
    "超尺寸": "OVERSIZE", "超重": "OVERWEIGHT", "回单驳回": "POD_REVIEW_REJECTED",
    "代收失败": "COD_COLLECTION_FAILED", "代收金额不符": "COD_AMOUNT_MISMATCH", "退货申请": "RETURN_REQUESTED",
    "延误": "DELIVERY_FAILED", "其他": "DELIVERY_FAILED",
}


def _login() -> str:
    r = httpx.post(f"{BASE}/auth/login", json={"phone": PHONE, "password": PASSWORD}, timeout=15)
    r.raise_for_status()
    d = r.json()
    if d.get("access_token"):
        return d["access_token"]
    comps = d.get("companies") or []
    comp = next((c for c in comps if c["tenant_name"] == TENANT), comps[0] if comps else None)
    if not comp:
        raise RuntimeError("TMS 登录成功但无可用租户")
    r = httpx.post(f"{BASE}/auth/select-tenant", json={"ticket": d["ticket"], "tenant_id": comp["tenant_id"]}, timeout=15)
    r.raise_for_status()
    return r.json()["access_token"]


def _req(method: str, path: str, **kw) -> httpx.Response:
    global _token
    with _lock:
        if _token is None:
            _token = _login()
        tok = _token
    def _headers(t: str) -> dict:
        h = {"Authorization": f"Bearer {t}"}
        if method.upper() in ("POST", "PUT", "PATCH", "DELETE"):
            h["X-Idempotency-Key"] = str(uuid.uuid4())
        return h

    r = httpx.request(method, f"{BASE}{path}", headers=_headers(tok), timeout=30, **kw)
    if r.status_code == 401:
        with _lock:
            _token = _login()
            tok = _token
        r = httpx.request(method, f"{BASE}{path}", headers=_headers(tok), timeout=30, **kw)
    return r


def _api(method: str, path: str, **kw) -> dict | list:
    r = _req(method, path, **kw)
    if r.status_code >= 400:
        try:
            err = r.json()
        except ValueError:
            err = r.text[:200]
        return {"error": f"TMS 接口 {path} 返回 {r.status_code}", "detail": err}
    return r.json() if r.text else {}


def _items(res) -> tuple[list, int]:
    """兼容 {data, pagination} 与 {items, total} 两种列表响应。"""
    if isinstance(res, list):
        return res, len(res)
    items = res.get("data") or res.get("items") or []
    pg = res.get("pagination") or {}
    total = pg.get("total") or res.get("total") or len(items)
    return items, total


def _brief(w: dict) -> dict:
    return {
        "waybill_no": w.get("waybill_no"), "status": STATUS_ZH.get(w.get("status"), w.get("status")),
        "origin_site": w.get("origin_site_name") or w.get("origin_site_id"),
        "destination_site": w.get("destination_site_name") or w.get("destination_site_id"),
        "customer": w.get("customer_name_snapshot"),
        "receiver": (w.get("destination_address") or {}).get("contact"),
        "items": w.get("items_summary"), "payment_type": w.get("payment_type"),
        "receivable_amount": w.get("receivable_amount"), "created_at": w.get("created_at"),
    }


def _find_waybill(waybill_no: str) -> dict | None:
    res = _api("GET", "/waybills", params={"keyword": waybill_no.strip(), "page_size": 5})
    if isinstance(res, dict) and res.get("error"):
        return res
    items, _ = _items(res)
    exact = [w for w in items if w.get("waybill_no") == waybill_no.strip()]
    return (exact or items or [None])[0]


@tool(
    "query_waybill",
    "按运单号查询 TMS 系统中的运单详情（起止网点、客户、货物、费用、当前状态）。",
    {"type": "object", "properties": {"waybill_no": {"type": "string", "description": "运单号"}},
     "required": ["waybill_no"]},
)
def query_waybill(waybill_no: str):
    w = _find_waybill(waybill_no)
    if not w:
        return {"error": f"未找到运单 {waybill_no}"}
    if w.get("error"):
        return w
    detail = _api("GET", f"/waybills/{w['id']}")
    if isinstance(detail, dict) and not detail.get("error"):
        w = {**w, **{k: v for k, v in detail.items() if v is not None}}
    out = _brief(w)
    out["is_cod"] = w.get("is_cod")
    out["fee_breakdown"] = w.get("fee_breakdown")
    return out


@tool(
    "track_waybill",
    "查询运单的全程时间线/物流轨迹（TMS 系统真实事件）。",
    {"type": "object", "properties": {"waybill_no": {"type": "string"}}, "required": ["waybill_no"]},
)
def track_waybill(waybill_no: str):
    w = _find_waybill(waybill_no)
    if not w or w.get("error"):
        return w or {"error": f"未找到运单 {waybill_no}"}
    tl = _api("GET", f"/waybills/{w['id']}/timeline")
    if isinstance(tl, dict) and tl.get("error"):
        return tl
    events = tl if isinstance(tl, list) else tl.get("items") or tl.get("events") or []
    slim = []
    for e in events[-20:]:
        slim.append({k: e.get(k) for k in ("event_at", "occurred_at", "created_at", "ulsc_code", "event_type", "title", "site_name", "notes", "description") if e.get(k) is not None})
    return {"waybill_no": w.get("waybill_no"), "status": STATUS_ZH.get(w.get("status"), w.get("status")), "events": slim}


@tool(
    "search_waybills",
    "按关键词（客户名/电话/运单号片段）或状态搜索运单列表。",
    {"type": "object", "properties": {
        "keyword": {"type": "string", "description": "关键词，可选"},
        "phone": {"type": "string", "description": "寄/收件人手机号，可选"},
        "name": {"type": "string", "description": "寄/收件人或客户姓名，可选"},
        "status": {"type": "string", "enum": list(STATUS_ZH.values()), "description": "运单状态，可选"},
    }},
)
def search_waybills(keyword: str = "", status: str = "", phone: str = "", name: str = ""):
    keyword = keyword or phone or name  # 兼容微调数据里的 phone/name 参数习惯
    params: dict = {"page_size": 10}
    if keyword:
        params["keyword"] = keyword
    if status:
        rev = {v: k for k, v in STATUS_ZH.items()}
        params["status"] = rev.get(status, status)
    res = _api("GET", "/waybills", params=params)
    if isinstance(res, dict) and res.get("error"):
        return res
    items, total = _items(res)
    return {"total": total, "waybills": [_brief(w) for w in items]}


@tool(
    "waybill_status_counts",
    "统计当前各状态的运单数量（运营概览）。",
    {"type": "object", "properties": {}},
)
def waybill_status_counts():
    res = _api("GET", "/waybills/status-counts")
    if isinstance(res, dict) and res.get("error"):
        return res
    counts = res.get("counts") or res
    if isinstance(counts, dict):
        return {STATUS_ZH.get(k, k): v for k, v in counts.items() if isinstance(v, int)}
    return counts


@tool(
    "report_exception",
    "为运单在 TMS 中登记异常工单。type 可选：破损/丢失/地址错误/拒收/派送失败/扣货/超重/超尺寸/延误/其他。",
    {"type": "object", "properties": {
        "waybill_no": {"type": "string"},
        "type": {"type": "string", "enum": list(EXC_CODE)},
        "description": {"type": "string"},
    }, "required": ["waybill_no", "type"]},
)
def report_exception(waybill_no: str, type: str, description: str = ""):
    w = _find_waybill(waybill_no)
    if not w or w.get("error"):
        return w or {"error": f"未找到运单 {waybill_no}"}
    body = {"waybill_id": w["id"], "code": EXC_CODE.get(type, "DELIVERY_FAILED"),
            "severity": "high" if type in ("丢失", "破损") else "medium", "note": description or type}
    res = _api("POST", "/exceptions", json=body)
    if isinstance(res, dict) and res.get("error"):
        return res
    return {"exception_id": res.get("id"), "waybill_no": w.get("waybill_no"), "code": body["code"],
            "status": res.get("status", "open"), "note": "已登记异常工单，运营会在工作台跟进处理"}


def _all_sites() -> list | dict:
    res = _api("GET", "/sites", params={"page_size": 100})
    if isinstance(res, dict) and res.get("error"):
        return res
    sites = [s for s in _items(res)[0] if s.get("status") in (None, "active")]
    # 名称去重（环境残留的重复网点取最早创建的一个）
    seen: dict = {}
    for s2 in sorted(sites, key=lambda x: x.get("created_at") or ""):
        seen.setdefault(s2.get("name"), s2)
    return list(seen.values())


def _find_site(name: str) -> dict | None:
    sites = _all_sites()
    if isinstance(sites, dict):
        return sites
    q = name.strip()
    exact = [s for s in sites if s.get("name") == q]
    if exact:
        return exact[0]
    sub = [s for s in sites if q in s.get("name", "") or s.get("name", "") in q]
    if len(sub) == 1:
        return sub[0]
    # 模糊：去掉「网点/站/仓」后缀，按城市/关键词包含匹配
    core = q.replace("网点", "").replace("营业部", "").replace("站", "").replace("仓", "")
    fuzzy = [s for s in sites if core and core in s.get("name", "")]
    if len(fuzzy) == 1:
        return fuzzy[0]
    if len(fuzzy) > 1:
        return {"error": f"网点「{name}」有多个匹配，请让用户确认", "candidates": [s["name"] for s in fuzzy]}
    return None


@tool(
    "calc_freight",
    "用 TMS 报价引擎试算两个网点之间的零担运费。origin/destination 传网点名称（如 杭州网点）。",
    {"type": "object", "properties": {
        "origin": {"type": "string", "description": "起运网点名称"},
        "destination": {"type": "string", "description": "目的网点名称"},
        "weight_kg": {"type": "number"},
        "volume_m3": {"type": "number", "description": "体积方数，可选"},
        "piece_count": {"type": "integer", "description": "件数，可选"},
        "declared_value": {"type": "number", "description": "声明价值，可选"},
    }, "required": ["origin", "destination", "weight_kg"]},
)
def calc_freight(origin: str, destination: str, weight_kg: float, volume_m3: float | None = None,
                 piece_count: int | None = None, declared_value: float | None = None):
    fs, ts = _find_site(origin), _find_site(destination)
    if not fs or fs.get("error") or not ts or ts.get("error"):
        bad = fs if (not fs or fs.get("error")) else ts
        if bad and bad.get("error"):
            return bad
        sites = _all_sites()
        names = [s["name"] for s in sites] if isinstance(sites, list) else []
        return {"error": f"未找到网点「{origin if not fs or fs.get('error') else destination}」",
                "available_sites": names, "hint": "请用 available_sites 里的准确名称重新调用 calc_freight"}
    body: dict = {"from_site_id": fs["id"], "to_site_id": ts["id"], "weight_kg": str(weight_kg)}
    if volume_m3:
        body["volume_m3"] = str(volume_m3)
    if piece_count:
        body["piece_count"] = piece_count
    if declared_value:
        body["declared_value"] = str(declared_value)
    res = _api("POST", "/pricing/trial-quote", json=body)
    if isinstance(res, dict) and res.get("error"):
        return res
    # 映射成模型熟悉的字段：total_fee 为最终应收，breakdown 为明细说明
    parts = "；".join(f"{b.get('label')} {b.get('amount')} 元" for b in res.get("breakdown") or [])
    return {
        "from_site": fs["name"], "to_site": ts["name"],
        "billable_weight_kg": weight_kg, "volume_m3": volume_m3,
        "product": res.get("applied_product_name"),
        "total_fee": float(res.get("total") or res.get("discounted_freight") or 0),
        "breakdown": parts or f"运费 {res.get('total')} 元",
        "note": "以上为 TMS 报价引擎试算结果，请直接引用 total_fee，不要自行按首重续重公式重算",
    }

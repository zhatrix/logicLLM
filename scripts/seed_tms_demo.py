"""往本地零担通 TMS（Dev Tenant）种一批演示数据：3 个网点、1 个客户、路由规则、3 张运单、若干轨迹事件。
幂等：按名称/单号存在即跳过。用法：uv run python scripts/seed_tms_demo.py
"""
from __future__ import annotations

import datetime
import sys

from logicllm.tools.tms import BASE, _api, _items  # 复用登录与请求封装

SITES = [
    {"name": "杭州西湖网点", "roles": ["origin", "destination"], "addr": ("浙江", "杭州", "西湖")},
    {"name": "深圳南山网点", "roles": ["origin", "destination"], "addr": ("广东", "深圳", "南山")},
    {"name": "成都金牛网点", "roles": ["origin", "destination"], "addr": ("四川", "成都", "金牛")},
]


def ensure_site(spec) -> dict:
    res = _api("GET", "/sites", params={"keyword": spec["name"], "page_size": 5})
    for s in _items(res)[0]:
        if s["name"] == spec["name"]:
            return s
    p, c, dst = spec["addr"]
    body = {
        "name": spec["name"], "node_type": "site", "cooperation_type": "direct", "roles": spec["roles"],
        "address": {"province": p, "city": c, "district": dst, "detail": f"{c}市{dst}区物流园1号"},
        "contact_admin": {"name": "站长", "phone": "13800000001"},
    }
    r = _api("POST", "/sites", json=body)
    if r.get("error"):
        sys.exit(f"创建网点失败: {r}")
    print("创建网点:", spec["name"])
    return r


def ensure_customer() -> dict:
    res = _api("GET", "/customers", params={"page_size": 20})
    for cu in _items(res)[0]:
        if cu["name"] == "杭州云货科技有限公司":
            return cu
    body = {
        "code": "CUST-DEMO-001", "name": "杭州云货科技有限公司", "short_name": "云货科技",
        "contacts": [{"name": "王小明", "phone": "13812345678", "role": "logistics"}],
        "settlement": {"cycle": "cash_on_delivery"}, "credit": {"limit": "0"},
    }
    r = _api("POST", "/customers", json=body)
    if r.get("error"):
        sys.exit(f"创建客户失败: {r}")
    print("创建客户:", body["name"])
    return r


def ensure_route(frm: dict, to: dict):
    res = _api("GET", "/route-rules", params={"page_size": 50})
    for rr in _items(res)[0]:
        if rr.get("from_site_id") == frm["id"] and rr.get("to_site_id") == to["id"]:
            return rr
    body = {
        "from_site_id": frm["id"], "to_site_id": to["id"], "priority": 0,
        "valid_from": datetime.date.today().isoformat(),
        "legs": [{"from_node_id": frm["id"], "to_node_id": to["id"], "leg_type": "linehaul", "carrier_type": "hq_fleet"}],
    }
    r = _api("POST", "/route-rules", json=body)
    if r.get("error"):
        print(f"路由规则 {frm['name']}→{to['name']} 失败: {r}")
        return None
    print(f"创建路由: {frm['name']} → {to['name']}")
    return r


def ensure_waybill(cust: dict, frm: dict, to: dict, weight: str, fee: str, receiver: tuple[str, str]):
    body = {
        "customer_id": cust["id"], "customer_name_snapshot": cust["name"],
        "origin_site_id": frm["id"], "destination_site_id": to["id"],
        "origin_address": None,
        "destination_address": {"contact": receiver[0], "phone": receiver[1],
                                "province": to["address"]["province"] if isinstance(to.get("address"), dict) else "四川",
                                "city": to["name"][:2], "district": "城区"},
        "items_summary": {"count": 3, "weight_kg": weight, "volume_m3": "0.5"},
        "payment_type": "cash_on_delivery", "settlement_snapshot": {"cycle": "cash_on_delivery"},
        "is_cod": False, "cod_amount": "0", "receivable_amount": fee, "cod_service_fee_amount": "0",
        "surcharges": [], "fee_breakdown": {"base": fee, "surcharges": []},
        "pricing_snapshot": {"product_id": None, "discount_rate": "1.0"},
        "business_mode": "ltl_network",
    }
    r = _api("POST", "/waybills", json=body)
    if r.get("error"):
        print(f"创建运单失败 ({frm['name']}→{to['name']}): {r}")
        return None
    print("创建运单:", r.get("waybill_no"), f"{frm['name']} → {to['name']}")
    return r


def main():
    print("TMS:", BASE)
    sites = {s["name"]: ensure_site(s) for s in SITES}
    cust = ensure_customer()
    hz, sz, cd = sites["杭州西湖网点"], sites["深圳南山网点"], sites["成都金牛网点"]
    for a, b in [(hz, sz), (hz, cd), (sz, cd)]:
        ensure_route(a, b)
    existing, total = _items(_api("GET", "/waybills", params={"page_size": 5}))
    if total >= 3:
        print("已有运单，跳过创建:", [w["waybill_no"] for w in existing[:5]])
    else:
        ensure_waybill(cust, hz, sz, "120", "260.00", ("陈静", "15901112222"))
        ensure_waybill(cust, hz, cd, "50", "200.00", ("王经理", "13800000001"))
        ensure_waybill(cust, sz, cd, "800", "980.00", ("刘强", "15633334444"))
    wb, _t = _items(_api("GET", "/waybills", params={"page_size": 10}))
    print("\n当前运单：")
    for w in wb:
        print(" ", w["waybill_no"], w["status"], w.get("origin_site_name"), "→", w.get("destination_site_name"))


if __name__ == "__main__":
    main()

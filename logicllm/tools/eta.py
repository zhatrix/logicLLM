"""时效估算。"""
from __future__ import annotations

from datetime import datetime, timedelta

from logicllm.tools import tool
from logicllm.tools.geo import CITY_PROVINCE, REMOTE_PROVINCES, distance_km


@tool(
    "estimate_eta",
    "估算从寄件城市到收件城市的快递时效（天数）及预计送达日期。",
    {"type": "object", "properties": {
        "origin": {"type": "string"}, "destination": {"type": "string"},
        "service": {"type": "string", "enum": ["标准快递", "特快", "经济", "零担"]},
        "ship_date": {"type": "string", "description": "寄出日期 YYYY-MM-DD，默认今天"},
    }, "required": ["origin", "destination"]},
)
def estimate_eta(origin: str, destination: str, service: str = "标准快递", ship_date: str | None = None):
    if origin not in CITY_PROVINCE or destination not in CITY_PROVINCE:
        return {"error": "未收录城市", "supported": sorted(CITY_PROVINCE)}
    d = distance_km(origin, destination) or 1500
    same_city = origin == destination
    if same_city:
        days = 1
    elif d <= 500:
        days = 2
    elif d <= 1500:
        days = 3
    elif d <= 2500:
        days = 4
    else:
        days = 5
    if CITY_PROVINCE[destination] in REMOTE_PROVINCES:
        days += 2
    adj = {"标准快递": 0, "特快": -1, "经济": 2, "零担": 3}[service]
    days = max(1, days + adj)
    start = datetime.strptime(ship_date, "%Y-%m-%d") if ship_date else datetime.now()
    return {
        "origin": origin, "destination": destination, "service": service,
        "distance_km": round(d), "est_days": days,
        "est_delivery_date": (start + timedelta(days=days)).strftime("%Y-%m-%d"),
        "note": "偏远地区已含 +2 天" if CITY_PROVINCE[destination] in REMOTE_PROVINCES else "",
    }

"""运费计算：首重续重 + 体积重 + 偏远附加 + 服务类型。"""
from __future__ import annotations

from logicllm.tools import tool
from logicllm.tools.geo import CITY_PROVINCE, REMOTE_PROVINCES, distance_km

VOLUME_DIVISOR = 6000  # cm³/kg，快递常用
SERVICE_MULT = {"标准快递": 1.0, "特快": 1.6, "经济": 0.8, "零担": 0.5}


def _zone(origin: str, dest: str) -> str:
    po, pd = CITY_PROVINCE[origin], CITY_PROVINCE[dest]
    if po == pd:
        return "同省"
    d = distance_km(origin, dest) or 1500
    return "邻近" if d <= 800 else "跨区"


FIRST_KG = {"同省": 8.0, "邻近": 10.0, "跨区": 12.0}
ADD_KG = {"同省": 2.0, "邻近": 4.0, "跨区": 6.0}


@tool(
    "calc_freight",
    "按首重/续重规则计算运费。考虑体积重（长×宽×高/6000）、计费区域（同省/邻近/跨区）、偏远地区附加费与服务类型。",
    {"type": "object", "properties": {
        "origin": {"type": "string", "description": "寄件城市"},
        "destination": {"type": "string", "description": "收件城市"},
        "weight_kg": {"type": "number", "description": "实际重量 kg"},
        "length_cm": {"type": "number"}, "width_cm": {"type": "number"}, "height_cm": {"type": "number"},
        "service": {"type": "string", "enum": list(SERVICE_MULT), "description": "服务类型，默认 标准快递"},
        "declared_value": {"type": "number", "description": "保价声明价值（元），可选"},
    }, "required": ["origin", "destination", "weight_kg"]},
)
def calc_freight(origin: str, destination: str, weight_kg: float, length_cm: float = 0,
                 width_cm: float = 0, height_cm: float = 0, service: str = "标准快递",
                 declared_value: float = 0):
    if service not in SERVICE_MULT:
        return {"error": f"不支持的服务类型 {service}", "supported": list(SERVICE_MULT)}
    unknown = [c for c in (origin, destination) if c not in CITY_PROVINCE]
    if unknown:
        return {"error": f"未收录城市: {'、'.join(unknown)}（仅支持国内已开通城市）", "supported": sorted(CITY_PROVINCE)}
    vol_w = (length_cm * width_cm * height_cm) / VOLUME_DIVISOR if length_cm and width_cm and height_cm else 0
    bill_w = max(weight_kg, vol_w)
    import math
    bill_w = math.ceil(bill_w * 2) / 2  # 向上取 0.5kg
    zone = _zone(origin, destination)
    base = FIRST_KG[zone] + max(0, bill_w - 1) * ADD_KG[zone]
    base *= SERVICE_MULT[service]
    remote = 10.0 if CITY_PROVINCE.get(destination) in REMOTE_PROVINCES else 0.0
    oversize = 30.0 if (weight_kg > 50 or max(length_cm, width_cm, height_cm) > 150) else 0.0
    insurance = max(1.0, declared_value * 0.005) if declared_value else 0.0
    total = round(base + remote + oversize + insurance, 2)
    return {
        "billable_weight_kg": bill_w, "volume_weight_kg": round(vol_w, 2), "zone": zone,
        "service": service, "base_fee": round(base, 2), "remote_surcharge": remote,
        "oversize_surcharge": oversize,
        "insurance_fee": round(insurance, 2), "total_fee": total,
        "breakdown": f"首重{FIRST_KG[zone]}元 + 续重{max(0, bill_w-1)}kg×{ADD_KG[zone]}元，×{SERVICE_MULT[service]}({service})"
                     + (f" + 偏远附加{remote}元" if remote else "")
                     + (f" + 超长超重操作费{oversize}元" if oversize else "")
                     + (f" + 保价费{insurance:.2f}元" if insurance else ""),
    }

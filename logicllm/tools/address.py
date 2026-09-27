"""中文地址解析：省/市/区/详细 + 姓名电话（规则实现，供模型做结构化抽取时校验/补充）。"""
from __future__ import annotations

import re

from logicllm.tools import tool
from logicllm.tools.geo import CITY_PROVINCE

PROVINCES = ["北京", "上海", "天津", "重庆", "河北", "山西", "辽宁", "吉林", "黑龙江", "江苏", "浙江", "安徽",
             "福建", "江西", "山东", "河南", "湖北", "湖南", "广东", "海南", "四川", "贵州", "云南", "陕西",
             "甘肃", "青海", "台湾", "内蒙古", "广西", "西藏", "宁夏", "新疆", "香港", "澳门"]
PHONE_RE = re.compile(r"(?<!\d)(1[3-9]\d{9})(?!\d)")
NAME_RE = re.compile(r"(?:收件人|寄件人|联系人|姓名)[:：]?\s*([一-龥]{2,4})")


def _normalize_phone_text(text: str) -> str:
    """把 +86 138 0000 1234 / 139-1234-5678 这类写法归一为 11 位连续数字，便于正则识别与输出规范化。"""
    text = re.sub(r"\+?86[\s-]?(?=1[3-9]\d)", "", text)
    return re.sub(r"(?<=\d)[\s-]+(?=\d)", "", text)


def parse_address(text: str) -> dict:
    raw = _normalize_phone_text(text.strip())
    phone = PHONE_RE.search(raw)
    name = NAME_RE.search(raw)
    body = PHONE_RE.sub(" ", raw)
    body = NAME_RE.sub(" ", body)
    province = city = district = ""
    for p in PROVINCES:
        if p in body:
            province = p
            body = re.sub(rf"{p}(省|市|自治区|壮族自治区|回族自治区|维吾尔自治区)?", " ", body, count=1)
            break
    m = re.search(r"([一-龥]{2,5}?市)", body)
    if province in ("北京", "上海", "天津", "重庆"):  # 直辖市：市即省，避免把"南京西路"识别成南京市
        city = province
    elif m:
        city = m.group(1).rstrip("市")
        body = body.replace(m.group(1), " ", 1)
    else:
        for c in CITY_PROVINCE:
            if c in body:
                city = c
                body = body.replace(c, " ", 1)
                break
    if not province and city:
        province = CITY_PROVINCE.get(city, "")
    m = re.search(r"([一-龥]{1,5}?(?:区|县|旗))", body)
    if m:
        district = m.group(1)
        body = body.replace(district, " ", 1)
    detail = re.sub(r"\s+", " ", body).strip(" ,，")
    return {
        "name": name.group(1) if name else "",
        "phone": phone.group(1) if phone else "",
        "province": province, "city": city, "district": district, "detail": detail,
        "remote": province in {"新疆", "西藏", "青海", "内蒙古", "甘肃", "宁夏"},
    }


@tool(
    "parse_address",
    "把一段中文收寄件信息解析为 姓名/电话/省/市/区县/详细地址 结构。",
    {"type": "object", "properties": {"text": {"type": "string", "description": "原始地址文本"}},
     "required": ["text"]},
)
def _parse_address(text: str):
    return parse_address(text)

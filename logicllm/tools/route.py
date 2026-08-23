"""干线路径规划：Dijkstra，可选途经/回避。"""
from __future__ import annotations

import heapq
from collections import defaultdict

from logicllm.tools import tool
from logicllm.tools.geo import EDGES, CITY_PROVINCE

GRAPH: dict[str, dict[str, int]] = defaultdict(dict)
for a, b, d in EDGES:
    GRAPH[a][b] = d
    GRAPH[b][a] = d


def _dijkstra(src: str, dst: str, avoid: set[str]) -> tuple[float, list[str]] | None:
    dist = {src: 0.0}
    prev: dict[str, str] = {}
    pq = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == dst:
            path = [u]
            while u in prev:
                u = prev[u]
                path.append(u)
            return d, path[::-1]
        if d > dist.get(u, float("inf")):
            continue
        for v, w in GRAPH[u].items():
            if v in avoid and v != dst:
                continue
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    return None


def shortest_path(origin: str, destination: str, via: list[str] | None = None,
                  avoid: list[str] | None = None) -> dict:
    for c in [origin, destination, *(via or []), *(avoid or [])]:
        if c not in CITY_PROVINCE:
            return {"error": f"未收录城市: {c}", "supported": sorted(CITY_PROVINCE)}
    stops = [origin, *(via or []), destination]
    total, full = 0.0, [origin]
    for a, b in zip(stops, stops[1:]):
        r = _dijkstra(a, b, set(avoid or []))
        if r is None:
            return {"error": f"{a} 到 {b} 无可达路径（回避条件下）"}
        total += r[0]
        full += r[1][1:]
    hours = total / 70  # 干线货车平均 70km/h（含休息）
    return {
        "origin": origin, "destination": destination, "path": full,
        "distance_km": round(total), "est_drive_hours": round(hours, 1),
        "legs": len(full) - 1,
    }


@tool(
    "plan_route",
    "规划两城市之间的公路干线路径，返回途经城市、总里程与预计行驶时长。可指定必经城市与回避城市。",
    {"type": "object", "properties": {
        "origin": {"type": "string", "description": "起点城市，如 上海"},
        "destination": {"type": "string", "description": "终点城市，如 成都"},
        "via": {"type": "array", "items": {"type": "string"}, "description": "必经城市列表，可选"},
        "avoid": {"type": "array", "items": {"type": "string"}, "description": "回避城市列表，可选"},
    }, "required": ["origin", "destination"]},
)
def plan_route(origin: str, destination: str, via=None, avoid=None):
    return shortest_path(origin, destination, via, avoid)


@tool(
    "optimize_delivery_order",
    "给定一个出发城市和多个配送城市，用最近邻+2-opt 给出总里程较短的配送顺序（简单 TSP）。",
    {"type": "object", "properties": {
        "depot": {"type": "string", "description": "出发/仓库所在城市"},
        "stops": {"type": "array", "items": {"type": "string"}, "description": "需要配送的城市列表"},
        "return_to_depot": {"type": "boolean", "description": "是否回到出发地，默认 true"},
    }, "required": ["depot", "stops"]},
)
def optimize_delivery_order(depot: str, stops: list[str], return_to_depot: bool = True):
    nodes = [depot, *stops]
    for c in nodes:
        if c not in CITY_PROVINCE:
            return {"error": f"未收录城市: {c}"}
    n = len(nodes)
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            r = _dijkstra(nodes[i], nodes[j], set())
            D[i][j] = D[j][i] = r[0] if r else 1e9
    # 最近邻
    order, left = [0], set(range(1, n))
    while left:
        last = order[-1]
        nxt = min(left, key=lambda j: D[last][j])
        order.append(nxt)
        left.remove(nxt)
    if return_to_depot:
        order.append(0)

    def length(o):
        return sum(D[a][b] for a, b in zip(o, o[1:]))

    # 2-opt
    improved = True
    while improved:
        improved = False
        for i in range(1, len(order) - 2):
            for j in range(i + 1, len(order) - 1):
                cand = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                if length(cand) + 1e-6 < length(order):
                    order, improved = cand, True
    return {
        "sequence": [nodes[i] for i in order],
        "total_km": round(length(order)),
        "est_drive_hours": round(length(order) / 70, 1),
    }

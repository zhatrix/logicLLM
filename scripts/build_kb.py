"""构建知识库向量索引：uv run python scripts/build_kb.py"""
from logicllm.rag.kb import build_index, KnowledgeBase
from logicllm import config

n = build_index()
print(f"索引完成：{n} 个切块 → {config.KB_INDEX}")
kb = KnowledgeBase()
for q in ["没保价丢了赔多少", "充电宝能不能寄", "FOB是什么"]:
    hits = kb.search(q, top_k=2)
    print(f"\nQ: {q}")
    for h in hits:
        print(f"  {h['score']:.3f}  {h['title']}")

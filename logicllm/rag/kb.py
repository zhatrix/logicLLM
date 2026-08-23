"""知识库：Markdown 按标题切块 → bge-m3 向量 → numpy 余弦检索。"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from logicllm import config


@dataclass
class Chunk:
    doc: str
    title: str
    text: str


def split_markdown(path: Path, max_chars: int = 700) -> list[Chunk]:
    text = path.read_text(encoding="utf-8")
    doc = path.stem
    parts = re.split(r"\n(?=#{1,3} )", text)
    chunks: list[Chunk] = []
    h1 = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        first, _, body = part.partition("\n")
        if first.startswith("# "):
            h1 = first[2:].strip()
        title = first.lstrip("# ").strip()
        full_title = f"{h1} / {title}" if h1 and title != h1 else title
        body = body.strip()
        if not body:
            continue
        # 过长段落按空行再切
        buf = ""
        for para in re.split(r"\n\s*\n", body):
            if len(buf) + len(para) > max_chars and buf:
                chunks.append(Chunk(doc, full_title, buf.strip()))
                buf = ""
            buf += para + "\n\n"
        if buf.strip():
            chunks.append(Chunk(doc, full_title, buf.strip()))
    return chunks


_embedder = None


def embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(config.EMBED_MODEL, device="mps")
    return _embedder


def build_index(kb_dir: Path = config.KB_DIR, out: Path = config.KB_INDEX) -> int:
    chunks: list[Chunk] = []
    for p in sorted(kb_dir.glob("*.md")):
        chunks.extend(split_markdown(p))
    texts = [f"{c.title}\n{c.text}" for c in chunks]
    emb = embedder().encode(texts, normalize_embeddings=True, batch_size=16, show_progress_bar=True)
    np.savez_compressed(out, emb=emb.astype(np.float32),
                        doc=np.array([c.doc for c in chunks]),
                        title=np.array([c.title for c in chunks]),
                        text=np.array([c.text for c in chunks]))
    return len(chunks)


class KnowledgeBase:
    def __init__(self, index: Path = config.KB_INDEX):
        self.available = index.exists()
        if self.available:
            z = np.load(index, allow_pickle=False)
            self.emb, self.doc, self.title, self.text = z["emb"], z["doc"], z["title"], z["text"]

    def search(self, query: str, top_k: int = config.RAG_TOP_K, min_score: float = config.RAG_MIN_SCORE) -> list[dict]:
        if not self.available:
            return []
        q = embedder().encode([query], normalize_embeddings=True)[0]
        scores = self.emb @ q
        idx = np.argsort(-scores)[:top_k]
        return [
            {"doc": str(self.doc[i]), "title": str(self.title[i]), "text": str(self.text[i]), "score": float(scores[i])}
            for i in idx if scores[i] >= min_score
        ]

    @staticmethod
    def format(hits: list[dict]) -> str:
        return "\n\n".join(f"【{h['title']}】\n{h['text']}" for h in hits)

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

from .schema import Evidence


TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


class KnowledgeBase:
    """A tiny BM25-style retriever over local Markdown files.

    This keeps the demo dependency-light. The class is intentionally small so it
    can later be replaced by FAISS, Milvus, or a LangChain retriever.
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.documents = self._load_documents()
        self.doc_tokens = [tokenize(doc["content"]) for doc in self.documents]
        self.avgdl = sum(len(tokens) for tokens in self.doc_tokens) / max(len(self.doc_tokens), 1)
        self.doc_freq = self._document_frequency()

    def _load_documents(self) -> list[dict[str, str]]:
        docs: list[dict[str, str]] = []
        for path in sorted(self.data_dir.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            chunks = self._split_markdown(text)
            for idx, chunk in enumerate(chunks, start=1):
                title = chunk.splitlines()[0].lstrip("# ").strip() if chunk.splitlines() else path.stem
                docs.append(
                    {
                        "source": path.name,
                        "title": title or f"{path.stem}-{idx}",
                        "content": chunk.strip(),
                    }
                )
        return docs

    def _split_markdown(self, text: str) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        for line in text.splitlines():
            if line.startswith("## ") and current:
                chunks.append("\n".join(current))
                current = [line]
            else:
                current.append(line)
        if current:
            chunks.append("\n".join(current))
        return [chunk for chunk in chunks if len(tokenize(chunk)) >= 4]

    def _document_frequency(self) -> Counter[str]:
        df: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            df.update(set(tokens))
        return df

    def search(self, query: str, top_k: int = 3) -> list[Evidence]:
        query_tokens = tokenize(query)
        scored: list[Evidence] = []
        for doc, tokens in zip(self.documents, self.doc_tokens):
            score = self._bm25_score(query_tokens, tokens)
            if score > 0:
                scored.append(
                    Evidence(
                        source=doc["source"],
                        title=doc["title"],
                        content=doc["content"],
                        score=round(score, 4),
                    )
                )
        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]

    def _bm25_score(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        if not query_tokens or not doc_tokens:
            return 0.0
        counts = Counter(doc_tokens)
        score = 0.0
        total_docs = len(self.documents)
        k1 = 1.5
        b = 0.75
        for token in query_tokens:
            if token not in counts:
                continue
            df = self.doc_freq.get(token, 0)
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            tf = counts[token]
            denom = tf + k1 * (1 - b + b * len(doc_tokens) / max(self.avgdl, 1))
            score += idf * (tf * (k1 + 1) / denom)
        return score

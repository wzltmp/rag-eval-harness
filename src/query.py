"""Retrieve + rerank + answer.

Usage:
    python -m src.query "What does Section 3.1 say about X?"
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any

from anthropic import Anthropic
from anthropic.types import TextBlock
from dotenv import load_dotenv
from openai import OpenAI
from pgvector.psycopg import register_vector
from psycopg import Connection, connect
from sentence_transformers import CrossEncoder

load_dotenv()

EMBED_MODEL = "text-embedding-3-small"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "claude-haiku-4-5-20251001"  # cheap; swap to sonnet for quality
TOP_K_RETRIEVE = 20
TOP_K_RERANK = 5


@dataclass
class Chunk:
    source: str
    page: int
    text: str
    score: float = 0.0


def retrieve(question: str, conn: Connection[Any], top_k: int = TOP_K_RETRIEVE) -> list[Chunk]:
    client = OpenAI()
    qvec = client.embeddings.create(model=EMBED_MODEL, input=[question]).data[0].embedding
    rows = conn.execute(
        "SELECT source, page, text FROM chunks ORDER BY embedding <=> %s::vector LIMIT %s",
        (qvec, top_k),
    ).fetchall()
    return [Chunk(source=r[0], page=r[1], text=r[2]) for r in rows]


_RERANKER: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = CrossEncoder(RERANK_MODEL)
    return _RERANKER


def rerank(question: str, chunks: list[Chunk]) -> list[Chunk]:
    model = _get_reranker()
    pairs = [(question, c.text) for c in chunks]
    scores = model.predict(pairs)
    for c, s in zip(chunks, scores, strict=True):
        c.score = float(s)
    chunks.sort(key=lambda c: c.score, reverse=True)
    return chunks[:TOP_K_RERANK]


def answer(question: str, chunks: list[Chunk]) -> str:
    context = "\n\n".join(
        f"[{i + 1}] (source: {c.source}, page {c.page})\n{c.text}" for i, c in enumerate(chunks)
    )
    prompt = (
        "Answer the question using only the sources below. Cite sources inline like [1], [2]. "
        "If the answer isn't in the sources, say so.\n\n"
        f"Sources:\n{context}\n\nQuestion: {question}"
    )
    client = Anthropic()
    msg = client.messages.create(
        model=LLM_MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    for block in msg.content:
        if isinstance(block, TextBlock):
            return str(block.text)
    raise RuntimeError(f"no text block in response (stop_reason={msg.stop_reason!r})")


def ask(question: str, rerank_on: bool = True, top_k_retrieve: int = TOP_K_RETRIEVE) -> str:
    with connect(os.environ["DATABASE_URL"]) as conn:
        register_vector(conn)
        chunks = retrieve(question, conn, top_k=top_k_retrieve)
    chunks = rerank(question, chunks) if rerank_on else chunks[:TOP_K_RERANK]
    return answer(question, chunks)


if __name__ == "__main__":
    print(ask(sys.argv[1] if len(sys.argv) > 1 else "What is this document about?"))

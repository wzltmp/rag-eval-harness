"""Ingest PDFs into pgvector.

Usage:
    python -m src.ingest data/pdfs/
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pgvector.psycopg import register_vector
from psycopg import connect
from pypdf import PdfReader

load_dotenv()

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def init_db() -> None:
    with connect(os.environ["DATABASE_URL"]) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id SERIAL PRIMARY KEY,
                source TEXT NOT NULL,
                page INT,
                text TEXT NOT NULL,
                embedding vector({EMBED_DIM})
            );
            """
        )
        conn.commit()


def read_pdf(path: Path) -> list[tuple[int, str]]:
    """Returns [(page_num, page_text), ...]."""
    reader = PdfReader(str(path))
    return [(i, (p.extract_text() or "")) for i, p in enumerate(reader.pages)]


def read_text(path: Path) -> list[tuple[int, str]]:
    """Returns [(0, full_text)] — no pagination for plain text."""
    return [(0, path.read_text(encoding="utf-8", errors="ignore"))]


def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_text(text)


def embed(texts: list[str], client: OpenAI) -> list[list[float]]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def iter_docs(folder: Path):
    for path in sorted(folder.glob("**/*")):
        if path.suffix.lower() == ".pdf":
            yield path, read_pdf(path)
        elif path.suffix.lower() in (".txt", ".md"):
            yield path, read_text(path)


def ingest_folder(folder: Path, *, reset: bool = False) -> None:
    init_db()
    if reset:
        with connect(os.environ["DATABASE_URL"]) as conn:
            conn.execute("TRUNCATE chunks RESTART IDENTITY")
            conn.commit()

    client = OpenAI()

    with connect(os.environ["DATABASE_URL"]) as conn:
        register_vector(conn)

        for path, pages in iter_docs(folder):
            print(f"Ingesting {path.name}")
            for page_num, page_text in pages:
                chunks = chunk_text(page_text)
                if not chunks:
                    continue
                vectors = embed(chunks, client)
                with conn.cursor() as cur:
                    for chunk, vec in zip(chunks, vectors):
                        cur.execute(
                            "INSERT INTO chunks (source, page, text, embedding) VALUES (%s, %s, %s, %s)",
                            (path.name, page_num, chunk, vec),
                        )
                conn.commit()


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    reset = "--reset" in sys.argv
    folder = Path(args[0]) if args else Path("data/pdfs")
    ingest_folder(folder, reset=reset)
    print("Done.")

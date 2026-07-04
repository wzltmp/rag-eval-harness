# Chat-with-Your-Docs RAG

[![CI](https://github.com/wzltmp/rag-eval-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/wzltmp/rag-eval-harness/actions/workflows/ci.yml)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![mypy: strict](https://img.shields.io/badge/mypy-strict-2a6db2.svg)](http://mypy-lang.org/)

A retrieval-augmented chat app over a corpus of Paul Graham essays, with a hand-curated 30-question eval set and an A/B comparison of vector-only retrieval vs. vector + cross-encoder rerank.

> **Live demo:** https://rag-eval-harness.streamlit.app
>
> **Source:** this repo. Built as project 1 of a 4-project AI engineer portfolio.
>
> Stack in production: Streamlit Cloud (chat UI) + Supabase (managed Postgres with pgvector) + OpenAI embeddings + cross-encoder rerank + Anthropic Claude.

## What this project demonstrates

- End-to-end RAG: PDF/text ingest → chunking → embeddings → pgvector → vector search → cross-encoder rerank → LLM answer with inline citations.
- A real, measured eval. 30 hand-written Q&A pairs grounded in specific essays, scored with an LLM-as-judge (Claude Haiku).
- A real finding (see [Results](#results) — rerank didn't help on this corpus, and the reason is interesting).

## Stack

| Layer | Choice |
|---|---|
| Embeddings | OpenAI `text-embedding-3-small` (1536-d) |
| Vector store | Postgres + `pgvector` (cosine distance) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (HuggingFace) |
| LLM | Anthropic Claude (`claude-haiku-4-5` for cost, swap to Sonnet for quality) |
| UI | Streamlit |
| Eval judge | Claude Haiku, prompt-locked to YES/NO |

## Results

Corpus: 28 Paul Graham essays (greatwork, wealth, ramen-profitable, do-things-that-dont-scale, etc.), ingested as 695 chunks of ≤1000 chars with 150-char overlap.

Eval: 30 hand-written Q&A pairs in `data/eval_set.jsonl`. Each scored YES/NO by an LLM judge.

| Config | Retrieval | Rerank | Accuracy | Δ vs. A |
|---|---|---|---|---|
| **A** | top-20 vector | — | **24/30 = 80.0%** | — |
| **B** | top-20 vector | cross-encoder → top-5 | 23/30 = 76.7% | −3.3 pp |
| **C** | top-40 vector | cross-encoder → top-5 | 23/30 = 76.7% | −3.3 pp |

### What I learned

1. **Cross-encoder rerank did not help on this corpus.** With 28 documents and 695 chunks, top-20 vector search already contained the gold chunk for almost every question; the reranker had little to fix and occasionally promoted lexically-similar but less-relevant chunks. The mental model: rerank's value scales with retrieval scope. On a 10k-document corpus where top-20 misses the gold chunk 30% of the time, the same reranker should pay off.

2. **LLM-as-judge variance is non-trivial.** Re-running the same eval gave ±1 question swings (a 3% delta) without any code change. With 30 items, that's noise you can't ignore — meaningful comparisons need either a larger eval set, a stricter judge prompt, multiple judge runs averaged, or a deterministic scoring rule. Building this taught me to budget for eval reliability, not just pipeline quality.

3. **Six questions flip pass↔fail between configs.** They're not a constant 24 — six different items pass A but fail B, and six pass B but fail A. So rerank *is* doing something; it's just not net-positive here. Worth inspecting which questions flip and why for the next iteration.

## File layout

```
01-chat-with-your-docs-rag/
├── README.md
├── requirements.txt
├── docker-compose.yml          # postgres + pgvector on port 5433
├── .env.example
├── .gitignore
├── src/
│   ├── ingest.py               # pdf/.txt → chunks → embeddings → pgvector
│   ├── query.py                # question → retrieve → rerank → answer
│   ├── eval.py                 # LLM-judge scoring over a Q&A set
│   └── chat.py                 # streamlit chat UI
├── scripts/
│   ├── fetch_pg_essays.py      # download ~30 PG essays as .txt
│   └── run_eval.py             # run 3 configs (A/B/C), save eval_results.json
├── data/
│   ├── pdfs/                   # corpus (.txt; gitignored — regen with fetch script)
│   ├── eval_set.jsonl          # 30 hand-curated Q&A pairs (tracked)
│   └── eval_results.json       # latest eval numbers (gitignored)
└── tests/
    └── test_chunking.py        # 5 tests covering the chunker
```

## How to run

### 1. One-time setup

```bash
cd 01-chat-with-your-docs-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then fill in real ANTHROPIC_API_KEY and OPENAI_API_KEY
docker compose up -d  # starts pgvector on localhost:5433
```

### 2. Build the corpus + ingest

```bash
python scripts/fetch_pg_essays.py        # ~28 PG essays into data/pdfs/
python -m src.ingest data/pdfs --reset   # ~695 chunks into pgvector
```

### 3. Try a question

```bash
python -m src.query "What is the difference between maker's schedule and manager's schedule?"
```

### 4. Run the eval

```bash
python scripts/run_eval.py
# writes data/eval_results.json
```

### 5. Run the chat UI

```bash
streamlit run src/chat.py
```

### 6. Run the tests

```bash
pytest tests/ -v
```

## Defensible interview talking points

These are real, reproducible from this repo:

- **"I built a vector + rerank RAG pipeline over 28 PG essays and benchmarked it on a hand-written 30-question eval set. Vector-only got 80%. Adding a cross-encoder reranker didn't help — it actually dropped 3.3 percentage points. I traced it to retrieval recall already being saturated: with a small corpus, top-20 contains the gold chunk almost always, so the reranker just shuffles noise."**
- **"I noticed LLM-as-judge variance on re-runs — about ±3% with no code change. So I treat any single-digit delta as noise unless I average across multiple judge runs."**
- **"I used pgvector instead of Chroma so the vector store is the same Postgres that already holds app state. One DB, fewer moving parts to deploy."**
- **"The chunker is `RecursiveCharacterTextSplitter` with 1000/150. Tests in `tests/test_chunking.py` lock in size bounds, non-empty chunks, and overlap."**

## Next experiments (not yet run)

- Re-test on a 10× larger corpus (e.g., a 300-document set) where retrieval recall *isn't* saturated. Expect rerank to flip positive.
- Replace the LLM judge with a multi-vote majority across 3 runs to cut variance.
- Add a "parent-document" retriever — fetch surrounding chunks for context once retrieval lands the right one.

---

Built as project 1 of a 4-project AI engineer portfolio. See the top-level repo for the full plan.

"""Eval harness. Reads data/eval_set.jsonl with {"question": ..., "expected": ...} per line.

Usage:
    python -m src.eval --rerank
    python -m src.eval --no-rerank
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

from src.query import ask

load_dotenv()

JUDGE_MODEL = "claude-haiku-4-5-20251001"


def judge(question: str, expected: str, actual: str) -> bool:
    """LLM-as-judge. Returns True if the actual answer contains the key facts in expected."""
    client = Anthropic()
    prompt = (
        "You are grading a question-answering system. Return ONLY 'YES' or 'NO'.\n\n"
        f"Question: {question}\n"
        f"Expected key facts: {expected}\n"
        f"Actual answer: {actual}\n\n"
        "Does the actual answer correctly contain the expected key facts? YES or NO."
    )
    msg = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=4,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip().upper().startswith("YES")


def run(rerank: bool, top_k_retrieve: int = 20) -> dict:
    path = Path("data/eval_set.jsonl")
    items = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    correct = 0
    misses: list[str] = []
    for item in items:
        actual = ask(item["question"], rerank_on=rerank, top_k_retrieve=top_k_retrieve)
        ok = judge(item["question"], item["expected"], actual)
        correct += int(ok)
        marker = "PASS" if ok else "FAIL"
        print(f"  [{marker}] {item['question'][:70]}")
        if not ok:
            misses.append(item["question"])
    total = len(items)
    acc = correct / total
    print(f"\nAccuracy: {correct}/{total} = {acc:.1%}  (rerank={rerank}, top_k_retrieve={top_k_retrieve})")
    return {
        "correct": correct,
        "total": total,
        "accuracy": acc,
        "misses": misses,
        "rerank": rerank,
        "top_k_retrieve": top_k_retrieve,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rerank", action="store_true", default=True)
    p.add_argument("--no-rerank", dest="rerank", action="store_false")
    args = p.parse_args()
    run(args.rerank)

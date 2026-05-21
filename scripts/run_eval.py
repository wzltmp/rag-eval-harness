"""Run the eval set twice — once without reranker, once with — and save results.

Usage:
    python scripts/run_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from src.eval import run

load_dotenv()


def main() -> None:
    print("\n=== Config A: no rerank, top_k=20 ===")
    no_rerank = run(rerank=False, top_k_retrieve=20)
    print("\n=== Config B: rerank, top_k=20 ===")
    with_rerank_20 = run(rerank=True, top_k_retrieve=20)
    print("\n=== Config C: rerank, top_k=40 ===")
    with_rerank_40 = run(rerank=True, top_k_retrieve=40)

    out = {
        "no_rerank_k20": no_rerank,
        "rerank_k20": with_rerank_20,
        "rerank_k40": with_rerank_40,
        "delta_rerank_pp": (with_rerank_20["accuracy"] - no_rerank["accuracy"]) * 100,
        "delta_wider_pp": (with_rerank_40["accuracy"] - no_rerank["accuracy"]) * 100,
    }
    Path("data").mkdir(exist_ok=True)
    Path("data/eval_results.json").write_text(json.dumps(out, indent=2))
    print("\n=== Summary ===")
    print(f"  A. no rerank,   k=20: {no_rerank['correct']}/{no_rerank['total']} = {no_rerank['accuracy']:.1%}")
    print(f"  B. with rerank, k=20: {with_rerank_20['correct']}/{with_rerank_20['total']} = {with_rerank_20['accuracy']:.1%}  (delta vs A: {out['delta_rerank_pp']:+.1f} pp)")
    print(f"  C. with rerank, k=40: {with_rerank_40['correct']}/{with_rerank_40['total']} = {with_rerank_40['accuracy']:.1%}  (delta vs A: {out['delta_wider_pp']:+.1f} pp)")
    print("\nSaved to data/eval_results.json")


if __name__ == "__main__":
    main()

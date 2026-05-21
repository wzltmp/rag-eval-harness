"""Fetch ~30 well-known Paul Graham essays as .txt files.

Saves to data/pdfs/<slug>.txt — the existing folder name is kept for simplicity
even though we're using .txt now. ingest.py reads both .pdf and .txt.

Usage:
    python scripts/fetch_pg_essays.py
"""
from __future__ import annotations

import html
import re
import sys
import time
import urllib.request
from pathlib import Path

# Curated list of well-known PG essays. URLs are stable on paulgraham.com.
ESSAYS = [
    "greatwork.html",
    "do.html",
    "hwh.html",
    "wealth.html",
    "ds.html",
    "makersschedule.html",
    "startupideas.html",
    "really.html",
    "founders.html",
    "good.html",
    "hp.html",
    "growth.html",
    "ramenprofitable.html",
    "startuplessons.html",
    "disagree.html",
    "boss.html",
    "todo.html",
    "before.html",
    "buz.html",
    "noob.html",
    "smart.html",
    "early.html",
    "founders2.html",
    "schlep.html",
    "ambitious.html",
    "fix.html",
    "love.html",
    "hubs.html",
    "convince.html",
    "vb.html",
]

BASE = "https://paulgraham.com/"
OUT_DIR = Path("data/pdfs")


def strip_html(raw: str) -> str:
    # PG essays are largely plain text inside a <font> tag with <br><br> paragraph breaks.
    # Drop scripts / styles, then normalize tags to spaces / newlines, then unescape entities.
    raw = re.sub(r"<script[\s\S]*?</script>", "", raw, flags=re.I)
    raw = re.sub(r"<style[\s\S]*?</style>", "", raw, flags=re.I)
    raw = re.sub(r"<br\s*/?>", "\n", raw, flags=re.I)
    raw = re.sub(r"</p>", "\n\n", raw, flags=re.I)
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = html.unescape(raw)
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"\n[ \t]+", "\n", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()


def fetch(slug: str) -> str | None:
    url = BASE + slug
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (portfolio-rag-demo)"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.read().decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"  ! {slug}: {e}", file=sys.stderr)
        return None


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0
    for slug in ESSAYS:
        out = OUT_DIR / slug.replace(".html", ".txt")
        if out.exists() and out.stat().st_size > 1000:
            print(f"  = {slug} (cached)")
            saved += 1
            continue
        raw = fetch(slug)
        if not raw:
            continue
        text = strip_html(raw)
        if len(text) < 500:
            print(f"  ! {slug}: too short ({len(text)} chars), skipping")
            continue
        out.write_text(text, encoding="utf-8")
        print(f"  + {slug} ({len(text):,} chars)")
        saved += 1
        time.sleep(0.5)  # be polite

    print(f"\nSaved {saved}/{len(ESSAYS)} essays to {OUT_DIR}/")


if __name__ == "__main__":
    main()

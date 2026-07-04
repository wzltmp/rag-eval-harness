"""Tests for the chunker in src.ingest."""
from __future__ import annotations

from itertools import pairwise

from src.ingest import CHUNK_OVERLAP, CHUNK_SIZE, chunk_text


def test_short_text_produces_one_chunk():
    text = "This is a short paragraph. " * 5  # ~135 chars, well under CHUNK_SIZE
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0].strip()


def test_long_text_produces_multiple_chunks():
    text = "Lorem ipsum dolor sit amet. " * 200  # ~5,600 chars > 5 x CHUNK_SIZE
    chunks = chunk_text(text)
    assert len(chunks) > 4


def test_chunks_respect_size_bound():
    text = "Sentence number one. " * 500
    chunks = chunk_text(text)
    for c in chunks:
        assert len(c) <= CHUNK_SIZE + CHUNK_OVERLAP, f"chunk too large: {len(c)}"


def test_no_empty_chunks():
    text = "Real content here. " * 100
    chunks = chunk_text(text)
    assert all(c.strip() for c in chunks)


def test_consecutive_chunks_overlap():
    """RecursiveCharacterTextSplitter aims for overlap; verify some shared substring exists."""
    text = ("Paragraph A. " * 100) + ("Paragraph B. " * 100)
    chunks = chunk_text(text)
    if len(chunks) < 2:
        return  # only assertable when we actually got multiple chunks
    # Check that at least one adjacent pair shares a non-trivial substring (overlap).
    found = False
    for a, b in pairwise(chunks):
        tail = a[-min(len(a), 40):]
        if tail.strip() and tail.strip() in b:
            found = True
            break
    assert found, "expected consecutive chunks to share overlapping text"

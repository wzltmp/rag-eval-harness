"""Streamlit chat UI.

Usage:
    streamlit run src/chat.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.query import _get_reranker, ask

st.set_page_config(page_title="Chat with your docs", page_icon=":books:")
st.title("Chat with your docs")
st.caption("RAG over 28 Paul Graham essays. pgvector + cross-encoder rerank + Claude Haiku.")


@st.cache_resource
def _warm_reranker():
    return _get_reranker()


with st.spinner("Loading reranker model (first boot only)..."):
    _warm_reranker()

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if q := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            a = ask(q, rerank_on=True)
        st.markdown(a)
    st.session_state.messages.append({"role": "assistant", "content": a})

"""Streamlit chat UI.

Usage:
    streamlit run src/chat.py
"""
from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

from src.query import ask

load_dotenv()

st.set_page_config(page_title="Chat with your docs", page_icon=":books:")
st.title("Chat with your docs")

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

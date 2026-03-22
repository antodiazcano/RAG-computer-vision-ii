"""
Streamlit frontend for the RAG Knowledge Base.
"""

import os
import sys

import streamlit as st
from google import genai
from pinecone.db_data.index import Index


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.chatbot import generate_answer
from src.config import config
from src.save_docs_to_db import save_all_files
from src.utils import file_hash, get_gen_ai_client, get_index_vector_db, load_registry


# Cached resources


@st.cache_resource
def get_genai_client() -> genai.Client:
    """
    Caches the chatbot client.

    Returns:
        Chatbot client.
    """

    return get_gen_ai_client()


@st.cache_resource
def get_pinecone_index() -> Index:
    """
    Caches the pinecone index.

    Returns:
        Pinecone Index.
    """

    return get_index_vector_db()


GEN_AI_CLIENT = get_genai_client()
PINECONE_IDX = get_pinecone_index()


# Start of the page

st.set_page_config(
    page_title="Computer Vision II Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* ── global ── */
    [data-testid="stAppViewContainer"] { background: #080c10; }
    [data-testid="stSidebar"]          { background: #0d1117; border-right: 1px solid
                                         #30363d; }
    .block-container                   { padding-top: 1.5rem; padding-bottom: 1rem; }

    /* ── typography ── */
    html, body, [class*="css"]  { font-family: 'IBM Plex Sans', sans-serif; color:
                                  #e6edf3; }
    h1, h2, h3                  { font-family: 'IBM Plex Mono', monospace !important; }

    /* ── chat bubbles ── */
    .bubble-user {
        background: rgba(124,106,247,.12);
        border: 1px solid rgba(124,106,247,.30);
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 6px;
        color: #e6edf3;
    }
    .bubble-bot {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 6px;
        color: #e6edf3;
        line-height: 1.65;
    }

    /* ── source card header ── */
    .src-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        color: #00e5ff;
        background: rgba(0,229,255,.06);
        border: 1px solid rgba(0,229,255,.25);
        border-radius: 4px;
        padding: 4px 10px;
        margin-bottom: 6px;
        display: inline-block;
    }

    /* ── pill badges ── */
    .badge {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 10px;
        padding: 2px 8px;
        border-radius: 3px;
        border: 1px solid;
        letter-spacing: .04em;
        display: inline-block;
        margin: 2px 3px;
    }
    .badge-embed { color:#00e5ff; border-color:rgba(0,229,255,.3);
                   background:rgba(0,229,255,.06); }
    .badge-llm   { color:#7c6af7; border-color:rgba(124,106,247,.3);
                   background:rgba(124,106,247,.06); }
    .badge-pine  { color:#3fb950; border-color:rgba(63,185,80,.3);
                   background:rgba(63,185,80,.06); }

    /* hide Streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# Create memory for the chatbot

if "messages" not in st.session_state:
    st.session_state.messages = []
    # List of {"role": "user" | "assistant", "content": str, "sources": list}


# ======================================================================================
# ====================================== Sidebar =======================================
# ======================================================================================


with st.sidebar:
    st.markdown(
        """
        <div style="padding:4px 0 16px">
          <span style="font-family:'IBM Plex Mono',monospace;font-size:15px;
                       font-weight:700;color:#00e5ff;letter-spacing:.08em">
            RAG · Knowledge Base
          </span>
        </div>
        <div style="margin-bottom:14px">
          <span class="badge badge-embed">gemini-embedding-2-preview</span>
          <span class="badge badge-llm">gemini-2.5-flash</span>
          <span class="badge badge-pine">Pinecone</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Number of documents and number of indexed documents in the db
    st.markdown(
        "<p style='font-family:IBM Plex Mono,monospace;font-size:11px;"
        "color:#8b949e;text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px'>"
        "Corpus</p>",
        unsafe_allow_html=True,
    )
    all_files = [
        p
        for p in config.paths.documents_folder.iterdir()
        if p.suffix.lower() in config.paths.supported_extensions
    ]
    reg = load_registry()
    indexed_count = sum(1 for p in all_files if reg.get(p.name) == file_hash(p))
    col1, col2 = st.columns(2)
    col1.metric("Total documents", len(all_files))
    col2.metric("Indexed", indexed_count)

    # List of documents
    if all_files:
        with st.expander("📄 Documents", expanded=True):
            for p in sorted(all_files):
                h = file_hash(p)
                indexed = reg.get(p.name) == h
                dot = "🟢" if indexed else "🟡"
                kb = round(p.stat().st_size / 1024, 1)
                st.markdown(
                    f"{dot} `{p.name}` <span style='color:#484f58;font-size:11px'>{kb} "
                    f"KB</span>",
                    unsafe_allow_html=True,
                )
    else:
        st.info("No documents found. Upload some below.")

    st.divider()

    # Upload documents
    st.markdown(
        "<p style='font-family:IBM Plex Mono,monospace;font-size:11px;"
        "color:#8b949e;text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px'>"
        "Upload</p>",
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Drop PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded:
        for uf in uploaded:
            dest = config.paths.documents_folder / uf.name
            dest.write_bytes(uf.read())
        st.success(f"Saved {len(uploaded)} file(s). Click **Index** below.")
        st.rerun()

    st.divider()

    # Index documents
    st.markdown(
        "<p style='font-family:IBM Plex Mono,monospace;font-size:11px;"
        "color:#8b949e;text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px'>"
        "Indexing</p>",
        unsafe_allow_html=True,
    )
    col_a, col_b = st.columns([3, 1])
    run_index = col_a.button("⚙ Index new docs", use_container_width=True)
    if run_index:
        with st.spinner("Indexing... this may take a while for large PDFs."):
            save_all_files()
        st.rerun()

    # Chat controls
    st.divider()
    if st.button("🗑 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    top_k = st.slider("Sources per answer (top-k)", 1, 10, 5)


# ======================================================================================
# =================================== Main chat area ===================================
# ======================================================================================


st.markdown(
    "<h2 style='margin-bottom:4px'>Knowledge Base Chat</h2>"
    "<p style='color:#8b949e;font-size:13px;margin-bottom:20px'>"
    "Every answer includes the exact source page rendered from the original PDF.</p>",
    unsafe_allow_html=True,
)

# Render conversation history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f"<div class='bubble-user'>🧑 {msg['content']}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='bubble-bot'>{msg['content']}</div>", unsafe_allow_html=True
        )

        sources = msg.get("sources", [])
        if sources:
            st.markdown(
                f"<p style='font-family:IBM Plex Mono,monospace;font-size:10px;"
                f"color:#484f58;text-transform:uppercase;letter-spacing:.12em;"
                f"margin:12px 0 8px'>Sources · {len(sources)} reference(s)</p>",
                unsafe_allow_html=True,
            )
            cols = st.columns(min(len(sources), 4))
            for col, src in zip(cols, sources):
                with col:
                    col.markdown(
                        "<div style='background:#1c2330;border:1px solid #30363d;"
                        "border-radius:4px;padding:20px;text-align:center;"
                        "color:#484f58;font-family:IBM Plex Mono,monospace;font-size:"
                        "10px'>No preview</div>",
                        unsafe_allow_html=True,
                    )
                col.markdown(
                    f"<div class='src-header'>{src['source']}<br>"
                    f"Pag. {src['page']} / {src['total_pages']} "
                    f"<span style='color:#484f58'>· {round(src['score']*100)}%</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

# Welcome hint when chat is empty
if not st.session_state.messages:
    st.markdown(
        """
        <div style="text-align:center;padding:60px 20px;color:#484f58">
          <div style="font-size:48px;margin-bottom:16px">🔍</div>
          <p style="font-family:'IBM Plex Mono',monospace;font-size:14px;color:#8b949e">
            Index your documents and start asking questions
          </p>
          <p style="font-size:12px;margin-top:8px">
            Supported: PDF · Each answer shows the rendered source page
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Chat input

question = st.chat_input("Ask something about your documents!")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question, "sources": []}
    )
    st.markdown(f"<div class='bubble-user'>🧑 {question}</div>", unsafe_allow_html=True)

    with st.spinner("Retrieving & generating..."):
        answer, chunks = generate_answer(question)
        if not chunks:
            sources = []
        else:
            # Deduplicate by (source, page), keep highest score
            seen: dict[tuple, dict] = {}
            for c in chunks:
                key = (c["source"], c["page"])
                if key not in seen or c["score"] > seen[key]["score"]:
                    seen[key] = c
            sources = sorted(seen.values(), key=lambda x: -x["score"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )
    st.rerun()

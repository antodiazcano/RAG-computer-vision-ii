"""
Streamlit frontend for the RAG Knowledge Base.
"""

import json
import os
import sys

# Set LangSmith env vars before any LangChain imports
# Streamlit Cloud exposes secrets via st.secrets, not os.environ
try:
    import streamlit as st

    for key in (
        "LANGSMITH_API_KEY",
        "LANGSMITH_TRACING",
        "LANGSMITH_PROJECT",
        "LANGSMITH_ENDPOINT",
    ):
        if key not in os.environ and key in st.secrets:
            os.environ[key] = st.secrets[key]
except Exception:
    import streamlit as st

from pinecone.db_data.index import Index


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.chatbot.clients import CHAT_CLIENTS, chat, create_chat_client
from src.chatbot.rag import RAGTool
from src.config import config
from src.utils import get_embedding_client, get_index_vector_db


# Cached resources


@st.cache_resource
def get_pinecone_index() -> Index:
    """
    Caches the pinecone index.

    Returns:
        Pinecone Index.
    """

    return get_index_vector_db()


@st.cache_resource
def get_cached_embedding_client():
    """
    Caches the Gemini embedding client.
    """

    return get_embedding_client()


PINECONE_IDX = get_pinecone_index()
EMBEDDING_CLIENT = get_cached_embedding_client()


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
    .bubble-user, .bubble-bot {
        display: flex;
        gap: 10px;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 6px;
        color: #e6edf3;
    }
    .bubble-user {
        background: rgba(124,106,247,.12);
        border: 1px solid rgba(124,106,247,.30);
    }
    .bubble-bot {
        background: #161b22;
        border: 1px solid #30363d;
        line-height: 1.65;
    }
    .bubble-icon {
        flex-shrink: 0;
        width: 20px;
        height: 20px;
        font-size: 20px;
        line-height: 20px;
        text-align: center;
        margin-top: 2px;
    }
    .bubble-content {
        flex: 1;
        min-width: 0;
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
    #MainMenu, footer { visibility: hidden; }
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
            RAG - Computer Vision II
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Provider and API key
    st.markdown(
        "<p style='font-family:IBM Plex Mono,monospace;font-size:11px;"
        "color:#8b949e;text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px'>"
        "Chat Model</p>",
        unsafe_allow_html=True,
    )
    provider = st.selectbox(
        "Provider",
        options=list(CHAT_CLIENTS.keys()),
        index=list(CHAT_CLIENTS.keys()).index("Groq"),
        label_visibility="collapsed",
    )
    use_default_groq = provider == "Groq"
    api_key = st.text_input(
        "API Key",
        type="password",
        label_visibility="collapsed",
        placeholder=(
            "Using free Llama model"
            if use_default_groq
            else f"Paste your {provider} API key"
        ),
        disabled=use_default_groq,
    )
    if use_default_groq:
        api_key = config.chat_model.groq_api_key  # type: ignore

    chat_model = config.chat_model.providers[provider]
    st.markdown(
        f"<div style='margin:14px 0'>"
        f"<span style='color:#8b949e;font-size:10px'>Embedding Model:</span> "
        f"<span class='badge badge-embed'>gemini-embedding-2-preview</span><br>"
        f"<span style='color:#8b949e;font-size:10px'>Chatbot Model:</span> "
        f"<span class='badge badge-llm'>{chat_model}</span><br>"
        f"<span style='color:#8b949e;font-size:10px'>Vector DB:</span> "
        f"<span class='badge badge-pine'>Pinecone</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # Corpus index (table of contents)
    if config.paths.corpus_index_path.exists():
        try:
            corpus = json.loads(
                config.paths.corpus_index_path.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, FileNotFoundError):
            corpus = []

        if corpus:
            st.markdown(
                "<div style='text-align:center;margin-bottom:12px'>"
                "<span style='font-size:32px'>📚</span><br>"
                "<span style='font-family:IBM Plex Mono,monospace;"
                "font-size:16px;font-weight:700;color:#00e5ff;"
                "letter-spacing:.08em'>Table of Contents</span>"
                "</div>",
                unsafe_allow_html=True,
            )
            chapters = [
                entry
                for entry in corpus
                if any(h["level"] == "chapter" for h in entry.get("toc", []))
            ]
            for entry in chapters:
                toc = entry["toc"]
                chapter_items = [h for h in toc if h["level"] == "chapter"]
                if not chapter_items:
                    continue
                chapter_name = chapter_items[0]["name"]
                chapter_id = chapter_items[0]["id"]

                sections = [h for h in toc if h["level"] == "section"]
                with st.expander(f"📖  {chapter_id}. {chapter_name}", expanded=False):
                    for sec in sections:
                        subsections = [
                            h
                            for h in toc
                            if h["level"] == "subsection"
                            and h["id"].startswith(sec["id"] + ".")
                        ]
                        st.markdown(
                            f"<div style='color:#e6edf3;font-size:12px;"
                            f"font-weight:600;margin:8px 0 4px'>"
                            f"§ {sec['id']}  {sec['name']}</div>",
                            unsafe_allow_html=True,
                        )
                        for sub in subsections:
                            st.markdown(
                                f"<div style='color:#8b949e;font-size:11px;"
                                f"margin-left:16px;line-height:1.8'>"
                                f"§ {sub['id']}  {sub['name']}</div>",
                                unsafe_allow_html=True,
                            )

            # PDFs without TOC
            pdf_entries = [entry for entry in corpus if not entry.get("toc")]
            if pdf_entries:
                for entry in pdf_entries:
                    st.markdown(
                        f"<div style='color:#8b949e;font-size:11px;"
                        f"margin:4px 0'>📄 {entry['source']}</div>",
                        unsafe_allow_html=True,
                    )

    st.divider()

    if st.button("🗑 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    top_k = st.slider("Sources per answer (top-k)", 1, 5, 3)


# ======================================================================================
# =================================== Main chat area ===================================
# ======================================================================================


st.markdown(
    "<h2 style='margin-bottom:4px'>Computer Vision II Assistant 🤖</h2>"
    "<p style='color:#8b949e;font-size:13px;margin-bottom:20px'>"
    "Every answer includes the exact source page rendered from the original PDF.</p>",
    unsafe_allow_html=True,
)

# Render conversation history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f"<div class='bubble-user'>"
            f"<span class='bubble-icon'>🧑</span>"
            f"<span class='bubble-content'>{msg['content']}</span></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='bubble-bot'>"
            f"<span class='bubble-icon'>🤖</span>"
            f"<span class='bubble-content'>{msg['content']}</span></div>",
            unsafe_allow_html=True,
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
                score_pct = round(src["score"] * 100)
                bar_color = (
                    "#3fb950"
                    if score_pct >= 75
                    else "#d29922" if score_pct >= 50 else "#f85149"
                )
                col.markdown(
                    f"<div style='background:#1c2330;border:1px solid #30363d;"
                    f"border-radius:6px;padding:14px 12px;text-align:center'>"
                    f"<div style='font-size:28px;margin-bottom:6px'>📄</div>"
                    f"<div style='font-family:IBM Plex Mono,monospace;font-size:11px;"
                    f"color:#e6edf3;white-space:nowrap;overflow:hidden;"
                    f"text-overflow:ellipsis'>{src['source']}</div>"
                    f"<div style='color:#8b949e;font-size:10px;margin:4px 0 8px'>"
                    f"{'§ ' + str(src['location']) if src.get('doc_type') == 'tex'
                       else 'Page ' + str(src['location']) + ' / '
                       + str(src['total_locations'])}</div>"
                    f"<div style='background:#21262d;border-radius:3px;height:6px;"
                    f"overflow:hidden'>"
                    f"<div style='width:{score_pct}%;height:100%;"
                    f"background:{bar_color};border-radius:3px'></div></div>"
                    f"<div style='font-family:IBM Plex Mono,monospace;font-size:10px;"
                    f"color:{bar_color};margin-top:4px'>{score_pct}% match</div>"
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
            Ask something about the Computer Vision II course!
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Chat input

question = st.chat_input("Ask anything")

if question:
    if not api_key:
        st.warning("Please enter your API key in the sidebar.")
    else:
        st.session_state.messages.append(
            {"role": "user", "content": question, "sources": []}
        )
        st.markdown(
            f"<div class='bubble-user'>"
            f"<span class='bubble-icon'>🧑</span>"
            f"<span class='bubble-content'>{question}</span></div>",
            unsafe_allow_html=True,
        )

        with st.spinner("Thinking..."):
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]
            client = create_chat_client(provider, api_key)
            rag_tool = RAGTool(EMBEDDING_CLIENT, PINECONE_IDX, top_k=top_k)
            answer, chunks = chat(
                client, question, rag_tool=rag_tool, chat_history=history
            )
            sources = sorted(chunks, key=lambda x: -x["score"]) if chunks else []

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
            }
        )
        st.rerun()

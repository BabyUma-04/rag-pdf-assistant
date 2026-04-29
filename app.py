"""
app.py — Streamlit UI for RAG PDF Assistant
Run with: streamlit run app.py
"""

import logging
import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

# Add src/ to path so imports work
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_pipeline import RAGPipeline

# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ─────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG PDF Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Import fonts */
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

  /* Root variables */
  :root {
    --ink: #0f0f11;
    --paper: #fafaf8;
    --accent: #e8420a;
    --accent2: #2563eb;
    --muted: #6b7280;
    --border: #e5e5e0;
    --surface: #f5f5f2;
    --radius: 10px;
  }

  /* Global */
  html, body, [data-testid="stAppViewContainer"] {
    background: var(--paper) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--ink);
  }

  /* Hide Streamlit branding */
  #MainMenu, footer, header { visibility: hidden; }

  /* Title */
  .rag-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.2rem;
    letter-spacing: -0.03em;
    line-height: 1.1;
    color: var(--ink);
    margin-bottom: 0;
  }
  .rag-subtitle {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
  }

  /* Status badge */
  .badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.04em;
  }
  .badge-ok  { background: #dcfce7; color: #15803d; }
  .badge-err { background: #fee2e2; color: #b91c1c; }
  .badge-warn{ background: #fef9c3; color: #a16207; }

  /* Chat messages */
  .msg-user {
    background: var(--ink);
    color: white;
    border-radius: var(--radius) var(--radius) 2px var(--radius);
    padding: 14px 18px;
    margin: 8px 0;
    font-size: 0.95rem;
    max-width: 80%;
    margin-left: auto;
  }
  .msg-assistant {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius) var(--radius) var(--radius) 2px;
    padding: 16px 20px;
    margin: 8px 0;
    font-size: 0.95rem;
    max-width: 90%;
    line-height: 1.65;
  }
  .msg-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--muted);
    margin-bottom: 6px;
  }

  /* Citation cards */
  .citation-card {
    background: white;
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent2);
    border-radius: 6px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.83rem;
  }
  .citation-source {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: var(--accent2);
    font-weight: 500;
    margin-bottom: 4px;
  }
  .citation-snippet {
    color: var(--muted);
    font-size: 0.82rem;
    line-height: 1.5;
  }

  /* Stat cards */
  .stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 14px 16px;
    text-align: center;
  }
  .stat-number {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
  }
  .stat-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--muted);
    margin-top: 4px;
  }

  /* Divider */
  hr { border: none; border-top: 1px solid var(--border); margin: 20px 0; }

  /* Buttons */
  .stButton > button {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
  }

  /* Input */
  .stTextInput input, .stTextArea textarea {
    font-family: 'DM Sans', sans-serif !important;
    border-radius: 8px !important;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: var(--ink) !important;
  }
  [data-testid="stSidebar"] * {
    color: #e5e5e0 !important;
  }
  [data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    color: white !important;
  }

  /* Expander */
  .streamlit-expanderHeader {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Session State Init
# ─────────────────────────────────────────────────────────────
if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []  # {role, content, citations}

if "pdf_indexed" not in st.session_state:
    st.session_state.pdf_indexed = False

if "pdf_stats" not in st.session_state:
    st.session_state.pdf_stats = []


pipeline: RAGPipeline = st.session_state.pipeline


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📚 RAG PDF Assistant")
    st.markdown('<p style="font-size:0.75rem;color:#9ca3af;font-family:\'DM Mono\',monospace">LOCAL · PRIVATE · FREE</p>', unsafe_allow_html=True)
    st.markdown("---")

    # System Status
    st.markdown("**System Status**")
    status = RAGPipeline.get_system_status()

    if status["ollama_running"]:
        st.markdown('<span class="badge badge-ok">✓ Ollama Running</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-err">✗ Ollama Offline</span>', unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:0.75rem;color:#f87171;margin-top:6px">Run: <code>ollama serve</code></p>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Model selector
    st.markdown("**LLM Model**")
    available_models = status.get("available_models", [])
    if available_models:
        selected_model = st.selectbox(
            "Select model",
            available_models,
            index=0,
            label_visibility="collapsed",
        )
        pipeline.current_model = selected_model
    else:
        st.markdown(
            '<p style="font-size:0.78rem;color:#f87171">No models found.<br>'
            'Run: <code>ollama pull llama3</code></p>',
            unsafe_allow_html=True
        )
        selected_model = "llama3"

    st.markdown("---")

    # Retrieval settings
    st.markdown("**Retrieval Settings**")
    top_k = st.slider("Top-K chunks", min_value=1, max_value=10, value=5, step=1)
    st.markdown(
        f'<p style="font-size:0.72rem;color:#9ca3af">Retrieve {top_k} most relevant passage(s)</p>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    # Indexed documents info
    if st.session_state.pdf_indexed and pipeline.indexed_sources:
        st.markdown("**Indexed Documents**")
        for src in pipeline.indexed_sources:
            st.markdown(f'<p style="font-size:0.78rem;color:#a7f3d0">📄 {src}</p>', unsafe_allow_html=True)
        st.markdown("---")

    # Actions
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            pipeline.clear_history()
            st.rerun()
    with col2:
        if st.button("↺ Reset All", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pdf_indexed = False
            st.session_state.pdf_stats = []
            pipeline.reset()
            st.rerun()


# ─────────────────────────────────────────────────────────────
# Main Content
# ─────────────────────────────────────────────────────────────
st.markdown('<h1 class="rag-title">PDF Question & Answer</h1>', unsafe_allow_html=True)
st.markdown('<p class="rag-subtitle">Powered by Ollama · FAISS · HuggingFace · LangChain</p>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Upload Section ──────────────────────────────────────────
if not st.session_state.pdf_indexed:
    st.markdown("#### Step 1 — Upload your PDF(s)")
    uploaded_files = st.file_uploader(
        "Drop PDF files here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} file(s) selected**")
        for f in uploaded_files:
            size_mb = f.size / (1024 * 1024)
            st.markdown(
                f'<span style="font-size:0.83rem;color:#6b7280">📄 {f.name} — {size_mb:.1f} MB</span>',
                unsafe_allow_html=True
            )

        if st.button("🚀 Index Documents", type="primary", use_container_width=True):
            if not status["ollama_running"]:
                st.error("⚠️ Ollama is not running. Start it with `ollama serve`.")
            else:
                # Save uploaded files to temp dir
                tmp_paths = []
                with tempfile.TemporaryDirectory() as tmpdir:
                    for uf in uploaded_files:
                        tmp_path = os.path.join(tmpdir, uf.name)
                        with open(tmp_path, "wb") as f:
                            f.write(uf.getbuffer())
                        tmp_paths.append(tmp_path)

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def update_progress(msg, pct):
                        progress_bar.progress(pct)
                        status_text.markdown(
                            f'<p style="font-size:0.85rem;color:#6b7280">{msg}</p>',
                            unsafe_allow_html=True
                        )

                    success, message = pipeline.index_pdfs(
                        tmp_paths,
                        progress_callback=update_progress,
                    )

                if success:
                    st.session_state.pdf_indexed = True
                    st.session_state.pdf_stats = pipeline.pdf_stats
                    progress_bar.progress(1.0)
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
else:
    # ── Stats Row ─────────────────────────────────────────────
    stats = st.session_state.pdf_stats
    total_pages = sum(s.get("total_pages", 0) for s in stats if "total_pages" in s)
    total_chunks = sum(s.get("total_chunks", 0) for s in stats if "total_chunks" in s)
    total_docs = len([s for s in stats if "error" not in s])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{total_docs}</div><div class="stat-label">Documents</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{total_pages}</div><div class="stat-label">Pages</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{total_chunks}</div><div class="stat-label">Chunks</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{top_k}</div><div class="stat-label">Top-K</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Step 2 — Ask anything about your documents")

    # ── Chat History ─────────────────────────────────────────
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="msg-user"><div class="msg-label">You</div>{msg["content"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="msg-assistant"><div class="msg-label">Assistant · {pipeline.current_model}</div>{msg["content"]}</div>',
                    unsafe_allow_html=True
                )
                # Show citations if present
                if msg.get("citations"):
                    with st.expander(f"📎 {len(msg['citations'])} source(s) used", expanded=False):
                        for cite in msg["citations"]:
                            st.markdown(
                                f'<div class="citation-card">'
                                f'<div class="citation-source">Source {cite["index"]}: {cite["source"]} · Page {cite["page"]}</div>'
                                f'<div class="citation-snippet">{cite["snippet"]}</div>'
                                f'</div>',
                                unsafe_allow_html=True
                            )

    # ── Input ─────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.form("question_form", clear_on_submit=True):
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            user_question = st.text_input(
                "Ask a question",
                placeholder="e.g. What are the main conclusions of this paper?",
                label_visibility="collapsed",
            )
        with col_btn:
            submitted = st.form_submit_button("Ask →", type="primary", use_container_width=True)

    if submitted and user_question.strip():
        if not status["ollama_running"]:
            st.error("⚠️ Ollama is not running. Please start it with `ollama serve`.")
        else:
            # Add user message to history
            st.session_state.messages.append({
                "role": "user",
                "content": user_question.strip(),
                "citations": [],
            })

            # Generate answer with streaming
            with st.spinner("Thinking..."):
                answer_placeholder = st.empty()
                full_answer = []
                citations = []

                try:
                    # Retrieve chunks first for citations
                    from vector_store import format_retrieved_context, retrieve_relevant_chunks
                    docs = retrieve_relevant_chunks(pipeline.vectorstore, user_question, top_k=top_k)
                    context, citations = format_retrieved_context(docs)

                    # Stream the answer
                    from llm_handler import generate_answer_streaming
                    for token in generate_answer_streaming(context, user_question, pipeline.current_model):
                        full_answer.append(token)
                        answer_placeholder.markdown(
                            f'<div class="msg-assistant"><div class="msg-label">Assistant · {pipeline.current_model}</div>{"".join(full_answer)}</div>',
                            unsafe_allow_html=True
                        )

                    complete_answer = "".join(full_answer)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": complete_answer,
                        "citations": citations,
                    })

                except Exception as e:
                    st.error(f"❌ Error: {e}")

            st.rerun()

    # ── Example Questions ─────────────────────────────────────
    if not st.session_state.messages:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p style="font-size:0.8rem;color:#9ca3af;font-family:\'DM Mono\',monospace">EXAMPLE QUESTIONS</p>', unsafe_allow_html=True)
        example_qs = [
            "What is the main topic of this document?",
            "Summarize the key findings.",
            "What are the conclusions?",
            "List the main points discussed.",
        ]
        cols = st.columns(2)
        for i, q in enumerate(example_qs):
            with cols[i % 2]:
                if st.button(q, key=f"ex_{i}", use_container_width=True):
                    st.session_state._pending_question = q
                    st.rerun()

    # Handle example question click
    if hasattr(st.session_state, "_pending_question"):
        q = st.session_state._pending_question
        del st.session_state._pending_question
        st.session_state.messages.append({"role": "user", "content": q, "citations": []})
        st.rerun()

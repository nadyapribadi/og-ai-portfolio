import streamlit as st
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from retrieval import (
    ask,
    corpus_sources,
    load_vectorstore,
    rerank_strategy,
    validate_models,
)
from config import (
    APP_TITLE,
    DOCUMENT_SOURCES, SOURCE_FRIENDLY, LLM_MODEL_QUALITY,
    corpus_content,
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Streamlit Cloud stores secrets in st.secrets; the pipeline reads os.environ.
# Bridge the two so the same code runs locally (demo1/.env) and hosted.
try:
    for _key, _value in st.secrets.items():
        os.environ.setdefault(_key, str(_value))
except Exception:
    pass          # no secrets configured — local .env is used instead

# Retrieval works without a key; answering does not. Without this check the app
# renders perfectly and then fails on the first question with a client-library
# message ("the api_key client option must be set") that never says where to set
# it — which is exactly what a visitor to the deploy saw.
HAS_LLM_KEY = bool(os.getenv("GROQ_API_KEY"))

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg-primary: #0d1117;
    --bg-secondary: #161b22;
    --bg-card: #1c2128;
    --bg-card-hover: #21262d;
    --amber: #e6a817;
    --amber-dim: #b8841a;
    --amber-glow: rgba(230, 168, 23, 0.12);
    --amber-border: rgba(230, 168, 23, 0.35);
    --text-primary: #e6edf3;
    --text-secondary: #8b949e;
    --text-muted: #484f58;
    --border: #30363d;
    --border-hover: #484f58;
    --green: #3fb950;
    --red: #f85149;
    --blue: #58a6ff;
    --blue-glow: rgba(88, 166, 255, 0.1);
}

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: var(--bg-primary);
    color: var(--text-primary);
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.block-container {
    padding: 1.5rem 2rem;
    max-width: 920px;
}

/* ── APP HEADER ── */
.app-header {
    border-bottom: 1px solid var(--border);
    padding-bottom: 1rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
}
.app-title {
    font-size: 1.4rem;
    font-weight: 600;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    margin: 0;
}
.app-title span { color: var(--amber); }
.app-subtitle {
    font-size: 0.75rem;
    color: var(--text-secondary);
    font-family: 'IBM Plex Mono', monospace;
    margin-top: 0.25rem;
}
.index-badge {
    font-size: 0.68rem;
    font-family: 'IBM Plex Mono', monospace;
    color: var(--text-muted);
    text-align: right;
    line-height: 1.6;
}
.index-badge span {
    color: var(--green);
    font-weight: 500;
}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background-color: var(--bg-secondary);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .block-container {
    padding: 1.2rem 1rem;
    max-width: 100%;
}

.sidebar-label {
    font-size: 0.63rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.5rem;
    font-family: 'IBM Plex Mono', monospace;
}

.doc-item {
    font-size: 0.72rem;
    color: var(--text-secondary);
    padding: 0.25rem 0 0.25rem 0.6rem;
    border-left: 2px solid var(--border);
    margin-bottom: 0.25rem;
    transition: all 0.12s ease;
}
.doc-item:hover {
    border-left-color: var(--amber);
    color: var(--text-primary);
}

/* Category headers in sidebar */
.cat-header {
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-top: 0.9rem;
    margin-bottom: 0.35rem;
    display: flex;
    align-items: center;
    gap: 0.3rem;
}

/* ── BUTTONS ── */
div[data-testid="stButton"] button {
    background-color: #1c2128 !important;
    color: #8b949e !important;
    border: 1px solid #3d4451 !important;
    border-radius: 4px !important;
    font-size: 0.72rem !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    padding: 0.35rem 0.6rem !important;
    text-align: left !important;
    width: 100% !important;
    transition: all 0.15s ease !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.04) !important;
}
div[data-testid="stButton"] button:hover {
    background-color: var(--amber-glow) !important;
    border-color: var(--amber-border) !important;
    color: var(--text-primary) !important;
    transform: translateX(2px) !important;
}
div[data-testid="stButton"] button:active {
    opacity: 0.7 !important;
    transform: translateX(0px) !important;
}

/* Clear button — distinct style */
.clear-btn div[data-testid="stButton"] button {
    background-color: transparent !important;
    border-color: var(--border) !important;
    color: var(--text-muted) !important;
    font-size: 0.7rem !important;
}
.clear-btn div[data-testid="stButton"] button:hover {
    border-color: var(--red) !important;
    color: var(--red) !important;
    background-color: rgba(248, 81, 73, 0.08) !important;
    transform: none !important;
}

/* ── EMPTY STATE ── */
.empty-state {
    margin-top: 2rem;
}
.empty-state-title {
    font-size: 1rem;
    font-weight: 500;
    color: var(--text-secondary);
    margin-bottom: 1.2rem;
    text-align: center;
}
.capability-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
    margin-bottom: 2rem;
}
.capability-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.9rem 1rem;
    transition: border-color 0.15s ease;
}
.capability-card:hover {
    border-color: var(--border-hover);
}
.capability-card-title {
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.35rem;
}
.capability-card-desc {
    font-size: 0.7rem;
    color: var(--text-secondary);
    line-height: 1.5;
}
.empty-hint {
    text-align: center;
    font-size: 0.72rem;
    color: var(--text-muted);
    font-family: 'IBM Plex Mono', monospace;
    margin-top: 0.5rem;
}

/* ── CHAT MESSAGES ── */
div[data-testid="stChatMessage"] {
    background-color: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    margin-bottom: 0.8rem;
    padding: 0.8rem 1rem;
}

/* ── CHAT INPUT ── */
div[data-testid="stChatInput"] {
    border-top: 1px solid var(--border);
    padding-top: 1rem;
}
div[data-testid="stChatInput"] textarea {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--text-primary) !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.875rem !important;
}
div[data-testid="stChatInput"] textarea:focus {
    border-color: var(--amber-dim) !important;
    box-shadow: 0 0 0 2px var(--amber-glow) !important;
}

/* ── BADGES ── */
.lang-badge {
    display: inline-block;
    font-size: 0.68rem;
    font-family: 'IBM Plex Mono', monospace;
    padding: 0.12rem 0.45rem;
    border-radius: 3px;
    background: var(--amber-glow);
    border: 1px solid var(--amber-border);
    color: var(--amber);
    margin-bottom: 0.5rem;
    margin-right: 0.4rem;
}
.source-tag {
    display: inline-block;
    font-size: 0.65rem;
    font-family: 'IBM Plex Mono', monospace;
    padding: 0.1rem 0.4rem;
    border-radius: 3px;
    background: var(--blue-glow);
    border: 1px solid rgba(88, 166, 255, 0.3);
    color: var(--blue);
    margin-right: 0.3rem;
    margin-top: 0.4rem;
}

/* ── COPY BUTTON ── */
.copy-row {
    margin-top: 0.5rem;
}

/* ── EXPANDER ── */
details {
    background-color: var(--bg-secondary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    margin-top: 0.5rem !important;
}
summary {
    font-size: 0.72rem !important;
    color: var(--text-secondary) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    padding: 0.5rem !important;
    cursor: pointer !important;
}
summary:hover { color: var(--text-primary) !important; }

/* ── POWERED BY ── */
.powered-by {
    font-size: 0.62rem;
    font-family: 'IBM Plex Mono', monospace;
    color: var(--text-muted);
    line-height: 1.8;
    padding-top: 0.5rem;
}
.powered-by span {
    color: var(--text-secondary);
}

/* ── MISC ── */
hr {
    border-color: var(--border) !important;
    margin: 0.75rem 0 !important;
}
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }
div[data-testid="stAlert"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    font-size: 0.8rem !important;
}

/* Loading indicator */
.loading-text {
    font-size: 0.75rem;
    font-family: 'IBM Plex Mono', monospace;
    color: var(--amber);
    margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CONSTANTS — loaded from config.py
# Edit config.py to customize for your documents
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None
if "confirm_clear" not in st.session_state:
    st.session_state.confirm_clear = False


# ─────────────────────────────────────────────
# CACHED RESOURCES
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading document index...")
def get_vectorstore():
    # A fresh clone has no index. Build one from whatever documents exist
    # instead of failing — that is what makes the demo portable across hosts.
    import ingest
    import retrieval

    with st.spinner("Building the document index — first run only, ~1 minute..."):
        rebuilt = ingest.ensure_vectorstore()
    if rebuilt:
        # A rebuild deletes and recreates the directory. Any client opened
        # before that point is now holding a deleted database, so drop it —
        # otherwise the next write fails with SQLITE_READONLY_DBMOVED.
        retrieval._vs_cache.clear()
    return load_vectorstore("en")


@st.cache_resource(show_spinner=False)
def get_corpus_families():
    from retrieval import corpus_families
    return corpus_families()


@st.cache_resource(show_spinner=False)
def get_model_status():
    """Validate model IDs once per app instance, not on every rerun."""
    return validate_models()


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def stream_text(text):
    import time
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.018)

def friendly_source(filename):
    return SOURCE_FRIENDLY.get(filename, filename)

def render_sources(sources):
    html = ""
    for s in sources:
        name = friendly_source(s)
        html += f'<span class="source-tag">📄 {name}</span>'
    return html

def render_chunks_expander(chunks, key_suffix=""):
    with st.expander(f"🔍 View source excerpts · {len(chunks)} retrieved"):
        for i, chunk in enumerate(chunks, 1):
            raw_source = chunk.metadata.get("source_file", "unknown")
            page = chunk.metadata.get("page", "?")
            friendly = friendly_source(raw_source)
            st.markdown(f"**[{i}]** `{friendly}` — Page {page}")
            content = chunk.page_content
            if len(content) > 600:
                content = content[:600] + "..."
            st.code(content, language=None)
            if i < len(chunks):
                st.divider()


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

# Build or load the index before anything else touches it. The sidebar needs to
# know which corpus is indexed in order to offer matching sample questions, and
# opening Chroma before the build would leave a client holding a directory that
# the build then replaces.
try:
    vectorstore = get_vectorstore()
    chunk_count = vectorstore._collection.count()
    # The sidebar lists what is indexed, not what is configured to be indexed:
    # on a host without the copyrighted PDFs the index holds the bundled
    # samples, and claiming the full document pack there would be a lie a
    # visitor can check.
    indexed_sources = corpus_sources()
    listed_sources = (
        [friendly_source(name) for name in indexed_sources]
        if indexed_sources
        else list(DOCUMENT_SOURCES)
    )
    doc_count = len(listed_sources)
except Exception as exc:
    # The index is built on first run, so reaching here means the build itself
    # failed. Show the real traceback instead of a hint that hides it — that is
    # the only way to diagnose this on a host you cannot shell into.
    st.error("**Could not load or build the document index.**")
    st.exception(exc)
    st.stop()

# What the sidebar offers and what the empty state promises must match the
# corpus that is actually indexed — questions, capability cards and subtitle.
question_set, capability_cards, subtitle = corpus_content(get_corpus_families())

with st.sidebar:
    # Branding
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <div style="font-size: 1rem; font-weight: 600; color: #e6edf3; letter-spacing: -0.01em;">
            🛢️ O&G Document Intelligence
        </div>
        <div style="font-size: 0.68rem; color: #8b949e; font-family: 'IBM Plex Mono', monospace; margin-top: 0.2rem;">
            Built by Nadya Boyke Pribadi
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Document sources
    st.markdown('<div class="sidebar-label">📚 Document Sources</div>', unsafe_allow_html=True)
    for name in listed_sources:
        st.markdown(f'<div class="doc-item">{name}</div>', unsafe_allow_html=True)

    st.divider()

    # Sample questions — with visual category separation
    st.markdown('<div class="sidebar-label">💡 Try asking</div>', unsafe_allow_html=True)
    for category, questions in question_set.items():
        st.markdown(f'<div class="cat-header">{category}</div>', unsafe_allow_html=True)
        for q in questions:
            if st.button(q, key=f"btn_{hash(q)}", use_container_width=True):
                st.session_state.pending_question = q
                st.session_state.confirm_clear = False

    st.divider()

    # Clear conversation — with confirmation
    if not st.session_state.confirm_clear:
        st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
        if st.button("🗑️ Clear conversation", use_container_width=True):
            if len(st.session_state.messages) == 0:
                pass  # nothing to clear
            else:
                st.session_state.confirm_clear = True
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Clear all messages?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Yes", use_container_width=True):
                st.session_state.messages = []
                st.session_state.confirm_clear = False
                st.rerun()
        with col2:
            if st.button("❌ No", use_container_width=True):
                st.session_state.confirm_clear = False
                st.rerun()

    st.divider()

    # Powered by
    rerank_label = (
        "mmarco cross-encoder"
        if rerank_strategy() == "cross-encoder"
        else "late interaction · memory-tight host"
    )
    st.markdown(f"""
    <div class="powered-by">
        POWERED BY<br>
        <span>Groq</span> · <span>{LLM_MODEL_QUALITY}</span><br>
        <span>ChromaDB</span> · <span>ONNX Runtime</span><br>
        <span>LangChain</span> · <span>Streamlit</span>
        <br><br>RETRIEVAL<br>
        <span>multilingual-e5-small</span><br>
        <span>{rerank_label}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size: 0.62rem; color: #484f58; margin-top: 0.8rem; line-height: 1.5;">
    ⚠️ Answers may be incomplete. Verify against original IOGP documents.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────

# Verify the configured LLM models still exist before the user asks anything.
# Groq retires model IDs; without this the app dies mid-question with a 404.
if not HAS_LLM_KEY:
    st.error(
        "**No Groq API key configured — the app can search, but it cannot answer.**\n\n"
        "Add `GROQ_API_KEY` and restart:\n\n"
        "- **Streamlit Cloud:** open the app menu → *Settings* → *Secrets* and add\n"
        '  `GROQ_API_KEY = "gsk_..."`\n'
        "- **Local:** copy `demo1_doc_intelligence/.env.example` to "
        "`demo1_doc_intelligence/.env` and fill it in.\n\n"
        "A free key takes a minute: https://console.groq.com/keys"
    )

_missing_models, _available_models = get_model_status()
if _missing_models:
    st.error(
        "**Configured model(s) are no longer available on this account.**\n\n"
        + "\n".join(f"- `{m}`" for m in _missing_models)
        + "\n\n**Available on this account:**\n\n"
        + ", ".join(f"`{m}`" for m in _available_models)
        + "\n\nUpdate `LLM_MODEL_FAST` / `LLM_MODEL_QUALITY` in `config.py` "
          "and restart the app."
    )
    st.stop()

# Load vectorstore
# Header with index stats
st.markdown(f"""
<div class="app-header">
    <div>
        <div class="app-title">🛢️ {APP_TITLE}</div>
        <div class="app-subtitle">{subtitle}</div>
    </div>
    <div class="index-badge">
        <span>●</span> {doc_count} documents<br>
        <span>{chunk_count}</span> passages indexed
    </div>
</div>
""", unsafe_allow_html=True)

# ── EMPTY STATE ──
if not st.session_state.messages:
    cards_html = "".join([
        f"""<div class="capability-card">
                <div class="capability-card-title">{c["title"]}</div>
                <div class="capability-card-desc">{c["desc"]}</div>
            </div>"""
        for c in capability_cards
    ])
    st.markdown(f"""
    <div class="empty-state">
        <div class="empty-state-title">What can I help you with?</div>
        <div class="capability-grid">{cards_html}</div>
        <div class="empty-hint">← Click a sample question or type below to start</div>
    </div>
    """, unsafe_allow_html=True)

# ── CHAT HISTORY ──
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            if msg.get("sources"):
                st.markdown(render_sources(msg["sources"]), unsafe_allow_html=True)

            if msg.get("chunks"):
                render_chunks_expander(msg["chunks"])

            # Copy button
            st.button(
                "📋 Copy answer",
                key=f"copy_{msg.get('id', id(msg))}",
                on_click=lambda m=msg: st.session_state.update({"clipboard": m["content"]}),
                help="Copy answer text to clipboard"
            )

# ── INPUT HANDLING ──
question = st.chat_input(
    "Ask about upstream O&G standards, safety rules, equipment specifications...",
    disabled=not HAS_LLM_KEY,
)

if not question and st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None

if question:
    # Add and display user message
    msg_id = len(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question, "id": msg_id})

    with st.chat_message("user"):
        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        status_placeholder.markdown(
            '<div class="loading-text">⟳ Searching documents...</div>',
            unsafe_allow_html=True
        )

        try:
            result = ask(question)
            status_placeholder.empty()

            # Stream answer
            st.write_stream(stream_text(result["answer"]))

            # Verified quotes — the evidence behind each claim
            if result.get("claims"):
                with st.expander(f"✅ Verified sources · {len(result['claims'])} claims"):
                    for claim in result["claims"]:
                        st.markdown(
                            f"`{claim.source_file}` · "
                            f"§{claim.clause_id or '-'} · p.{claim.page}"
                        )
                        st.code(claim.quote, language=None)

            # "Not found" with rejected claims underneath is the one failure a
            # visitor cannot diagnose from the outside — the excerpts looked
            # right and the answer still was not there. Show what the guardrail
            # threw away, so the next report of this is evidence instead of a
            # guess.
            if result.get("not_found") and result.get("rejected"):
                with st.expander(
                    f"🔎 Why “not found” · {len(result['rejected'])} claim(s) rejected"
                ):
                    st.markdown(
                        "The model proposed these, but they could not be verified "
                        "against the retrieved excerpts:"
                    )
                    for claim, reason in result["rejected"]:
                        st.markdown(
                            f"- `{claim.source_file}` · §{claim.clause_id or '-'} · "
                            f"p.{claim.page} — _{reason}_"
                        )
                        st.code(claim.quote, language=None)

            # Sources — friendly names
            if result["sources"]:
                st.markdown(render_sources(result["sources"]), unsafe_allow_html=True)

            # Chunks expander
            if result.get("chunks"):
                render_chunks_expander(result["chunks"])

            # Save to history
            new_msg_id = len(st.session_state.messages)
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
                "chunks": result["chunks"],
                "id": new_msg_id,
            })

        except Exception as e:
            status_placeholder.empty()
            error_msg = str(e)
            if "429" in error_msg or "rate_limit" in error_msg.lower():
                st.warning(
                    "⏳ **Daily limit reached.**\n\n"
                    "The free Groq tier allows 100,000 tokens/day on the 70B model. "
                    "This resets automatically — please try again in a few minutes.\n\n"
                    "💡 Tip: the daily limit resets in rolling windows, not just at midnight."
                )
            else:
                st.error(
                    f"Something went wrong. Please try again.\n\n"
                    f"Details: {error_msg}"
                )

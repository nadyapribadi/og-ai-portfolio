import streamlit as st
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from retrieval import ask, load_vectorstore

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="O&G Document Intelligence",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
# CONSTANTS
# ─────────────────────────────────────────────
SAMPLE_QUESTIONS = {
    "🦺 HSE Rules": [
        "What are the life saving rules?",
        "What must I confirm before entering a confined space?",
        "What are the hot work requirements in a hazardous area?",
    ],
    "📊 Process Safety": [
        "What is the difference between Tier 1 and Tier 2 process safety events?",
        "How are process safety KPIs measured?",
        "What does LOPC stand for and what are its consequences?",
    ],
    "⚙️ Equipment Standards": [
        "What does IOGP S-737 specify for deluge skid design?",
        "What standards does S-737 reference for electrical installations?",
        "What does IOGP S-717 cover for noise emitting equipment?",
    ],
    "🇮🇩 Bahasa Indonesia": [
        "Apa saja aturan keselamatan jiwa menurut IOGP?",
        "Apa yang harus dilakukan sebelum memasuki ruang tertutup?",
        "Apa perbedaan antara kejadian keselamatan proses Tier 1 dan Tier 2?",
    ],
}

DOCUMENT_SOURCES = [
    "IOGP Report 459 — Life-Saving Rules",
    "IOGP Report 456 — Process Safety KPIs",
    "JIP33 S-737 — Deluge Skids (TRS + QRS)",
    "JIP33 S-717 — Noise Equipment (TRS + QRS)",
    "JIP33 S-719 — Water Mist Fire Protection",
]

# Friendly names for raw filenames shown in source tags
SOURCE_FRIENDLY = {
    "459.pdf": "IOGP 459 Life-Saving Rules",
    "456.pdf": "IOGP 456 Process Safety KPIs",
    "S-737v2026-03 TRS.pdf": "S-737 Deluge Skids (Technical)",
    "S-737Qv2026-03 QRS.pdf": "S-737 Deluge Skids (Quality)",
    "S-717v2025-03 TRS.pdf": "S-717 Noise Equipment (Technical)",
    "S-717Qv2020-06 QRS.pdf": "S-717 Noise Equipment (Quality)",
    "S-719v2025-01 TRS.pdf": "S-719 Water Mist (Technical)",
    "S-719Qv2025-01 QRS.pdf": "S-719 Water Mist (Quality)",
    "S-719Jv2025-01 TRS with Justification.pdf": "S-719 Water Mist (Justification)",
}

LANG_FLAGS = {
    "en": "🇬🇧 English",
    "id": "🇮🇩 Bahasa Indonesia",
    "ar": "🇸🇦 Arabic",
    "ms": "🇲🇾 Malay",
}

LANG_OPTIONS = {
    "Auto-detect": "auto",
    "English": "en",
    "Bahasa Indonesia": "id",
}


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None
if "lang_override" not in st.session_state:
    st.session_state.lang_override = "auto"
if "confirm_clear" not in st.session_state:
    st.session_state.confirm_clear = False


# ─────────────────────────────────────────────
# CACHED RESOURCES
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading document index...")
def get_vectorstore():
    return load_vectorstore("en")


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
    for name in DOCUMENT_SOURCES:
        st.markdown(f'<div class="doc-item">{name}</div>', unsafe_allow_html=True)

    st.divider()

    # Language
    st.markdown('<div class="sidebar-label">🌐 Language</div>', unsafe_allow_html=True)
    lang_choice = st.selectbox(
        "Language",
        options=list(LANG_OPTIONS.keys()),
        index=0,
        label_visibility="collapsed",
    )
    st.session_state.lang_override = LANG_OPTIONS[lang_choice]
    if st.session_state.lang_override != "auto":
        st.caption(f"Override active: {lang_choice}")
    else:
        st.caption("Auto-detecting from your question")

    st.divider()

    # Sample questions — with visual category separation
    st.markdown('<div class="sidebar-label">💡 Try asking</div>', unsafe_allow_html=True)
    for category, questions in SAMPLE_QUESTIONS.items():
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
    st.markdown("""
    <div class="powered-by">
        POWERED BY<br>
        <span>Groq</span> · <span>LLaMA 3.3 70B</span><br>
        <span>ChromaDB</span> · <span>sentence-transformers</span><br>
        <span>LangChain</span> · <span>Streamlit</span>
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

# Load vectorstore
try:
    vectorstore = get_vectorstore()
    chunk_count = vectorstore._collection.count()
    doc_count = len(DOCUMENT_SOURCES)
except Exception as e:
    st.error(f"Failed to load document index. Run `ingest.py` first.\n\n{e}")
    st.stop()

# Header with index stats
st.markdown(f"""
<div class="app-header">
    <div>
        <div class="app-title">🛢️ O&G <span>Document Intelligence</span></div>
        <div class="app-subtitle">Ask questions from IOGP standards · JIP33 specifications · Process safety guidelines</div>
    </div>
    <div class="index-badge">
        <span>●</span> {doc_count} documents<br>
        <span>{chunk_count}</span> pages indexed
    </div>
</div>
""", unsafe_allow_html=True)

# ── EMPTY STATE ──
if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-state-title">What can I help you with?</div>
        <div class="capability-grid">
            <div class="capability-card">
                <div class="capability-card-title">🦺 HSE & Safety Rules</div>
                <div class="capability-card-desc">
                    Ask about IOGP Life-Saving Rules, confined space entry,
                    hot work requirements, energy isolation, working at height.
                </div>
            </div>
            <div class="capability-card">
                <div class="capability-card-title">📊 Process Safety KPIs</div>
                <div class="capability-card-desc">
                    Tier 1 and Tier 2 process safety events, LOPC definitions,
                    consequence thresholds, KPI measurement frameworks.
                </div>
            </div>
            <div class="capability-card">
                <div class="capability-card-title">⚙️ Equipment Specifications</div>
                <div class="capability-card-desc">
                    JIP33 S-737 deluge skids, S-717 noise equipment,
                    S-719 water mist fire protection — technical and quality requirements.
                </div>
            </div>
            <div class="capability-card">
                <div class="capability-card-title">🇮🇩 Bahasa Indonesia</div>
                <div class="capability-card-desc">
                    Tanya dalam Bahasa Indonesia. Sistem mendeteksi bahasa
                    otomatis dan menjawab dalam bahasa yang sama.
                </div>
            </div>
        </div>
        <div class="empty-hint">← Click a sample question or type below to start</div>
    </div>
    """, unsafe_allow_html=True)

# ── CHAT HISTORY ──
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            lang_display = LANG_FLAGS.get(msg.get("language", "en"), "🌐")
            st.markdown(f'<span class="lang-badge">{lang_display}</span>', unsafe_allow_html=True)

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
    "Ask about upstream O&G standards, safety rules, equipment specifications..."
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
            result = ask(question, lang_override=st.session_state.lang_override)
            status_placeholder.empty()

            # Language badge
            lang_display = LANG_FLAGS.get(result["language"], f"🌐 {result['language']}")
            st.markdown(f'<span class="lang-badge">{lang_display}</span>', unsafe_allow_html=True)

            # Stream answer
            st.write_stream(stream_text(result["answer"]))

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
                "language": result["language"],
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
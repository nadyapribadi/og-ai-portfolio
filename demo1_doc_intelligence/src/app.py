import streamlit as st
import html
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
st.markdown(
    "<style>\n"
    + (Path(__file__).with_name("app.css")).read_text(encoding="utf-8")
    + "\n</style>",
    unsafe_allow_html=True,
)


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
def friendly_source(filename):
    return SOURCE_FRIENDLY.get(filename, filename)


def render_sources(sources):
    """Source chips. File names come from the documents, so escape them: a PDF
    titled `<script>` is unlikely, but a citation chip is not the place to find
    out."""
    return "".join(
        f'<span class="source-tag">{html.escape(friendly_source(name))}</span>'
        for name in sources
    )


def render_chunks_expander(chunks, key_suffix=""):
    with st.expander(f"Source excerpts, {len(chunks)} retrieved"):
        for i, chunk in enumerate(chunks, 1):
            raw_source = chunk.metadata.get("source_file", "unknown")
            page = chunk.metadata.get("page", "?")
            friendly = friendly_source(raw_source)
            st.markdown(f"**[{i}]** `{friendly}`, page {page}")
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
    st.markdown(
        '<div class="brand">O&amp;G Document Intelligence</div>'
        '<div class="brand-by">Built by Nadya Boyke Pribadi</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # Document sources
    st.markdown('<div class="sidebar-label">Document sources</div>', unsafe_allow_html=True)
    for name in listed_sources:
        st.markdown(
            f'<div class="doc-item">{html.escape(str(name))}</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Sample questions — with visual category separation
    st.markdown('<div class="sidebar-label">Try asking</div>', unsafe_allow_html=True)
    for category, questions in question_set.items():
        st.markdown(f'<div class="cat-header">{category}</div>', unsafe_allow_html=True)
        for q in questions:
            if st.button(q, key=f"btn_{hash(q)}", use_container_width=True):
                st.session_state.pending_question = q
                st.session_state.confirm_clear = False

    st.divider()

    # Clear conversation — with confirmation
    if not st.session_state.confirm_clear:
        if st.button("Clear conversation", use_container_width=True):
            if len(st.session_state.messages) == 0:
                pass  # nothing to clear
            else:
                st.session_state.confirm_clear = True
                st.rerun()
    else:
        st.warning("Clear all messages?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear", use_container_width=True):
                st.session_state.messages = []
                st.session_state.confirm_clear = False
                st.rerun()
        with col2:
            if st.button("Cancel", use_container_width=True):
                st.session_state.confirm_clear = False
                st.rerun()

    st.divider()

    # Powered by
    rerank_label = (
        "mmarco cross-encoder"
        if rerank_strategy() == "cross-encoder"
        else "late interaction, memory-tight host"
    )
    st.markdown(f"""
    <div class="powered-by">
        <strong>Powered by</strong><br>
        Groq, {LLM_MODEL_QUALITY}<br>
        ChromaDB, ONNX Runtime<br>
        LangChain, Streamlit<br><br>
        <strong>Retrieval</strong><br>
        multilingual-e5-small<br>
        {rerank_label}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="note">
    Answers can be incomplete. Verify every answer against the original document
    before acting on it.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────

# Verify the configured LLM models still exist before the user asks anything.
# Groq retires model IDs; without this the app dies mid-question with a 404.
if not HAS_LLM_KEY:
    st.error(
        "**No Groq API key configured. The app can search, but it cannot answer.**\n\n"
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
        <div class="app-title">{APP_TITLE}</div>
        <div class="app-subtitle">{subtitle}</div>
    </div>
    <div class="index-badge">
        {doc_count} documents<br>
        {chunk_count} passages indexed
    </div>
</div>
""", unsafe_allow_html=True)

# ── EMPTY STATE ──
if not st.session_state.messages:
    # An index of what this corpus answers, not a grid of four identical cards:
    # the term carries the weight and the description defers to it.
    rows = "".join(
        '<div class="index-row">'
        f'<div class="index-term">{html.escape(c["title"])}</div>'
        f'<div class="index-desc">{html.escape(c["desc"])}</div>'
        "</div>"
        for c in capability_cards
    )
    st.markdown(f"""
    <div class="empty-state">
        <div class="empty-state-title">What this corpus can answer</div>
        <div class="empty-state-hint">
            Pick a question on the left, or type your own below.
        </div>
        <div class="index-list">{rows}</div>
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
                "Copy answer",
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
        status_placeholder.markdown("Searching the documents.")

        try:
            result = ask(question)
            status_placeholder.empty()

            # Rendered in one piece. It used to stream word by word at 18ms per
            # word, which delayed reading by seconds to animate text the reader
            # had already waited for.
            st.markdown(result["answer"])

            # Verified quotes: the evidence behind each claim, set apart so a
            # claim can always be checked against its source.
            if result.get("claims"):
                with st.expander(f"Verified sources, {len(result['claims'])} claims"):
                    for claim in result["claims"]:
                        st.markdown(
                            '<div class="evidence">'
                            f'<div class="evidence-head">{html.escape(str(claim.source_file))}, '
                            f'clause {html.escape(str(claim.clause_id or "-"))}, '
                            f'page {claim.page}</div>'
                            f"{html.escape(claim.quote)}</div>",
                            unsafe_allow_html=True,
                        )

            # "Not found" with rejected claims underneath is the one failure a
            # visitor cannot diagnose from the outside — the excerpts looked
            # right and the answer still was not there. Show what the guardrail
            # threw away, so the next report of this is evidence instead of a
            # guess.
            if result.get("not_found") and result.get("rejected"):
                with st.expander(
                    f"Why this says not found, {len(result['rejected'])} claim(s) rejected"
                ):
                    st.markdown(
                        "The model proposed these, but they could not be verified "
                        "against the retrieved excerpts:"
                    )
                    for claim, reason in result["rejected"]:
                        st.markdown(
                            f"- `{claim.source_file}`, clause {claim.clause_id or '-'}, "
                            f"page {claim.page}. _{reason}_"
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
                    "**Daily limit reached.**\n\n"
                    "The free Groq tier allows 100,000 tokens/day on the 70B model. "
                    "It resets in rolling windows, not just at midnight, so try "
                    "again in a few minutes.\n\n"
                    "If it happens often, a shorter question or fewer source "
                    "excerpts per answer will use less of the daily budget."
                )
            else:
                st.error(
                    f"Something went wrong. Please try again.\n\n"
                    f"Details: {error_msg}"
                )

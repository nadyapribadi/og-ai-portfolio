import os
import tempfile
from pathlib import Path

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
DEMO_ROOT      = Path(__file__).resolve().parent.parent


def _default_vectorstore():
    """Prefer the repo's data directory, fall back to a writable temp path.

    Hosted platforms mount the repository read-only, so building the index
    inside it fails. Writing to a temp directory instead is what lets the same
    code run locally and on Streamlit Cloud without a committed index.
    """
    preferred = DEMO_ROOT / "data" / "vectorstore_en"
    candidates = [preferred, Path(tempfile.gettempdir()) / "demo1_vectorstore_en"]
    for candidate in candidates:
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            # A real write test: create a directory and a file inside it, the
            # way the vector store will. A bare touch() can pass on a mount
            # that then refuses to create new content.
            probe_dir = candidate.parent / ".write_probe"
            probe_dir.mkdir(exist_ok=True)
            probe_file = probe_dir / "probe"
            probe_file.write_text("x", encoding="utf-8")
            probe_file.unlink()
            probe_dir.rmdir()
            return candidate
        except OSError:
            continue
    return preferred


VECTORSTORE_DIR = Path(
    os.getenv("DEMO1_VECTORSTORE") or _default_vectorstore()
)

# Where documents are read from. The real corpus (raw_docs) wins whenever it
# holds anything; sample_docs is the committed, licence-free fallback that makes
# a fresh clone runnable. See ingest.select_docs_dirs().
RAW_DOCS_DIR    = DEMO_ROOT / "data" / "raw_docs"
SAMPLE_DOCS_DIR = DEMO_ROOT / "sample_docs"
# Override to point at a mounted volume or a different corpus, e.g. in a
# container:  DEMO1_DOCS_DIR=/data/specs
DOCS_DIR_OVERRIDE = os.getenv("DEMO1_DOCS_DIR")

# ─────────────────────────────────────────────
# Demo 1 — Configuration
# Edit this file to adapt the app to your own documents.
# No changes needed in app.py.
# ─────────────────────────────────────────────

# App branding
APP_TITLE    = "O&G Document Intelligence"
APP_SUBTITLE = "Ask questions from IOGP standards · JIP33 specifications · Process safety guidelines"

# Document sources shown in sidebar
# Update this list to match your actual PDFs
DOCUMENT_SOURCES = [
    "IOGP Report 459 — Life-Saving Rules",
    "IOGP Report 456 — Process Safety KPIs",
    "JIP33 S-737 — Deluge Skids (TRS + QRS)",
    "JIP33 S-717 — Noise Equipment (TRS + QRS)",
    "JIP33 S-719 — Water Mist Fire Protection",
]

# Sample questions shown as clickable buttons in sidebar
# Replace with questions relevant to your own documents
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

# Shown instead when the index was built from the bundled sample corpus, so the
# first thing a new user clicks actually works.
SAMPLE_QUESTIONS_SAMPLE = {
    "📐 Technical requirements": [
        "What is the minimum design pressure for the deluge skid?",
        "What ingress protection rating is required for electrical enclosures?",
        "What surface preparation standard applies to steel structures?",
    ],
    "✅ Quality requirements": [
        "What pressure test is required and for how long?",
        "What documents must the manufacturing record book contain?",
        "When must conformity assessment be completed?",
    ],
    "🇮🇩 Bahasa Indonesia": [
        "Berapa tekanan desain minimum untuk deluge skid?",
        "Apa yang harus disiapkan sebelum pengujian tekanan?",
    ],
}

# Capability cards shown on empty state (first load)
# Replace with descriptions relevant to your own documents
CAPABILITY_CARDS = [
    {
        "title": "🦺 HSE & Safety Rules",
        "desc": (
            "Ask about IOGP Life-Saving Rules, confined space entry, "
            "hot work requirements, energy isolation, working at height."
        ),
    },
    {
        "title": "📊 Process Safety KPIs",
        "desc": (
            "Tier 1 and Tier 2 process safety events, LOPC definitions, "
            "consequence thresholds, KPI measurement frameworks."
        ),
    },
    {
        "title": "⚙️ Equipment Specifications",
        "desc": (
            "JIP33 S-737 deluge skids, S-717 noise equipment, "
            "S-719 water mist fire protection — technical and quality requirements."
        ),
    },
    {
        "title": "🇮🇩 Bahasa Indonesia",
        "desc": (
            "Tanya langsung dalam Bahasa Indonesia — pencarian dan jawaban "
            "ditangani dalam bahasa yang sama, tanpa terjemahan."
        ),
    },
]

# Friendly display names for raw PDF filenames
# Used in source citations — maps filename → readable name
# Add entries for your own PDFs
SOURCE_FRIENDLY = {
    "459.pdf":                                    "IOGP 459 Life-Saving Rules",
    "456.pdf":                                    "IOGP 456 Process Safety KPIs",
    "S-737v2026-03 TRS.pdf":                      "S-737 Deluge Skids (Technical)",
    "S-737Qv2026-03 QRS.pdf":                     "S-737 Deluge Skids (Quality)",
    "S-717v2025-03 TRS.pdf":                      "S-717 Noise Equipment (Technical)",
    "S-717Qv2020-06 QRS.pdf":                     "S-717 Noise Equipment (Quality)",
    "S-719v2025-01 TRS.pdf":                      "S-719 Water Mist (Technical)",
    "S-719Qv2025-01 QRS.pdf":                     "S-719 Water Mist (Quality)",
    "S-719Jv2025-01 TRS with Justification.pdf":  "S-719 Water Mist (Justification)",
}

# ─────────────────────────────────────────────
# LLM settings
# ─────────────────────────────────────────────
# Groq retires model IDs over time. These are validated at startup (see
# validate_models() in retrieval.py) so a retired model produces a readable
# message listing what IS available, instead of a bare 404 mid-question.
#
# When a model disappears, update the IDs here — nowhere else.
LLM_PROVIDER      = "groq"
LLM_MODEL_FAST    = "openai/gpt-oss-20b"    # query expansion — cheap and quick
LLM_MODEL_QUALITY = "openai/gpt-oss-120b"   # answering — quality matters

# Tried in order if the primary model is retired, rate-limited or overloaded.
# This is what stops a provider deprecation from silently killing the app.
LLM_FALLBACK_MODELS = [
    m for m in os.getenv("DEMO1_FALLBACK_MODELS", "openai/gpt-oss-120b,qwen/qwen3.8-27b").split(",")
    if m.strip()
]

# ─────────────────────────────────────────────
# Embeddings + chunking
# ─────────────────────────────────────────────
# The chunk budget MUST stay under the embedding model's window. If it does
# not, the tail of every chunk is silently dropped at embedding time and never
# becomes searchable — that was the original defect: 3,500-character chunks
# against a 256-token model left 53% of the corpus invisible.
# Chosen by measurement, not by preference — see the bake-off table in
# docs/plans/2026-09-18-demo1-overhaul.md. The English-only all-MiniLM-L6-v2
# scored 50% page hit-rate; this multilingual model scores 78% and fixes the
# Bahasa path outright. Override with DEMO1_EMBED_MODEL=... to re-run the
# comparison without editing code.
EMBED_MODEL = os.getenv(
    "DEMO1_EMBED_MODEL", "intfloat/multilingual-e5-small"
)

# Models that expect task prefixes. E5-family models are trained this way and
# lose accuracy without them; everything else embeds raw text.
EMBED_PROMPTS = {
    "intfloat/multilingual-e5-small": {"query": "query: ", "passage": "passage: "},
    "intfloat/multilingual-e5-base": {"query": "query: ", "passage": "passage: "},
}

# Fraction of the model's window a chunk may occupy, leaving room for the
# "[file | §clause | p.N]" label. The window itself is read from the model's own
# tokenizer at ingest time — never hardcoded.
CHUNK_WINDOW_FRACTION = 0.85

# Cross-encoder used to rerank candidates. The multilingual mMARCO model trades
# a little English accuracy for a lot of Bahasa — 83%/67% versus 92%/33% — and
# lifts overall page MRR from 0.529 to 0.624.
RERANK_MODEL = os.getenv(
    "DEMO1_RERANK_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
)

# How many chunks to hand the answering model. Chunks are now clause-sized
# (~200 tokens) rather than page-sized, so retrieval depth — not a fixed count —
# is what keeps the context budget comparable to before.
FINAL_TOP_K         = 12

# Candidates each retriever contributes before fusion and reranking.
CANDIDATE_K         = 50

# Chroma collection name — changing it forces a clean rebuild.
COLLECTION_NAME     = "og_docs"

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
    "IOGP Report 459: Life-Saving Rules",
    "IOGP Report 456: Process Safety KPIs",
    "JIP33 S-737: Deluge Skids (TRS + QRS)",
    "JIP33 S-717: Noise Equipment (TRS + QRS)",
    "JIP33 S-719: Water Mist Fire Protection",
]

# Sample questions shown as clickable buttons in sidebar
# Replace with questions relevant to your own documents
SAMPLE_QUESTIONS = {
    "HSE rules": [
        "What are the life saving rules?",
        "What must I confirm before entering a confined space?",
        "What are the hot work requirements in a hazardous area?",
    ],
    "Process safety": [
        "What is the difference between Tier 1 and Tier 2 process safety events?",
        "How are process safety KPIs measured?",
        "What does LOPC stand for and what are its consequences?",
    ],
    "Equipment standards": [
        "What does IOGP S-737 specify for deluge skid design?",
        "What standards does S-737 reference for electrical installations?",
        "What does IOGP S-717 cover for noise emitting equipment?",
    ],
    "Bahasa Indonesia": [
        "Apa saja aturan keselamatan jiwa menurut IOGP?",
        "Apa yang harus dilakukan sebelum memasuki ruang tertutup?",
        "Apa perbedaan antara kejadian keselamatan proses Tier 1 dan Tier 2?",
    ],
}

# Shown instead when the index was built from the bundled sample corpus, so the
# first thing a new user clicks actually works.
SAMPLE_QUESTIONS_SAMPLE = {
    "Technical requirements": [
        "What is the minimum design pressure for the deluge skid?",
        "What ingress protection rating is required for electrical enclosures?",
        "What surface preparation standard applies to steel structures?",
    ],
    "Quality requirements": [
        "What pressure test is required and for how long?",
        "What documents must the manufacturing record book contain?",
        "When must conformity assessment be completed?",
    ],
    "Bahasa Indonesia": [
        "Berapa tekanan desain minimum untuk deluge skid?",
        "Berapa lama pengujian tekanan hidrostatik berlangsung?",
    ],
}

# Capability cards shown on empty state (first load)
# Replace with descriptions relevant to your own documents
CAPABILITY_CARDS = [
    {
        "title": "HSE and safety rules",
        "desc": (
            "Ask about IOGP Life-Saving Rules, confined space entry, "
            "hot work requirements, energy isolation, working at height."
        ),
    },
    {
        "title": "Process safety KPIs",
        "desc": (
            "Tier 1 and Tier 2 process safety events, LOPC definitions, "
            "consequence thresholds, KPI measurement frameworks."
        ),
    },
    {
        "title": "Equipment specifications",
        "desc": (
            "JIP33 S-737 deluge skids, S-717 noise equipment, "
            "S-719 water mist fire protection: technical and quality requirements."
        ),
    },
    {
        "title": "Bahasa Indonesia",
        "desc": (
            "Tanya langsung dalam Bahasa Indonesia: pencarian dan jawaban "
            "ditangani dalam bahasa yang sama, tanpa terjemahan."
        ),
    },
]

# Shown when the index holds the bundled sample corpus. The cards above describe
# what the full document pack can do; advertising IOGP life-saving rules and
# Tier 1/2 process safety events next to an index of two synthetic deluge-skid
# files is a promise the app cannot keep, and a visitor can check it with one
# click. Cards follow the corpus, exactly like the sample questions do.
CAPABILITY_CARDS_SAMPLE = [
    {
        "title": "Design requirements",
        "desc": (
            "Minimum design pressure, ingress protection rating, cable support "
            "and tag plates: the technical requirements of the bundled S-900 "
            "deluge skid specification."
        ),
    },
    {
        "title": "Inspection and testing",
        "desc": (
            "Hydrostatic pressure testing and its duration, the test medium, "
            "inspection points, material and welding certification."
        ),
    },
    {
        "title": "Documentation and conformity",
        "desc": (
            "Manufacturing record book contents, language of documentation, "
            "conformity assessment, concession requests."
        ),
    },
    {
        "title": "Bahasa Indonesia",
        "desc": (
            "Tanya langsung dalam Bahasa Indonesia: pencarian dan jawaban "
            "ditangani dalam bahasa yang sama, tanpa terjemahan."
        ),
    },
]


def corpus_content(families):
    """Questions, cards and subtitle for the corpus that is actually indexed.

    `families` comes from retrieval.corpus_families(). The IOGP reports are the
    marker: whoever indexes them has the full document pack and gets the full
    story; everyone else is running the bundled sample and gets the sample's.
    """
    full = "IOGP 459" in families
    if full:
        return SAMPLE_QUESTIONS, CAPABILITY_CARDS, APP_SUBTITLE
    return (
        SAMPLE_QUESTIONS_SAMPLE,
        CAPABILITY_CARDS_SAMPLE,
        "Ask questions from the bundled S-900 deluge skid specification "
        "(synthetic, licence-free)",
    )

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
# When a model disappears, update the IDs here, nowhere else.
#
# The public demo answers with the smaller model on purpose. It runs on a free
# key that anyone on the internet can spend, and one answer costs roughly 2-4k
# tokens (12 excerpts plus the prompt and the reply), so the daily budget buys
# about twice as many answers this way. For local work where answer quality
# matters more than the daily budget:
#
#     DEMO1_ANSWER_MODEL=openai/gpt-oss-120b python demo1_doc_intelligence/src/ingest.py
LLM_PROVIDER      = "groq"
LLM_MODEL_FAST    = "openai/gpt-oss-20b"
LLM_MODEL_QUALITY = os.getenv("DEMO1_ANSWER_MODEL", "openai/gpt-oss-20b")

# Tried in order if the primary model is retired, rate-limited or overloaded.
# This is what stops a provider deprecation from silently killing the app.
LLM_FALLBACK_MODELS = [
    m for m in os.getenv("DEMO1_FALLBACK_MODELS", "openai/gpt-oss-120b,qwen/qwen3.8-27b").split(",")
    if m.strip()
]

# ─────────────────────────────────────────────
# Cost guard
# ─────────────────────────────────────────────
# Cost contract for this call path, written before the code that enforces it:
#
#   provider          Groq, free tier. No payment method is attached, so the
#                     hard stop is the daily token limit, not a bill.
#   unit              ~2-4k tokens per answer: 12 excerpts, the prompt, the
#                     reply. Answers that fail before generating cost nothing.
#   calls per answer  at most 3 models (primary plus two fallbacks) x 2 attempts
#                     of ask(). 4xx responses are only ever re-asked of a
#                     different model, which consumes no tokens.
#   per session       8 answers (MAX_ANSWERS_PER_SESSION below).
#   per day           25 answers, about 75-100k tokens, which is meant to sit
#                     below the free daily budget rather than discover it.
#   provider cap      Groq daily token limit. Verify in the Groq console that no
#                     payment method is attached; then exhausting the demo
#                     cannot cost money, only availability.
#   concurrency       One Streamlit container, one answer per session, counted
#                     in a file. Two simultaneous visitors can race by an answer
#                     or two, which is accepted: the provider limit is the
#                     backstop.
#
# The guard is enforced in app.py before ask(); capability questions (the ones
# answered from configuration, no model call) do not spend anything.
QUOTA_GUARD = os.getenv("DEMO1_QUOTA_GUARD", "on").strip().lower() not in (
    "off", "0", "false", "no",
)
MAX_ANSWERS_PER_SESSION = int(os.getenv("DEMO1_MAX_ANSWERS_PER_SESSION") or 8)
MAX_ANSWERS_PER_DAY = int(os.getenv("DEMO1_MAX_ANSWERS_PER_DAY") or 25)
MIN_SECONDS_BETWEEN_ANSWERS = float(os.getenv("DEMO1_MIN_SECONDS") or 4)
QUOTA_FILE = Path(
    os.getenv("DEMO1_QUOTA_FILE")
    or (Path(tempfile.gettempdir()) / "demo1_quota.json")
)

# ─────────────────────────────────────────────
# Embeddings + chunking
# ─────────────────────────────────────────────
# Both models run on ONNX Runtime, not torch. The hosted app gets 690 MB–2.7 GB
# and throttles on overshoot; torch + sentence-transformers + two fp32 models
# measured ~1.35 GB resident, which is what produced "this app has gone over its
# resource limits". These repos are int8 ONNX ports of the same weights — see
# onnx_models.py for the measurements behind the choice.
#
# `*_ONNX_FILE` selects the precision. model_quantized.onnx is int8 (smallest,
# what deployment uses); model.onnx is fp32 (largest, for a local parity check).

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
    "DEMO1_EMBED_MODEL", "Xenova/multilingual-e5-small"
)
EMBED_ONNX_FILE = os.getenv("DEMO1_EMBED_ONNX_FILE", "onnx/model_quantized.onnx")

# Models that expect task prefixes. E5-family models are trained this way and
# lose accuracy without them; everything else embeds raw text.
EMBED_PROMPTS = {
    "Xenova/multilingual-e5-small": {"query": "query: ", "passage": "passage: "},
    "intfloat/multilingual-e5-small": {"query": "query: ", "passage": "passage: "},
    "intfloat/multilingual-e5-base": {"query": "query: ", "passage": "passage: "},
}

# Fraction of the model's window a chunk may occupy, leaving room for the
# "[file | §clause | p.N]" label. The window itself is read from the model's own
# tokenizer at ingest time — never hardcoded.
CHUNK_WINDOW_FRACTION = 0.85

# Cross-encoder used to rerank candidates. Reranking earns its place: measured
# on the final stack, fusion order alone gives page MRR 0.468, late interaction
# 0.531, and this cross-encoder 0.618. An English-only cross-encoder managed
# 0.485 — the multilingual model is what carries the Bahasa half of the golden
# set. It ships its own int8 ONNX exports; onnx_models picks the CPU variant.
RERANK_MODEL = os.getenv(
    "DEMO1_RERANK_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
)
# Empty means "pick the variant that matches this CPU".
RERANK_ONNX_FILE = os.getenv("DEMO1_RERANK_ONNX_FILE", "")

# The cross-encoder is the better ranker and a second set of weights: ~310 MB
# resident, measured, on a host that may hand the app only 690 MB total. So the
# strategy is a decision, not a constant:
#
#   auto (default)     cross-encoder when the container reports enough memory,
#                      late interaction on the embedding model already loaded
#                      when it does not
#   on / off           pin it, e.g. to reproduce either measurement
#
# Measured on the 18-question golden set: cross-encoder page MRR 0.618 and the
# right page in the context 100% of the time at k=20; late interaction 0.531
# and 89%. Late interaction costs nothing extra, which is the whole point.
RERANK_MODE = os.getenv("DEMO1_RERANK", "auto").strip().lower()

# Below this container ceiling, the second model is not affordable: 145 MB base
# + 440 MB embedder + 310 MB reranker leaves too little room inside 690 MB.
RERANK_MIN_LIMIT_MB = int(os.getenv("DEMO1_RERANK_MIN_LIMIT_MB") or 1400)

# How many chunks to hand the answering model. Chunks are now clause-sized
# (~200 tokens) rather than page-sized, so retrieval depth — not a fixed count —
# is what keeps the context budget comparable to before.
FINAL_TOP_K         = 12

# Depth used when the cross-encoder is not available. Without it the fused order
# is less precise, so the answering model gets more excerpts to choose from:
# measured page hit-rate 89% at 20 chunks versus 83% at 12.
DEEP_TOP_K          = 20

# Candidates each retriever contributes before fusion and reranking.
CANDIDATE_K         = 50

# Chroma collection name — changing it forces a clean rebuild.
COLLECTION_NAME     = "og_docs"

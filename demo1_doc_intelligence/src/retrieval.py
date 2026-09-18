import os
import re
import sys
from pathlib import Path
from dotenv import load_dotenv
from langdetect import detect, LangDetectException
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

sys.path.insert(0, str(Path(__file__).parent))
from config import LLM_MODEL_FAST, LLM_MODEL_QUALITY  # noqa: E402

# Load demo1's own .env, wherever the process was started from.
load_dotenv(Path(__file__).parent.parent / ".env")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

VECTORSTORE_EN       = Path(__file__).parent.parent / "data" / "vectorstore_en"
VECTORSTORE_MULTI    = Path(__file__).parent.parent / "data" / "vectorstore_multi"
MODEL_EN             = "all-MiniLM-L6-v2"
MODEL_MULTI          = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION           = "og_docs"
TOP_K                = 4
# Model IDs live in config.py — set them there, not here.
GROQ_MODEL_FAST      = LLM_MODEL_FAST
GROQ_MODEL_QUALITY   = LLM_MODEL_QUALITY
MULTILINGUAL_ENABLED = False

SYSTEM_PROMPT = """You are an expert assistant for upstream oil and gas operations,
procurement, HSE standards, and supply chain management.

You will be given numbered document excerpts and a question.

STRICT RULES — follow all of them without exception:

RULE 1 — USE ONLY THE EXCERPTS:
Every single statement in your answer must be directly traceable to a specific excerpt.
If you cannot point to an excerpt that supports a claim, do NOT make that claim.
Never use your general knowledge to fill gaps. If information is missing from excerpts, say so.

RULE 2 — CITE EVERY CLAIM:
After each statement or list item, add the source in parentheses: (Source: filename, Page: N)
If a claim comes from multiple excerpts, cite all of them.
Never make an uncited claim.

RULE 3 — HANDLE PROHIBITIONS CORRECTLY:
If the document prohibits an action entirely (e.g. "never walk under a suspended load"),
state the prohibition clearly and completely. Do not search for prerequisites that don't exist.
A prohibition IS the complete answer.

RULE 4 — NEVER HEDGE FALSELY:
If the information IS in the excerpts, state it confidently and directly.
Do not say "tidak jelas" or "not explicitly stated" if the answer IS in the excerpts.
Only say information is not found if you genuinely cannot locate it in any excerpt.

RULE 5 — LANGUAGE:
Answer in the same language the user used to ask the question.
For technical terms (e.g. Tier 1, LOPC, deluge skid) keep the original English term
even when answering in Bahasa Indonesia — do not translate technical terms.

RULE 6 — INCOMPLETE LISTS:
If a list appears incomplete because not all items are in the excerpts,
show only what IS in the excerpts and end with:
"Note: this list may be incomplete — only retrieved excerpts are shown."
Never complete or extend a list beyond what the excerpts contain.

RULE 7 — NOTHING FOUND:
If genuinely no excerpt contains relevant information, say exactly:
"This information is not found in the loaded documents."

Document excerpts:
{context}"""

_vs_cache = {}


def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return "en"


def _content_to_text(content):
    """Normalise a LangChain message payload to plain text.

    Some providers return a list of content blocks instead of a string;
    calling .strip() on that raises AttributeError.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text") or block.get("content") or ""))
        return "\n".join(parts)
    return "" if content is None else str(content)


_model_check_done = False


def validate_models(models=None, strict=False):
    """Check the configured models still exist on this account.

    Returns (missing, available). With strict=True raises RuntimeError instead
    of warning, so callers can fail loudly with a useful message.
    """
    wanted = list(dict.fromkeys(models or [GROQ_MODEL_FAST, GROQ_MODEL_QUALITY]))
    try:
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        available = sorted(m.id for m in client.models.list().data)
    except Exception as exc:                      # network, auth, quota ...
        if strict:
            raise RuntimeError(f"Could not verify Groq models: {exc}") from exc
        return [], []

    missing = [m for m in wanted if m not in available]
    if missing:
        message = (
            "These configured models are no longer available on this account: "
            + ", ".join(missing)
            + "\nAvailable models: "
            + ", ".join(available)
            + "\nUpdate LLM_MODEL_FAST / LLM_MODEL_QUALITY in config.py."
        )
        if strict:
            raise RuntimeError(message)
        print("WARNING: " + message)
    return missing, available


def translate_to_english(text):
    from deep_translator import GoogleTranslator
    return GoogleTranslator(source='auto', target='en').translate(text)


_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


def expand_query(question, max_variants=3):
    """Generate search variants. The original question is always first.

    Never raises on unexpected model output — if the model returns nothing
    usable, we fall back to the original question alone.
    """
    llm = ChatGroq(
        model=GROQ_MODEL_FAST,
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    messages = [
        SystemMessage(content=(
            "You are helping search an oil and gas document database. "
            f"Given a question, generate {max_variants} search queries that would find the answer. "
            "Make each query use DIFFERENT vocabulary than the original — "
            "focus on the specific content, actions, or procedures the answer would contain. "
            f"Return only the {max_variants} queries, one per line, nothing else."
        )),
        HumanMessage(content=question),
    ]
    text = _content_to_text(llm.invoke(messages).content)

    variants = []
    for line in text.splitlines():
        line = _BULLET_RE.sub("", line).strip()
        if not line or line.lower() == question.strip().lower():
            continue
        if line not in variants:
            variants.append(line)

    return [question] + variants[:max_variants]

def load_vectorstore(lang):
    if not MULTILINGUAL_ENABLED:
        lang = "en"
    if lang in _vs_cache:
        return _vs_cache[lang]

    model_name = MODEL_EN if lang == "en" else MODEL_MULTI
    persist_dir = VECTORSTORE_EN if lang == "en" else VECTORSTORE_MULTI

    print(f"  Language: {lang} → using {model_name}")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )
    print(f"  Vectorstore loaded: {vectorstore._collection.count()} chunks")
    _vs_cache[lang] = vectorstore
    return vectorstore


def search_chunks(vectorstore, question):
    from sentence_transformers import CrossEncoder

    # Step 1 — fetch broad candidate set via MMR
    seen_ids = set()
    all_chunks = []

    variants = expand_query(question)
    print(f"  Query variants: {len(variants)}")

    for variant in variants:
        results = vectorstore.max_marginal_relevance_search(
            variant, k=TOP_K + 4, fetch_k=30
        )
        for r in results:
            if len(r.page_content.strip()) >= 100:
                doc_id = (
                    f"{r.metadata.get('source_file')}_"
                    f"{r.metadata.get('page')}_"
                    f"{r.page_content[:50]}"
                )
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_chunks.append(r)

    if not all_chunks:
        return []

    # Step 2 — rerank with cross-encoder
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [[question, chunk.page_content] for chunk in all_chunks]
    scores = reranker.predict(pairs)

    # Sort by score, return top 6
    ranked = sorted(zip(scores, all_chunks), key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in ranked[:6]]


def build_context(chunks):
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.metadata.get("source_file", "unknown")
        page   = chunk.metadata.get("page", "?")
        parts.append(
            f"[{i}] Source: {source} | Page: {page}\n{chunk.page_content}"
        )
    return "\n\n".join(parts)


def ask(question, lang_override="auto"):
    global _model_check_done
    if not _model_check_done:
        validate_models(strict=True)
        _model_check_done = True

    lang = lang_override if lang_override != "auto" else detect_language(question)
    search_question = question

    if lang != "en":
        print(f"  Translating from {lang} to English for search...")
        search_question = translate_to_english(question)
        print(f"  Translated: {search_question}")

    vectorstore = load_vectorstore(lang)
    chunks      = search_chunks(vectorstore, search_question)
    context     = build_context(chunks)

    llm = ChatGroq(
        model=GROQ_MODEL_QUALITY,
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    messages = [
        SystemMessage(content=SYSTEM_PROMPT.format(context=context)),
        HumanMessage(content=question),
    ]
    response = llm.invoke(messages)
    sources  = list({c.metadata.get("source_file", "unknown") for c in chunks})

    return {
        "answer":   response.content,
        "sources":  sources,
        "chunks":   chunks,
        "language": lang,
    }


if __name__ == "__main__":
    test_questions = [
        # Previously failing — should now work after ingest fix
        "What does IOGP S-737 specify for deluge skid inspection?",
        "What does S-717 cover?",
        "What does LOPC stand for and what are its consequences?",
        # Regression check — must still pass
        "What are the life saving rules?",
        "Apa perbedaan antara kejadian keselamatan proses Tier 1 dan Tier 2?",
        "What standards does S-737 reference for electrical installations?",
    ]
    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        result = ask(q)
        print(f"A: {result['answer']}")
        print(f"Sources: {result['sources']}")
        print(f"Language detected: {result['language']}")

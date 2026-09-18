"""Rank clauses and answer with citations.

One index, one query path. Chunks are already clause-sized and page-bounded by
ingest.py, so this module only has to rank them well. Queries run in whatever
language the user writes — the embedding model is multilingual, so there is no
translation step and no language detection to get wrong.
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

sys.path.insert(0, str(Path(__file__).parent))
from config import (  # noqa: E402
    COLLECTION_NAME,
    EMBED_MODEL,
    FINAL_TOP_K,
    LLM_MODEL_FAST,
    LLM_MODEL_QUALITY,
    RERANK_MODEL,
    VECTORSTORE_DIR,
)
from embeddings import build_embeddings  # noqa: E402

# Load demo1's own .env, wherever the process was started from.
load_dotenv(Path(__file__).parent.parent / ".env")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

GROQ_MODEL_FAST    = LLM_MODEL_FAST
GROQ_MODEL_QUALITY = LLM_MODEL_QUALITY

MULTILINGUAL_QUERY = True   # queries are embedded natively, never translated

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
_reranker = None
_model_check_done = False


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


def validate_models(models=None, strict=False):
    """Check the configured models still exist on this account.

    Groq retires models; without this the app dies mid-question with a bare 404
    that tells the user nothing. Returns (missing, available).
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


def load_vectorstore(_lang=None):
    """Load the single clause index. `_lang` is accepted and ignored."""
    if "vs" in _vs_cache:
        return _vs_cache["vs"]

    print(f"  embedding: {EMBED_MODEL}")
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=build_embeddings(),
        persist_directory=str(VECTORSTORE_DIR),
    )
    print(f"  vectorstore loaded: {vectorstore._collection.count()} chunks")
    _vs_cache["vs"] = vectorstore
    return vectorstore


def get_reranker():
    """Load the cross-encoder once per process, not once per question."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def search_chunks(vectorstore, question):
    """Retrieve a broad candidate set, then rerank it down to FINAL_TOP_K."""
    seen_ids = set()
    candidates = []

    results = vectorstore.max_marginal_relevance_search(
        question, k=FINAL_TOP_K * 2, fetch_k=50
    )
    for result in results:
        doc_id = (
            f"{result.metadata.get('source_file')}_"
            f"{result.metadata.get('page')}_"
            f"{result.metadata.get('clause_id')}_"
            f"{result.page_content[:40]}"
        )
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            candidates.append(result)

    if not candidates:
        return []

    reranker = get_reranker()
    scores = reranker.predict([[question, c.page_content] for c in candidates])
    ranked = sorted(zip(scores, candidates), key=lambda pair: pair[0], reverse=True)
    return [chunk for _, chunk in ranked[:FINAL_TOP_K]]


def build_context(chunks):
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.metadata.get("source_file", "unknown")
        page = chunk.metadata.get("page", "?")
        clause = chunk.metadata.get("clause_id") or "-"
        parts.append(
            f"[{i}] Source: {source} | Clause: {clause} | Page: {page}\n"
            f"{chunk.page_content}"
        )
    return "\n\n".join(parts)


def ask(question):
    global _model_check_done
    if not _model_check_done:
        validate_models(strict=True)
        _model_check_done = True

    vectorstore = load_vectorstore()
    chunks = search_chunks(vectorstore, question)
    context = build_context(chunks)

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

    sources = list({c.metadata.get("source_file", "unknown") for c in chunks})
    clauses = list(
        {c.metadata.get("clause_id", "") for c in chunks if c.metadata.get("clause_id")}
    )

    return {
        "answer": _content_to_text(response.content),
        "sources": sources,
        "clauses": clauses,
        "chunks": chunks,
    }


if __name__ == "__main__":
    result = ask("What are the life saving rules?")
    print(result["answer"][:400])
    print("Sources:", result["sources"])

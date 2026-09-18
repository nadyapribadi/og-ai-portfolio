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
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

sys.path.insert(0, str(Path(__file__).parent))
import answer as answer_mod  # noqa: E402
import hybrid  # noqa: E402
import routing  # noqa: E402
from config import (  # noqa: E402
    CANDIDATE_K,
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

# Below this many routed candidates we fall back to searching everything.
MIN_ROUTED_CANDIDATES = 3


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


def _chunk_key(meta, text):
    return (
        f"{meta.get('source_file')}|{meta.get('page')}|"
        f"{meta.get('clause_id')}|{text[:40]}"
    )


def _corpus(vectorstore):
    """Load the whole index once so keyword search can run in memory."""
    if "corpus" not in _vs_cache:
        data = vectorstore.get(include=["documents", "metadatas"])
        texts = data["documents"] or []
        metas = data["metadatas"] or []
        keys = [_chunk_key(meta, text) for meta, text in zip(metas, texts)]
        bm25 = hybrid.BM25([hybrid.tokenize(text) for text in texts])
        _vs_cache["corpus"] = (keys, texts, metas, bm25)
    return _vs_cache["corpus"]


def _vector_keys(vectorstore, question, where=None):
    """Keys of the vector-search hits, optionally filtered by metadata."""
    try:
        hits = vectorstore.similarity_search(
            question, k=CANDIDATE_K, filter=where or None
        )
    except Exception:
        # A filter the backend rejects must never lose the search entirely.
        if not where:
            raise
        hits = vectorstore.similarity_search(question, k=CANDIDATE_K)
    return [_chunk_key(hit.metadata, hit.page_content) for hit in hits]


def _allowed_indices(metas, family, role):
    """Corpus indices permitted by the inferred route, or None for no filter."""
    if not family and not role:
        return None
    allowed = {
        i
        for i, meta in enumerate(metas)
        if not family or meta.get("doc_family") == family
    }
    if role:
        by_role = {i for i in allowed if metas[i].get("doc_role") == role}
        if len(by_role) >= MIN_ROUTED_CANDIDATES:
            allowed = by_role
    return allowed if len(allowed) >= MIN_ROUTED_CANDIDATES else None


def search_chunks(vectorstore, question, limit=None):
    """Route, retrieve with two strategies, fuse by rank, then rerank.

    Each stage is a separate signal: routing narrows the corpus, BM25 catches
    exact tokens, vector search catches meaning, RRF combines two rankings that
    are not otherwise comparable, and the cross-encoder makes the final,
    expensive judgement on a short list.
    """
    keys, texts, metas, bm25 = _corpus(vectorstore)
    key_to_index = {key: index for index, key in enumerate(keys)}
    by_key = {
        key: Document(page_content=text, metadata=meta)
        for key, text, meta in zip(keys, texts, metas)
    }

    family = routing.infer_family(question)
    role = routing.infer_role(question)
    allowed = _allowed_indices(metas, family, role)

    vector_keys = _vector_keys(
        vectorstore, question, where={"doc_family": family} if family else None
    )
    if allowed is not None:
        vector_keys = [
            key
            for key in vector_keys
            if key_to_index.get(key) in allowed
        ]
    if len(vector_keys) < MIN_ROUTED_CANDIDATES:
        vector_keys = _vector_keys(vectorstore, question)

    keyword_keys = [keys[i] for i in bm25.top_k(question, CANDIDATE_K, allowed)]

    fused = hybrid.reciprocal_rank_fusion([vector_keys, keyword_keys])
    candidates = [by_key[key] for key in fused if key in by_key]
    if not candidates:
        return []

    reranker = get_reranker()
    scores = reranker.predict([[question, c.page_content] for c in candidates])
    ranked = sorted(zip(scores, candidates), key=lambda pair: pair[0], reverse=True)
    return [chunk for _, chunk in ranked[: limit or FINAL_TOP_K]]


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


NOT_FOUND = "This information is not found in the loaded documents."


def _render(claims):
    """Turn validated claims into the text the user reads."""
    if not claims:
        return NOT_FOUND
    if len(claims) == 1:
        claim = claims[0]
        return (
            f"{claim.text} (Source: {claim.source_file}, Page: {claim.page})"
        )
    lines = []
    for claim in claims:
        lines.append(
            f"- {claim.text} (Source: {claim.source_file}, Page: {claim.page})"
        )
    return "\n".join(lines)


def _generate(question, chunks):
    """Ask for claims, then keep only the ones that verify against the excerpts."""
    llm = ChatGroq(
        model=GROQ_MODEL_QUALITY,
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    structured = answer_mod.structured_llm(llm)
    if structured is None:
        raise RuntimeError("the configured model cannot return structured output")

    draft = structured.invoke(answer_mod.build_messages(question, build_context(chunks)))
    accepted, rejected = answer_mod.validate_claims(draft, chunks)
    return {
        "answer": _render(accepted),
        "claims": accepted,
        "rejected": rejected,
        "not_found": bool(draft.not_found) or not accepted,
        "sources": list(
            {c.metadata.get("source_file", "unknown") for c in chunks}
        ),
        "clauses": list(
            {
                c.metadata.get("clause_id", "")
                for c in chunks
                if c.metadata.get("clause_id")
            }
        ),
        "chunks": chunks,
    }


def ask(question):
    """Answer with citations that the system verified, not ones it was asked for."""
    global _model_check_done
    if not _model_check_done:
        validate_models(strict=True)
        _model_check_done = True

    vectorstore = load_vectorstore()
    result = _generate(question, search_chunks(vectorstore, question))

    # Bounded retry: only when retrieval or the model produced nothing usable do
    # we pay for a second attempt, and then with a wider net.
    if not result["claims"] and result["rejected"]:
        wider = search_chunks(vectorstore, question, limit=FINAL_TOP_K * 2)
        retry = _generate(question, wider)
        if retry["claims"]:
            retry["retried"] = True
            return retry

    return result


if __name__ == "__main__":
    result = ask("What are the life saving rules?")
    print(result["answer"][:400])
    print("Sources:", result["sources"])

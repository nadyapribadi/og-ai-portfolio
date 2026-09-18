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

sys.path.insert(0, str(Path(__file__).parent))
import answer as answer_mod  # noqa: E402
import hybrid  # noqa: E402
import llm  # noqa: E402
import routing  # noqa: E402
from config import (  # noqa: E402
    CANDIDATE_K,
    COLLECTION_NAME,
    DEEP_TOP_K,
    EMBED_MODEL,
    FINAL_TOP_K,
    LLM_MODEL_FAST,
    LLM_MODEL_QUALITY,
    RERANK_MODEL,
    RERANK_MIN_LIMIT_MB,
    RERANK_MODE,
    RERANK_ONNX_FILE,
    VECTORSTORE_DIR,
)
from embeddings import build_embeddings, clear_chroma_client_cache  # noqa: E402
from memory import container_memory_limit_mb, inside_container  # noqa: E402
from onnx_models import (  # noqa: E402
    OnnxCrossEncoder,
    OnnxLateInteraction,
    preferred_onnx_file,
)

# Load demo1's own .env, wherever the process was started from.
load_dotenv(Path(__file__).parent.parent / ".env")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

GROQ_MODEL_FAST    = LLM_MODEL_FAST
GROQ_MODEL_QUALITY = LLM_MODEL_QUALITY

_vs_cache = {}
_reranker = None
_late_reranker = None
_strategy = None
_model_check_done = False

# Below this many routed candidates we fall back to searching everything.
MIN_ROUTED_CANDIDATES = 3

# How much of the fused list late interaction rewrites. MaxSim needs every
# candidate's token vectors, so the pool is what the rerank costs in memory and
# time; the tail keeps its fusion order. Relevant passages are effectively
# always inside the first forty of the fused list (page hit-rate is 83% at 12),
# so the ceiling costs almost nothing in quality and bounds the work.
LATE_RERANK_POOL = 40


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
    """Check the configured models still exist — delegates to the provider layer."""
    wanted = list(dict.fromkeys(models or [GROQ_MODEL_FAST, GROQ_MODEL_QUALITY]))
    return llm.validate_models(wanted, strict=strict)


def load_vectorstore(_lang=None):
    """Load the single clause index. `_lang` is accepted and ignored."""
    if "vs" in _vs_cache:
        return _vs_cache["vs"]

    # A rebuild may have replaced the directory behind a cached client.
    clear_chroma_client_cache()
    print(f"  embedding: {EMBED_MODEL}")
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=build_embeddings(),
        persist_directory=str(VECTORSTORE_DIR),
    )
    count = vectorstore._collection.count()
    print(f"  vectorstore loaded: {count} chunks")
    # Do not cache an empty index: it means the store is missing or was replaced
    # underneath us, and the next caller should open a fresh one rather than
    # keep a dead handle.
    if count:
        _vs_cache["vs"] = vectorstore
    return vectorstore


def corpus_families():
    """Document families present in the index — used to pick sample questions."""
    vectorstore = load_vectorstore()
    metas = vectorstore.get(include=["metadatas"]).get("metadatas") or []
    return {meta.get("doc_family") for meta in metas if meta.get("doc_family")}


def corpus_sources():
    """Source files actually in the index, for a sidebar that cannot lie.

    The deployed app falls back to the bundled sample corpus when the real PDFs
    are absent (they are copyrighted and never committed), so listing the
    configured document pack would advertise five IOGP/JIP33 reports while
    holding two sample files. Read the index instead.
    """
    vectorstore = load_vectorstore()
    metas = vectorstore.get(include=["metadatas"]).get("metadatas") or []
    return sorted({meta.get("source_file") for meta in metas if meta.get("source_file")})


def get_reranker():
    """Load the cross-encoder once per process, not once per question."""
    global _reranker
    if _reranker is None:
        _reranker = OnnxCrossEncoder(
            RERANK_MODEL,
            filename=RERANK_ONNX_FILE or preferred_onnx_file(RERANK_MODEL),
        )
    return _reranker


def get_late_reranker():
    """Rerank with the embedding model already resident, at no extra cost."""
    global _late_reranker
    if _late_reranker is None:
        _late_reranker = OnnxLateInteraction(build_embeddings())
    return _late_reranker


def rerank_strategy():
    """Which reranker this host gets, and why.

    The cross-encoder ranks better — measured page MRR 0.618 against 0.531 for
    late interaction — but it is a second set of weights, ~310 MB resident, and
    Streamlit Community Cloud may hand the app as little as 690 MB in total.
    Late interaction scores with token vectors the loaded embedder already
    produces, so it is free; the caller deepens the context to DEEP_TOP_K to
    compensate for its looser ordering.

    `DEMO1_RERANK=on|off` pins the choice; `auto`, the default, reads the
    container's own memory ceiling.
    """
    global _strategy
    if _strategy is None:
        limit = container_memory_limit_mb()
        if RERANK_MODE in ("on", "off"):
            _strategy = "cross-encoder" if RERANK_MODE == "on" else "late-interaction"
        elif limit is None:
            # No declared ceiling. That is a laptop, unless we are plainly in a
            # container — where the ceiling is enforced from outside and cannot
            # be read, so assume it is tight rather than gamble on it.
            _strategy = (
                "late-interaction" if inside_container() else "cross-encoder"
            )
        else:
            _strategy = (
                "late-interaction"
                if limit < RERANK_MIN_LIMIT_MB
                else "cross-encoder"
            )
        print(f"  rerank: {_strategy} (container memory limit: {limit} MB)")
    return _strategy


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
        # Clause titles are short and semantically dense — "8.1 Protective
        # coatings" is a far stronger signal for that question than the body
        # text of §1, which merely cites the same standard.
        title_bm25 = hybrid.BM25(
            [
                hybrid.tokenize(
                    f"{meta.get('clause_id') or ''} "
                    f"{meta.get('heading') or meta.get('title') or ''}"
                )
                for meta in metas
            ]
        )
        _vs_cache["corpus"] = (keys, texts, metas, bm25, title_bm25)
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
    keys, texts, metas, bm25, title_bm25 = _corpus(vectorstore)
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
    title_keys = [keys[i] for i in title_bm25.top_k(question, CANDIDATE_K, allowed)]

    fused = hybrid.reciprocal_rank_fusion([vector_keys, keyword_keys, title_keys])
    candidates = [by_key[key] for key in fused if key in by_key]
    if not candidates:
        return []

    if rerank_strategy() == "cross-encoder":
        scores = get_reranker().predict(
            [[question, c.page_content] for c in candidates]
        )
        ranked = sorted(zip(scores, candidates), key=lambda pair: pair[0], reverse=True)
        return [chunk for _, chunk in ranked[: limit or FINAL_TOP_K]]

    # Late interaction orders the close calls against the question's own token
    # vectors. It is less precise than the cross-encoder, so the cut is deeper:
    # the same evidence reaches the model in 89% of golden questions at 20
    # chunks, against 83% at 12.
    pool = candidates[:LATE_RERANK_POOL]
    scores = get_late_reranker().predict([[question, c.page_content] for c in pool])
    ranked = sorted(zip(scores, pool), key=lambda pair: pair[0], reverse=True)
    ordered = [chunk for _, chunk in ranked] + candidates[LATE_RERANK_POOL:]
    return ordered[: limit or DEEP_TOP_K]


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
    messages = answer_mod.build_messages(question, chunks)

    def call(model):
        structured = answer_mod.structured_llm(model)
        if structured is None:
            raise RuntimeError("the configured model cannot return structured output")
        return structured.invoke(messages)

    draft = llm.with_fallback(GROQ_MODEL_QUALITY, call)
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
    if result["claims"]:
        return result

    # One bounded second attempt, and only on failure. Which way to adjust
    # depends on why it failed: rejected claims mean the answer was not in the
    # retrieved set, so widen it; an empty answer means the model could not find
    # it in a broad context, so focus it. The model is not perfectly
    # deterministic, so a single retry also smooths over a bad draw.
    limit = FINAL_TOP_K * 2 if result["rejected"] else max(4, FINAL_TOP_K // 2)
    retry = _generate(question, search_chunks(vectorstore, question, limit=limit))
    if retry["claims"]:
        retry["retried"] = True
        return retry
    return result


if __name__ == "__main__":
    result = ask("What are the life saving rules?")
    print(result["answer"][:400])
    print("Sources:", result["sources"])

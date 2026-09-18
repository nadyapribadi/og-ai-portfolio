"""Single place that constructs the embedding model and reports its window.

Both ingest and retrieval go through here so the two can never drift apart —
the chunk size is derived from the same window the embedder actually honours.
"""
from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBED_MODEL, EMBED_PROMPTS


def build_embeddings(model_name=None):
    name = model_name or EMBED_MODEL
    kwargs = {"model_name": name}
    prompts = EMBED_PROMPTS.get(name)
    if prompts:
        kwargs["model_kwargs"] = {"prompts": prompts}
        kwargs["encode_kwargs"] = {
            "normalize_embeddings": True,
            "prompt_name": "passage",
        }
        kwargs["query_encode_kwargs"] = {
            "normalize_embeddings": True,
            "prompt_name": "query",
        }
    return HuggingFaceEmbeddings(**kwargs)


def embedding_window(model_name=None):
    """Token window the model actually honours, with safety caps."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name or EMBED_MODEL)
    window = getattr(tokenizer, "model_max_length", None) or 512
    # Tokenizers use a huge sentinel when the window is unknown.
    return window if 0 < window <= 8192 else 512


def clear_chroma_client_cache():
    """Drop Chroma's per-path client cache.

    Chroma keys a persistent client by its path and hands back the same
    underlying System for the same path, connection included. Once an index
    directory has been deleted and rebuilt — which happens whenever the app
    rebuilds an index it found unusable — that reused connection still points
    at the deleted database, and the next write fails with
    SQLITE_READONLY_DBMOVED (1032), reported as "attempt to write a readonly
    database".

    Clearing the cache forces a fresh client for the rebuilt directory.
    """
    try:
        from chromadb.api.shared_system_client import SharedSystemClient

        SharedSystemClient.clear_system_cache()
    except Exception:                # noqa: BLE001 - never fail on cleanup
        pass

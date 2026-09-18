"""Single place that constructs the embedding model and reports its window.

Both ingest and retrieval go through here so the two can never drift apart —
the chunk size is derived from the same window the embedder actually honours.

The model executes on ONNX Runtime, not torch. The hosted app has between
690 MB and 2.7 GB of memory and throttles on overshoot; torch plus two fp32
transformers measured ~1.35 GB resident. See onnx_models.py for the numbers and
the reasoning.
"""
from config import EMBED_MODEL, EMBED_ONNX_FILE, EMBED_PROMPTS
from onnx_models import OnnxEmbeddings, build_tokenizer, model_max_length

# One embedder per (repo, file) for the whole process. Chroma is opened twice on
# a cold start — once by the index build, once by retrieval — and a second
# session of the same weights is a second 440 MB, measured, for no benefit.
_embedders = {}


def build_embeddings(model_name=None):
    name = model_name or EMBED_MODEL
    key = (name, EMBED_ONNX_FILE)
    if key not in _embedders:
        _embedders[key] = OnnxEmbeddings(
            name,
            filename=EMBED_ONNX_FILE,
            prompts=EMBED_PROMPTS.get(name),
            max_length=embedding_window(name),
        )
    return _embedders[key]


def embedding_window(model_name=None):
    """Token window the model actually honours, with safety caps."""
    return model_max_length(model_name or EMBED_MODEL)


def token_counter(model_name=None):
    """Count tokens the way the embedder will, for chunk-size decisions."""
    tokenizer = build_tokenizer(model_name or EMBED_MODEL)

    def count(text):
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    return count


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

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

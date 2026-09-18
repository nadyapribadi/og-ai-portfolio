"""Local inference on ONNX Runtime - the same models, without torch.

Why this exists
---------------
The deployed app loads two transformer models on every cold start. Measured on
the real import path, torch + sentence-transformers + both fp32 models cost
~1.35 GB resident:

    torch                     236 MB
    sentence-transformers     199 MB   (import side)
    multilingual-e5-small     478 MB   (weights)
    mmarco cross-encoder      437 MB   (weights)

Streamlit Community Cloud gives an app between 690 MB and 2.7 GB and throttles
it on overshoot, which is how this demo ended up serving "this app has gone
over its resource limits" instead of answers. The same two models as int8 ONNX
graphs are 118 MB and 119 MB on disk, and run on onnxruntime - already in the
dependency tree, because chromadb requires it. The retrieval design does not
change; only the backend that executes the matrices does.

Worth knowing before trusting that word "int8": onnxruntime's resident cost is
not the file size. Measured on linux/amd64 it expands a 118 MB int8 graph to
~440 MB once loaded - the weights stay int8 on disk and in the optimized graph,
but the runtime keeps more than one copy. Two models therefore still do not fit
in a 690 MB container, which is why retrieval.py loads the second one only when
memory.py says the container can afford it.

The classes below are shaped like the two interfaces the rest of the code
already expects - LangChain's Embeddings and CrossEncoder.predict(pairs) - so
ingest.py and retrieval.py never learn which backend is running.
"""
from __future__ import annotations

import os
import platform
from pathlib import Path

import numpy as np
from langchain_core.embeddings import Embeddings

# Tokenizers spawn their own thread pool, which then fights onnxruntime's for
# the two cores a free Community Cloud app gets. Nothing here needs it.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Weights are downloaded once per container into the Hugging Face cache. Point
# this somewhere writable when the home directory is not.
MODEL_CACHE = os.getenv("DEMO1_MODEL_CACHE") or None

# Threads onnxruntime may use per session. Fewer threads keep a shared,
# bandwidth-limited container responsive and its memory arena small.
THREADS = int(os.getenv("DEMO1_ORT_THREADS") or 2)


def _download(repo, filename):
    """Path to a file from a Hugging Face repo, downloaded once per host."""
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(repo_id=repo, filename=filename, cache_dir=MODEL_CACHE)
    )


def _session(onnx_path):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = THREADS
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # The arena caches freed activation blocks across runs. It is the faster
    # default and the more expensive one here: measured peak on the query path,
    # arena on versus off was 1162 MB versus 961 MB for late interaction and
    # 1246 MB versus 1088 MB for the cross-encoder. Predictable beats fast on a
    # host that throttles on resident memory.
    options.enable_cpu_mem_arena = False
    return ort.InferenceSession(
        str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def _tokenizer(repo, max_length):
    """A Rust tokenizer loaded from the repo's tokenizer.json.

    Padding is re-enabled explicitly: some exports ship a fixed 512-token
    padding strategy in tokenizer.json, which would waste a forward pass on
    every short batch.
    """
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(_download(repo, "tokenizer.json")))
    tokenizer.enable_truncation(max_length=max_length)
    pad_id = tokenizer.token_to_id("<pad>")
    pad_token = "<pad>"
    if pad_id is None:
        pad_id = tokenizer.token_to_id("[PAD]")
        pad_token = "[PAD]"
    tokenizer.enable_padding(pad_id=pad_id or 0, pad_token=pad_token)
    return tokenizer


def _batched(items, size):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def build_tokenizer(repo, max_length=None):
    """Public tokenizer handle — used for window checks and token counting."""
    tokenizer = _tokenizer(repo, max_length or 512)
    if max_length is None:
        # Counting must not silently cap at the truncation length.
        tokenizer.no_truncation()
    return tokenizer


def model_max_length(repo, default=512):
    """The window the encoder itself declares, in tokens.

    Read from the model's own config rather than hardcoded, because the chunk
    budget is derived from it and a wrong window silently drops the tail of
    every chunk.
    """
    import json

    config = json.loads(_download(repo, "config.json").read_text())
    for key in ("max_position_embeddings", "n_positions", "max_seq_length"):
        value = config.get(key)
        if isinstance(value, int) and 0 < value <= 8192:
            return value
    return default


def _cpu_flags():
    """CPU feature string, or "" where the platform has no cpuinfo."""
    try:
        return Path("/proc/cpuinfo").read_text()
    except OSError:
        return ""


def preferred_onnx_file(repo):
    """The graph variant that matches this CPU's vector instructions.

    Exports differ by publisher: the transformers.js ports ship one portable
    onnx/model_quantized.onnx, while sentence-transformers' own exports ship a
    graph per instruction set (model_qint8_arm64, model_quint8_avx2,
    model_qint8_avx512_vnni, ...). An int8 graph the CPU cannot execute
    natively still runs, but through slower generic kernels, so the match is
    what keeps inference on the fast path.
    """
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        return "onnx/model_qint8_arm64.onnx"
    flags = _cpu_flags()
    if "avx512_vnni" in flags:
        return "onnx/model_qint8_avx512_vnni.onnx"
    if "avx512f" in flags:
        return "onnx/model_qint8_avx512.onnx"
    # AVX2 is the floor on any x86 host that runs a current cloud image.
    return "onnx/model_quint8_avx2.onnx"


def _feeds(session, tokenizer, texts):
    """Token ids, attention mask, and type ids when the graph asks for them."""
    encodings = tokenizer.encode_batch(texts)
    feeds = {
        "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
        "attention_mask": np.array(
            [e.attention_mask for e in encodings], dtype=np.int64
        ),
    }
    if any(i.name == "token_type_ids" for i in session.get_inputs()):
        feeds["token_type_ids"] = np.array(
            [e.type_ids for e in encodings], dtype=np.int64
        )
    return feeds


def _mean_pool(hidden, mask):
    """Average token vectors, ignoring padding, then scale to unit length."""
    weights = mask[..., None].astype(np.float32)
    summed = (hidden * weights).sum(axis=1)
    counts = np.clip(weights.sum(axis=1), 1e-9, None)
    vectors = summed / counts
    norms = np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9, None)
    return (vectors / norms).astype(np.float32)


class OnnxEmbeddings(Embeddings):
    """Sentence embeddings from an int8 ONNX encoder, in batches."""

    def __init__(
        self,
        repo,
        filename="onnx/model_quantized.onnx",
        prompts=None,
        max_length=512,
        batch_size=8,
    ):
        self.repo = repo
        self.filename = filename
        self.prompts = prompts or {}
        self.batch_size = batch_size
        self.session = _session(_download(repo, filename))
        self.tokenizer = _tokenizer(repo, max_length)

    def _encode(self, texts, role):
        prefix = self.prompts.get(role, "")
        vectors = []
        for batch in _batched([f"{prefix}{text}" for text in texts], self.batch_size):
            feeds = _feeds(self.session, self.tokenizer, batch)
            hidden = self.session.run(None, feeds)[0]
            vectors.append(_mean_pool(hidden, feeds["attention_mask"]))
        if not vectors:
            return np.zeros((0, 1), dtype=np.float32)
        return np.vstack(vectors)

    def embed_documents(self, texts):
        return self._encode(list(texts), "passage").tolist()

    def embed_query(self, text):
        return self._encode([text], "query")[0].tolist()

    def token_vectors(self, texts, role):
        """L2-normalised vector per token, padding stripped."""
        prefix = self.prompts.get(role, "")
        vectors = []
        for batch in _batched([f"{prefix}{text}" for text in texts], self.batch_size):
            feeds = _feeds(self.session, self.tokenizer, batch)
            hidden = self.session.run(None, feeds)[0]
            for row, mask in zip(hidden, feeds["attention_mask"]):
                kept = row[mask.astype(bool)]
                norms = np.clip(
                    np.linalg.norm(kept, axis=1, keepdims=True), 1e-9, None
                )
                vectors.append((kept / norms).astype(np.float32))
        return vectors


class OnnxLateInteraction:
    """ColBERT-style MaxSim reranking on the embedder that is already loaded.

    A trained cross-encoder ranks better, but it is a second set of weights and
    the hosted app cannot afford two. MaxSim scores every query token against
    every passage token using vectors the running embedder already produces, so
    it adds no resident memory - only one forward pass per candidate.
    """

    def __init__(self, embedder):
        self.embedder = embedder

    def predict(self, pairs):
        questions = list(dict.fromkeys(question for question, _ in pairs))
        passages = list(dict.fromkeys(passage for _, passage in pairs))
        query_vectors = dict(
            zip(questions, self.embedder.token_vectors(questions, "query"))
        )
        passage_vectors = dict(
            zip(passages, self.embedder.token_vectors(passages, "passage"))
        )
        scores = []
        for question, passage in pairs:
            similarity = query_vectors[question] @ passage_vectors[passage].T
            scores.append(float(similarity.max(axis=1).mean()))
        return scores


class OnnxCrossEncoder:
    """Relevance scores from an int8 ONNX cross-encoder.

    Exposes the same predict([[query, passage], ...]) shape
    sentence-transformers does, so retrieval.py is unchanged.
    """

    def __init__(
        self,
        repo,
        filename="onnx/model_quantized.onnx",
        max_length=512,
        batch_size=8,
    ):
        self.repo = repo
        self.session = _session(_download(repo, filename))
        self.tokenizer = _tokenizer(repo, max_length)
        self.batch_size = batch_size

    def predict(self, pairs):
        scores = []
        for batch in _batched([tuple(pair) for pair in pairs], self.batch_size):
            logits = self.session.run(
                None, _feeds(self.session, self.tokenizer, batch)
            )[0]
            # Single-logit regression heads are the norm; a two-class head is
            # the other shape this has to survive.
            column = logits[:, 0] if logits.shape[-1] == 1 else logits[:, -1]
            scores.extend(float(value) for value in column)
        return scores

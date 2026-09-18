"""End-to-end guards for demo1.

The audit found that app.py called ask() with an argument retrieval.py did not
accept, and that both configured models had been retired — neither was caught
because nothing exercised the contract. These tests exist to catch that class
of failure.
"""
import inspect
import os
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import config  # noqa: E402
import retrieval  # noqa: E402


def test_ask_takes_exactly_the_arguments_app_passes():
    """app.py calls ask(question). Keep the two in step."""
    params = [
        p
        for name, p in inspect.signature(retrieval.ask).parameters.items()
        if p.default is inspect.Parameter.empty
    ]
    assert [p.name for p in params] == ["question"]


def test_model_ids_have_one_source_of_truth():
    assert retrieval.GROQ_MODEL_QUALITY == config.LLM_MODEL_QUALITY
    assert retrieval.GROQ_MODEL_FAST == config.LLM_MODEL_FAST


def test_content_to_text_handles_every_provider_shape():
    """Newer providers return content blocks, not a string."""
    assert retrieval._content_to_text("a\nb") == "a\nb"
    assert "a" in retrieval._content_to_text([{"type": "text", "text": "a"}])
    assert retrieval._content_to_text(None) == ""
    assert retrieval._content_to_text(42) == "42"


def test_only_questions_about_the_assistant_are_treated_as_such():
    """A capability answer is canned, so it must not swallow document questions.

    "apa saja yg akan dicek di sini?" is a real question about inspection in the
    specification and has to keep going to retrieval; "apa saja yg bisa dibahas
    di chat ini?" is about the assistant and has no excerpt to retrieve.
    """
    about_the_app = [
        "apa saja yg bisa dibahas di chat ini?",
        "apa yang bisa ditanyakan?",
        "bisa dibahas apa saja?",
        "What can you do?",
        "what questions can I ask?",
        "what is this app?",
        "topik apa yang tersedia?",
    ]
    about_the_documents = [
        "apa saja yg akan dicek di sini?",
        "apa yang harus disiapkan sebelum pengujian tekanan?",
        "What pressure test is required and for how long?",
        "What is the minimum design pressure for the deluge skid?",
        "Apa persyaratan desain deluge skid?",
    ]
    for question in about_the_app:
        assert retrieval.is_capability_question(question), question
    for question in about_the_documents:
        assert not retrieval.is_capability_question(question), question


def test_vectorstore_path_is_configurable(monkeypatch):
    """Deployments and model bake-offs relocate the index via environment."""
    assert isinstance(config.VECTORSTORE_DIR, Path)


def test_index_presence_is_checked_without_opening_chroma(tmp_path, monkeypatch):
    """A store counts as present only when it actually holds vectors.

    A file-size check is not enough: opening Chroma on a directory initialises
    a full schema, so an empty index looks healthy. And opening a client at all
    is what caused SQLITE_READONLY_DBMOVED in deployment.
    """
    import sqlite3

    import ingest

    store = tmp_path / "store"
    store.mkdir()
    monkeypatch.setattr(ingest, "VECTORSTORE", store)
    monkeypatch.setattr(ingest, "index_signature", lambda: "signature-current")
    assert ingest.index_present() is False

    database = store / "chroma.sqlite3"
    database.write_bytes(b"not a database at all")
    assert ingest.index_present() is False, "an unreadable file is not an index"

    (store / ingest.SIGNATURE_FILE).write_text("signature-current")

    database.unlink()
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE embeddings (id TEXT)")
    connection.commit()
    connection.close()
    assert ingest.index_present() is False, "an empty table is not an index"

    connection = sqlite3.connect(database)
    connection.execute("INSERT INTO embeddings VALUES ('a')")
    connection.commit()
    connection.close()
    assert ingest.index_present() is True


def test_an_index_from_older_code_is_rebuilt(tmp_path, monkeypatch):
    """A parser fix has to reach the deployed app.

    The container keeps its filesystem between updates, so an index built by the
    previous parser stayed in place and the demo kept answering from stale,
    mis-attributed chunks after the fix was deployed.
    """
    import sqlite3

    import ingest

    store = tmp_path / "store"
    store.mkdir()
    monkeypatch.setattr(ingest, "VECTORSTORE", store)

    connection = sqlite3.connect(store / "chroma.sqlite3")
    connection.execute("CREATE TABLE embeddings (id TEXT)")
    connection.execute("INSERT INTO embeddings VALUES ('a')")
    connection.commit()
    connection.close()

    monkeypatch.setattr(ingest, "index_signature", lambda: "signature-now")
    assert ingest.index_present() is False, "no signature: built by older code"

    (store / ingest.SIGNATURE_FILE).write_text("signature-then")
    assert ingest.index_present() is False, "different signature: code changed"

    (store / ingest.SIGNATURE_FILE).write_text("signature-now\n")
    assert ingest.index_present() is True, "matching signature: keep the index"


def test_chroma_client_cache_can_be_cleared():
    """Chroma reuses a client per path; a rebuilt path needs a fresh one."""
    from chromadb.api.shared_system_client import SharedSystemClient

    import embeddings

    SharedSystemClient._identifier_to_system["probe-identifier"] = object()
    embeddings.clear_chroma_client_cache()
    assert "probe-identifier" not in SharedSystemClient._identifier_to_system


@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="requires GROQ_API_KEY")
def test_live_answer_is_cited():
    result = retrieval.ask("What are the life saving rules?")
    assert result["answer"].strip()
    assert result["sources"]


def test_container_limit_is_read_from_cgroup_files(tmp_path, monkeypatch):
    """The rerank decision depends on this reading being right."""
    import memory

    limit_file = tmp_path / "memory.max"
    monkeypatch.setattr(memory, "LIMIT_FILES", (limit_file,))

    limit_file.write_text("max\n")
    assert memory.container_memory_limit_mb() is None, "cgroup v2 spells it 'max'"

    limit_file.write_text("724828160\n")
    assert memory.container_memory_limit_mb() == 691

    limit_file.write_text("9223372036854771712\n")
    assert memory.container_memory_limit_mb() is None, "cgroup v1 sentinel is not a limit"

    limit_file.write_text("not a number\n")
    assert memory.container_memory_limit_mb() is None

    limit_file.unlink()
    assert memory.container_memory_limit_mb() is None


def test_rerank_strategy_follows_the_container_ceiling(monkeypatch):
    """A 690 MB container must not load a second ~310 MB model.

    Loading it anyway is the failure this whole change exists to prevent:
    Streamlit Community Cloud serves "over its resource limits" instead of
    answers once the app goes past its allocation.
    """
    import retrieval

    def strategy_for(limit, container=False):
        monkeypatch.setattr(retrieval, "_strategy", None)
        monkeypatch.setattr(
            retrieval, "container_memory_limit_mb", lambda: limit
        )
        monkeypatch.setattr(
            retrieval, "inside_container", lambda: container
        )
        return retrieval.rerank_strategy()

    assert strategy_for(690) == "late-interaction"
    assert strategy_for(1400) == "cross-encoder"
    assert strategy_for(2700) == "cross-encoder"
    assert strategy_for(None) == "cross-encoder", "no ceiling means a laptop"
    assert strategy_for(None, container=True) == "late-interaction", (
        "an unreadable ceiling inside a container is not evidence of room"
    )
    monkeypatch.setattr(retrieval, "_strategy", None)


def test_the_deployed_path_never_imports_torch():
    """Torch is still installed (the bake-off scripts use it), so a stray import
    would be silent — and would put 236 MB back on the hosted budget."""
    import subprocess
    import sys as _sys

    script = (
        f"import sys; sys.path.insert(0, {str(SRC)!r});"
        "import retrieval, embeddings, onnx_models;"
        "print(sorted(m for m in ('torch', 'sentence_transformers', 'transformers')"
        " if m in sys.modules))"
    )
    result = subprocess.run(
        [_sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]", result.stdout

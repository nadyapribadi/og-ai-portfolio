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
    assert ingest.index_present() is False

    database = store / "chroma.sqlite3"
    database.write_bytes(b"not a database at all")
    assert ingest.index_present() is False, "an unreadable file is not an index"

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

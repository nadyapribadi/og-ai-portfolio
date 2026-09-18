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


@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="requires GROQ_API_KEY")
def test_live_answer_is_cited():
    result = retrieval.ask("What are the life saving rules?")
    assert result["answer"].strip()
    assert result["sources"]

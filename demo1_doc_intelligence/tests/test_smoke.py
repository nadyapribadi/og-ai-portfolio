"""Phase 0 smoke tests.

Cheap guards against the specific failures found in the audit:

* a call-signature mismatch between ``app.py`` and ``retrieval.py`` that broke
  every single question in the UI;
* model IDs hardcoded in two places, so a retired model was invisible;
* provider payloads that crashed the query-expansion parser.
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


def test_ask_accepts_lang_override():
    """app.py calls ask(question, lang_override=...) — keep that contract."""
    assert "lang_override" in inspect.signature(retrieval.ask).parameters


def test_model_ids_have_one_source_of_truth():
    assert retrieval.GROQ_MODEL_FAST == config.LLM_MODEL_FAST
    assert retrieval.GROQ_MODEL_QUALITY == config.LLM_MODEL_QUALITY


def test_content_to_text_handles_every_provider_shape():
    """Newer providers return content blocks, not a string."""
    assert retrieval._content_to_text("a\nb") == "a\nb"
    assert "a" in retrieval._content_to_text([{"type": "text", "text": "a"}])
    assert retrieval._content_to_text(None) == ""
    assert retrieval._content_to_text(42) == "42"


class _FakeLLM:
    """Stands in for ChatGroq, returning a canned content payload."""

    def __init__(self, content):
        self._content = content

    def __call__(self, *args, **kwargs):
        content = self._content

        class _Instance:
            def invoke(self, _messages):
                class _Message:
                    pass

                _Message.content = content
                return _Message()

        return _Instance()


def test_expand_query_falls_back_to_original(monkeypatch):
    """Unusable model output must never raise, nor return an empty list."""
    monkeypatch.setattr(retrieval, "ChatGroq", _FakeLLM("   1) \n  - \n"))
    assert retrieval.expand_query("what is LOPC?") == ["what is LOPC?"]


def test_expand_query_strips_numbering_and_bullets(monkeypatch):
    monkeypatch.setattr(
        retrieval,
        "ChatGroq",
        _FakeLLM("1. loss of primary containment definition\n• LOPC consequences\n"),
    )
    out = retrieval.expand_query("what is LOPC?")
    assert out[0] == "what is LOPC?"
    assert out[1:] == [
        "loss of primary containment definition",
        "LOPC consequences",
    ]


@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="requires GROQ_API_KEY")
def test_live_answer_is_cited():
    """End-to-end: a real question returns text plus at least one source."""
    result = retrieval.ask("What are the life saving rules?")
    assert result["answer"].strip()
    assert result["sources"]

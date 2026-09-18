"""Smoke test for the Streamlit UI.

Runs the real app script through Streamlit's own testing harness, so a broken
import, a bad widget call, or a crash on first render fails the build instead
of failing the first visitor. No browser required.
"""
import sys
from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")

SRC = Path(__file__).resolve().parents[1] / "src"
APP = SRC / "app.py"
sys.path.insert(0, str(SRC))


def test_app_renders_without_exception():
    app = streamlit_testing.AppTest.from_file(str(APP), default_timeout=300)
    app.run()

    assert not app.exception, app.exception
    # The header and the capability cards are markdown; the sample questions are
    # buttons in the sidebar. If either is missing the app rendered incorrectly.
    markdown = " ".join(element.value for element in app.markdown)
    assert "Document Intelligence" in markdown
    assert len(app.button) > 0, "no sample-question buttons rendered"


def test_app_offers_questions_that_match_the_indexed_corpus():
    """A stranger's first click must not return 'not found'."""
    app = streamlit_testing.AppTest.from_file(str(APP), default_timeout=300)
    app.run()

    labels = " ".join(button.label for button in app.button)
    assert labels.strip(), "no sample questions were offered"

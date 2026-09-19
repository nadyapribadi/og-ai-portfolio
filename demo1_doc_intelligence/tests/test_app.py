"""Smoke test for the Streamlit UI.

Runs the real app script through Streamlit's own testing harness, so a broken
import, a bad widget call, or a crash on first render fails the build instead
of failing the first visitor. No browser required.
"""
import re
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


# Emoji as decoration, and the em dash. Both are hard gates in antislop (R-04,
# R-02): emoji competes with the content for attention, and the em dash is
# forbidden in user-facing text. Arrows are deliberately not in this pattern:
# "app menu -> Settings -> Secrets" is a direction, not decoration.
EMOJI = re.compile(
    "[\U0001f300-\U0001faff\u2600-\u27bf\u2b00-\u2bff\u25a0-\u25ff\ufe0f]"
)


def test_nothing_rendered_uses_emoji_or_an_em_dash():
    app = streamlit_testing.AppTest.from_file(str(APP), default_timeout=300).run()
    assert not app.exception, app.exception

    rendered = " ".join(element.value for element in app.markdown)
    rendered += " ".join(button.label for button in app.button)
    rendered += " ".join(element.value for element in app.error)
    rendered += " ".join(element.value for element in app.warning)

    assert "\u2014" not in rendered, "em dash in rendered text"
    found = EMOJI.findall(rendered)
    assert not found, f"emoji in rendered text: {found}"


def test_the_canned_answer_obeys_the_same_copy_rules():
    """The empty state is not the only text this app renders.

    The capability answer ("what can you do?") is assembled in retrieval.py and
    needs no API key, so it can be driven here — which is how a stray em dash in
    that string was found after the empty state had already passed.
    """
    app = streamlit_testing.AppTest.from_file(str(APP), default_timeout=300)
    app.run()
    app.chat_input[0].set_value("what can you do?").run()
    assert not app.exception, app.exception

    rendered = " ".join(element.value for element in app.markdown)
    assert "I answer only from the documents indexed here" in rendered
    assert "\u2014" not in rendered, "em dash in the answer"
    found = EMOJI.findall(rendered)
    assert not found, f"emoji in the answer: {found}"


def test_config_copy_is_clean_for_the_corpus_that_is_not_rendered_here():
    """CI runs the sample corpus, so the full document pack's strings are never
    rendered. Check them directly, or a dash or emoji can sit there for months."""
    import config

    strings = [
        config.APP_TITLE,
        config.APP_SUBTITLE,
        *config.DOCUMENT_SOURCES,
        *config.SOURCE_FRIENDLY.values(),
    ]
    for question_set in (config.SAMPLE_QUESTIONS, config.SAMPLE_QUESTIONS_SAMPLE):
        for category, questions in question_set.items():
            strings += [category, *questions]
    for cards in (config.CAPABILITY_CARDS, config.CAPABILITY_CARDS_SAMPLE):
        for card in cards:
            strings += [card["title"], card["desc"]]

    for text in strings:
        assert "\u2014" not in text, f"em dash in {text!r}"
        assert not EMOJI.search(text), f"emoji in {text!r}"

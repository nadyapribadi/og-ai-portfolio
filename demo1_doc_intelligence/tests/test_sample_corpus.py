"""Does a stranger's first click work — on the corpus the deploy actually has?

The hosted app answers from the bundled sample corpus, because the real IOGP
and JIP33 PDFs are copyrighted and never committed. Every question the sidebar
offers there must therefore retrieve its own answer: a first click that returns
"not found" is the demo failing in public.

This runs only when a sample index has been built, which is what CI does before
pytest:

    DEMO1_DOCS_DIR=demo1_doc_intelligence/sample_docs \\
    DEMO1_VECTORSTORE=/tmp/vs python demo1_doc_intelligence/src/ingest.py
    DEMO1_VECTORSTORE=/tmp/vs python -m pytest demo1_doc_intelligence/tests -q

Each row is a sidebar question and the sentence that answers it. The check is
two-fold, and both halves are the parts that failed in production: the sentence
has to be in a retrieved excerpt, and a claim citing the clause that sentence
belongs to has to survive validate_claims().
"""
import os
import sys
from pathlib import Path

import pytest
from langchain_core.documents import Document

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

if not os.getenv("DEMO1_VECTORSTORE"):
    pytest.skip(
        "needs a built sample index; set DEMO1_VECTORSTORE", allow_module_level=True
    )

import answer  # noqa: E402
import config  # noqa: E402
import retrieval  # noqa: E402

TRS = "S-900v2026-01 TRS.md"
QRS = "S-900Qv2026-01 QRS.md"

CASES = [
    ("What is the minimum design pressure for the deluge skid?", TRS, 4, "4.2.1",
     "The deluge skid shall be designed for a minimum design pressure of 16 bar"),
    ("What ingress protection rating is required for electrical enclosures?", TRS, 4,
     "4.1.1",
     "Electrical equipment enclosures shall be at least IP56 in accordance with IEC 60529."),
    ("What surface preparation standard applies to steel structures?", TRS, 5, "5.1.1",
     "Surface preparation for steel structures shall be in accordance with ISO 12944-4."),
    ("What pressure test is required and for how long?", QRS, 3, "3.2.1",
     "Each skid shall be hydrostatically tested at 1.5 times the design pressure for a minimum of 30 minutes."),
    ("What documents must the manufacturing record book contain?", QRS, 4, "4.1.1",
     "The manufacturer shall provide a manufacturing record book containing material certificates"),
    ("When must conformity assessment be completed?", QRS, 4, "5.1.1",
     "Conformity assessment activities shall be completed before shipment."),
    ("Berapa tekanan desain minimum untuk deluge skid?", TRS, 4, "4.2.1",
     "The deluge skid shall be designed for a minimum design pressure of 16 bar"),
    ("Berapa lama pengujian tekanan hidrostatik berlangsung?", QRS, 3, "3.2.1",
     "Each skid shall be hydrostatically tested at 1.5 times the design pressure for a minimum of 30 minutes."),
]


def test_every_offered_question_is_answerable():
    offered = [
        question
        for questions in config.SAMPLE_QUESTIONS_SAMPLE.values()
        for question in questions
    ]
    assert [case[0] for case in CASES] == offered, (
        "the sidebar and this test have drifted apart"
    )


def test_what_can_i_ask_is_answered_without_the_model():
    """No excerpt answers a question about the app, so the model said "not found".

    It is answered from configuration instead — which also means this test needs
    no API key.
    """
    result = retrieval.ask("apa saja yg bisa dibahas di chat ini?")
    assert not result["not_found"]
    assert "S-900Qv2026-01 QRS.md" in result["answer"]
    assert "What pressure test is required and for how long?" in result["answer"]
    assert result["claims"] == [], "a canned answer claims nothing from a document"


@pytest.mark.parametrize("question,source,page,clause,quote", CASES)
def test_question_retrieves_its_own_answer(question, source, page, clause, quote):
    chunks = retrieval.search_chunks(retrieval.load_vectorstore("en"), question)
    assert chunks, "no excerpts retrieved at all"

    # "Supported by", not "contains as a substring": each chunk opens with a
    # label that repeats its heading, and on this corpus the heading is often the
    # first line of a wrapped sentence, so the label sits between "with" and the
    # rest. quote_supported() is the pipeline's own contract — ask it, rather
    # than a stricter rule the product never promised.
    hit = [
        chunk
        for chunk in chunks
        if answer.quote_supported(quote, chunk.page_content)
    ]
    assert hit, (
        f"no retrieved excerpt supports the answer to {question!r}; got "
        + ", ".join(
            f"{c.metadata.get('source_file')} p.{c.metadata.get('page')}"
            for c in chunks
        )
    )
    assert any(
        chunk.metadata.get("source_file") == source
        and int(chunk.metadata.get("page", 0)) == page
        for chunk in hit
    ), f"the supporting excerpt is not {source} p.{page}"

    # A correct model cites the clause the sentence belongs to — which is not
    # always the excerpt's own id. That claim has to survive the guardrail.
    draft = answer.DraftAnswer(
        claims=[
            answer.Claim(
                text=quote,
                clause_id=clause,
                source_file=source,
                page=page,
                quote=quote,
            )
        ]
    )
    accepted, rejected = answer.validate_claims(draft, chunks)
    assert accepted, f"the guardrail rejected a correct citation: {rejected}"

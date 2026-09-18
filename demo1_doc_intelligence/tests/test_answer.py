"""Tests for the citation guardrail.

These exist because the product promise is "every answer can be verified
against the source". A model that invents a page number, or quotes a sentence
that is not in the document, must be caught here rather than by the user.
"""
import sys
from pathlib import Path

from langchain_core.documents import Document

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import answer  # noqa: E402


def chunk(text, clause="8.1", page=29, source="S-737v2026-03 TRS.pdf"):
    return Document(
        page_content=text,
        metadata={
            "source_file": source,
            "clause_id": clause,
            "page": page,
            "doc_family": "S-737",
            "doc_role": "TRS",
        },
    )


EXCERPT = (
    "Surface preparation for steel structures shall be in accordance "
    "with ISO 12944-4."
)


def test_verbatim_quote_is_supported():
    assert answer.quote_supported(
        "Surface preparation for steel structures shall be in accordance with ISO 12944-4.",
        EXCERPT,
    )


def test_light_paraphrase_still_supported():
    """Whitespace and case differences must not reject a real quote."""
    assert answer.quote_supported(
        "surface   preparation for STEEL structures shall be", EXCERPT
    )


def test_unrelated_quote_is_rejected():
    assert not answer.quote_supported(
        "Deluge valves shall be rated for the design pressure of the system.",
        EXCERPT,
    )


def test_fabricated_citation_is_rejected():
    """A page the retrieval layer never returned must not be citable."""
    chunks = [chunk(EXCERPT, page=29)]
    draft = answer.DraftAnswer(
        claims=[
            answer.Claim(
                text="Coating shall follow ISO 12944-4.",
                clause_id="8.1",
                source_file="S-737v2026-03 TRS.pdf",
                page=99,                      # never retrieved
                quote="Surface preparation for steel structures",
            )
        ]
    )
    accepted, rejected = answer.validate_claims(draft, chunks)
    assert accepted == []
    assert rejected and "not one of the retrieved excerpts" in rejected[0][1]


def test_fabricated_quote_is_rejected():
    chunks = [chunk(EXCERPT, page=29)]
    draft = answer.DraftAnswer(
        claims=[
            answer.Claim(
                text="Deluge valves shall be painted blue.",
                clause_id="8.1",
                source_file="S-737v2026-03 TRS.pdf",
                page=29,
                quote="Deluge valves shall be painted blue before shipment",
            )
        ]
    )
    accepted, rejected = answer.validate_claims(draft, chunks)
    assert accepted == []
    assert rejected and "quote does not appear" in rejected[0][1]


def test_valid_claim_survives_validation():
    chunks = [chunk(EXCERPT, page=29)]
    draft = answer.DraftAnswer(
        claims=[
            answer.Claim(
                text="Surface preparation follows ISO 12944-4.",
                clause_id="8.1",
                source_file="S-737v2026-03 TRS.pdf",
                page=29,
                quote="Surface preparation for steel structures shall be in accordance with ISO 12944-4.",
            )
        ]
    )
    accepted, rejected = answer.validate_claims(draft, chunks)
    assert len(accepted) == 1 and rejected == []


def test_clause_inside_a_chunk_is_citable():
    """Chunks are clause-sized, not clause-exclusive.

    The deployed sample corpus served "what pressure test is required and for
    how long?" with the right excerpt — a chunk whose own id is 3.1.1 but whose
    body carries 3.2.1, the clause that states the test. The model cited 3.2.1,
    the guardrail demanded 3.1.1, and the app answered "not found" while holding
    the answer. A clause number that really appears in the excerpt is citable.
    """
    body = (
        "[S-900Qv2026-01 QRS.md | §3.1.1 The purchaser shall be given access | p.3]\n"
        "3.1.2 Inspection points shall be recorded in the inspection and test plan.\n"
        "### 3.2 Pressure testing\n"
        "3.2.1 Each skid shall be hydrostatically tested at 1.5 times the design\n"
        "pressure for a minimum of 30 minutes."
    )
    chunks = [chunk(body, clause="3.1.1", page=3, source="S-900Qv2026-01 QRS.md")]
    draft = answer.DraftAnswer(
        claims=[
            answer.Claim(
                text="Each skid is hydrostatically tested for at least 30 minutes.",
                clause_id="3.2.1",
                source_file="S-900Qv2026-01 QRS.md",
                page=3,
                quote="Each skid shall be hydrostatically tested at 1.5 times the design pressure for a minimum of 30 minutes.",
            )
        ]
    )
    accepted, rejected = answer.validate_claims(draft, chunks)
    assert len(accepted) == 1, rejected
    assert rejected == []
    assert answer.contained_clauses(chunks[0]) == ["3.1.2", "3.2", "3.2.1"]


def test_clause_absent_from_the_excerpt_is_still_rejected():
    """Tolerating finer citations must not tolerate invented ones."""
    body = "[X.md | §3.1.1 label | p.3]\n3.1.1 A test certificate shall be provided."
    chunks = [chunk(body, clause="3.1.1", page=3, source="X.md")]
    draft = answer.DraftAnswer(
        claims=[
            answer.Claim(
                text="Each skid is hydrostatically tested for 30 minutes.",
                clause_id="3.2.1",                     # not in this excerpt
                source_file="X.md",
                page=3,
                quote="A test certificate shall be provided.",
            )
        ]
    )
    accepted, rejected = answer.validate_claims(draft, chunks)
    assert accepted == []
    assert rejected and "not one of the retrieved excerpts" in rejected[0][1]

    # A number that merely contains the cited one must not count either.
    assert not answer.clause_present("3.2.1", "13.2.14 is a different clause")
    # Nor may a bare clause number match the "1" of a decimal.
    assert not answer.clause_present("1", "tested at 1.5 times the design pressure")


def test_excerpt_header_offers_the_sections_it_contains():
    """Front matter often has no clause id, only a markdown section heading."""
    body = (
        "[S-900v2026-01 TRS.md | (preamble) | p.2]\n"
        "## 1 Scope\n"
        "This specification covers the design of deluge skid assemblies.\n"
        "### 2.1 Normative references\n"
        "The following documents are referred to in the text."
    )
    excerpt_chunk = chunk(body, clause="", page=2, source="S-900v2026-01 TRS.md")
    assert answer.contained_clauses(excerpt_chunk) == ["1", "2.1"]
    assert answer.clause_present("1", excerpt_chunk.page_content)

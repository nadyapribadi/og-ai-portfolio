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

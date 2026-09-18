"""Regression tests for the PDF → clause parser.

Every test here corresponds to a defect that was observed on the real corpus,
not a hypothetical one.
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import parse  # noqa: E402


def page(number, text, name="X.pdf"):
    return parse.Page(name, number, text)


def words(text):
    return len(text.split())


def test_running_header_is_stripped():
    """A header repeated on most pages made every page look alike."""
    pages = [page(i, f"Specification for Widgets\nbody {i}") for i in range(1, 6)]
    cleaned = parse.strip_running_headers(pages)
    assert all("Specification for Widgets" not in p.text for p in cleaned)
    assert all(f"body {i}" in cleaned[i - 1].text for i in range(1, 6))


def test_bare_sub_clause_needs_a_seen_parent():
    """'7.2.1' alone on a line is a clause only when 7.2 (or 7) came first."""
    pages = [page(1, "7.2 Welding\n7.2.1\nShall be qualified.\n")]
    ids = [c.clause_id for c in parse.split_into_clauses(pages)]
    assert "7.2.1" in ids


def test_bare_sub_clause_inherits_the_parent_title():
    """'8.1.2' has no heading of its own; without the parent title its chunk
    loses the context that makes it findable."""
    pages = [page(1, "8.1 Protective coatings\n8.1.1\nShall follow ISO 12944-4.\n")]
    clause = next(c for c in parse.split_into_clauses(pages) if c.clause_id == "8.1.1")
    assert clause.title == ""
    assert clause.parent_title == "Protective coatings"
    assert "Protective coatings" in clause.context()


def test_reference_number_is_not_a_clause():
    """'29 CFR 1910.147' used to be parsed as clause 1910.147 and swallow a whole section."""
    pages = [page(1, "1910.147\nSome body text about OSHA rules.\n")]
    ids = [c.clause_id for c in parse.split_into_clauses(pages)]
    assert "1910.147" not in ids


def test_clause_ids_do_not_leak_between_documents():
    """Clause numbering is per document."""
    pages = [
        page(1, "7.2 Welding\nbody\n", name="A.pdf"),
        page(1, "7.2.1\nbody\n", name="B.pdf"),
    ]
    clauses = parse.split_into_clauses(pages)
    b_ids = [c.clause_id for c in clauses if c.source_file == "B.pdf"]
    assert "7.2.1" not in b_ids


def test_table_of_contents_is_skipped():
    """The TOC repeats every heading, so retrieval used to return it as an answer."""
    toc = "Contents\nForeword .......... 4\nScope .......... 6\nReferences .......... 9\n"
    clauses = parse.split_into_clauses([page(1, toc)])
    assert clauses == []


def test_heading_styles_across_the_three_document_families():
    assert parse.detect_heading("8.1 Protective coatings") == ("8.1", "Protective coatings")
    assert parse.detect_heading("2. The Life-Saving Rules") == ("2", "The Life-Saving Rules")
    assert parse.detect_heading("Part C - Tier 1 and Tier 2 Indicators") == (
        "Part C",
        "Tier 1 and Tier 2 Indicators",
    )
    assert parse.detect_heading("Scope") == ("", "Scope")


def test_chunks_never_span_a_page_break():
    """A chunk carrying text from two pages cannot be cited accurately."""
    clause = parse.Clause(
        source_file="X.pdf",
        doc_family="X",
        doc_role="spec",
        clause_id="1",
        section="",
        title="Scope",
        lines=[(1, "alpha " * 40), (2, "bravo " * 40)],
    )
    chunks = parse.chunk_clause(clause, words, 60)
    assert chunks, "expected at least one chunk"
    for text, _ in chunks:
        assert not ("alpha" in text and "bravo" in text)
    assert {p for _, p in chunks} == {1, 2}


def test_chunks_respect_the_window():
    clause = parse.Clause(
        source_file="X.pdf",
        doc_family="X",
        doc_role="spec",
        clause_id="1",
        section="",
        title="Scope",
        lines=[(1, "word " * 300)],
    )
    chunks = parse.chunk_clause(clause, words, 100)
    assert chunks
    assert all(words(text) <= 100 for text, _ in chunks)


def test_describe_document_maps_families_and_roles():
    assert parse.describe_document("459.pdf") == ("IOGP 459", "Report")
    assert parse.describe_document("S-737v2026-03 TRS.pdf") == ("S-737", "TRS")
    assert parse.describe_document("S-737Qv2026-03 QRS.pdf") == ("S-737", "QRS")
    assert parse.describe_document("S-719Jv2025-01 TRS with Justification.pdf") == (
        "S-719",
        "TRS+Justification",
    )

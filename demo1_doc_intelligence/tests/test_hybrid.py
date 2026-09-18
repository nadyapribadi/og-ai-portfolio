"""Tests for keyword search, rank fusion and document routing."""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import hybrid  # noqa: E402
import routing  # noqa: E402


def test_tokenizer_keeps_document_codes_intact():
    """"S-737" must stay one token, not become "s" and "737"."""
    assert "s-737" in hybrid.tokenize("What does S-737 require?")
    assert "12944-4" in hybrid.tokenize("coating to ISO 12944-4")


def test_tokenizer_matches_singular_and_plural():
    """BM25 has no stemmer, so "coating" must match a clause titled "coatings"."""
    assert "coating" in hybrid.tokenize("protective coatings")
    assert "coatings" in hybrid.tokenize("protective coatings")
    assert "rule" in hybrid.tokenize("life saving rules")
    # Short words and double-s words must not be mangled.
    assert hybrid.tokenize("gas") == ["gas"]
    assert hybrid.tokenize("class") == ["class"]


def test_bm25_matches_across_number():
    corpus = [
        hybrid.tokenize("noise emitting equipment"),
        hybrid.tokenize("protective coatings"),
    ]
    bm25 = hybrid.BM25(corpus)
    assert bm25.top_k("protective coating", 2) == [1]


def test_bm25_ranks_the_matching_document_first():
    corpus = [
        hybrid.tokenize("deluge skid design requirements"),
        hybrid.tokenize("noise emitting equipment"),
        hybrid.tokenize("water mist fire protection"),
    ]
    bm25 = hybrid.BM25(corpus)
    assert bm25.top_k("deluge skid", 3)[0] == 0
    assert bm25.top_k("water mist", 3)[0] == 2


def test_bm25_returns_nothing_for_absent_terms():
    bm25 = hybrid.BM25([hybrid.tokenize("deluge skid")])
    assert bm25.top_k("photosynthesis", 3) == []


def test_bm25_respects_an_allowed_subset():
    corpus = [
        hybrid.tokenize("deluge skid"),
        hybrid.tokenize("deluge skid again"),
    ]
    bm25 = hybrid.BM25(corpus)
    assert bm25.top_k("deluge skid", 3, allowed={1}) == [1]


def test_rrf_combines_two_rankings():
    fused = hybrid.reciprocal_rank_fusion([["a", "b", "c"], ["c", "a"]])
    assert set(fused) == {"a", "b", "c"}
    assert fused[0] == "a"          # ranked well by both


def test_family_inference_from_code_and_from_domain_words():
    assert routing.infer_family("What does S-737 require?") == "S-737"
    assert routing.infer_family("What are the life saving rules?") == "459"
    assert routing.infer_family("What is LOPC?") == "456"
    assert routing.infer_family("What noise limits apply?") == "S-717"
    assert routing.infer_family("What is the weather today?") is None


def test_role_inference():
    assert routing.infer_role("What are the quality requirements?") == "QRS"
    assert routing.infer_role("What is the design requirement?") == "TRS"
    assert routing.infer_role("Tell me about deluge skids") is None

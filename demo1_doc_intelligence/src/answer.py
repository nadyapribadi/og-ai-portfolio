"""Structured, verifiable answers.

The model does not get to invent citations. It may only select from the clause
ids the retrieval layer already found, and every claim must quote text that is
actually present in the cited excerpt. Claims that fail either check are
discarded before the user ever sees them.

This is the difference between "the model was asked to cite" and "the system
guarantees the citation".
"""
from __future__ import annotations

import re
from typing import List

from pydantic import BaseModel, Field

# How many consecutive words of a quote must appear verbatim in the excerpt.
# Requiring the whole quote would reject harmless paraphrase; requiring a single
# word would accept fabrication.
QUOTE_WINDOW = 6


class Claim(BaseModel):
    """A single statement, tied to the clause it came from."""

    text: str = Field(
        description="One statement that answers the question, in the user's language"
    )
    clause_id: str = Field(
        description="Clause id exactly as shown in the excerpts, e.g. '8.1'"
    )
    source_file: str = Field(description="Document filename exactly as shown")
    page: int = Field(description="Page number exactly as shown in the excerpt header")
    quote: str = Field(
        description="A sentence copied verbatim from that excerpt supporting the claim"
    )


class DraftAnswer(BaseModel):
    claims: List[Claim] = Field(default_factory=list)
    not_found: bool = Field(
        default=False,
        description="True only when no excerpt contains the answer",
    )


def _normalise(text):
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def quote_supported(quote, excerpt, window=QUOTE_WINDOW):
    """Is the quote actually present in the excerpt?

    Any `window` consecutive words matching is enough, which tolerates light
    paraphrase while still rejecting text that is not in the source.
    """
    words = _normalise(quote).split()
    haystack = _normalise(excerpt)
    if not words or not haystack:
        return False
    if len(words) < window:
        return " ".join(words) in haystack
    return any(
        " ".join(words[i : i + window]) in haystack
        for i in range(len(words) - window + 1)
    )


def validate_claims(draft, chunks):
    """Split draft claims into (accepted, rejected-with-reason)."""
    index = {
        (
            chunk.metadata.get("source_file"),
            int(chunk.metadata.get("page", 0)),
            chunk.metadata.get("clause_id") or "",
        ): chunk
        for chunk in chunks
    }

    accepted, rejected = [], []
    for claim in draft.claims:
        chunk = index.get((claim.source_file, claim.page, claim.clause_id))
        if chunk is None:
            rejected.append((claim, "citation is not one of the retrieved excerpts"))
            continue
        if not quote_supported(claim.quote, chunk.page_content):
            rejected.append((claim, "quote does not appear in the cited excerpt"))
            continue
        accepted.append(claim)
    return accepted, rejected


SYSTEM_PROMPT = """You answer questions about upstream oil and gas standards.

You are given numbered excerpts. Each excerpt header carries its document, its
clause id and its page.

Return your answer as structured claims. For every claim you MUST:
  * copy clause_id, source_file and page exactly from the excerpt header;
  * put a sentence in quote that is copied verbatim from that excerpt.

Every part of your answer must come from the excerpts. Never use general
knowledge to fill gaps. If no excerpt answers the question, return no claims and
set not_found to true.

Answer in the same language the question was asked in. Keep technical terms
(Tier 1, LOPC, deluge skid) in English even when answering in Bahasa Indonesia.

Excerpts:
{context}"""


def build_messages(question, context):
    from langchain_core.messages import HumanMessage, SystemMessage

    return [
        SystemMessage(content=SYSTEM_PROMPT.format(context=context)),
        HumanMessage(content=question),
    ]


def structured_llm(llm):
    """Ask the provider for a schema-constrained response when it supports it."""
    try:
        return llm.with_structured_output(DraftAnswer)
    except Exception:
        return None

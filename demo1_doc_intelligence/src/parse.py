"""Turn IOGP / JIP33 PDFs into a document → section → clause structure.

The corpus mixes three heading styles:

* JIP33 (S-737 / S-717 / S-719) — decimal clause numbers, e.g.
  ``8.1 Protective coatings``. Sub-clause numbers sometimes sit alone on a line
  with the body text starting on the next line.
* IOGP 459 — ``2. The Life-Saving Rules``
* IOGP 456 — ``Part C - Tier 1 and Tier 2 Indicators`` plus unnumbered named
  sections such as ``Scope`` and ``Definitions``.

Every page also repeats a running header ("Specification for Deluge Skids",
"Process safety − Recommended practice on Key Performance Indicators", ...).
Those headers appear on most pages, so every page looked alike to the embedding
model — stripping them is what lets retrieval tell pages apart.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class Page:
    source_file: str
    page: int
    text: str


@dataclass
class Clause:
    source_file: str
    doc_family: str
    doc_role: str
    clause_id: str
    section: str
    title: str
    # Nearest titled ancestor. A bare "8.1.2" on its own line carries no title
    # of its own, so without this its chunk loses the context that makes it
    # findable — "Protective coatings" lived only on the "8.1" heading.
    parent_title: str = ""
    # (page, line) pairs. Keeping the page per line — rather than one page per
    # clause — is what makes citations land on the page the text is actually
    # printed on: a clause heading can start on one page and run onto the next.
    lines: list = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(line for _, line in self.lines)

    @property
    def page(self) -> int:
        return self.lines[0][0] if self.lines else 0

    def context(self, page=None) -> str:
        """Short label prepended to every chunk so it is self-describing."""
        bits = [self.source_file]
        if self.clause_id:
            label = f"§{self.clause_id}"
            heading = self.title or self.parent_title
            if heading:
                label += f" {heading}"
            bits.append(label)
        elif self.title or self.parent_title:
            bits.append(self.title or self.parent_title)
        bits.append(f"p.{self.page if page is None else page}")
        return "[" + " | ".join(bits) + "]"


# ── running headers / footers ────────────────────────────────────────────────
# A bare page number, roman numeral, or "Page 3 of 20".
NOISE_LINE = re.compile(r"^\s*(?:page\s+)?\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?\s*$", re.I)


def strip_running_headers(pages, min_page_share=0.5, max_len=120):
    """Drop lines repeated across most pages of the same document.

    Operates per document, because a header is only boilerplate relative to the
    document it repeats in.
    """
    by_doc = {}
    for page in pages:
        by_doc.setdefault(page.source_file, []).append(page)

    cleaned = []
    for doc, doc_pages in by_doc.items():
        counts = Counter()
        for page in doc_pages:
            seen = {ln.strip() for ln in page.text.split("\n") if ln.strip()}
            for line in seen:
                if len(line) <= max_len and not line.startswith("|"):
                    counts[line] += 1

        threshold = max(2, int(len(doc_pages) * min_page_share))
        repeated = {line for line, n in counts.items() if n >= threshold}

        for page in doc_pages:
            kept = [
                line
                for line in page.text.split("\n")
                if line.strip()
                and line.strip() not in repeated
                and not NOISE_LINE.match(line)
            ]
            cleaned.append(Page(doc, page.page, "\n".join(kept)))
    return cleaned


# ── heading detection ────────────────────────────────────────────────────────
NAMED_SECTIONS = {
    "scope", "definitions", "introduction", "foreword", "references",
    "abbreviations", "purpose", "general", "normative references",
    "table of contents", "contents", "revision history", "acknowledgements",
}

# Sections that are navigation or front matter, not answers. The table of
# contents is the important one: it repeats every heading in the document, so
# it is a magnet for retrieval and was previously returned instead of the page
# that actually holds the answer.
SKIP_TITLES = {
    "table of contents", "contents", "revision history", "acknowledgements",
}

# A table of contents renders as "Foreword .......... 4".
DOT_LEADER = re.compile(r"\.{4,}\s*\d")


def looks_like_toc(text, sample=1200):
    return len(DOT_LEADER.findall(text[:sample])) >= 3


HEADING_PATTERNS = [
    # "8.1 Protective coatings" / "7.2.1 Surface preparation"
    (re.compile(r"^(\d+(?:\.\d+){0,5})\s+([A-Z][^.!?]{2,90})$"), "numbered"),
    # "7.2.1" alone — body text follows on subsequent lines
    (re.compile(r"^(\d+(?:\.\d+){1,5})$"), "bare"),
    # "2. The Life-Saving Rules"
    (re.compile(r"^(\d+)\.\s+([A-Z][^.!?]{2,90})$"), "dotted"),
    # "Part C - Tier 1 and Tier 2 Indicators"
    (re.compile(r"^(Part\s+[A-Z])\s*[-\u2013\u2014]\s*(.+)$"), "part"),
]


def detect_heading(line, seen_ids=None):
    """Return (clause_id, title) if the line is a heading, else None.

    ``seen_ids`` guards the bare-number pattern: a line holding just "7.2.1" is
    a sub-clause only if a parent such as "7.2" or "7" was already seen. Without
    that check, references inside the body text — "29 CFR 1910.147", for
    instance — are mistaken for clause numbers and swallow the real content.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("|"):
        return None

    for pattern, kind in HEADING_PATTERNS:
        match = pattern.match(stripped)
        if not match:
            continue
        if kind == "bare":
            number = match.group(1)
            segments = number.split(".")
            if any(len(seg) > 2 for seg in segments):
                continue
            if seen_ids is not None:
                ancestors = [
                    ".".join(segments[:i]) for i in range(1, len(segments))
                ]
                if not any(a in seen_ids for a in ancestors):
                    continue
            return number, ""
        number, title = match.group(1), match.group(2).strip()
        # "754 API published the third edition ..." is a reference entry.
        if kind != "part" and "." not in number and len(number) > 2:
            continue
        return number, title

    if stripped.rstrip(":").lower() in NAMED_SECTIONS:
        return "", stripped.rstrip(":")
    return None


def parent_of(clause_id):
    """'8.1.2' -> '8.1'; '8' -> ''."""
    return clause_id.rsplit(".", 1)[0] if "." in clause_id else ""


def describe_document(filename):
    """Map a filename to (family, role) — e.g. S-737 TRS, IOGP 459 report."""
    match = re.match(
        r"^(S-\d+)[A-Za-z]?v[\d-]+\s*(.*)\.[A-Za-z0-9]+$", filename, re.I
    )
    if match:
        family, tail = match.group(1).upper(), match.group(2).strip()
        role = "TRS" if "TRS" in tail.upper() else (
            "QRS" if "QRS" in tail.upper() else (
                "IRS" if "IRS" in tail.upper() else tail or "spec"
            )
        )
        if "justification" in tail.lower():
            role += "+Justification"
        return family, role
    if match := re.match(r"^(\d{3})\.[A-Za-z0-9]+$", filename):
        return f"IOGP {match.group(1)}", "Report"
    return filename.rsplit(".", 1)[0], "document"


def split_into_clauses(pages):
    """Group page text into clauses, carrying the heading across page breaks."""
    clauses = []
    current = None
    seen_ids = set()
    titles = {}
    current_doc = None

    def close():
        if current and current.text.strip() and not is_skippable(current):
            clauses.append(current)

    def is_skippable(clause):
        if clause.title.rstrip(":").lower() in SKIP_TITLES:
            return True
        return looks_like_toc(clause.text)

    for page in pages:
        if page.source_file != current_doc:
            # Clause numbering is per document — never let one document's ids
            # validate another's bare sub-clause numbers.
            seen_ids = set()
            titles = {}
            current_doc = page.source_file
        family, role = describe_document(page.source_file)
        for line in page.text.split("\n"):
            heading = detect_heading(line, seen_ids)
            if heading:
                close()
                clause_id, title = heading
                if clause_id:
                    seen_ids.add(clause_id)
                    if title:
                        titles[clause_id] = title
                parent_title = ""
                if not title and clause_id:
                    # Walk up: 8.1.2 → 8.1 → 8
                    segments = clause_id.split(".")
                    for depth in range(len(segments) - 1, 0, -1):
                        ancestor = ".".join(segments[:depth])
                        if ancestor in titles:
                            parent_title = titles[ancestor]
                            break
                current = Clause(
                    source_file=page.source_file,
                    doc_family=family,
                    doc_role=role,
                    clause_id=clause_id,
                    section=parent_of(clause_id),
                    title=title,
                    parent_title=parent_title,
                )
                continue
            if current is None:
                # Preamble before the first heading (cover page, foreword page).
                current = Clause(
                    source_file=page.source_file,
                    doc_family=family,
                    doc_role=role,
                    clause_id="",
                    section="",
                    title="(preamble)",
                )
            current.lines.append((page.page, line.strip()))

    close()
    return clauses


def split_long_line(line, count_tokens, budget):
    """Break a single over-long line (often a table row) into budget-sized parts."""
    if count_tokens(line) <= budget:
        return [line]
    parts, current, size = [], [], 0
    for word in line.split(" "):
        tokens = count_tokens(word + " ")
        if size and size + tokens > budget:
            parts.append(" ".join(current))
            current, size = [], 0
        current.append(word)
        size += tokens
    if current:
        parts.append(" ".join(current))
    return parts or [line]


def chunk_clause(clause, count_tokens, max_tokens, min_tokens=20):
    """Split a clause into (chunk_text, page) pairs that fit the model window.

    Each chunk records the page its own text starts on — not the page the clause
    started on — so a citation can be trusted. The context label is counted
    against the budget and repeated on every piece.
    """
    header_budget = count_tokens(clause.context()) + 4
    budget = max(min_tokens, max_tokens - header_budget)

    pieces, buffer, size, start_page = [], [], 0, None
    for page, line in clause.lines:
        if not line:
            continue
        # Never let a chunk span a page break: a chunk that starts on p.14 but
        # carries text from p.15 cannot be cited accurately. Bounding every
        # chunk to a single page makes the citation exact by construction.
        if start_page is not None and page != start_page and buffer:
            pieces.append((start_page, buffer))
            buffer, size, start_page = [], 0, None
        for part in split_long_line(line, count_tokens, budget):
            part_tokens = count_tokens(part)
            if start_page is None:
                start_page = page
            if size and size + part_tokens > budget:
                pieces.append((start_page, buffer))
                buffer, size, start_page = [], 0, page
            buffer.append(part)
            size += part_tokens
    if buffer:
        pieces.append((start_page, buffer))

    out = []
    for page, lines in pieces:
        body = "\n".join(lines)
        if count_tokens(body) < min_tokens and len(pieces) > 1:
            continue  # drop near-empty fragments produced by splitting
        out.append((f"{clause.context(page)}\n{body}", page))
    return out

"""Decide which document family a question is about.

The corpus holds four unrelated document families (IOGP 459, IOGP 456,
JIP33 S-717/S-719, JIP33 S-737). Without routing they compete in one pool, so a
question about deluge skids can be answered from a noise specification.

Routing is deliberately deterministic and conservative: it only narrows the
search when the question gives a clear signal, and the caller falls back to the
full corpus whenever the narrowed set is too small to answer from. A wrong
narrowing is worse than no narrowing.
"""
import re

# Words that identify a family even when the document code is not mentioned.
FAMILY_KEYWORDS = {
    "459": ("life saving", "life-saving", "lifesaving", "life saving rules"),
    "456": ("lopc", "process safety event", "tier 1", "tier 2", "tier 1 and 2"),
    "S-737": ("deluge", "deluge skid"),
    "S-717": ("noise", "noise emitting", "sound pressure"),
    "S-719": ("water mist", "watermist"),
}

# Words that identify the role a document plays in a JIP33 pack.
ROLE_KEYWORDS = {
    "TRS": ("design", "technical", "shall be provided", "installation", "fabrication"),
    "QRS": ("quality", "inspection", "test certificate", "conformity", "qrs"),
}

CODE_RE = re.compile(r"s[-\s]?(\d{3})|\b(459|456)\b", re.I)


def infer_family(question):
    """Return a document family code, or None when the question is ambiguous."""
    match = CODE_RE.search(question)
    if match:
        number = match.group(1) or match.group(2)
        return f"S-{number}" if match.group(1) else number

    lowered = question.lower()
    for family, keywords in FAMILY_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return family
    return None


def infer_role(question):
    """Return 'TRS', 'QRS', or None."""
    lowered = question.lower()
    for role, keywords in ROLE_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return role
    return None


def apply_route(candidates, family=None, role=None, min_keep=3):
    """Narrow candidates to a family and/or role.

    Returns the original list unchanged when the narrowing would leave too
    little to work with — an empty answer is never better than a broad one.
    """
    if not family and not role:
        return candidates

    narrowed = [
        chunk
        for chunk in candidates
        if (not family or chunk.metadata.get("doc_family") == family)
    ]
    if role:
        by_role = [c for c in narrowed if c.metadata.get("doc_role") == role]
        if len(by_role) >= min_keep:
            narrowed = by_role

    if len(narrowed) < min_keep:
        return candidates
    return narrowed

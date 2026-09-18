"""Keyword search and rank fusion.

Vector search is good at concepts and bad at exact tokens. This corpus is full
of exact tokens — ``S-737``, ``LOPC``, ``Tier 1``, ``ISO 12944-4`` — so a
keyword ranking runs alongside it and the two are fused.

BM25 is implemented here rather than pulled in as a dependency: it is thirty
lines, it keeps the install lean, and it is unit-tested.
"""
import math
import re
from collections import Counter

# Keep hyphens and dots inside a token so "s-737" and "12944-4" survive as one
# term instead of being shredded into fragments.
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-.]*")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class BM25:
    """Standard Okapi BM25 over an in-memory corpus."""

    def __init__(self, corpus, k1=1.5, b=0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.n_docs = len(corpus)
        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0
        self.doc_freqs = [Counter(doc) for doc in corpus]

        document_frequency = Counter()
        for counts in self.doc_freqs:
            document_frequency.update(counts.keys())
        self.idf = {
            term: math.log((self.n_docs - freq + 0.5) / (freq + 0.5) + 1)
            for term, freq in document_frequency.items()
        }

    def get_scores(self, query_tokens):
        scores = [0.0] * self.n_docs
        if not self.avgdl:
            return scores
        for term in set(query_tokens):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for index, counts in enumerate(self.doc_freqs):
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * self.doc_len[index] / self.avgdl
                )
                scores[index] += idf * frequency * (self.k1 + 1) / denominator
        return scores

    def top_k(self, query, k, allowed=None):
        """Return the indices of the k best-matching documents.

        `allowed` optionally restricts the search to a subset of indices, which
        is how document routing is applied to the keyword side.
        """
        scores = self.get_scores(tokenize(query))
        if allowed is not None:
            scores = [s if i in allowed else 0.0 for i, s in enumerate(scores)]
        order = sorted(range(self.n_docs), key=lambda i: scores[i], reverse=True)
        return [i for i in order[:k] if scores[i] > 0]


def reciprocal_rank_fusion(rankings, k=60):
    """Fuse several ranked id lists into one.

    RRF scores by rank rather than by raw score, so a BM25 score and a cosine
    distance — which are not comparable — can be combined safely.
    """
    scores = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)

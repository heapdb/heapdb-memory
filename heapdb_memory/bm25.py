"""A small incremental Okapi BM25 index (stdlib only).

Scoring follows Robertson & Zaragoza (2009), "The Probabilistic Relevance
Framework: BM25 and Beyond", with the non-negative IDF used by Lucene::

    idf(t)      = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
    score(q, d) = sum_{t in set(q)} idf(t) * tf(t,d) * (k1 + 1)
                                     / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))

Each distinct query term counts once (query term frequency is ignored).
Terms are summed in sorted order, so scores do not depend on hash seeds.
Documents can be added and removed at any time; statistics are maintained
incrementally, so a query costs O(sum of the posting-list lengths of the
query terms).
"""

from __future__ import annotations

import heapq
import math
from collections import Counter
from typing import Hashable, Iterable, Sequence

__all__ = ["BM25Index"]


class BM25Index:
    """Incremental BM25 over pre-tokenised documents.

    Args:
        k1: Term-frequency saturation (default 1.2).
        b: Length normalisation (0 = none, 1 = full; default 0.75).
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        if k1 < 0 or not 0.0 <= b <= 1.0:
            raise ValueError("need k1 >= 0 and 0 <= b <= 1")
        self.k1 = float(k1)
        self.b = float(b)
        self._postings: dict[str, dict[Hashable, int]] = {}
        self._doc_terms: dict[Hashable, Counter[str]] = {}
        self._doc_len: dict[Hashable, int] = {}
        self._total_len = 0

    # -- maintenance ----------------------------------------------------------

    def __len__(self) -> int:
        return len(self._doc_len)

    def __contains__(self, doc_id: Hashable) -> bool:
        return doc_id in self._doc_len

    @property
    def avgdl(self) -> float:
        return self._total_len / len(self._doc_len) if self._doc_len else 0.0

    def df(self, term: str) -> int:
        return len(self._postings.get(term, ()))

    def add(self, doc_id: Hashable, tokens: Sequence[str]) -> None:
        """Index a document (replaces an existing one with the same id)."""
        if doc_id in self._doc_len:
            self.remove(doc_id)
        tf = Counter(tokens)
        self._doc_terms[doc_id] = tf
        self._doc_len[doc_id] = len(tokens)
        self._total_len += len(tokens)
        for t, n in tf.items():
            self._postings.setdefault(t, {})[doc_id] = n

    def remove(self, doc_id: Hashable) -> bool:
        """Drop a document. Returns False if it was not indexed."""
        tf = self._doc_terms.pop(doc_id, None)
        if tf is None:
            return False
        self._total_len -= self._doc_len.pop(doc_id)
        for t in tf:
            plist = self._postings.get(t)
            if plist is not None:
                plist.pop(doc_id, None)
                if not plist:
                    del self._postings[t]
        return True

    # -- scoring --------------------------------------------------------------

    def idf(self, term: str) -> float:
        n = len(self._doc_len)
        df = self.df(term)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def scores(self, query_tokens: Iterable[str]) -> dict[Hashable, float]:
        """BM25 score of every document matching at least one query term."""
        out: dict[Hashable, float] = {}
        if not self._doc_len:
            return out
        avgdl = self.avgdl or 1.0
        k1, b = self.k1, self.b
        for t in sorted(set(query_tokens)):  # fixed order: bit-reproducible sums
            plist = self._postings.get(t)
            if not plist:
                continue
            idf = self.idf(t)
            for d, tf in plist.items():
                norm = k1 * (1.0 - b + b * self._doc_len[d] / avgdl)
                out[d] = out.get(d, 0.0) + idf * tf * (k1 + 1.0) / (tf + norm)
        return out

    def score(self, query_tokens: Iterable[str], doc_id: Hashable) -> float:
        """BM25 score of one document (0.0 if it matches nothing)."""
        tf_doc = self._doc_terms.get(doc_id)
        if tf_doc is None:
            return 0.0
        avgdl = self.avgdl or 1.0
        s = 0.0
        for t in sorted(set(query_tokens)):  # fixed order: bit-reproducible sums
            tf = tf_doc.get(t, 0)
            if tf:
                norm = self.k1 * (1.0 - self.b + self.b * self._doc_len[doc_id] / avgdl)
                s += self.idf(t) * tf * (self.k1 + 1.0) / (tf + norm)
        return s

    def top(self, query_tokens: Iterable[str], n: int = 10) -> list[tuple[Hashable, float]]:
        """The ``n`` best ``(doc_id, score)`` pairs, best first (ties: by id)."""
        sc = self.scores(query_tokens)
        return heapq.nsmallest(n, sc.items(), key=lambda kv: (-kv[1], str(kv[0])))

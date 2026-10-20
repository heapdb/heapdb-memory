"""BM25 correctness against a hand computation, text features, embeddings."""

from __future__ import annotations

import math

import pytest

from heapdb_memory import BM25Index, HashingEmbedder, VectorIndex
from heapdb_memory.text import memory_features, pair_features, query_type, stem, terms


def brute_bm25(query, docs, k1=1.2, b=0.75):
    n = len(docs)
    avgdl = sum(len(d) for d in docs.values()) / n
    out = {}
    for did, d in docs.items():
        s = 0.0
        for t in set(query):
            df = sum(1 for x in docs.values() if t in x)
            tf = d.count(t)
            if not tf:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            s += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / avgdl))
        if s:
            out[did] = s
    return out


DOCS = {
    "a": "the cat sat on the mat".split(),
    "b": "the dog sat on the log the dog".split(),
    "c": "cats and dogs and birds".split(),
    "d": "a dog a dog a dog".split(),
}


def test_bm25_matches_formula() -> None:
    idx = BM25Index()
    for k, v in DOCS.items():
        idx.add(k, v)
    for q in (["dog"], ["sat", "dog"], ["the", "cat", "cat"], ["unknown"], ["birds", "mat"]):
        want = brute_bm25(q, DOCS)
        got = idx.scores(q)
        assert set(got) == set(want)
        for k in want:
            assert got[k] == pytest.approx(want[k])
        for k in DOCS:
            assert idx.score(q, k) == pytest.approx(want.get(k, 0.0))


def test_bm25_incremental_remove_and_replace() -> None:
    idx = BM25Index()
    for k, v in DOCS.items():
        idx.add(k, v)
    idx.remove("d")
    rest = {k: v for k, v in DOCS.items() if k != "d"}
    assert idx.scores(["dog"]) == pytest.approx(brute_bm25(["dog"], rest))
    assert not idx.remove("d")
    idx.add("a", "dog dog".split())          # replace
    rest["a"] = ["dog", "dog"]
    assert idx.scores(["dog"]) == pytest.approx(brute_bm25(["dog"], rest))
    assert len(idx) == 3 and "a" in idx and idx.avgdl == pytest.approx(sum(map(len, rest.values())) / 3)
    top = idx.top(["dog"], 2)
    assert [d for d, _ in top] == sorted(rest, key=lambda k: -brute_bm25(["dog"], rest).get(k, 0))[:2]


def test_bm25_rejects_bad_params() -> None:
    with pytest.raises(ValueError):
        BM25Index(b=1.5)


def test_stem_and_terms() -> None:
    assert stem("running") == stem("run") == "run"
    assert stem("studies") == stem("studied") == stem("study")
    assert stem("hiking") == stem("hike") == stem("hikes")
    assert terms("The cats were RUNNING to the park!") == ["cat", "run", "park"]


def test_memory_features_fixed_slots() -> None:
    f = memory_features("We went hiking in the Alps last summer with my 2 kids", "episode",
                        source="Caroline", tags=["Trip"])
    slots = sorted(x.split(":")[0] for x in f)
    assert slots == ["date", "ent", "kind", "len", "num", "q", "source", "tag", "topic"]
    assert {"kind:episode", "source:caroline", "date:yes", "num:yes", "ent:yes", "tag:trip"} <= f
    g = memory_features("ok?", "note")
    assert {"q:yes", "len:short", "source:unknown", "topic:none", "ent:no"} <= g


def test_query_type_and_pair_features() -> None:
    assert query_type("When did Caroline go hiking?") == "when"
    assert query_type("What year did they marry?") == "when"
    assert query_type("How many kids does Mel have?") == "howmany"
    assert query_type("Would Caroline likely enjoy camping?") == "hypothetical"
    assert query_type("Did Mel paint a sunrise?") == "yesno"
    f = memory_features("Caroline: I went on 7 May", "episode", source="Caroline")
    p = pair_features("When did Caroline go?", f)
    assert "qt:when&date:yes" in p and "qsrc:yes" in p and f <= p
    assert pair_features(None, f) == f


def test_hashing_embedder_and_vector_index() -> None:
    e = HashingEmbedder(128)
    v1, v2, v3 = e.embed(["quarterly roadmap review", "quartrly roadmp revew", "bananas are yellow"])
    assert sum(x * x for x in v1) == pytest.approx(1.0)
    idx = VectorIndex()
    for i, v in enumerate([v1, v3]):
        idx.add(i, v)
    assert idx.top(v2, 1)[0][0] == 0
    assert idx.remove(0) and not idx.remove(0) and len(idx) == 1
    assert idx.top(v2, 5)[0][0] == 1

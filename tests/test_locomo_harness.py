"""The LoCoMo harness on a tiny synthetic fixture in LoCoMo's format (no download)."""

from __future__ import annotations

import json

import pytest

from evals.locomo import data, report, run


def _conv(sample_id: str, who: tuple[str, str], topic_words: list[str]) -> dict:
    a, b = who
    conv = {"speaker_a": a, "speaker_b": b}
    qa = []
    for s in range(1, 4):
        conv[f"session_{s}_date_time"] = f"1:5{s} pm on {s} May, 2023"
        turns = []
        for i, w in enumerate(topic_words):
            spk = a if i % 2 == 0 else b
            turns.append({"speaker": spk, "dia_id": f"D{s}:{i + 1}",
                          "text": f"In session {s} {spk} talked about {w} number {s}{i}."})
        conv[f"session_{s}"] = turns
        for i, w in enumerate(topic_words[:2]):
            qa.append({"question": f"What did they say about {w} number {s}{i}?", "answer": "x",
                       "evidence": [f"D{s}:{i + 1}"], "category": 4 if i == 0 else 2})
    qa.append({"question": "Unanswerable?", "answer": "x", "evidence": [], "category": 3})
    qa.append({"question": f"Both {topic_words[0]} things?", "answer": "x",
               "evidence": ["D1:1; D2:1", "D:3:1"], "category": 1})
    return {"sample_id": sample_id, "conversation": conv, "qa": qa}


@pytest.fixture
def fixture_path(tmp_path):
    convs = [_conv(f"conv-{i}", ("Ann", "Bob"), ["guitar", "garden", "cooking", "travel"]) for i in range(3)]
    p = tmp_path / "locomo_tiny.json"
    p.write_text(json.dumps(convs))
    return p


def test_load_parses_evidence_and_excludes(fixture_path) -> None:
    convs = data.load(fixture_path)
    assert len(convs) == 3 and len(convs[0].turns) == 12
    c = convs[0]
    assert c.excluded["no_evidence"] == 1
    multi = [q for q in c.qa if q.category == 1][0]
    assert multi.evidence == {"D1:1", "D2:1", "D3:1"}
    assert c.now > max(t.time for t in c.turns)
    assert c.turns[0].text.startswith("Ann: ")


def test_run_and_report(fixture_path, tmp_path) -> None:
    convs = data.load(fixture_path)
    logs = []
    res = run.run(convs, None, only={"bm25", "recency", "heapdb", "heapdb+learned",
                                     "abl:bm25+learned"}, log=logs.append)
    m = res["methods"]
    assert set(m) == {"bm25", "recency", "heapdb", "heapdb+learned", "abl:bm25+learned"}
    n = sum(len(c.qa) for c in convs)
    for name, r in m.items():
        assert r["overall"]["n"] == n
        assert 0.0 <= r["overall"]["R@1"] <= r["overall"]["R@5"] <= r["overall"]["R@10"] <= 1.0
        assert len(r["per_conversation"]) == 3
    # The questions name the evidence turn uniquely: BM25 finds it at rank 1 for single-hop.
    assert m["bm25"]["per_category"]["single-hop"]["R@1"] == pytest.approx(1.0)
    assert len(res["train_feedback_events_per_fold"]) == 3 * 2
    res["meta"] = {"dataset_sha256": "x", "conversations": 3, "memories": 36, "questions": n,
                   "questions_per_category": {c: 1 for c in data.CATEGORIES.values()},
                   "excluded": {"no_evidence": 3}, "embedder": None, "depth": 50,
                   "train_shown": 10, "python": "3", "machine": "x", "runtime_s": 0}
    md = report.render(res) + report.analysis(res)
    assert "<!-- summary:start -->" in md and "| `heapdb+learned` |" in md and "ΔMRR" in md


def test_no_test_leakage(fixture_path, monkeypatch) -> None:
    """Training feedback never touches the held-out conversation's namespace."""
    convs = data.load(fixture_path)
    seen: list[tuple[str, set]] = []
    orig = run.train

    def spy(mem, train_convs, cfg, *args, **kw):
        seen.append({c.sample_id for c in train_convs})
        return orig(mem, train_convs, cfg, *args, **kw)

    monkeypatch.setattr(run, "train", spy)
    run.run(convs, None, only={"heapdb+learned"}, log=lambda *_: None)
    ids = [c.sample_id for c in convs]
    assert seen == [set(ids) - {t} for t in ids]


def test_paired_bootstrap() -> None:
    a = {"per_question": [["c", 4, [1], 1]] * 20}
    b = {"per_question": [["c", 4, [2], 1]] * 20}
    d, lo, hi = report.paired(a, b)
    assert d == pytest.approx(0.5) and lo == pytest.approx(0.5) and hi == pytest.approx(0.5)

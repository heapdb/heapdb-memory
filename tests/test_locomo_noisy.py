"""Noisy training feedback for the LoCoMo harness: the oracle is unchanged,
the noise is seeded and deterministic, and tuning and leakage guarantees hold
under every feedback model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_locomo_harness import _conv

from evals.locomo import data, report, run
from evals.locomo.run import ORACLE, Feedback

LEXICAL = {"bm25", "abl:bm25+learned", "heapdb+learned+tuned"}


@pytest.fixture
def convs(tmp_path):
    raw = [_conv(f"conv-{i}", ("Ann", "Bob"), ["guitar", "garden", "cooking", "travel"]) for i in range(3)]
    p = tmp_path / "locomo_tiny.json"
    p.write_text(json.dumps(raw))
    return data.load(p)


def _ranks(res: dict) -> dict:
    return {n: [q[2] for q in m["per_question"]] for n, m in res["methods"].items()}


def _quiet(*_a) -> None:
    pass


def _train_before_noise(mem, convs, cfg, *args, **kw) -> int:
    """`train` as committed before feedback models existed (reference)."""
    n = 0
    for c in convs:
        for q in c.qa:
            res = mem.recall(q.question, k=run.TRAIN_SHOWN, namespace=c.sample_id, now=c.now,
                             config=cfg, explain=False, track=False)
            ev = {f"{c.sample_id}|{d}" for d in q.evidence}
            used = [r.id for r in res if r.id in ev]
            ign = [r.id for r in res if r.id not in ev]
            if used:
                mem.mark_used(used, now=c.now, query=q.question)
            if ign:
                mem.mark_ignored(ign, now=c.now, query=q.question)
            n += len(used) + len(ign)
    return n


def test_oracle_matches_pre_noise_training(convs, monkeypatch) -> None:
    """Oracle mode (and every zero-noise spelling of it) is the old protocol."""
    new = run.run(convs, None, only=LEXICAL, log=_quiet)
    for fb in (Feedback(seed=123), Feedback.parse("miss=0,false_pos=0,sparse=1"),
               Feedback(missed_as="unmarked")):
        assert fb.is_oracle and fb.label == "oracle"
        assert _ranks(run.run(convs, None, only=LEXICAL, log=_quiet, feedback=fb)) == _ranks(new)
    monkeypatch.setattr(run, "train", _train_before_noise)
    old = run.run(convs, None, only=LEXICAL, log=_quiet)
    assert _ranks(new) == _ranks(old)
    assert new["train_feedback_events_per_fold"] == old["train_feedback_events_per_fold"]
    assert new["tuned_choices"] == old["tuned_choices"]


def test_tune_memo_picks_the_same_configs(convs) -> None:
    mem = run.build_store(convs, None, max(c.now for c in convs))[0]
    cfg = run.ScoringConfig()
    memo: dict = {}
    for i in range(len(convs)):
        tr = [c for j, c in enumerate(convs) if j != i]
        assert run.tune(mem, tr, cfg, False) == run.tune(mem, tr, cfg, False, memo=memo)
    assert memo


SHOWN = [f"c|m{i}" for i in range(10)]
EVIDENCE = {"c|m1", "c|m4", "c|m7"}


def test_judge_semantics() -> None:
    others = [s for s in SHOWN if s not in EVIDENCE]
    assert ORACLE.judge("c", 0, SHOWN, EVIDENCE) == (["c|m1", "c|m4", "c|m7"], others)
    # Sparse 0: never any feedback.
    assert Feedback(sparse=0.0).judge("c", 0, SHOWN, EVIDENCE) == ([], [])
    # Missing everything: missed evidence is ignored (default) or left unmarked.
    used, ign = Feedback(miss=1.0).judge("c", 0, SHOWN, EVIDENCE)
    assert used == [] and ign == SHOWN
    used, ign = Feedback(miss=1.0, missed_as="unmarked").judge("c", 0, SHOWN, EVIDENCE)
    assert used == [] and set(ign) == set(SHOWN) - EVIDENCE
    # Citing everything shown.
    used, ign = Feedback(false_pos=1.0).judge("c", 0, SHOWN, EVIDENCE)
    assert used == SHOWN and ign == []
    # One source: the highest-ranked evidence; other evidence unmarked.
    used, ign = Feedback(cite="top1").judge("c", 0, SHOWN, EVIDENCE)
    assert used == ["c|m1"] and set(ign) == set(SHOWN) - EVIDENCE
    used, ign = Feedback(cite="top1").judge("c", 0, ["c|m4", "c|m1"], EVIDENCE)
    assert used == ["c|m4"] and ign == []
    assert Feedback.parse("cite_top1").label == "cite_top1"
    assert Feedback.parse("miss=0.5", missed_as="unmarked").label == "miss=0.5(unmarked)"
    with pytest.raises(ValueError):
        Feedback.parse("miss=1.5")
    with pytest.raises(ValueError):
        Feedback.parse("noise=0.1")


def test_noise_is_seeded_deterministic_and_nested() -> None:
    def draws(fb: Feedback) -> list:
        return [fb.judge(f"conv-{c}", q, SHOWN, EVIDENCE) for c in range(5) for q in range(200)]

    fb = Feedback(miss=0.5, false_pos=0.1, sparse=0.5, seed=1)
    assert draws(fb) == draws(Feedback.parse("miss=0.5,false_pos=0.1,sparse=0.5", seed=1))
    assert draws(fb) != draws(Feedback(miss=0.5, false_pos=0.1, sparse=0.5, seed=2))
    # Rates come out as asked (1000 questions x 3 evidence / 7 other turns).
    n_ev = 3 * 1000
    missed = sum(len(EVIDENCE - set(u)) for u, _ in draws(Feedback(miss=0.3, seed=4)))
    assert abs(missed / n_ev - 0.3) < 0.03
    fps = sum(len(set(u) - EVIDENCE) for u, _ in draws(Feedback(false_pos=0.1, seed=4)))
    assert abs(fps / (7 * 1000) - 0.1) < 0.015
    fed = sum(1 for u, i in draws(Feedback(sparse=0.25, seed=4)) if u or i)
    assert abs(fed / 1000 - 0.25) < 0.04
    # Nested levels: what is missed at 0.3 is also missed at 0.5 (same seed).
    lo, hi = draws(Feedback(miss=0.3, seed=4)), draws(Feedback(miss=0.5, seed=4))
    assert all(set(u_hi) <= set(u_lo) for (u_lo, _), (u_hi, _) in zip(lo, hi))
    # Independent of the order in which turns are shown.
    u, _ = fb.judge("conv-0", 7, SHOWN, EVIDENCE)
    u2, _ = fb.judge("conv-0", 7, SHOWN[::-1], EVIDENCE)
    assert set(u) == set(u2)


def test_noisy_run_is_deterministic_and_tuning_ignores_feedback(convs) -> None:
    fb = Feedback.parse("miss=0.5,false_pos=0.2,sparse=0.5", seed=3)
    a = run.run(convs, None, only=LEXICAL, log=_quiet, feedback=fb)
    b = run.run(convs, None, only=LEXICAL, log=_quiet, feedback=fb)
    assert _ranks(a) == _ranks(b) and a["train_feedback_stats"] == b["train_feedback_stats"]
    oracle = run.run(convs, None, only=LEXICAL, log=_quiet)
    assert a["train_feedback_stats"]["used"] != oracle["train_feedback_stats"]["used"]
    # Tuned separately (no shared memo): the same weights under every feedback model.
    assert a["tuned_choices"] == oracle["tuned_choices"]
    assert a["feedback"]["label"] == "miss=0.5,false_pos=0.2,sparse=0.5"
    # bm25 reads no usefulness: feedback cannot change it.
    assert _ranks(a)["bm25"] == _ranks(oracle)["bm25"]


@pytest.mark.parametrize("spec", ["miss=0.5,false_pos=0.1,sparse=0.5", "cite_top1"])
def test_no_test_leakage_under_noise(convs, monkeypatch, spec) -> None:
    seen: list[set] = []
    orig = run.train

    def spy(mem, train_convs, cfg, *args, **kw):
        seen.append({c.sample_id for c in train_convs})
        return orig(mem, train_convs, cfg, *args, **kw)

    monkeypatch.setattr(run, "train", spy)
    run.run(convs, None, only={"heapdb+learned"}, log=_quiet, feedback=Feedback.parse(spec, seed=1))
    ids = [c.sample_id for c in convs]
    assert seen == [set(ids) - {t} for t in ids]


def test_sweep_collect_and_report(convs, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(run, "SWEEP", [("oracle", "oracle", "ignored"),
                                       ("miss", "miss=0.5", "ignored"),
                                       ("no feedback", "sparse=0", "ignored"),
                                       ("realistic", run.REALISTIC, "ignored")])
    d = tmp_path / "runs"
    for i in range(2):  # two shards, then a resumed no-op pass
        run.sweep(convs, None, d, shard=(i, 2), seeds=(0, 1), only=run.SWEEP_METHODS, log=_quiet)
    assert len(list(d.glob("*.json"))) == 1 + 2 + 1 + 2
    run.sweep(convs, None, d, seeds=(0, 1), log=_quiet)
    meta = {"questions": 21, "embedder": None, "seeds": [0, 1],
            "questions_per_category": {c: 1 for c in report.CATS}}
    res = run.collect(d, seeds=(0, 1), meta=meta)
    assert [s["label"] for s in res["settings"]] == ["oracle", "miss=0.5", "sparse=0", run.REALISTIC]
    assert res["settings"][1]["seeds"] == [0, 1] and len(res["settings"][1]["ranks"]["abl:bm25+learned"]) == 2
    md = report.noisy(res)
    assert "| `oracle` |" in md and "## Reading" in md and "Threats to validity" in md
    # Seed-averaged bootstrap equals `paired` when there is one seed.
    a = {"per_question": [[q[0], q[1], r, q[2]] for q, r in
                          zip(res["questions"], res["settings"][0]["ranks"]["abl:bm25+learned"][0])]}
    b = {"per_question": [[q[0], q[1], r, q[2]] for q, r in
                          zip(res["questions"], res["settings"][0]["ranks"]["bm25"][0])]}
    d_ = [report._rr([0, 0, ra]) - report._rr([0, 0, rb]) for ra, rb in
          zip(res["settings"][0]["ranks"]["abl:bm25+learned"][0], res["settings"][0]["ranks"]["bm25"][0])]
    assert report.boot(d_) == report.paired(a, b)


COMMITTED = Path(__file__).parents[1] / "evals" / "locomo" / "results.json"


@pytest.mark.skipif(not (data.cache_dir() / "locomo" / "locomo10.json").exists(),
                    reason="LoCoMo not in the local cache (tests never download)")
def test_oracle_reproduces_committed_results() -> None:
    """With the real dataset cached: oracle mode gives the committed per-question
    ranks for BM25 and learned usefulness (lexical, about a minute)."""
    path = data.cache_dir() / "locomo" / "locomo10.json"
    if data.sha256(path) != data.SHA256:
        pytest.skip("cached LoCoMo is not the evaluated version")
    committed = json.loads(COMMITTED.read_text())
    res = run.run(data.load(path), None, only={"bm25", "abl:bm25+learned"}, log=_quiet)
    for n in ("bm25", "abl:bm25+learned"):
        assert _ranks(res)[n] == [q[2] for q in committed["methods"][n]["per_question"]], n
        assert res["methods"][n]["overall"]["MRR"] == pytest.approx(
            committed["methods"][n]["overall"]["MRR"], abs=1e-12)

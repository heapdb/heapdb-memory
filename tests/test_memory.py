"""SDK tests: add / recall / feedback learning / supersede / forget / persistence."""

from __future__ import annotations

import json

import pytest

from heapdb_memory import HashingEmbedder, Memory, ScoringConfig

T0 = 1_700_000_000.0
DAY = 86400.0


class Clock:
    def __init__(self, t: float = T0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def mem(clock: Clock) -> Memory:
    return Memory(namespace="user:alice", clock=clock)


def test_add_and_recall_ranks_relevant_first(mem: Memory) -> None:
    a = mem.add("Alice is vegetarian and allergic to peanuts", kind="preference")
    mem.add("Alice's flight to Lisbon leaves on Friday", kind="fact")
    mem.add("The deploy pipeline uses GitHub Actions", kind="fact")
    res = mem.recall("what can Alice eat? any allergies?", k=3)
    assert res[0].id == a.id
    assert res.ids[0] == a.id and res.query.startswith("what")
    r = res[0]
    assert r.score == pytest.approx(r.relevance * r.recency * r.importance * r.usefulness * r.rule_multiplier)
    line = r.explain()
    assert "rel" in line and "recency" in line and "importance" in line and "useful" in line
    assert r.feature_links and all(0 < s < 1 for _, s in r.feature_links)
    d = r.to_dict()
    assert d["id"] == a.id and "explanation" in d and d["components"]["usefulness"] == r.usefulness


def test_recall_validation_and_empty(mem: Memory) -> None:
    assert list(mem.recall("anything")) == []
    with pytest.raises(ValueError):
        mem.recall("x", k=0)
    with pytest.raises(ValueError):
        mem.add("   ")
    with pytest.raises(ValueError):
        mem.add("x", importance=0)


def test_no_lexical_match_falls_back_to_priority(mem: Memory, clock: Clock) -> None:
    mem.add("old note about gardening", created_at=T0 - 90 * DAY)
    new = mem.add("fresh note about cycling")
    res = mem.recall("zzzz qqqq", k=2)
    assert [r.id for r in res][0] == new.id     # recency x importance decides
    assert all(r.relevance == 1.0 for r in res)


def test_importance_is_a_dose(mem: Memory) -> None:
    lo = mem.add("project codename is falcon", importance=0.5)
    hi = mem.add("project deadline for falcon is March", importance=2.0)
    res = mem.recall("falcon project", k=2)
    assert res[0].id == hi.id and res[1].id == lo.id
    assert mem.get(lo.id).importance_source == "explicit"
    h = mem.add("ok")
    assert h.importance_source == "heuristic" and h.importance < 1.0


def test_recency_decays_and_use_refreshes(clock: Clock) -> None:
    cfg = ScoringConfig(half_life_days=10, recency_floor=0.0, w_importance=0.0)
    mem = Memory(namespace="n", clock=clock, config=cfg)
    old = mem.add("standup is at 9am", created_at=T0 - 20 * DAY)
    r = mem.recall("standup", k=1)[0]
    assert r.recency == pytest.approx(0.25, rel=1e-6)
    mem.mark_used([old.id])
    assert mem.get(old.id).last_used == T0 and mem.get(old.id).use_count == 1
    assert mem.recall("standup", k=1)[0].recency == pytest.approx(1.0)


def test_feedback_learning_changes_ranking(clock: Clock) -> None:
    mem = Memory(namespace="agent", clock=clock)
    # Two equally relevant memories of different kinds; tool output first by importance? no:
    tool = mem.add("weather api returned rain for Paris tomorrow", kind="note", source="tool",
                   importance=1.0)
    pref = mem.add("user dislikes rain and prefers indoor plans in Paris", kind="preference",
                   source="user", importance=1.0)
    first = mem.recall("plans in Paris if rain", k=2)
    before = {r.id: r.score for r in first}
    # The agent keeps using the user's preferences and ignoring tool output.
    for _ in range(8):
        mem.recall("plans in Paris if rain", k=2)
        mem.feedback(used=[pref.id])
    after = mem.recall("plans in Paris if rain", k=2)
    assert after[0].id == pref.id
    assert after[0].usefulness > first[[r.id for r in first].index(pref.id)].usefulness
    sc = {r.id: r.score for r in after}
    assert sc[tool.id] / sc[pref.id] < before[tool.id] / before[pref.id]
    rep = mem.explain_usefulness()
    assert rep.row("source:user").strength > rep.row("source:tool").strength
    assert mem.get(tool.id).ignore_count == 8


def test_learning_transfers_to_new_memories_and_namespaces(clock: Clock) -> None:
    mem = Memory(namespace="a", clock=clock)
    for i in range(10):
        p = mem.add(f"user prefers option {i} style answers", kind="preference")
        t = mem.add(f"tool log line {i} answers style", kind="tool_result", source="tool")
        mem.mark_used([p.id])
        mem.mark_ignored([t.id])
    # A brand-new namespace borrows the pooled (global) links.
    p2 = mem.add("prefers short answers", kind="preference", namespace="b")
    t2 = mem.add("tool: short answers enabled", kind="tool_result", source="tool", namespace="b")
    cfg = ScoringConfig(w_importance=0.0, w_recency=0.0)
    res = {r.id: r for r in mem.recall("short answers", k=2, namespace="b", config=cfg)}
    assert res[p2.id].usefulness > res[t2.id].usefulness


def test_supersede_by_key(mem: Memory, clock: Clock) -> None:
    old = mem.add("Alice lives in Porto", key="city")
    clock.t += DAY
    new = mem.add("Alice moved to Lisbon", key="city")
    assert mem.get(old.id).superseded_by == new.id and not mem.get(old.id).active
    assert mem.find_by_key("city").id == new.id
    res = mem.recall("where does Alice live? Porto or Lisbon", k=2)
    by = {r.id: r for r in res}
    assert by[old.id].rule_multiplier == pytest.approx(0.1)
    assert by[old.id].rules == ("superseded",) and "superseded" in by[old.id].explain()
    assert res[0].id == new.id
    assert old.id not in mem.recall("Porto", include_superseded=False).ids
    assert [m.id for m in mem.list()] == [new.id]
    assert len(mem.list(include_superseded=True)) == 2


def test_update_creates_superseding_memory(mem: Memory) -> None:
    a = mem.add("standup at 9am", kind="instruction", tags=["team"], key="standup")
    b = mem.update(a.id, "standup at 10am")
    assert b.kind == "instruction" and b.tags == ("team",) and b.key == "standup"
    assert mem.get(a.id).superseded_by == b.id
    assert mem.find_by_key("standup").id == b.id


def test_forget_ids_and_namespace(mem: Memory) -> None:
    a = mem.add("secret token is hunter2")
    b = mem.add("token rotation every 90 days")
    assert mem.forget(a.id)["memories"] == 1
    assert a.id not in mem and mem.get(a.id) is None
    assert a.id not in mem.recall("token", k=5).ids
    mem.mark_used([b.id])
    other = mem.add("bob likes tea", namespace="bob")
    out = mem.forget(namespace="user:alice")
    assert out["memories"] == 1 and out["events"] >= 1
    assert mem.list() == [] and other.id in mem
    assert "user:alice" not in mem.db.entities
    with pytest.raises(KeyError):
        mem.mark_used([a.id])


def test_kinds_and_tags_filters(mem: Memory) -> None:
    mem.add("use metric units", kind="preference", tags=["units"])
    i = mem.add("always answer in metric units", kind="instruction", tags=["units", "style"])
    assert mem.recall("metric units", kinds=["instruction"]).ids == [i.id]
    assert mem.recall("metric units", tags=["style"]).ids == [i.id]
    assert [m.id for m in mem.list(kind="instruction")] == [i.id]
    assert [m.id for m in mem.list(tags=["style"])] == [i.id]


def test_feedback_by_recall_id(mem: Memory) -> None:
    a = mem.add("coffee order: flat white")
    b = mem.add("coffee shop near office: Blue Door")
    res = mem.recall("coffee", k=2)
    mem.recall("something else entirely coffee", k=1)
    out = mem.feedback(used=[a.id], recall_id=res.recall_id)
    assert out == {"used": [a.id], "ignored": [b.id]}
    with pytest.raises(KeyError):
        mem.feedback(used=[a.id], recall_id="nope")


def test_persistence_round_trip(tmp_path, clock: Clock) -> None:
    d = tmp_path / "store"
    mem = Memory(d, namespace="u", clock=clock, embedder=HashingEmbedder(64))
    a = mem.add("Alice is vegetarian", kind="preference", tags=["diet"], key="diet")
    b = mem.add("Alice now eats fish", kind="preference", key="diet")
    c = mem.add("delete me")
    for _ in range(3):
        mem.recall("what does Alice eat", k=2)
        mem.feedback(used=[b.id])
    mem.forget(c.id)
    before = mem.recall("what does Alice eat", k=3)
    mem.save()
    files = {p.name for p in d.iterdir()}
    assert {"heapdb.jsonl", "memories.jsonl", "memory.json", "vectors.jsonl"} <= files

    back = Memory.load(d, clock=clock, embedder=HashingEmbedder(64))
    after = back.recall("what does Alice eat", k=3)
    assert after.ids == before.ids
    assert [r.score for r in after] == pytest.approx([r.score for r in before])
    assert back.get(a.id).superseded_by == b.id
    assert back.get(b.id).use_count == 3 and back.find_by_key("diet").id == b.id
    assert c.id not in back and len(back) == 2
    assert back.config == mem.config
    # The memory text never goes into the HeapDB snapshot.
    snap = (d / "heapdb.jsonl").read_text()
    assert "vegetarian" not in snap and "fish" not in snap
    assert json.loads((d / "memory.json").read_text())["version"] == 1


def test_autosave(tmp_path, clock: Clock) -> None:
    mem = Memory(tmp_path, autosave=True, clock=clock)
    m = mem.add("remember the milk")
    assert Memory(tmp_path, clock=clock).get(m.id).text == "remember the milk"


def test_load_missing_dir(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        Memory.load(tmp_path / "nothing")


def test_hybrid_recall_with_embedder(clock: Clock) -> None:
    mem = Memory(namespace="n", clock=clock, embedder=HashingEmbedder(256))
    a = mem.add("the quarterly roadmap review is on Thursday")
    mem.add("bananas are yellow")
    # "roadmaps" / "reviewing" do not share BM25 stems with ... they do; use a typo-ish variant
    res = mem.recall("quartrly roadmp revew", k=2)
    assert res[0].id == a.id and res[0].semantic is not None and res[0].cosine is not None


def test_add_many_and_stats(mem: Memory) -> None:
    items = mem.add_many([{"text": "a fact about cats", "kind": "fact"},
                          {"text": "user prefers dogs", "kind": "preference"}])
    assert len(items) == 2 and len(mem) == 2
    mem.mark_used([items[1].id])
    st = mem.stats()
    assert st["memories"] == 2 and st["kinds"] == {"fact": 1, "preference": 1}
    assert st["uses"] == 1 and st["most_useful_features"]


def test_config_validation_and_round_trip() -> None:
    with pytest.raises(ValueError):
        ScoringConfig(lexical_weight=2)
    with pytest.raises(ValueError):
        ScoringConfig(half_life_days=0)
    c = ScoringConfig(w_recency=0.5, prior=(0.2, 5.0))
    assert ScoringConfig.from_dict(json.loads(json.dumps(c.to_dict()))) == c


def test_zero_weight_switches_component_off(mem: Memory) -> None:
    mem.add("alpha beta", importance=3.0, created_at=T0 - 400 * DAY)
    r = mem.recall("alpha", config=ScoringConfig(w_recency=0, w_importance=0, w_usefulness=0))[0]
    assert r.score == pytest.approx(r.relevance)

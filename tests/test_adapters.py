"""LangGraph store and LlamaIndex memory block adapters (skipped if not installed)."""

from __future__ import annotations

import asyncio

import pytest

from heapdb_memory import Memory

T0 = 1_700_000_000.0


def test_langgraph_store() -> None:
    pytest.importorskip("langgraph.store.base")
    from heapdb_memory.adapters.langgraph import HeapDBStore

    store = HeapDBStore(Memory(clock=lambda: T0))
    ns = ("user", "alice")
    store.put(ns, "diet", {"text": "Alice is vegetarian", "kind": "preference", "tags": ["food"]})
    store.put(ns, "city", {"content": "Alice lives in Porto"})
    store.put(ns, "city", {"content": "Alice moved to Lisbon"})       # replace
    store.put(("user", "bob"), "diet", {"text": "Bob eats everything"})
    it = store.get(ns, "city")
    assert it.value == {"content": "Alice moved to Lisbon"} and it.key == "city"
    assert tuple(it.namespace) == ns
    hits = store.search(("user",), query="what does alice eat? vegetarian", limit=2)
    assert hits[0].key == "diet" and tuple(hits[0].namespace) == ns and hits[0].score > 0
    assert [h.key for h in store.search(ns, query="Lisbon Porto", limit=10)] == ["city"]  # replaced, not duplicated
    assert [h.key for h in store.search(ns, filter={"kind": "preference"})] == ["diet"]
    assert sorted(store.list_namespaces()) == [("user", "alice"), ("user", "bob")]
    assert store.list_namespaces(max_depth=1) == [("user",)]
    assert store.list_namespaces(prefix=("user", "bob")) == [("user", "bob")]
    assert store.mark_used([hits[0]]) == 1
    assert store.mark_ignored([(ns, "city")]) == 1
    store.delete(ns, "diet")
    assert store.get(ns, "diet") is None
    assert asyncio.run(store.aget(ns, "city")).value["content"] == "Alice moved to Lisbon"


def test_llamaindex_block() -> None:
    pytest.importorskip("llama_index.core.memory")
    from llama_index.core.base.llms.types import ChatMessage
    from llama_index.core.memory import Memory as LIMemory

    from heapdb_memory.adapters.llamaindex import HeapDBMemoryBlock

    mem = Memory(clock=lambda: T0)
    block = HeapDBMemoryBlock(memory=mem, namespace="alice", k=2)
    asyncio.run(block.aput([ChatMessage(role="user", content="I love hiking in the Alps"),
                            ChatMessage(role="assistant", content="Noted, mountains it is."),
                            ChatMessage(role="system", content="ignored system text")]))
    assert len(mem.list("alice")) == 2
    out = asyncio.run(block.aget([ChatMessage(role="user", content="plan a hiking weekend")]))
    assert "hiking in the Alps" in out and out.startswith("- [")
    shown = block.last_recall_ids
    fb = block.feedback(used=[shown[0]])
    assert fb["used"] == [shown[0]] and set(fb["ignored"]) == set(shown[1:])
    assert asyncio.run(block.aget([])) == ""
    li = LIMemory.from_defaults(session_id="s", memory_blocks=[block])
    assert li.memory_blocks[0] is block

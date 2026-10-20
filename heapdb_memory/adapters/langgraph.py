"""LangGraph long-term memory store backed by :class:`heapdb_memory.Memory`.

``HeapDBStore`` implements ``langgraph.store.base.BaseStore`` (``batch`` /
``abatch`` over ``GetOp``, ``PutOp``, ``SearchOp``, ``ListNamespacesOp``), so
it can be passed wherever LangGraph takes a store::

    from heapdb_memory import Memory
    from heapdb_memory.adapters.langgraph import HeapDBStore

    store = HeapDBStore(Memory("./memory"))
    graph = builder.compile(store=store)
    # in a node:
    store.put(("user", "alice"), "diet", {"text": "Alice is vegetarian", "kind": "preference"})
    hits = store.search(("user", "alice"), query="what can alice eat?", limit=3)
    store.mark_used([hits[0]])      # learning signal (not part of BaseStore)

Mapping:

* A LangGraph namespace tuple becomes the Memory namespace ``"/".join(ns)``.
* ``put(ns, key, value)`` stores one memory whose text is ``value[text_field]``
  (first of ``text``, ``content``, ``memory``, ``data`` present, else the JSON of
  the value). ``value["kind"]``, ``value["importance"]`` and ``value["tags"]``
  are honoured if present. The whole value is kept in metadata and returned
  unchanged. Re-putting a key *replaces* the memory (LangGraph semantics); the
  new memory keeps the old one's usage history in the learned links.
* ``search(prefix, query=...)`` is :meth:`Memory.recall` in each namespace
  under ``prefix``, merged by score; ``SearchItem.score`` is the HeapDB score.
  Without a query it lists newest first. ``filter`` is exact match on value
  fields.
* ``delete`` / ``put(..., None)`` is :meth:`Memory.forget`.

``index`` and ``ttl`` arguments are accepted and ignored (indexing is always
on; there is no TTL).
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
)

from ..memory import Memory, MemoryItem

__all__ = ["HeapDBStore"]

TEXT_FIELDS = ("text", "content", "memory", "data")


def _ns_str(ns: Sequence[str]) -> str:
    return "/".join(ns)


def _dt(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


class HeapDBStore(BaseStore):
    """A LangGraph ``BaseStore`` over a :class:`Memory`.

    Args:
        memory: The memory store (persistence, scoring config, embedder).
        default_kind: Kind for values without a ``kind`` field.
    """

    supports_ttl = False

    def __init__(self, memory: Memory, default_kind: str = "fact") -> None:
        self.memory = memory
        self.default_kind = default_kind

    # -- helpers ---------------------------------------------------------------

    def _find(self, ns: tuple[str, ...], key: str) -> MemoryItem | None:
        return self.memory.find_by_key(key, namespace=_ns_str(ns))

    def _item(self, it: MemoryItem, score: float | None = None, search: bool = False) -> Item:
        ns = tuple(it.metadata.get("lg_namespace", it.namespace.split("/")))
        value = it.metadata.get("lg_value", {"text": it.text})
        created = _dt(it.metadata.get("lg_created_at", it.created_at))
        updated = _dt(it.created_at)
        if search:
            return SearchItem(namespace=ns, key=it.key or it.id, value=value,
                              created_at=created, updated_at=updated, score=score)
        return Item(value=value, key=it.key or it.id, namespace=ns,
                    created_at=created, updated_at=updated)

    @staticmethod
    def _text(value: dict[str, Any]) -> str:
        for f in TEXT_FIELDS:
            v = value.get(f)
            if isinstance(v, str) and v.strip():
                return v
        return json.dumps(value, sort_keys=True, default=str)

    def _put(self, op: PutOp) -> None:
        ns = tuple(op.namespace)
        old = self._find(ns, op.key)
        if op.value is None:
            if old is not None:
                self.memory.forget(old.id)
            return
        value = dict(op.value)
        created_at = old.metadata.get("lg_created_at", old.created_at) if old else None
        if old is not None:
            self.memory.forget(old.id)
        tags = value.get("tags")
        meta = {"lg_namespace": list(ns), "lg_value": value}
        it = self.memory.add(
            self._text(value), kind=str(value.get("kind", self.default_kind)),
            importance=value.get("importance"), tags=tags if isinstance(tags, list) else (),
            namespace=_ns_str(ns), key=op.key, metadata=meta)
        it.metadata["lg_created_at"] = created_at if created_at is not None else it.created_at

    def _namespaces(self) -> list[tuple[str, ...]]:
        seen = {}
        for it in self.memory:
            ns = tuple(it.metadata.get("lg_namespace", it.namespace.split("/")))
            seen[ns] = None
        return sorted(seen)

    @staticmethod
    def _match(value: dict[str, Any], flt: dict[str, Any] | None) -> bool:
        return not flt or all(value.get(k) == v for k, v in flt.items())

    def _search(self, op: SearchOp) -> list[SearchItem]:
        prefix = tuple(op.namespace_prefix)
        spaces = [ns for ns in self._namespaces() if ns[: len(prefix)] == prefix]
        rows: list[tuple[float, float, SearchItem]] = []
        need = op.offset + op.limit
        for ns in spaces:
            name = _ns_str(ns)
            if op.query:
                res = self.memory.recall(op.query, k=max(need * 3, 10), namespace=name,
                                         include_superseded=False, explain=False)
                for r in res:
                    v = r.memory.metadata.get("lg_value", {})
                    if self._match(v, op.filter):
                        rows.append((r.score, r.memory.created_at,
                                     self._item(r.memory, r.score, search=True)))
            else:
                for it in self.memory.list(name):
                    if self._match(it.metadata.get("lg_value", {}), op.filter):
                        rows.append((0.0, it.created_at, self._item(it, None, search=True)))
        rows.sort(key=lambda r: (-r[0], -r[1]))
        return [r[2] for r in rows[op.offset: op.offset + op.limit]]

    def _list_ns(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        out = self._namespaces()
        for cond in op.match_conditions or ():
            path = tuple(cond.path)

            def ok(ns: tuple[str, ...]) -> bool:
                part = ns[: len(path)] if cond.match_type == "prefix" else ns[-len(path):]
                return len(ns) >= len(path) and all(p == "*" or p == q for p, q in zip(path, part))
            out = [ns for ns in out if ok(ns)]
        if op.max_depth is not None:
            out = sorted({ns[: op.max_depth] for ns in out})
        return out[op.offset: op.offset + op.limit]

    # -- BaseStore -------------------------------------------------------------

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        results: list[Result] = []
        for op in ops:
            if isinstance(op, GetOp):
                it = self._find(tuple(op.namespace), op.key)
                results.append(self._item(it) if it is not None else None)
            elif isinstance(op, PutOp):
                self._put(op)
                results.append(None)
            elif isinstance(op, SearchOp):
                results.append(self._search(op))
            elif isinstance(op, ListNamespacesOp):
                results.append(self._list_ns(op))
            else:
                raise ValueError(f"unknown operation {type(op).__name__}")
        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        ops = list(ops)
        return await asyncio.get_running_loop().run_in_executor(None, self.batch, ops)

    # -- learning (beyond BaseStore) ---------------------------------------------

    def _ids(self, items: Iterable[Item | tuple[tuple[str, ...], str]]) -> list[str]:
        out = []
        for x in items:
            ns, key = (x.namespace, x.key) if isinstance(x, Item) else x
            it = self._find(tuple(ns), key)
            if it is not None:
                out.append(it.id)
        return out

    def mark_used(self, items: Iterable[Item | tuple[tuple[str, ...], str]]) -> int:
        """Items the agent actually used (``Item`` or ``(namespace, key)``)."""
        return self.memory.mark_used(self._ids(items))

    def mark_ignored(self, items: Iterable[Item | tuple[tuple[str, ...], str]]) -> int:
        """Items that were retrieved but not used."""
        return self.memory.mark_ignored(self._ids(items))

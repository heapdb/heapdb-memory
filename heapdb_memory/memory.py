"""The :class:`Memory` store: long-term memory for AI agents on HeapDB.

Model
-----
* A **namespace** (HeapDB *entity*) owns memories: a user, an agent, a project.
* A **memory** is a HeapDB *event* of type ``memory`` in its namespace, at its
  creation time, with ``weight`` = importance (the *dose*) and a small fixed
  set of shape *features* (``kind:preference``, ``source:tool``,
  ``len:long``, ``date:yes``, ``topic:food``, ``tag:diet`` ...; see
  :func:`heapdb_memory.text.memory_features`). The text itself lives in this
  module's store (and the BM25 / vector indexes), never in HeapDB.
* The HeapDB effect is ``useful``: "the agent actually used this memory".
  Feedback is recorded on **retrieval events**: when the agent reports that a
  recalled memory was used (:meth:`Memory.mark_used`) or ignored
  (:meth:`Memory.mark_ignored`), a HeapDB event of type ``retrieval`` is
  inserted with the memory's features plus those features crossed with the
  query type (``qt:when&date:yes``, :func:`~heapdb_memory.text.pair_features`)
  and confirmed / rejected with ``HeapDB.feedback``. So the Beta links per
  (feature, ``useful``) learn *which kinds of memory are useful for which
  kinds of question*, per namespace with a pooled global layer (HeapDB's
  hierarchical priors). Retrieval events carry no text.

Scoring (see ``docs/use-cases/agent-memory.md``)
-----------------------------------------------
For a query ``q`` at time ``now`` every candidate memory ``m`` gets::

    score(m) = R(q, m)^w_rel * T(m)^w_rec * I(m)^w_imp * U(m)^w_use * rules(m)

    R  relevance     = a * bm25(q, m) / max_c bm25(q, c)
                       + (1 - a) * (cos(q, m) - min_c cos) / (max_c cos - min_c cos)
    T  recency       = floor + (1 - floor) * 0.5 ** ((now - last_used(m)) / half_life)
    I  importance    = the memory's dose (explicit, or the kind/length heuristic)
    U  usefulness    = noisy-OR over the retrieval features of (q, m) of the
                       learned link strengths P(useful | f) in m's namespace
                       (HeapDB feature_risk)
    rules            = HeapDB rule multipliers (superseded memories x0.1)

It is a log-linear (product-of-experts) model: each exponent ``w`` is the
weight of one component in ``log score``; with all ``w = 1`` it is HeapDB's
own ``kernel x dose x links x rules`` score with a relevance factor in front.
Multiplication makes every component scale-free (multiplying a component by
a constant for all candidates never changes the ranking), so the components
need no per-query normalisation beyond relevance. ``last_used`` is the
creation time until the memory is first marked used; each confirmed use
refreshes it, so memories that keep being useful decay slowly (spaced
reinforcement) and unused ones fade to the ``floor``.

Candidates are the union of the top ``candidates`` memories by BM25 and (if an
embedder is configured) by cosine similarity, so cost per recall is bounded
by the candidate pool, not the store size.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import threading
import time as _time
import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping

from heapdb import ExponentialDecay, HeapDB

from .bm25 import BM25Index
from .embeddings import Embedder, VectorIndex
from .text import content_terms, memory_features, pair_features, terms

__all__ = [
    "KINDS",
    "Memory",
    "MemoryItem",
    "Recalled",
    "RecallResult",
    "ScoringConfig",
]

#: The memory kinds with built-in importance defaults. Any other string is
#: accepted as a kind too (importance 1.0).
KINDS = ("fact", "preference", "instruction", "episode", "summary", "tool_result", "note")

EFFECT = "useful"
EVENT_TYPE = "memory"
RETRIEVAL_TYPE = "retrieval"
DAY = 86400.0

STATE_VERSION = 1
SNAPSHOT_FILE = "heapdb.jsonl"
MEMORIES_FILE = "memories.jsonl"
VECTORS_FILE = "vectors.jsonl"
META_FILE = "memory.json"


def _to_ts(t: float | datetime | None, default: Callable[[], float]) -> float:
    if t is None:
        return float(default())
    if isinstance(t, datetime):
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return t.timestamp()
    return float(t)


@dataclass
class ScoringConfig:
    """Weights and constants of the recall score (all documented in the module
    docstring). Defaults are reasoned, not tuned on any benchmark.

    Attributes:
        w_relevance, w_recency, w_importance, w_usefulness: exponents of the
            log-linear score (0 switches a component off).
        lexical_weight: share ``a`` of BM25 in hybrid relevance; the rest is
            embedding cosine. Ignored (treated as 1.0) without an embedder.
        memory_links: also learn a link per memory (feature ``mem:<id>`` on its
            retrieval events), i.e. "this particular memory keeps being
            useful", on top of the per-kind links.
        half_life_days: recency half-life on time since last use.
        recency_floor: lower bound of the recency factor, so an old memory is
            down-weighted at most ``1/floor`` times relative to a fresh one.
        candidates: top-N per retriever (BM25, embeddings) unioned into the
            candidate pool.
        superseded_multiplier: HeapDB rule multiplier for superseded memories.
        prior: Beta prior of every (feature, useful) link. The default (1, 9)
            (mean 0.1) is sceptical so the noisy-OR over ~8 features does not
            saturate.
        global_pseudocount: how many pseudo-observations a namespace borrows
            from the pooled evidence of all namespaces (HeapDB hierarchical
            links). Larger = new namespaces trust population learning more.
        kind_importance: default importance per kind.
        length_importance: if True, heuristic importance is scaled by
            ``0.6 + 0.4 * min(1, content_terms / 8)`` (short chit-chat like
            "ok thanks" gets less weight).
        bm25_k1, bm25_b: BM25 parameters.
    """

    w_relevance: float = 1.0
    w_recency: float = 1.0
    w_importance: float = 1.0
    w_usefulness: float = 1.0
    lexical_weight: float = 0.5
    memory_links: bool = True
    half_life_days: float = 30.0
    recency_floor: float = 0.5
    candidates: int = 50
    superseded_multiplier: float = 0.1
    prior: tuple[float, float] = (0.1, 9.9)
    global_pseudocount: float = 20.0
    kind_importance: dict[str, float] = field(default_factory=lambda: {
        "instruction": 1.5, "preference": 1.3, "fact": 1.0, "summary": 1.2,
        "note": 1.0, "episode": 0.8, "tool_result": 0.6,
    })
    length_importance: bool = True
    bm25_k1: float = 1.2
    bm25_b: float = 0.75

    def __post_init__(self) -> None:
        if not 0.0 <= self.lexical_weight <= 1.0:
            raise ValueError("lexical_weight must be in [0, 1]")
        if not 0.0 <= self.recency_floor <= 1.0:
            raise ValueError("recency_floor must be in [0, 1]")
        if self.half_life_days <= 0:
            raise ValueError("half_life_days must be > 0")
        if self.candidates < 1:
            raise ValueError("candidates must be >= 1")
        if not 0.0 <= self.superseded_multiplier <= 1.0:
            raise ValueError("superseded_multiplier must be in [0, 1]")
        self.prior = (float(self.prior[0]), float(self.prior[1]))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["prior"] = list(self.prior)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ScoringConfig":
        known = {f.name for f in fields(cls)}
        kw = {k: v for k, v in d.items() if k in known}
        if "prior" in kw:
            kw["prior"] = tuple(kw["prior"])
        return cls(**kw)


@dataclass
class MemoryItem:
    """One stored memory. ``created_at``/``last_used`` are Unix seconds."""

    id: str
    namespace: str
    text: str
    kind: str
    importance: float
    importance_source: str           # "explicit" | "heuristic"
    created_at: float
    last_used: float
    features: tuple[str, ...]
    event_id: int                    # HeapDB event id
    tags: tuple[str, ...] = ()
    source: str | None = None
    key: str | None = None           # supersede key ("user.diet")
    superseded_by: str | None = None
    use_count: int = 0
    ignore_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def active(self) -> bool:
        return self.superseded_by is None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["features"] = list(self.features)
        d["tags"] = list(self.tags)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "MemoryItem":
        known = {f.name for f in fields(cls)}
        kw = {k: v for k, v in d.items() if k in known}
        kw["features"] = tuple(kw.get("features", ()))
        kw["tags"] = tuple(kw.get("tags", ()))
        return cls(**kw)


@dataclass(frozen=True)
class Recalled:
    """One recalled memory with its score broken down (EXPLAIN).

    ``score = relevance^w_rel * recency^w_rec * importance^w_imp *
    usefulness^w_use * rule_multiplier``.
    """

    rank: int
    memory: MemoryItem
    score: float
    relevance: float
    lexical: float            # BM25 / max BM25 over candidates, in [0, 1]
    semantic: float | None    # cosine / max cosine over candidates (None: no embedder)
    bm25: float               # raw BM25
    cosine: float | None      # raw cosine
    age_days: float           # since last use (or creation)
    recency: float
    importance: float
    usefulness: float         # noisy-OR of learned links
    rule_multiplier: float
    rules: tuple[str, ...] = ()
    feature_links: tuple[tuple[str, float], ...] = ()   # (feature, P(useful|f)), strongest first

    @property
    def id(self) -> str:
        return self.memory.id

    @property
    def text(self) -> str:
        return self.memory.text

    def explain(self, top_features: int = 3) -> str:
        """One-line explanation of the score."""
        rel = f"rel {self.relevance:.2f} (bm25 {self.lexical:.2f}"
        rel += f", cos {self.semantic:.2f})" if self.semantic is not None else ")"
        used = f", used {self.memory.use_count}x" if self.memory.use_count else ""
        feats = ", ".join(f"{f}={s:.2f}" for f, s in self.feature_links[:top_features])
        out = (f"{rel} x recency {self.recency:.2f} (age {self.age_days:.1f}d{used})"
               f" x importance {self.importance:.2f} x useful {self.usefulness:.2f}")
        if feats:
            out += f" [{feats}]"
        if self.rules:
            out += f" x rules {', '.join(self.rules)} x{self.rule_multiplier:g}"
        return out

    def to_dict(self, explain: bool = True) -> dict[str, Any]:
        d = {
            "id": self.memory.id, "rank": self.rank, "text": self.memory.text,
            "kind": self.memory.kind, "namespace": self.memory.namespace,
            "score": round(self.score, 6), "created_at": self.memory.created_at,
            "tags": list(self.memory.tags),
        }
        if self.memory.key:
            d["key"] = self.memory.key
        if explain:
            d["explanation"] = self.explain()
            d["components"] = {
                "relevance": self.relevance, "lexical": self.lexical, "semantic": self.semantic,
                "bm25": self.bm25, "cosine": self.cosine, "age_days": self.age_days,
                "recency": self.recency, "importance": self.importance,
                "usefulness": self.usefulness, "rule_multiplier": self.rule_multiplier,
                "feature_links": dict(self.feature_links),
            }
        return d


class RecallResult(list):
    """A list of :class:`Recalled` (best first) with the query context.

    ``recall_id`` identifies this recall for :meth:`Memory.feedback`.
    """

    def __init__(self, items: Iterable[Recalled], query: str, namespace: str,
                 at: float, recall_id: str, candidates: int) -> None:
        super().__init__(items)
        self.query = query
        self.namespace = namespace
        self.at = at
        self.recall_id = recall_id
        self.candidates = candidates

    @property
    def ids(self) -> list[str]:
        return [r.memory.id for r in self]

    def __str__(self) -> str:
        lines = [f"RECALL {self.query!r} IN {self.namespace} ({self.candidates} candidates)"]
        for r in self:
            lines.append(f"  {r.rank}. [{r.memory.id}] {r.memory.text[:100]}  score={r.score:.4f}")
            lines.append(f"      {r.explain()}")
        return "\n".join(lines)


class _Namespace:
    """Per-namespace indexes."""

    def __init__(self, k1: float, b: float) -> None:
        self.bm25 = BM25Index(k1=k1, b=b)
        self.vectors = VectorIndex()
        self.ids: set[str] = set()
        self.keys: dict[str, str] = {}      # supersede key -> active memory id


class Memory:
    """Long-term memory for agents: add, recall with EXPLAIN, learn from use.

    Args:
        path: Directory for persistence (created if needed). ``None`` keeps
            everything in memory. If the directory holds a saved store it is
            loaded.
        namespace: Default namespace (user / agent / project) for calls that
            do not pass one. Default: the saved one when loading, else
            ``"default"``.
        config: :class:`ScoringConfig`. When loading, the saved config is used
            unless one is passed.
        embedder: Optional :class:`~heapdb_memory.embeddings.Embedder` for
            hybrid (BM25 + cosine) recall. Vectors are cached on save, keyed by
            ``embedder.name``.
        autosave: Save after every mutating call (only with ``path``).
        clock: Time source (Unix seconds) used when ``now`` is not passed.

    Example::

        mem = Memory("./memory", namespace="user:alice")
        mem.add("Alice is vegetarian", kind="preference", key="diet")
        res = mem.recall("what should I cook for Alice?", k=3)
        print(res)                       # ranked, with one-line explanations
        mem.feedback(used=[res[0].id])   # used -> confirm; other shown -> reject
    """

    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        namespace: str | None = None,
        config: ScoringConfig | None = None,
        embedder: Embedder | None = None,
        autosave: bool = False,
        clock: Callable[[], float] = _time.time,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self._namespace_arg = namespace
        self.namespace = namespace or "default"
        self.embedder = embedder
        self.autosave = bool(autosave and self.path is not None)
        self.clock = clock
        self._lock = threading.RLock()
        self._items: dict[str, MemoryItem] = {}
        self._by_event: dict[int, str] = {}
        self._ns: dict[str, _Namespace] = {}
        self._superseded: set[str] = set()
        self._last_recall: dict[str, RecallResult] = {}
        self._recalls: dict[str, RecallResult] = {}
        self.counters = {"recalls": 0, "used": 0, "ignored": 0}

        loaded = self.path is not None and (self.path / META_FILE).exists()
        if loaded:
            self._load(config)
        else:
            self.config = config or ScoringConfig()
            self.db = self._new_db()
        self._install_rules()

    # -- set-up ------------------------------------------------------------------

    def _kernel(self) -> ExponentialDecay:
        # Long horizon: the floor handles very old memories, so the kernel
        # should not be truncated to 0 early.
        return ExponentialDecay(self.config.half_life_days * DAY, horizon_half_lives=60.0)

    def _new_db(self) -> HeapDB:
        db = HeapDB(bucket_seconds=DAY)
        db.register_effect(EFFECT, self._kernel(), default_prior=self.config.prior,
                           global_pseudocount=self.config.global_pseudocount)
        db.register_event_type(EVENT_TYPE)
        db.register_event_type(RETRIEVAL_TYPE)
        return db

    def _install_rules(self) -> None:
        sup = self._superseded
        self.db.add_rule(EFFECT, "superseded",
                         lambda ev: ev.attrs.get("mid") in sup,
                         self.config.superseded_multiplier, types=[EVENT_TYPE])

    def _space(self, namespace: str) -> _Namespace:
        sp = self._ns.get(namespace)
        if sp is None:
            sp = self._ns[namespace] = _Namespace(self.config.bm25_k1, self.config.bm25_b)
        return sp

    # -- writes ------------------------------------------------------------------

    def heuristic_importance(self, text: str, kind: str) -> float:
        """Default dose: kind prior x (optional) content-length factor."""
        imp = self.config.kind_importance.get(kind, 1.0)
        if self.config.length_importance:
            imp *= 0.6 + 0.4 * min(1.0, len(content_terms(text)) / 8.0)
        return imp

    def add(
        self,
        text: str,
        kind: str = "fact",
        importance: float | None = None,
        tags: Iterable[str] = (),
        namespace: str | None = None,
        entity: str | None = None,
        source: str | None = None,
        key: str | None = None,
        created_at: float | datetime | None = None,
        metadata: Mapping[str, Any] | None = None,
        id: str | None = None,
    ) -> MemoryItem:
        """Store a memory and return it.

        Args:
            text: The memory, as the agent should read it back.
            kind: ``fact``, ``preference``, ``instruction``, ``episode``,
                ``summary``, ``tool_result``, ``note`` or any other string.
            importance: Dose (> 0; 1.0 = normal). ``None`` uses
                :meth:`heuristic_importance`.
            tags: User tags; each becomes a ``tag:<t>`` feature.
            namespace: Owner (user / agent / project). ``entity`` is an alias.
            source: Where it came from (``user``, ``agent``, ``tool``, a
                speaker name...); becomes the ``source:`` feature.
            key: Supersede key. An active memory in the same namespace with
                the same key is superseded by this one (down-weighted by the
                HeapDB rule ``superseded``, still recallable).
            created_at: Unix seconds or datetime (default: now).
            metadata: Free JSON-serialisable data returned with the memory.
            id: Explicit id (must be unique); default a random 12-hex id.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        ns = namespace or entity or self.namespace
        tags = tuple(dict.fromkeys(t.strip().lower() for t in tags if t and t.strip()))
        with self._lock:
            mid = id or uuid.uuid4().hex[:12]
            if mid in self._items:
                raise ValueError(f"memory id {mid!r} already exists")
            t = _to_ts(created_at, self.clock)
            if importance is None:
                imp, imp_src = self.heuristic_importance(text, kind), "heuristic"
            else:
                imp, imp_src = float(importance), "explicit"
                if not imp > 0 or math.isinf(imp):
                    raise ValueError("importance must be a finite number > 0")
            feats = memory_features(text, kind, source, tags)
            ev = self.db.insert(ns, t, EVENT_TYPE, feats, weight=imp, attrs={"mid": mid})
            item = MemoryItem(
                id=mid, namespace=ns, text=text, kind=kind, importance=imp,
                importance_source=imp_src, created_at=t, last_used=t,
                features=tuple(sorted(feats)), event_id=ev.id, tags=tags, source=source,
                key=key, metadata=dict(metadata or {}),
            )
            self._index(item)
            if self.embedder is not None:
                self._space(ns).vectors.add(mid, self.embedder.embed([text])[0])
            if key is not None:
                old = self._space(ns).keys.get(key)
                if old is not None and old in self._items:
                    self._mark_superseded(old, mid)
                self._space(ns).keys[key] = mid
            self._maybe_save()
            return item

    def add_many(self, items: Iterable[Mapping[str, Any]]) -> list[MemoryItem]:
        """Add several memories (dicts of :meth:`add` keyword arguments),
        embedding them in one batch when an embedder is configured."""
        items = list(items)
        emb, self.embedder = self.embedder, None
        try:
            with self._lock:
                saved, self.autosave = self.autosave, False
                try:
                    out = [self.add(**it) for it in items]
                finally:
                    self.autosave = saved
        finally:
            self.embedder = emb
        if emb is not None and out:
            vecs = emb.embed([m.text for m in out])
            for m, v in zip(out, vecs):
                self._space(m.namespace).vectors.add(m.id, v)
        self._maybe_save()
        return out

    def _index(self, item: MemoryItem) -> None:
        self._items[item.id] = item
        self._by_event[item.event_id] = item.id
        sp = self._space(item.namespace)
        sp.ids.add(item.id)
        sp.bm25.add(item.id, terms(item.text))
        if item.superseded_by is not None:
            self._superseded.add(item.id)

    def _mark_superseded(self, old_id: str, new_id: str) -> None:
        old = self._items[old_id]
        old.superseded_by = new_id
        self._superseded.add(old_id)

    def supersede(self, old_id: str, new_id: str) -> None:
        """Mark ``old_id`` as superseded by ``new_id`` (same namespace)."""
        with self._lock:
            old, new = self._get(old_id), self._get(new_id)
            if old.namespace != new.namespace:
                raise ValueError("can only supersede within one namespace")
            self._mark_superseded(old_id, new_id)
            self._maybe_save()

    def update(self, memory_id: str, text: str, importance: float | None = None,
               tags: Iterable[str] | None = None, now: float | datetime | None = None) -> MemoryItem:
        """Replace a memory's text. HeapDB events are immutable, so this adds a
        new memory (same kind, key, source and, unless given, tags) that
        supersedes the old one, and returns the new memory."""
        with self._lock:
            old = self._get(memory_id)
            new = self.add(text, kind=old.kind, importance=importance,
                           tags=old.tags if tags is None else tags, namespace=old.namespace,
                           source=old.source, key=None, created_at=now, metadata=old.metadata)
            self._mark_superseded(old.id, new.id)
            if old.key is not None:
                new.key = old.key
                self._space(old.namespace).keys[old.key] = new.id
            self._maybe_save()
            return new

    def forget(self, memory_ids: str | Iterable[str] = (), namespace: str | None = None,
               scrub_global: bool = True) -> dict[str, int]:
        """Delete memories.

        * ``forget(ids)``: removes the text, metadata and index entries of
          each memory immediately. HeapDB has no per-event delete, so the
          memory's HeapDB event (features and dose only, no text) stays as a
          tombstone, hard-blocked from recall, until its namespace is forgotten.
        * ``forget(namespace="user:alice")``: everything about the namespace,
          including its HeapDB events and learned links
          (``HeapDB.forget_entity``; with ``scrub_global`` its contributions to
          the pooled global links are subtracted too).
        """
        if isinstance(memory_ids, str):
            memory_ids = [memory_ids]
        n = 0
        with self._lock:
            for mid in list(memory_ids):
                if self._remove(mid):
                    n += 1
            out = {"memories": n}
            if namespace is not None:
                for mid in [m for m, it in self._items.items() if it.namespace == namespace]:
                    self._remove(mid)
                    out["memories"] += 1
                out.update(self.db.forget_entity(namespace, scrub_global=scrub_global))
                self._ns.pop(namespace, None)
                self._last_recall.pop(namespace, None)
            self._maybe_save()
            return out

    def _remove(self, mid: str) -> bool:
        item = self._items.pop(mid, None)
        if item is None:
            return False
        self._by_event.pop(item.event_id, None)
        self._superseded.add(mid)          # tombstone: blocks it via the rule
        sp = self._ns.get(item.namespace)
        if sp is not None:
            sp.ids.discard(mid)
            sp.bm25.remove(mid)
            sp.vectors.remove(mid)
            if item.key is not None and sp.keys.get(item.key) == mid:
                del sp.keys[item.key]
        for it in self._items.values():
            if it.superseded_by == mid:
                it.superseded_by = None
                self._superseded.discard(it.id)
        return True

    # -- reads -------------------------------------------------------------------

    def _get(self, memory_id: str) -> MemoryItem:
        try:
            return self._items[memory_id]
        except KeyError:
            raise KeyError(f"unknown memory id {memory_id!r}") from None

    def get(self, memory_id: str) -> MemoryItem | None:
        return self._items.get(memory_id)

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, memory_id: object) -> bool:
        return memory_id in self._items

    def __iter__(self) -> Iterator[MemoryItem]:
        return iter(list(self._items.values()))

    @property
    def namespaces(self) -> list[str]:
        return sorted({it.namespace for it in self._items.values()})

    def list(self, namespace: str | None = None, kind: str | None = None,
             tags: Iterable[str] = (), include_superseded: bool = False,
             limit: int | None = None, offset: int = 0, all_namespaces: bool = False) -> list[MemoryItem]:
        """Memories, newest first, optionally filtered."""
        ns = None if all_namespaces else (namespace or self.namespace)
        want = {t.lower() for t in tags}
        out = [it for it in self._items.values()
               if (ns is None or it.namespace == ns)
               and (kind is None or it.kind == kind)
               and (include_superseded or it.active)
               and want <= set(it.tags)]
        out.sort(key=lambda it: (-it.created_at, it.id))
        out = out[offset:]
        return out[:limit] if limit is not None else out

    def find_by_key(self, key: str, namespace: str | None = None) -> MemoryItem | None:
        """The active memory with this supersede key, if any."""
        sp = self._ns.get(namespace or self.namespace)
        mid = sp.keys.get(key) if sp else None
        return self._items.get(mid) if mid else None

    # -- recall ------------------------------------------------------------------

    def _usefulness(self, feats: frozenset[str], ns: str, mid: str,
                    cache: dict[frozenset[str], float]) -> tuple[float, float]:
        """(noisy-OR of learned links over the retrieval features, HeapDB rule
        multiplier of the memory event)."""
        u = cache.get(feats)
        if u is None:
            # No `type`: typed rules do not apply, so this is the pure noisy-OR.
            # Rounded to 12 significant digits: HeapDB multiplies over a set, whose
            # order (and so the last float bits) depends on the hash seed; this
            # keeps rankings bit-reproducible across processes.
            u = cache[feats] = float(f"{self.db.feature_risk(EFFECT, feats, 1.0, entity=ns):.12g}")
        m = 1.0
        if mid in self._superseded:
            s = self.db.feature_risk(EFFECT, feats, 1.0, entity=ns, type=EVENT_TYPE,
                                     attrs={"mid": mid})
            m = s / u if u > 0 else 0.0
        return u, m

    def recall(
        self,
        query: str,
        k: int = 5,
        namespace: str | None = None,
        entity: str | None = None,
        now: float | datetime | None = None,
        explain: bool = True,
        kinds: Iterable[str] | None = None,
        tags: Iterable[str] = (),
        include_superseded: bool = True,
        min_score: float = 0.0,
        config: ScoringConfig | None = None,
        track: bool = True,
    ) -> RecallResult:
        """Top-``k`` memories for ``query``, best first, with EXPLAIN.

        Args:
            query: What the agent is about to do / was asked.
            k: How many to return.
            namespace: Whose memories (``entity`` is an alias).
            now: Query time (Unix seconds / datetime; default: clock).
            explain: Include per-feature usefulness links (slightly slower).
            kinds / tags: Only memories of these kinds / with all these tags.
            include_superseded: Superseded memories stay recallable but are
                down-weighted by the HeapDB rule; False drops them.
            min_score: Drop results below this score.
            config: Override the scoring config for this call (e.g. to compare
                weightings); the store's own config is not changed.
            track: Remember the shown ids for :meth:`feedback`.

        If no memory matches the query lexically (and there is no embedder),
        the namespace's memories are ranked with relevance 1 (i.e. by recency
        x importance x usefulness), so a recall never comes back empty while
        memories exist.
        """
        if k < 1:
            raise ValueError("k must be >= 1")
        cfg = config or self.config
        ns = namespace or entity or self.namespace
        at = _to_ts(now, self.clock)
        with self._lock:
            sp = self._ns.get(ns)
            rid = uuid.uuid4().hex[:12]
            if sp is None or not sp.ids:
                return RecallResult([], query, ns, at, rid, 0)
            qterms = terms(query)
            bm = dict(sp.bm25.top(qterms, cfg.candidates)) if qterms else {}
            cos: dict[str, float] = {}
            use_emb = self.embedder is not None and len(sp.vectors) > 0
            if use_emb:
                qv = self.embedder.embed([query])[0]
                sims = sp.vectors.similarities(qv)
                top = sorted(sims.items(), key=lambda kv: (-kv[1], kv[0]))[: cfg.candidates]
                cand = set(bm) | {d for d, _ in top}
                cos = {d: sims.get(d, 0.0) for d in cand}
            else:
                cand = set(bm)
            fallback = not cand
            if fallback:
                cand = set(sp.ids)
            # BM25 for embedding-only candidates (cheap: one doc each).
            for d in cand:
                if d not in bm:
                    bm[d] = sp.bm25.score(qterms, d) if qterms else 0.0
            a = cfg.lexical_weight if use_emb else 1.0
            bmax = max((bm[d] for d in cand), default=0.0)
            cmin = min((cos.get(d, 0.0) for d in cand), default=0.0)
            cspan = max((cos.get(d, 0.0) for d in cand), default=0.0) - cmin
            kind_set = set(kinds) if kinds is not None else None
            want_tags = {t.lower() for t in tags}
            kernel = ExponentialDecay(cfg.half_life_days * DAY, horizon_half_lives=1e9)
            cache: dict[frozenset[str], float] = {}
            rows = []
            for d in cand:
                it = self._items[d]
                if kind_set is not None and it.kind not in kind_set:
                    continue
                if want_tags and not want_tags <= set(it.tags):
                    continue
                if not include_superseded and not it.active:
                    continue
                lex = bm[d] / bmax if bmax > 0 else 0.0
                sem = ((cos.get(d, 0.0) - cmin) / cspan if cspan > 0 else 1.0) if use_emb else None
                rel = 1.0 if fallback else a * lex + (1.0 - a) * (sem or 0.0)
                age = max(0.0, at - it.last_used)
                rec = cfg.recency_floor + (1.0 - cfg.recency_floor) * kernel.value(age)
                u, m = self._usefulness(self._retrieval_features(query, it, cfg), ns, d, cache)
                score = (_pow(rel, cfg.w_relevance) * _pow(rec, cfg.w_recency)
                         * _pow(it.importance, cfg.w_importance) * _pow(u, cfg.w_usefulness) * m)
                if score < min_score or (score <= 0.0 and not fallback):
                    continue
                rows.append((score, d, rel, lex, sem, age, rec, u, m))
            rows.sort(key=lambda r: (-r[0], -self._items[r[1]].last_used, r[1]))
            out = []
            for rank, (score, d, rel, lex, sem, age, rec, u, m) in enumerate(rows[:k], 1):
                it = self._items[d]
                links: tuple[tuple[str, float], ...] = ()
                if explain:
                    _, ex = self.db.feature_risk(EFFECT, self._retrieval_features(query, it, cfg),
                                                 1.0, entity=ns, explain=True)
                    links = tuple((c.feature, c.strength) for c in ex.features)
                out.append(Recalled(
                    rank=rank, memory=it, score=score, relevance=rel, lexical=lex, semantic=sem,
                    bm25=bm.get(d, 0.0), cosine=cos.get(d) if use_emb else None,
                    age_days=age / DAY, recency=rec, importance=it.importance, usefulness=u,
                    rule_multiplier=m, rules=("superseded",) if m < 1.0 else (),
                    feature_links=links))
            res = RecallResult(out, query, ns, at, rid, len(cand))
            self.counters["recalls"] += 1
            if track:
                self._last_recall[ns] = res
                self._recalls[rid] = res
                if len(self._recalls) > 256:
                    self._recalls.pop(next(iter(self._recalls)))
            return res

    # -- feedback ----------------------------------------------------------------

    def _query_for(self, it: MemoryItem) -> str | None:
        """The query of the latest tracked recall that showed this memory."""
        for res in reversed(list(self._recalls.values())):
            if res.namespace == it.namespace and it.id in res.ids:
                return res.query
        return None

    def _retrieval_features(self, query: str | None, it: MemoryItem,
                            cfg: ScoringConfig | None = None) -> frozenset[str]:
        """Features of "memory ``it`` retrieved for ``query``": the memory's
        shape features, crossed with the query type, plus ``mem:<id>``."""
        f = pair_features(query, it.features)
        if (cfg or self.config).memory_links:
            f = f | {f"mem:{it.id}"}
        return f

    def _retrieval_event(self, it: MemoryItem, at: float, query: str | None) -> int:
        """Insert the HeapDB event "memory ``it`` was retrieved for a query like
        ``query``" and return its id. Its features are the memory's shape
        features crossed with the query type (:func:`pair_features`); no text."""
        q = query if query is not None else self._query_for(it)
        ev = self.db.insert(it.namespace, at, RETRIEVAL_TYPE, self._retrieval_features(q, it),
                            weight=1.0, attrs={"mid": it.id})
        return ev.id

    def _resolve(self, memory_ids: str | Iterable[str]) -> list[MemoryItem]:
        if isinstance(memory_ids, str):
            memory_ids = [memory_ids]
        return [self._get(m) for m in dict.fromkeys(memory_ids)]

    def mark_used(self, memory_ids: str | Iterable[str], now: float | datetime | None = None,
                  query: str | None = None) -> int:
        """The agent actually used (cited, acted on) these memories.

        For each memory a HeapDB ``retrieval`` event is inserted (the memory's
        features x the query type) and *confirmed* for effect ``useful``
        (proportional credit to its features, namespace + global layer), and
        the memory's ``last_used`` is refreshed, which resets its recency
        decay. ``query`` defaults to the query of the latest recall that showed
        the memory. Returns the number of memories marked.
        """
        at = _to_ts(now, self.clock)
        with self._lock:
            items = self._resolve(memory_ids)
            for it in items:
                t = max(at, it.created_at)
                eid = self._retrieval_event(it, t, query)
                self.db.feedback(it.namespace, EFFECT, t, confirmed_event_id=eid)
                it.last_used = max(it.last_used, t)
                it.use_count += 1
            self.counters["used"] += len(items)
            self._maybe_save()
        return len(items)

    def mark_ignored(self, memory_ids: str | Iterable[str], now: float | datetime | None = None,
                     query: str | None = None) -> int:
        """These memories were retrieved but not used: their ``retrieval``
        events are *rejected* (``beta += 1`` per feature). ``last_used`` is
        unchanged."""
        at = _to_ts(now, self.clock)
        with self._lock:
            items = self._resolve(memory_ids)
            by_ns: dict[str, list[tuple[int, float]]] = {}
            for it in items:
                t = max(at, it.created_at)
                by_ns.setdefault(it.namespace, []).append((self._retrieval_event(it, t, query), t))
                it.ignore_count += 1
            for ns, evs in by_ns.items():
                self.db.feedback(ns, EFFECT, max(t for _, t in evs),
                                 rejected_ids=[e for e, _ in evs])
            self.counters["ignored"] += len(items)
            self._maybe_save()
        return len(items)

    def feedback(self, used: Iterable[str] = (), recall_id: str | None = None,
                 namespace: str | None = None, now: float | datetime | None = None,
                 reject_unused: bool = True) -> dict[str, list[str]]:
        """Implicit feedback for a recall: ``used`` are confirmed, and (with
        ``reject_unused``) every other memory shown by that recall is rejected.

        ``recall_id`` picks the recall (default: the namespace's last one).
        Returns ``{"used": [...], "ignored": [...]}``.
        """
        used = [u for u in dict.fromkeys(used)]
        with self._lock:
            if recall_id is not None:
                res = self._recalls.get(recall_id)
                if res is None:
                    raise KeyError(f"unknown or expired recall_id {recall_id!r}")
            else:
                res = self._last_recall.get(namespace or self.namespace)
            shown = [m for m in (res.ids if res is not None else []) if m in self._items]
            ignored = [m for m in shown if m not in set(used)] if reject_unused else []
            q = res.query if res is not None else None
            if used:
                self.mark_used(used, now=now, query=q)
            if ignored:
                self.mark_ignored(ignored, now=now, query=q)
            return {"used": used, "ignored": ignored}

    # -- introspection -------------------------------------------------------------

    def explain_usefulness(self, namespace: str | None = None, min_support: float = 1.0):
        """HeapDB ``link_report`` for effect ``useful``: which kinds of memory
        turned out useful in this namespace (``None`` = all namespaces pooled
        when ``namespace="*"``)."""
        ns = None if namespace == "*" else (namespace or self.namespace)
        return self.db.link_report(EFFECT, entity=ns, min_support=min_support)

    def stats(self, namespace: str | None = None) -> dict[str, Any]:
        """Counts, kinds, feedback totals and the most / least useful features."""
        ns = namespace or self.namespace
        items = [it for it in self._items.values() if it.namespace == ns]
        kinds: dict[str, int] = {}
        for it in items:
            kinds[it.kind] = kinds.get(it.kind, 0) + 1
        rep = self.db.link_report(EFFECT, entity=ns, min_support=1.0)
        rows = [(r.feature, round(r.strength, 4), round(r.n, 2)) for r in rep]
        return {
            "namespace": ns,
            "memories": len(items),
            "active": sum(it.active for it in items),
            "superseded": sum(not it.active for it in items),
            "kinds": kinds,
            "uses": sum(it.use_count for it in items),
            "ignores": sum(it.ignore_count for it in items),
            "namespaces": len(self.namespaces),
            "total_memories": len(self._items),
            "embedder": getattr(self.embedder, "name", None),
            "most_useful_features": rows[:5],
            "least_useful_features": rows[-5:][::-1] if len(rows) > 5 else [],
            "session_counters": dict(self.counters),
        }

    # -- persistence -------------------------------------------------------------

    def _maybe_save(self) -> None:
        if self.autosave:
            self.save()

    def save(self, path: str | os.PathLike[str] | None = None) -> Path:
        """Write the store to a directory (atomically per file).

        Files: ``heapdb.jsonl`` (HeapDB snapshot: events, learned links),
        ``memories.jsonl`` (text and metadata), ``vectors.jsonl`` (embedding
        cache, if any) and ``memory.json`` (config, version).
        """
        target = Path(path) if path is not None else self.path
        if target is None:
            raise ValueError("no path: pass one or create the Memory with path=")
        target.mkdir(parents=True, exist_ok=True)
        with self._lock:
            _atomic(target / SNAPSHOT_FILE, lambda p: self.db.save(p))

            def write_items(p: Path) -> None:
                with open(p, "w", encoding="utf-8") as fh:
                    for it in sorted(self._items.values(), key=lambda i: i.event_id):
                        fh.write(json.dumps(it.to_dict(), ensure_ascii=False) + "\n")
            _atomic(target / MEMORIES_FILE, write_items)
            if self.embedder is not None:
                def write_vecs(p: Path) -> None:
                    with open(p, "w", encoding="utf-8") as fh:
                        for it in self._items.values():
                            v = self._space(it.namespace).vectors.get(it.id)
                            if v is not None:
                                fh.write(json.dumps({"id": it.id, "model": self.embedder.name,
                                                     "v": [round(x, 6) for x in v]}) + "\n")
                _atomic(target / VECTORS_FILE, write_vecs)
            meta = {"version": STATE_VERSION, "namespace": self.namespace,
                    "config": self.config.to_dict(), "counters": self.counters,
                    "keys": {ns: sp.keys for ns, sp in self._ns.items() if sp.keys}}
            _atomic(target / META_FILE, lambda p: p.write_text(json.dumps(meta, indent=1)))
        return target

    @classmethod
    def load(cls, path: str | os.PathLike[str], **kwargs: Any) -> "Memory":
        """Open a saved store (same as ``Memory(path, ...)`` on an existing directory)."""
        if not (Path(path) / META_FILE).exists():
            raise FileNotFoundError(f"no saved memory store in {path}")
        return cls(path, **kwargs)

    def _load(self, config: ScoringConfig | None) -> None:
        assert self.path is not None
        meta = json.loads((self.path / META_FILE).read_text())
        if int(meta.get("version", 0)) > STATE_VERSION:
            raise ValueError(f"store version {meta.get('version')} is newer than this library")
        self.config = config or ScoringConfig.from_dict(meta.get("config", {}))
        if self._namespace_arg is None:
            self.namespace = meta.get("namespace", self.namespace)
        self.counters.update(meta.get("counters", {}))
        self.db = HeapDB.load(self.path / SNAPSHOT_FILE)
        self.db.register_event_type(EVENT_TYPE)
        self.db.register_event_type(RETRIEVAL_TYPE)
        spec = self.db.effect(EFFECT)
        if config is not None:
            # Keep learned links, adopt the new kernel / rule settings.
            spec.kernel = self._kernel()
        with open(self.path / MEMORIES_FILE, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    self._index(MemoryItem.from_dict(json.loads(line)))
        for ns, keys in meta.get("keys", {}).items():
            self._space(ns).keys.update({k: v for k, v in keys.items() if v in self._items})
        # Tombstones: HeapDB events whose memory was forgotten stay blocked.
        known = set(self._items)
        for ev in self.db.events():
            mid = ev.attrs.get("mid")
            if ev.type == EVENT_TYPE and mid is not None and mid not in known:
                self._superseded.add(mid)
        if self.embedder is not None:
            cached: dict[str, list[float]] = {}
            vp = self.path / VECTORS_FILE
            if vp.exists():
                with open(vp, encoding="utf-8") as fh:
                    for line in fh:
                        if line.strip():
                            r = json.loads(line)
                            if r.get("model") == self.embedder.name:
                                cached[r["id"]] = r["v"]
            missing = [it for it in self._items.values() if it.id not in cached]
            if missing:
                for it, v in zip(missing, self.embedder.embed([m.text for m in missing])):
                    cached[it.id] = v
            for it in self._items.values():
                self._space(it.namespace).vectors.add(it.id, cached[it.id])


def _pow(x: float, w: float) -> float:
    if w == 0.0:
        return 1.0
    if w == 1.0:
        return x
    return x ** w if x > 0 else 0.0


def _atomic(path: Path, write: Callable[[Path], Any]) -> None:
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        write(Path(tmp))
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

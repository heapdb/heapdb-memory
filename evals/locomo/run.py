"""LoCoMo retrieval evaluation for heapdb-memory.

Task: every dialog turn of a LoCoMo conversation is a memory (namespace =
conversation, created at its session's timestamp). For every QA item with
evidence, query = question at ``now`` = one day after the conversation's last
session; a method returns a ranked list of turns (top 50); we score whether
the evidence turns are in it.

Metrics per question, averaged:
  R@k   evidence recall at k = |evidence in top k| / |evidence|  (k = 1, 5, 10)
  MRR   1 / rank of the first evidence turn in the top 50 (0 if none)

Learned usefulness uses leave-one-conversation-out cross-validation: for
each held-out conversation, feedback is given only on the other 9
conversations' QA items (recall top 10 with the not-yet-trained model,
``mark_used`` for shown evidence turns, ``mark_ignored`` for the other shown
turns) in their own namespaces. The held-out namespace receives no
feedback; it ranks with the global (pooled) links HeapDB lends it. Each fold
builds a fresh store, so nothing learned on a test conversation is ever used.

The default training feedback is *oracle* feedback (above). ``--feedback``
replaces it with a simulated, noisier agent (see :class:`Feedback`): missed
citations, false citations, sparse feedback, or citing only one source. The
noise is seeded and deterministic. ``--sweep`` runs the robustness sweep
behind ``NOISY_FEEDBACK.md``.

Usage::

    python -m evals.locomo.run [--onnx-model DIR] [--out evals/locomo/results.json]
    python -m evals.locomo.run --onnx-model DIR --feedback miss=0.5,false_pos=0.1 --seed 1 --out x.json
    python -m evals.locomo.run --onnx-model DIR --sweep [--shard 0/2]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Sequence

from heapdb_memory import Memory, ScoringConfig
from heapdb_memory.embeddings import Embedder

from .data import CATEGORIES, SHA256, Conversation, download, load, sha256

KS = (1, 5, 10)
DEPTH = 50
TRAIN_SHOWN = 10


# -- embedding cache ------------------------------------------------------------

class CachedEmbedder:
    """Wraps an embedder with an in-memory + on-disk cache keyed by text."""

    def __init__(self, inner: Embedder, cache_file: Path | None = None) -> None:
        self.inner = inner
        self.name = inner.name
        self.dim = inner.dim
        self.cache: dict[str, list[float]] = {}
        self.file = cache_file
        if cache_file is not None and cache_file.exists():
            for line in cache_file.read_text().splitlines():
                r = json.loads(line)
                if r["m"] == self.name:
                    self.cache[r["h"]] = r["v"]

    @staticmethod
    def _h(t: str) -> str:
        return hashlib.sha1(t.encode("utf-8")).hexdigest()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        missing = list(dict.fromkeys(t for t in texts if self._h(t) not in self.cache))
        if missing:
            new = self.inner.embed(missing)
            with (open(self.file, "a") if self.file else _Null()) as fh:
                for t, v in zip(missing, new):
                    self.cache[self._h(t)] = v
                    fh.write(json.dumps({"m": self.name, "h": self._h(t),
                                         "v": [round(x, 6) for x in v]}) + "\n")
        return [self.cache[self._h(t)] for t in texts]


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def write(self, _s: str) -> None:
        pass


# -- methods ---------------------------------------------------------------------

@dataclass
class Method:
    name: str
    description: str
    config: ScoringConfig | None = None   # None: special ranking (recency-only)
    embed: bool = False
    learned: bool = False
    group: str = "main"
    tuned: bool = False      # weights picked per fold on the training conversations
    grid: dict | None = None  # tuning grid (default: TUNE_GRID [+ TUNE_LEXICAL])


def _cfg(**kw) -> ScoringConfig:
    return ScoringConfig(**kw)


OFF = dict(w_recency=0.0, w_importance=0.0, w_usefulness=0.0)


def methods(have_embedder: bool) -> list[Method]:
    full = ScoringConfig()
    ms = [
        Method("bm25", "BM25 only", _cfg(**OFF)),
        Method("recency", "most recent first (no query)", None),
        Method("bm25+recency", "BM25 x fixed recency blend (half-life 30d, floor 0.5)",
               _cfg(w_importance=0.0, w_usefulness=0.0)),
        Method("heapdb", "HeapDB memory, no learning: relevance x recency x importance x prior usefulness",
               full),
        Method("heapdb+learned", "HeapDB memory with usefulness learned on the other 9 conversations",
               full, learned=True),
        Method("heapdb+learned+tuned", "as heapdb+learned, with recency/importance weights and "
               "half-life chosen per fold on the 9 training conversations (nested CV)",
               full, learned=True, tuned=True),
    ]
    if have_embedder:
        ms[3:3] = [
            Method("embedding", "MiniLM cosine only", _cfg(lexical_weight=0.0, **OFF), embed=True),
            Method("bm25+embedding", "hybrid relevance (0.5 BM25 + 0.5 cosine), nothing else",
                   _cfg(**OFF), embed=True),
        ]
        ms += [
            Method("bm25+embedding+tuned", "hybrid relevance only, BM25 share chosen per fold on "
                   "the training conversations (nested CV); fair baseline for the tuned HeapDB rows",
                   _cfg(**OFF), embed=True, learned=True, tuned=True,
                   grid={"lexical_weight": TUNE_LEXICAL}),
            Method("heapdb+emb", "HeapDB memory, hybrid relevance, no learning", full, embed=True),
            Method("heapdb+emb+learned", "HeapDB memory, hybrid relevance, learned usefulness",
                   full, embed=True, learned=True),
            Method("heapdb+emb+learned+tuned", "as heapdb+emb+learned, weights and BM25 share "
                   "tuned per fold on the training conversations", full, embed=True, learned=True,
                   tuned=True),
        ]
    # Ablations of the full lexical system (with learning).
    ab = [
        ("abl:-recency", replace(full, w_recency=0.0), True),
        ("abl:-importance", replace(full, w_importance=0.0), True),
        ("abl:-usefulness", replace(full, w_usefulness=0.0), False),
        ("abl:bm25+importance", _cfg(w_recency=0.0, w_usefulness=0.0), False),
        ("abl:bm25+learned", _cfg(w_recency=0.0, w_importance=0.0), True),
    ]
    for name, cfg, learned in ab:
        ms.append(Method(name, name.split(":", 1)[1], cfg, learned=learned, group="ablation"))
    return ms


# -- evaluation --------------------------------------------------------------------

@dataclass
class QResult:
    conv: str
    category: int
    ranks: list[int]            # 1-based ranks of evidence turns within DEPTH (missing: absent)
    n_evidence: int
    latency_ms: float

    def recall(self, k: int) -> float:
        return sum(1 for r in self.ranks if r <= k) / self.n_evidence

    @property
    def rr(self) -> float:
        return 1.0 / min(self.ranks) if self.ranks else 0.0


def build_store(convs: Sequence[Conversation], embedder: Embedder | None, now: float) -> tuple[Memory, dict]:
    mem = Memory(namespace="_", embedder=embedder, clock=lambda: now)
    id_map: dict[str, dict[str, str]] = {}
    rows = []
    for c in convs:
        for t in c.turns:
            rows.append(dict(text=t.text, kind="episode", source=t.speaker, namespace=c.sample_id,
                             created_at=t.time, id=f"{c.sample_id}|{t.dia_id}"))
    mem.add_many(rows)
    for c in convs:
        id_map[c.sample_id] = {t.dia_id: f"{c.sample_id}|{t.dia_id}" for t in c.turns}
    return mem, id_map


# -- training feedback models ------------------------------------------------------

@dataclass(frozen=True)
class Feedback:
    """How the simulated agent reports which of the top ``TRAIN_SHOWN``
    recalled turns it used, for one training question.

    Starting from the oracle (evidence turns used, other shown turns ignored):

    - ``sparse=r``: only a share ``r`` of training questions give any
      feedback at all; the rest are recalled and left unmarked.
    - ``miss=p``: each shown evidence turn is independently *not* cited with
      probability ``p``. ``missed_as`` decides what a missed turn becomes:
      ``"ignored"`` (default: an agent that does not cite something
      effectively signals ignore) or ``"unmarked"`` (no feedback for it).
    - ``cite="top1"``: of the (remaining) cited evidence turns only the
      highest-ranked one is marked used; the other evidence turns are left
      unmarked. Non-evidence turns are still ignored.
    - ``false_pos=q``: each shown non-evidence turn is independently marked
      used instead of ignored with probability ``q``.

    Randomness is a hash of ``(seed, conversation, question index, memory id)``,
    not a stream: the draw for a given question and turn is the same in every
    fold, method and run order, and the noise levels are nested (with one
    seed, the turns missed at ``miss=0.3`` are also missed at ``miss=0.5``).
    With every rate at its default this is exactly the oracle.
    """

    miss: float = 0.0
    false_pos: float = 0.0
    sparse: float = 1.0
    missed_as: str = "ignored"
    cite: str = "all"
    seed: int = 0

    def __post_init__(self) -> None:
        for k in ("miss", "false_pos", "sparse"):
            v = getattr(self, k)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{k}={v} must be in [0, 1]")
        if self.missed_as not in ("ignored", "unmarked"):
            raise ValueError(f"missed_as must be 'ignored' or 'unmarked', not {self.missed_as!r}")
        if self.cite not in ("all", "top1"):
            raise ValueError(f"cite must be 'all' or 'top1', not {self.cite!r}")

    @property
    def is_oracle(self) -> bool:
        return self.miss == 0 and self.false_pos == 0 and self.sparse == 1 and self.cite == "all"

    @property
    def stochastic(self) -> bool:
        return self.miss > 0 or self.false_pos > 0 or self.sparse < 1

    @property
    def label(self) -> str:
        """Setting name without the seed, e.g. ``miss=0.5,false_pos=0.1``."""
        if self.is_oracle:
            return "oracle"
        parts = []
        if self.cite == "top1":
            parts.append("cite_top1")
        if self.miss:
            parts.append(f"miss={self.miss:g}" + ("(unmarked)" if self.missed_as == "unmarked" else ""))
        if self.false_pos:
            parts.append(f"false_pos={self.false_pos:g}")
        if self.sparse < 1:
            parts.append(f"sparse={self.sparse:g}")
        return ",".join(parts)

    def to_dict(self) -> dict:
        return {"miss": self.miss, "false_pos": self.false_pos, "sparse": self.sparse,
                "missed_as": self.missed_as, "cite": self.cite, "seed": self.seed}

    @classmethod
    def parse(cls, spec: str, missed_as: str = "ignored", seed: int = 0) -> Feedback:
        """``"oracle"``, ``"cite_top1"`` or comma-separated ``key=value`` pairs
        (``miss``, ``false_pos``, ``sparse``), e.g. ``"miss=0.5,sparse=0.5"``."""
        kw: dict = {"missed_as": missed_as, "seed": seed}
        for part in filter(None, (p.strip() for p in spec.split(","))):
            if part == "oracle":
                continue
            if part == "cite_top1":
                kw["cite"] = "top1"
                continue
            k, sep, v = part.partition("=")
            if not sep or k not in ("miss", "false_pos", "sparse"):
                raise ValueError(f"bad feedback spec part {part!r}")
            kw[k] = float(v)
        return cls(**kw)

    def _u(self, tag: str, conv: str, qi: int, mid: str = "") -> float:
        h = hashlib.blake2b(f"{self.seed}|{tag}|{conv}|{qi}|{mid}".encode(), digest_size=8).digest()
        return int.from_bytes(h, "big") / 2.0 ** 64

    def judge(self, conv: str, qi: int, shown: Sequence[str], evidence: set[str]
              ) -> tuple[list[str], list[str]]:
        """(used, ignored) ids for one training question, both in rank order."""
        if self.sparse < 1.0 and self._u("sparse", conv, qi) >= self.sparse:
            return [], []
        used, ign = [], []
        top1_taken = False
        for mid in shown:
            if mid in evidence:
                if self.miss > 0.0 and self._u("miss", conv, qi, mid) < self.miss:
                    if self.missed_as == "ignored":
                        ign.append(mid)
                elif self.cite == "top1" and top1_taken:
                    pass  # cited only one source: other evidence stays unmarked
                else:
                    used.append(mid)
                    top1_taken = True
            elif self.false_pos > 0.0 and self._u("fp", conv, qi, mid) < self.false_pos:
                used.append(mid)
            else:
                ign.append(mid)
        return used, ign


ORACLE = Feedback()


def train(mem: Memory, convs: Sequence[Conversation], cfg: ScoringConfig,
          feedback: Feedback = ORACLE, stats: dict | None = None) -> int:
    """Feedback on training conversations only. Returns #feedback events.
    ``stats`` (optional) accumulates ``used`` / ``used_evidence`` (used and
    really evidence) / ``ignored`` / ``questions`` (questions that gave any
    feedback) / ``evidence_shown`` counts."""
    n = 0
    st = stats if stats is not None else {}
    for c in convs:
        for qi, q in enumerate(c.qa):
            res = mem.recall(q.question, k=TRAIN_SHOWN, namespace=c.sample_id, now=c.now,
                             config=cfg, explain=False, track=False)
            ev = {f"{c.sample_id}|{d}" for d in q.evidence}
            used, ign = feedback.judge(c.sample_id, qi, [r.id for r in res], ev)
            if used:
                mem.mark_used(used, now=c.now, query=q.question)
            if ign:
                mem.mark_ignored(ign, now=c.now, query=q.question)
            n += len(used) + len(ign)
            for k, v in (("used", len(used)), ("used_evidence", sum(1 for u in used if u in ev)),
                         ("ignored", len(ign)), ("questions", int(bool(used or ign))),
                         ("evidence_shown", sum(1 for r in res if r.id in ev))):
                st[k] = st.get(k, 0) + v
    return n


def evaluate(mem: Memory, conv: Conversation, method: Method) -> list[QResult]:
    out = []
    turns_by_recency = None
    if method.config is None:
        turns_by_recency = [f"{conv.sample_id}|{t.dia_id}"
                            for t in sorted(conv.turns, key=lambda t: (-t.time, t.dia_id))][:DEPTH]
    for q in conv.qa:
        ev = {f"{conv.sample_id}|{d}" for d in q.evidence}
        t0 = time.perf_counter()
        if turns_by_recency is not None:
            ranked = turns_by_recency
        else:
            res = mem.recall(q.question, k=DEPTH, namespace=conv.sample_id, now=conv.now,
                             config=method.config, explain=False, track=False)
            ranked = res.ids
        dt = (time.perf_counter() - t0) * 1000.0
        ranks = [i + 1 for i, m in enumerate(ranked) if m in ev]
        out.append(QResult(conv.sample_id, q.category, ranks, len(ev), dt))
    return out


#: Grid for nested-CV tuning (scored by MRR on the training conversations,
#: without learning, so tuning never sees the held-out conversation).
TUNE_GRID = {
    "w_recency": (0.0, 0.25, 0.5, 1.0),
    "half_life_days": (30.0, 180.0),
    "w_importance": (0.0, 1.0),
}
TUNE_LEXICAL = (0.3, 0.5, 0.7)


def tune(mem: Memory, train_convs: Sequence[Conversation], cfg: ScoringConfig,
         embed: bool, grid: dict | None = None, memo: dict | None = None) -> tuple[ScoringConfig, dict]:
    """Pick the grid point with the best MRR on ``train_convs``.

    ``mem`` must be a store that never received feedback, so a conversation's
    reciprocal ranks under a config do not depend on the fold. ``memo``
    (shared across folds, and across feedback models) caches them per
    ``(embed, config, conversation)``. The concatenation order is the same as
    without it, so the chosen config is identical, at a tenth of the cost.
    """
    import itertools
    if grid is None:
        grid = dict(TUNE_GRID)
        if embed:
            grid["lexical_weight"] = TUNE_LEXICAL
    best = None
    for vals in itertools.product(*grid.values()):
        kw = dict(zip(grid, vals))
        if kw.get("w_recency") == 0.0 and kw.get("half_life_days", 0) != grid["half_life_days"][0]:
            continue  # half-life is irrelevant without recency
        c = replace(cfg, **kw)
        probe = Method("tune", "", c)
        rrs: list[float] = []
        for tc in train_convs:
            key = (embed, _cfg_key(c), tc.sample_id)
            if memo is None or key not in memo:
                v = [r.rr for r in evaluate(mem, tc, probe)]
                if memo is None:
                    rrs += v
                    continue
                memo[key] = v
            rrs += memo[key]
        mrr = statistics.fmean(rrs)
        if best is None or mrr > best[0] + 1e-12:
            best = (mrr, c, kw)
    return best[1], {"params": best[2], "train_mrr": best[0]}


def summarize(rs: Sequence[QResult]) -> dict:
    if not rs:
        return {"n": 0}
    d = {"n": len(rs)}
    for k in KS:
        d[f"R@{k}"] = statistics.fmean(r.recall(k) for r in rs)
    d["MRR"] = statistics.fmean(r.rr for r in rs)
    return d


def run(convs: list[Conversation], embedder: Embedder | None, only: set[str] | None = None,
        log: Callable[[str], None] = print, feedback: Feedback = ORACLE,
        tune_memo: dict | None = None) -> dict:
    """Evaluate every method. ``feedback`` is the training feedback model of
    the ``+learned`` methods; ``tune_memo`` (see :func:`tune`) may be shared
    between calls on the same ``convs`` and embedder."""
    if tune_memo is None:
        tune_memo = {}
    ms = [m for m in methods(embedder is not None) if only is None or m.name in only]
    results: dict[str, list[QResult]] = {m.name: [] for m in ms}
    fold_stats: dict[str, list[dict]] = {m.name: [] for m in ms}
    train_events: list[int] = []
    train_stats: dict[str, int] = {}
    now = max(c.now for c in convs)

    # Methods without learning: one store, no feedback at all.
    static = [m for m in ms if not m.learned]
    if static:
        stores = {}
        for m in static:
            key = m.embed
            if key not in stores:
                stores[key] = build_store(convs, embedder if m.embed else None, now)[0]
            for c in convs:
                rs = evaluate(stores[key], c, m)
                results[m.name] += rs
                fold_stats[m.name].append(summarize(rs))
            log(f"  {m.name:<22} {_fmt(summarize(results[m.name]))}")

    # Learned methods: leave-one-conversation-out.
    learned = [m for m in ms if m.learned]
    tuned_choice: dict[str, list[dict]] = {m.name: [] for m in learned if m.tuned}
    base: dict[bool, Memory] = {}
    for i, test in enumerate(convs):
        tr = [c for c in convs if c.sample_id != test.sample_id]
        cache: dict[tuple, Memory] = {}
        for m in learned:
            cfg = m.config
            if m.tuned:
                if m.embed not in base:
                    base[m.embed] = build_store(convs, embedder if m.embed else None, now)[0]
                cfg, info = tune(base[m.embed], tr, m.config, m.embed, m.grid, memo=tune_memo)
                tuned_choice[m.name].append({"test": test.sample_id, **info})
            key = (m.embed, _cfg_key(cfg))
            if key not in cache:
                mem = build_store(convs, embedder if m.embed else None, now)[0]
                if cfg.w_usefulness > 0:   # no point training links nobody reads
                    train_events.append(train(mem, tr, cfg, feedback, stats=train_stats))
                cache[key] = mem
            rs = evaluate(cache[key], test, replace(m, config=cfg))
            results[m.name] += rs
            fold_stats[m.name].append(summarize(rs))
        log(f"  fold {i + 1}/{len(convs)} ({test.sample_id}) done")
    for m in learned:
        log(f"  {m.name:<22} {_fmt(summarize(results[m.name]))}")

    out = {"methods": {}}
    for m in ms:
        rs = results[m.name]
        per_cat = {CATEGORIES[c]: summarize([r for r in rs if r.category == c]) for c in sorted(CATEGORIES)}
        folds = fold_stats[m.name]
        lat = sorted(r.latency_ms for r in rs)
        out["methods"][m.name] = {
            "description": m.description, "group": m.group, "embed": m.embed,
            "learned": m.learned,
            "config": m.config.to_dict() if m.config else None,
            "overall": summarize(rs),
            "per_category": per_cat,
            "per_conversation": {c.sample_id: f for c, f in zip(convs, folds)},
            "across_conversations": {
                k: {"mean": statistics.fmean(f[k] for f in folds),
                    "std": statistics.stdev(f[k] for f in folds) if len(folds) > 1 else 0.0}
                for k in [f"R@{k}" for k in KS] + ["MRR"]},
            "latency_ms": {"median": lat[len(lat) // 2], "p95": lat[int(0.95 * (len(lat) - 1))],
                           "mean": statistics.fmean(lat)},
            "per_question": [[r.conv, r.category, r.ranks, r.n_evidence] for r in rs],
        }
    out["train_feedback_events_per_fold"] = train_events
    out["tuned_choices"] = tuned_choice
    out["feedback"] = {"label": feedback.label, **feedback.to_dict()}
    out["train_feedback_stats"] = train_stats
    return out


# -- noisy-feedback sweep ------------------------------------------------------------

#: Methods reported under each feedback model (``NOISY_FEEDBACK.md``).
#: ``bm25`` and ``bm25+embedding+tuned`` do not read usefulness, so feedback
#: cannot change them; the sweep asserts that.
SWEEP_METHODS = ("bm25", "bm25+embedding+tuned", "abl:bm25+learned", "heapdb+learned+tuned",
                 "heapdb+emb+learned+tuned")
FEEDBACK_FREE = ("bm25", "bm25+embedding+tuned")
SWEEP_SEEDS = (0, 1, 2)
REALISTIC = "miss=0.5,false_pos=0.1,sparse=0.5"
HARSH = "miss=0.7,false_pos=0.2,sparse=0.25"
#: (group, spec, missed_as). The oracle is miss=0 / false_pos=0 / sparse=1.
SWEEP = (
    [("oracle", "oracle", "ignored")]
    + [("miss", f"miss={p}", "ignored") for p in (0.3, 0.5, 0.7)]
    + [("miss (unmarked)", f"miss={p}", "unmarked") for p in (0.3, 0.5, 0.7)]
    + [("false_pos", f"false_pos={q}", "ignored") for q in (0.05, 0.1, 0.2)]
    + [("sparse", f"sparse={r}", "ignored") for r in (0.5, 0.25, 0.1)]
    + [("no feedback", "sparse=0", "ignored")]
    + [("cite_top1", "cite_top1", "ignored")]
    + [("realistic", REALISTIC, "ignored"), ("realistic", REALISTIC, "unmarked")]
    # Stress points beyond the planned grid, to find where the gain breaks.
    + [("miss", "miss=0.9", "ignored"), ("false_pos", "false_pos=0.4", "ignored"),
       ("sparse", "sparse=0.02", "ignored"), ("stress", HARSH, "ignored")]
)
STRESS = ("miss=0.9", "false_pos=0.4", "sparse=0.02", HARSH)


def sweep_runs(seeds: Sequence[int] = SWEEP_SEEDS) -> list[tuple[str, Feedback]]:
    """(group, feedback) for every run of the sweep; deterministic settings
    (oracle, cite_top1, sparse=0) run once."""
    out = []
    for group, spec, missed_as in SWEEP:
        fb = Feedback.parse(spec, missed_as=missed_as)
        for s in (seeds if fb.stochastic and fb.sparse > 0 else seeds[:1]):
            out.append((group, replace(fb, seed=s)))
    return out


def _run_file(d: Path, fb: Feedback) -> Path:
    slug = fb.label.replace(",", "__").replace("=", "-").replace("(", "_").replace(")", "")
    return d / f"{slug}__seed{fb.seed}.json"


def sweep(convs: list[Conversation], embedder: Embedder | None, run_dir: Path,
          shard: tuple[int, int] = (0, 1), seeds: Sequence[int] = SWEEP_SEEDS,
          only: Sequence[str] = SWEEP_METHODS, log: Callable[[str], None] = print) -> None:
    """Run the feedback sweep, one JSON per (setting, seed) in ``run_dir``.
    Resumable (finished runs are skipped) and shardable (``shard=(i, n)`` runs
    every n-th run). Folds, stores and tuning are those of :func:`run`; the
    tuning memo is shared across runs (tuning never sees feedback)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    memo: dict = {}
    names = [m.name for m in methods(embedder is not None) if m.name in only]
    for j, (group, fb) in enumerate(sweep_runs(seeds)):
        f = _run_file(run_dir, fb)
        if j % shard[1] != shard[0] or f.exists():
            continue
        t0 = time.time()
        res = run(convs, embedder, set(names), log=lambda *_: None, feedback=fb, tune_memo=memo)
        rec = {"group": group, "feedback": res["feedback"], "runtime_s": round(time.time() - t0, 1),
               "train_feedback_events_per_fold": res["train_feedback_events_per_fold"],
               "train_feedback_stats": res["train_feedback_stats"],
               "tuned_choices": res["tuned_choices"],
               "ranks": {n: [q[2] for q in res["methods"][n]["per_question"]] for n in names},
               "questions": [q[:2] + q[3:] for q in res["methods"][names[0]]["per_question"]]}
        tmp = f.with_suffix(".part")
        tmp.write_text(json.dumps(rec, separators=(",", ":")))
        tmp.replace(f)
        log(f"  {fb.label} seed {fb.seed}: {rec['runtime_s']} s  " + "  ".join(
            f"{n} {res['methods'][n]['overall']['MRR']:.3f}" for n in names))


def collect(run_dir: Path, seeds: Sequence[int] = SWEEP_SEEDS, meta: dict | None = None) -> dict:
    """Assemble the per-run files of :func:`sweep` into one results dict and
    check its invariants: same questions in every run, feedback-free methods
    and nested-CV tuning choices identical under every feedback model."""
    runs = []
    for group, fb in sweep_runs(seeds):
        f = _run_file(run_dir, fb)
        if not f.exists():
            raise SystemExit(f"missing sweep run {f}")
        runs.append(json.loads(f.read_text()))
    first = runs[0]
    for r in runs[1:]:
        if r["questions"] != first["questions"]:
            raise SystemExit("sweep runs disagree on the question list")
        if r["tuned_choices"] != first["tuned_choices"]:
            raise SystemExit(f"tuned choices differ under {r['feedback']['label']}")
        for n in FEEDBACK_FREE:
            if n in first["ranks"] and r["ranks"][n] != first["ranks"][n]:
                raise SystemExit(f"{n} changed under feedback {r['feedback']['label']}")
    settings: dict[str, dict] = {}
    for r in runs:
        fb = r["feedback"]
        key = fb["label"]
        s = settings.setdefault(key, {"group": r["group"], "label": key,
                                      "feedback": {k: v for k, v in fb.items() if k not in ("seed", "label")},
                                      "seeds": [], "ranks": {}, "train_feedback_events_per_fold": [],
                                      "train_feedback_stats": [], "runtime_s": 0.0})
        s["seeds"].append(fb["seed"])
        s["train_feedback_events_per_fold"].append(r["train_feedback_events_per_fold"])
        s["train_feedback_stats"].append(r["train_feedback_stats"])
        s["runtime_s"] += r["runtime_s"]
        for n, v in r["ranks"].items():
            if n in FEEDBACK_FREE:
                s["ranks"].setdefault(n, [v])
            else:
                s["ranks"].setdefault(n, []).append(v)
    return {"methods": list(first["ranks"]), "feedback_free": list(FEEDBACK_FREE),
            "stress": [Feedback.parse(s).label for s in STRESS],
            "questions": first["questions"], "tuned_choices": first["tuned_choices"],
            "settings": list(settings.values()), "meta": meta or {}}


def learned_links(convs: Sequence[Conversation], fold: int = 0, top: int = 12) -> dict:
    """Global `useful` links (shape features only) after training on every
    conversation except ``convs[fold]``, as in that fold of ``heapdb+learned``."""
    now = max(c.now for c in convs)
    mem = build_store(convs, None, now)[0]
    cfg = ScoringConfig()
    train(mem, [c for i, c in enumerate(convs) if i != fold], cfg)
    rows = [r for r in mem.db.link_report("useful", entity=None, min_support=30)
            if not r.feature.startswith("mem:")]
    fmt = lambda r: {"feature": r.feature, "strength": r.strength, "n": r.n}
    return {"held_out": convs[fold].sample_id, "prior_mean": cfg.prior[0] / sum(cfg.prior),
            "strongest": [fmt(r) for r in rows[:top]], "weakest": [fmt(r) for r in rows[-6:]]}


def _cfg_key(cfg: ScoringConfig | None) -> str:
    return json.dumps(cfg.to_dict() if cfg else None, sort_keys=True)


def _fmt(s: dict) -> str:
    return "  ".join(f"{k} {v:.3f}" for k, v in s.items() if k != "n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", type=Path, help="path to locomo10.json (default: download to cache)")
    p.add_argument("--onnx-model", type=Path, help="dir with all-MiniLM-L6-v2 model.onnx + tokenizer.json")
    p.add_argument("--out", type=Path,
                   help="output JSON (default: results.json; noisy_results.json with --sweep; "
                        "required with a non-oracle --feedback)")
    p.add_argument("--only", nargs="*", help="run only these methods")
    p.add_argument("--merge", action="store_true",
                   help="update the methods in an existing --out file instead of replacing it")
    p.add_argument("--feedback", default="oracle",
                   help="training feedback model: oracle (default), cite_top1, or key=value pairs "
                        "miss=P,false_pos=Q,sparse=R (see Feedback)")
    p.add_argument("--missed-as", choices=("ignored", "unmarked"), default="ignored",
                   help="what a missed evidence turn becomes under miss=P (default: ignored)")
    p.add_argument("--seed", type=int, default=0, help="seed of the feedback noise")
    p.add_argument("--sweep", action="store_true",
                   help="run the noisy-feedback sweep (SWEEP x --seeds) for SWEEP_METHODS")
    p.add_argument("--seeds", type=int, nargs="*", default=list(SWEEP_SEEDS),
                   help="seeds of the stochastic sweep settings")
    p.add_argument("--shard", default="0/1", help="with --sweep: run only shard i of n (i/n)")
    p.add_argument("--run-dir", type=Path,
                   help="with --sweep: per-run files (default: <cache>/locomo/noisy_runs)")
    p.add_argument("--collect", action="store_true",
                   help="with --sweep: only assemble finished runs into --out")
    a = p.parse_args(argv)
    fb = Feedback.parse(a.feedback, missed_as=a.missed_as, seed=a.seed)
    here = Path(__file__)
    if a.out is None:
        if a.sweep:
            a.out = here.with_name("noisy_results.json")
        elif fb.is_oracle:
            a.out = here.with_name("results.json")
        else:
            p.error("--out is required with a non-oracle --feedback (results.json holds the oracle run)")
    path = a.data or download()
    digest = sha256(path)
    if digest != SHA256:
        print(f"warning: dataset sha256 {digest} differs from the evaluated version {SHA256}",
              file=sys.stderr)
    convs = load(path)
    emb = None
    emb_info = None
    if a.onnx_model:
        from heapdb_memory.embeddings import OnnxEmbedder
        from .data import cache_dir
        inner = OnnxEmbedder(a.onnx_model)
        cf = cache_dir() / "locomo" / "vectors.jsonl"
        cf.parent.mkdir(parents=True, exist_ok=True)
        emb = CachedEmbedder(inner, cf)
        onnx = next(x for x in (a.onnx_model / "model.onnx", a.onnx_model / "onnx" / "model.onnx") if x.exists())
        emb_info = {"name": inner.name, "onnx_sha256": sha256(onnx)}
    t0 = time.time()
    if a.sweep:
        from .data import cache_dir
        run_dir = a.run_dir or cache_dir() / "locomo" / "noisy_runs"
        i, n = (int(x) for x in a.shard.split("/"))
        if not a.collect:
            sweep(convs, emb, run_dir, shard=(i, n), seeds=a.seeds,
                  only=a.only or SWEEP_METHODS)
            if n > 1:
                print(f"shard {i}/{n} done; run --sweep --collect once all shards finished")
                return 0
        meta = {"dataset_sha256": digest, "conversations": len(convs),
                "memories": sum(len(c.turns) for c in convs),
                "questions": sum(len(c.qa) for c in convs),
                "questions_per_category": {CATEGORIES[k]: sum(1 for c in convs for q in c.qa
                                                              if q.category == k)
                                           for k in sorted(CATEGORIES)},
                "embedder": emb_info, "depth": DEPTH, "train_shown": TRAIN_SHOWN,
                "seeds": list(a.seeds), "python": platform.python_version(),
                "machine": platform.machine()}
        res = collect(run_dir, a.seeds, meta)
        res["meta"]["runtime_s"] = round(sum(s["runtime_s"] for s in res["settings"]), 1)
        a.out.write_text(json.dumps(res, separators=(",", ":")))
        print(f"wrote {a.out}")
        return 0
    res = run(convs, emb, set(a.only) if a.only else None, feedback=fb)
    excluded: dict[str, int] = {}
    for c in convs:
        for k, v in c.excluded.items():
            excluded[k] = excluded.get(k, 0) + v
    res["meta"] = {
        "dataset_sha256": digest, "conversations": len(convs),
        "memories": sum(len(c.turns) for c in convs),
        "questions": sum(len(c.qa) for c in convs),
        "questions_per_category": {CATEGORIES[k]: sum(1 for c in convs for q in c.qa if q.category == k)
                                   for k in sorted(CATEGORIES)},
        "excluded": excluded, "embedder": emb_info, "depth": DEPTH, "train_shown": TRAIN_SHOWN,
        "python": platform.python_version(), "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "runtime_s": round(time.time() - t0, 1),
    }
    res["learned_links"] = learned_links(convs)
    if a.merge and a.out.exists():
        old = json.loads(a.out.read_text())
        if old["meta"]["dataset_sha256"] != digest:
            raise SystemExit("refusing to merge results from a different dataset version")
        order = [n for n in (m.name for m in methods(emb is not None)) if n in old["methods"] or n in res["methods"]]
        merged = {**old["methods"], **res["methods"]}
        old["methods"] = {n: merged[n] for n in order}
        old.setdefault("tuned_choices", {}).update(res.get("tuned_choices", {}))
        old["learned_links"] = res["learned_links"]
        old["meta"]["runtime_s"] = round(old["meta"]["runtime_s"] + res["meta"]["runtime_s"], 1)
        res = old
    a.out.write_text(json.dumps(res, indent=1))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

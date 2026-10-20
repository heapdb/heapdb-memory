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

Usage::

    python -m evals.locomo.run [--onnx-model DIR] [--out evals/locomo/results.json]
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


def train(mem: Memory, convs: Sequence[Conversation], cfg: ScoringConfig) -> int:
    """Feedback on training conversations only. Returns #feedback events."""
    n = 0
    for c in convs:
        for q in c.qa:
            res = mem.recall(q.question, k=TRAIN_SHOWN, namespace=c.sample_id, now=c.now,
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
         embed: bool, grid: dict | None = None) -> tuple[ScoringConfig, dict]:
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
        rs = [r for tc in train_convs for r in evaluate(mem, tc, probe)]
        mrr = statistics.fmean(r.rr for r in rs)
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
        log: Callable[[str], None] = print) -> dict:
    ms = [m for m in methods(embedder is not None) if only is None or m.name in only]
    results: dict[str, list[QResult]] = {m.name: [] for m in ms}
    fold_stats: dict[str, list[dict]] = {m.name: [] for m in ms}
    train_events: list[int] = []
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
                cfg, info = tune(base[m.embed], tr, m.config, m.embed, m.grid)
                tuned_choice[m.name].append({"test": test.sample_id, **info})
            key = (m.embed, _cfg_key(cfg))
            if key not in cache:
                mem = build_store(convs, embedder if m.embed else None, now)[0]
                if cfg.w_usefulness > 0:   # no point training links nobody reads
                    train_events.append(train(mem, tr, cfg))
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
    return out


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
    p.add_argument("--out", type=Path, default=Path(__file__).with_name("results.json"))
    p.add_argument("--only", nargs="*", help="run only these methods")
    p.add_argument("--merge", action="store_true",
                   help="update the methods in an existing --out file instead of replacing it")
    a = p.parse_args(argv)
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
    res = run(convs, emb, set(a.only) if a.only else None)
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

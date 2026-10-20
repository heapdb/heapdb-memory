# Agent memory on HeapDB: design

This document covers what `heapdb-memory` stores, how it scores a recall, how
it learns, and the decisions behind each choice. The README has the
quickstart.

## Why HeapDB for agent memory

Recalling a memory is the same question HeapDB answers: "of everything that
happened, which items matter now?" HeapDB already offers each piece that
question needs:

| HeapDB concept | agent memory |
|---|---|
| entity | namespace: a user, agent or project |
| event (type, features, time, weight) | a memory: shape features, creation time, importance as dose |
| lag kernel | recency decay on time since last use |
| effect + Beta links per (feature, effect) | `useful`: P(memory is used \| feature), learned |
| feedback confirm / reject (proportional credit) | `mark_used` / `mark_ignored` |
| hierarchical priors (global + entity layers) | a new user starts from what all users taught the store |
| rules | superseded memories x0.1, forgotten memories blocked |
| EXPLAIN, `link_report` | one-line explanation per result; which kinds of memory are useful |
| `forget_entity(scrub_global=True)` | erase a user, including their share of the global links |

The value is retrieval *quality* and *insight*, not speed: ranking signals
beyond similarity, an explanation for every result, and a store that learns
from what the agent actually uses.

## Data model

A memory (`MemoryItem`) has an id, a namespace, its text, a `kind`
(`fact`, `preference`, `instruction`, `episode`, `summary`, `tool_result`,
`note`, or any string), its importance (explicit, or the heuristic
`kind_importance[kind] x (0.6 + 0.4 x min(1, content_terms/8))`), tags, a
source, an optional supersede `key`, `created_at`, `last_used`, use and
ignore counts, and metadata.

In HeapDB it is one event of type `memory`, stored in its namespace at
`created_at` with weight = importance. Its features come from a **fixed set
of slots** (`heapdb_memory.text.memory_features`):

`kind:<kind>`, `source:<source>`, `len:short|medium|long`, `date:yes|no`,
`num:yes|no`, `q:yes|no` (the memory is itself a question), `ent:yes|no`
(names a proper noun), `topic:<one of 10>|none`, plus `tag:<t>` per user tag.

*Decision: words are not HeapDB features.* With thousands of word features
the noisy-OR saturates towards 1 for every memory, so usefulness would stop
telling memories apart, and the link table would hold a copy of the memory
content. Words go to the BM25 index; HeapDB learns over shape features
only. The memory text is never written into the HeapDB snapshot, and a test
checks this.

## Recall

1. **Candidate generation.** The top `candidates` (50) by BM25 (Okapi,
   k1=1.2, b=0.75, Lucene IDF, light stemmer, stop words) are unioned with
   the top 50 by cosine similarity, if an `Embedder` is configured. If
   nothing matches lexically and there is no embedder, every memory in the
   namespace is a candidate with relevance 1, so a recall never comes back
   empty while memories exist.
2. **Score**, a log-linear product of experts:

   ```
   score(q, m) = R^w_rel * T^w_rec * I^w_imp * U^w_use * rules(m)

   R = a * bm25(q,m)/max_c bm25 + (1-a) * (cos(q,m) - min_c cos)/(max_c cos - min_c cos)
   T = floor + (1 - floor) * 0.5 ** ((now - last_used(m)) / half_life)
   I = importance(m)                                   (HeapDB dose)
   U = 1 - prod_{f in F(q,m)} (1 - P(useful | f, namespace))   (HeapDB feature_risk)
   ```

   `F(q, m)` = the memory's features, plus `qt:<query type>&<slot>` for the
   date/num/len/q/ent slots, `qsrc:yes|no` (does the query mention the
   memory's source?), and `mem:<id>`. The query type is one of when, who,
   where, why, how, howmany, which, what, yesno, hypothetical or other.

   *Why a product:* with all `w = 1` it is HeapDB's own `kernel x dose x
   links x rules` score with relevance in front. Scaling any one component
   by a constant for every candidate never changes the ranking, so the
   components need no tuning against each other's units. Each `w` is the
   weight of that component in `log score`, and `w = 0` switches it off.
   Relevance is normalised per query because BM25 scale varies with the
   query.

   *Why recency counts from last use:* a memory the agent keeps using should
   not fade just because it is old (spaced reinforcement). Each `mark_used`
   sets `last_used = now`. The floor (0.5) bounds how far an unused memory
   can sink: at most 2x below a fresh one.
3. **EXPLAIN**: each `Recalled` carries R (BM25 and cosine parts), age,
   T, I, U, the per-feature links (strongest first) and any rules applied.
   `explain()` renders them as one line.

Defaults (`ScoringConfig`): all `w = 1`, `lexical_weight = 0.5`,
`half_life_days = 30`, `recency_floor = 0.5`, link prior Beta(0.1, 9.9)
(mean 0.01, i.e. weak), `global_pseudocount = 20`, `superseded_multiplier =
0.1`. These were reasoned out in advance, not tuned on LoCoMo. See the
evaluation for when to change them.

## Feedback loop

`mark_used(ids)` inserts, for each memory, a HeapDB `retrieval` event in its
namespace with features `F(q, m)`. Here `q` is the query of the latest recall
that showed the memory, or the explicit `query=`. The event is confirmed for
`useful`, which gives each feature proportional (EM) credit
`r_f = s_f / noisyOR`. The update goes to the namespace layer and the global
layer, and the memory's `last_used` is refreshed. `mark_ignored(ids)` rejects
the retrieval events instead (`beta += 1` per feature) and leaves
`last_used` alone. `feedback(used, recall_id)` confirms `used` and rejects
every other memory that recall showed.

*Why retrieval events rather than confirming the memory event itself:*
usefulness depends on the question, not only on the memory. A dated memory
is useful for "when" questions and noise for "who" questions. Recording
(query type x memory shape) lets HeapDB learn that. The memory event stays
the unit of dose, time and rules.

Because links are hierarchical, a namespace with no feedback of its own
borrows up to `global_pseudocount` pseudo-observations from the pooled
evidence of all namespaces. That is how the learned patterns reach new users
(and held-out conversations in the evaluation). `explain_usefulness()` (=
`link_report`) shows what was learned. These are associations under the
model, not causal claims.

Implicit learning through HeapDB's `observe` / `settle` was considered and
not used. An agent can say exactly which memories it used, and explicit
confirm/reject on what was shown is a cleaner signal than timing-based soft
credit.

## Update, supersede, forget

- `add(..., key=k)` supersedes the active memory with the same key in the
  namespace. `update(id, text)` adds a new version and supersedes the old
  one. Superseded memories stay recallable (useful for "what did Alice use
  to eat?") but get the HeapDB rule `superseded` x0.1, which shows in
  EXPLAIN. `recall(include_superseded=False)` drops them.
- `forget(ids)` removes text, metadata and index entries immediately. HeapDB
  events are immutable, so the memory's feature-only events remain as
  tombstones, blocked by the rule, until `forget(namespace=...)`. That call
  runs `HeapDB.forget_entity(scrub_global=True)`: events and learned links
  are dropped, and the namespace's contributions to the global links are
  subtracted exactly.

## Persistence

`save(dir)` writes four files, each atomically: `heapdb.jsonl` (HeapDB
snapshot: events, links, calibration), `memories.jsonl` (text and metadata),
`vectors.jsonl` (embedding cache, keyed by `embedder.name`) and
`memory.json` (version, config, default namespace, supersede keys).
`Memory(dir)` loads an existing store. With `autosave=True` (the MCP server
default) every write saves.

## MCP

`heapdb-memory-mcp` exposes `remember`, `recall`, `mark_used`
(`reject_others=True` = implicit feedback for the whole recall),
`mark_ignored`, `forget` and `list_memories`. The server instructions tell
the model to recall before answering and to report usage afterwards. Setup
for Claude Code, Claude Desktop and Cursor is in the README.

## Evaluation summary (LoCoMo)

Copied verbatim from [evals/locomo/RESULTS.md](../evals/locomo/RESULTS.md),
which also has per-category numbers, ablations, nested-CV choices, learned
links, latency and the full analysis.

<!-- summary:start -->
**LoCoMo evidence retrieval** (10 conversations, 5882 dialog turns as memories, 1982 questions with evidence). Recall@k = share of a question's evidence turns in the top k; MRR = 1/rank of the first evidence turn. *Per-conv MRR* = mean ± std over the 10 conversations (for `+learned` methods these are the 10 leave-one-conversation-out folds: usefulness learned only from the other 9 conversations' QA feedback).

| method | R@1 | R@5 | R@10 | MRR | per-conv MRR |
|---|---|---|---|---|---|
| `bm25` | 0.319 | 0.544 | 0.622 | 0.459 | 0.461 ± 0.029 |
| `recency` | 0.000 | 0.002 | 0.010 | 0.005 | 0.005 ± 0.003 |
| `bm25+recency` | 0.280 | 0.511 | 0.600 | 0.417 | 0.417 ± 0.020 |
| `embedding` | 0.141 | 0.331 | 0.425 | 0.263 | 0.261 ± 0.042 |
| `bm25+embedding` | 0.300 | 0.568 | 0.661 | 0.463 | 0.462 ± 0.036 |
| `heapdb` | 0.284 | 0.518 | 0.605 | 0.421 | 0.422 ± 0.018 |
| `heapdb+learned` | 0.317 | 0.542 | 0.622 | 0.454 | 0.455 ± 0.021 |
| `heapdb+learned+tuned` | 0.358 | 0.571 | 0.648 | 0.495 | 0.496 ± 0.027 |
| `bm25+embedding+tuned` | 0.321 | 0.573 | 0.654 | 0.474 | 0.474 ± 0.033 |
| `heapdb+emb` | 0.231 | 0.468 | 0.585 | 0.376 | 0.377 ± 0.036 |
| `heapdb+emb+learned` | 0.272 | 0.508 | 0.619 | 0.419 | 0.419 ± 0.036 |
| `heapdb+emb+learned+tuned` | 0.374 | 0.605 | 0.687 | 0.525 | 0.526 ± 0.032 |

Paired difference in MRR vs `bm25` (bootstrap over questions, 95% CI):

| method | ΔMRR | 95% CI | significant |
|---|---|---|---|
| `recency` | -0.454 | [-0.472, -0.435] | yes |
| `bm25+recency` | -0.042 | [-0.050, -0.034] | yes |
| `embedding` | -0.196 | [-0.216, -0.176] | yes |
| `bm25+embedding` | +0.005 | [-0.005, +0.014] | no |
| `heapdb` | -0.037 | [-0.046, -0.029] | yes |
| `heapdb+learned` | -0.005 | [-0.014, +0.004] | no |
| `heapdb+learned+tuned` | +0.037 | [+0.030, +0.044] | yes |
| `bm25+embedding+tuned` | +0.015 | [+0.010, +0.021] | yes |
| `heapdb+emb` | -0.082 | [-0.096, -0.069] | yes |
| `heapdb+emb+learned` | -0.040 | [-0.053, -0.026] | yes |
| `heapdb+emb+learned+tuned` | +0.066 | [+0.058, +0.075] | yes |

<!-- summary:end -->

**Reading it.** Learned usefulness is the main gain on held-out
conversations. HeapDB scoring plus learning adds +0.05 MRR over an equally
tuned BM25+MiniLM hybrid. The default 30-day recency hurts on LoCoMo,
because questions span the whole history and memories are never re-used, so
untuned defaults do not beat BM25. Use `w_recency=0` for archive-style
recall. Gains were measured with oracle feedback.

## Limitations

- Single process, in-memory indexes, JSONL persistence, exact vector search
  (fine for thousands of memories per namespace).
- The HeapDB log grows by one small `retrieval` event per feedback, and
  forgotten memories leave feature-only tombstones until their namespace is
  forgotten.
- Shape features, the stemmer and the query-type classifier are English
  heuristics.
- Recency and reinforcement are motivated by agent workloads, but the
  public benchmark does not test them. Learning was measured with oracle
  feedback.
- Learned links are associations, not causal effects.

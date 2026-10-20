# heapdb-memory

**Long-term memory for AI agents, built on [HeapDB](https://github.com/heapdb/heapdb-core).
Every recalled memory says why it ranked, and the store learns which memories
your agent actually uses.**

Most agent-memory stores rank by similarity alone. `heapdb-memory` ranks each
candidate memory by

```
score = relevance x recency-since-last-use x importance x learned usefulness x rules
```

and returns a one-line EXPLAIN for each result (from [examples/quickstart.py](examples/quickstart.py)):

```
RECALL 'is Alice vegetarian? what does she eat?' IN user:alice (4 candidates)
  1. [f4a74b8e042d] Alice eats fish now (pescatarian)  score=0.1621
      rel 0.93 (bm25 0.93) x recency 1.00 (age 0.0d) x importance 1.10 x useful 0.16 [date:no=0.02, ent:no=0.02, kind:preference=0.02]
  2. [0b4d4db47864] Alice's flight to Lisbon leaves Friday 9:40  score=0.0161
      rel 0.14 (bm25 0.14) x recency 0.90 (age 10.0d) x importance 0.95 x useful 0.14 [source:unknown=0.01, q:no=0.01, qt:what&date:yes=0.01]
  3. [08efc219e110] Alice is vegetarian and allergic to peanuts  score=0.0159
      rel 1.00 (bm25 1.00) x recency 0.90 (age 10.0d, used 1x) x importance 1.04 x useful 0.17 [date:no=0.02, ent:no=0.02, kind:preference=0.02] x rules superseded x0.1
```

When the agent tells it which recalled memories it used (`mark_used`) and
which it ignored (`mark_ignored`), HeapDB's Beta links learn *which kinds of
memory are useful for which kinds of question*. Examples: dated memories for
"when" questions, preferences over raw tool output, what the user said over
what the agent said. Each confirmed use also resets that memory's decay, so
memories that keep paying off stay fresh.

> **Status: alpha (v0.1).** It is small, readable, and pure Python apart from
> optional extras. It runs in a single process with file-backed storage.
> HeapDB core is being rewritten in Rust with the same Python API, and this
> package uses only that public API. APIs may change.

## Install

```bash
pip install "heapdb-memory[mcp] @ git+https://github.com/heapdb/heapdb-memory"
# extras: mcp, langgraph, llamaindex, embeddings (sentence-transformers), onnx (local ONNX encoder)
```

## SDK quickstart

```python
from heapdb_memory import Memory

mem = Memory("./memory", namespace="user:alice", autosave=True)   # dir is created / loaded

mem.add("Alice is vegetarian and allergic to peanuts", kind="preference", key="diet")
mem.add("Alice's flight to Lisbon leaves Friday 9:40", kind="fact", tags=["travel"])
mem.add("weather api: rain in Lisbon on Friday", kind="tool_result", source="tool")

res = mem.recall("plan Alice's Friday dinner", k=3)
print(res)                              # ranked, with explanations
for r in res:
    r.id, r.text, r.score, r.explain()  # also r.relevance, r.recency, r.importance, r.usefulness

mem.feedback(used=[res[0].id])          # used -> confirm; the other shown ones -> ignored
mem.add("Alice eats fish now", kind="preference", key="diet")   # supersedes the old diet memory
mem.forget(res[2].id)                   # or mem.forget(namespace="user:alice") to erase a user
print(mem.stats())
print(mem.explain_usefulness())         # which kinds of memory turned out useful
```

| method | what it does |
|---|---|
| `add(text, kind=, importance=, tags=, namespace=, source=, key=, created_at=, metadata=)` | store a memory. `importance` is the dose (default: kind/length heuristic). With `key`, it supersedes the older memory with the same key |
| `recall(query, k=5, namespace=, now=, kinds=, tags=, explain=True, config=)` | top-k `Recalled` items with the score broken down |
| `mark_used(ids)` / `mark_ignored(ids)` | learning signal (HeapDB confirm / reject). Used memories also refresh `last_used` |
| `feedback(used, recall_id=None)` | `used` are confirmed, the other memories of that recall are ignored |
| `update(id, text)`, `supersede(old, new)` | new versions; old ones are down-weighted by a HeapDB rule (x0.1) and stay recallable |
| `forget(ids)`, `forget(namespace=...)` | delete memories, or a whole namespace including its learned links (`forget_entity`) |
| `list(...)`, `get(id)`, `stats()`, `explain_usefulness()` | introspection |
| `save(path)`, `Memory.load(path)` | persistence (HeapDB snapshot + `memories.jsonl` + vector cache) |

Kinds with built-in importance defaults: `instruction` 1.5, `preference` 1.3,
`summary` 1.2, `fact` 1.0, `note` 1.0, `episode` 0.8, `tool_result` 0.6. Any
other string is accepted as a kind too.

## MCP server

`heapdb-memory-mcp` serves the store over MCP (stdio by default,
`--transport streamable-http` optional). It works with the official `mcp`
Python SDK 1.x and 2.x.

Tools: `remember`, `recall` (memories plus a one-line explanation each),
`mark_used`, `mark_ignored`, `forget`, `list_memories`. Each tool takes an
optional `namespace`.

| setting | flag | env | default |
|---|---|---|---|
| storage dir | `--dir` | `HEAPDB_MEMORY_DIR` | `~/.heapdb-memory` |
| default namespace | `--namespace` | `HEAPDB_MEMORY_NAMESPACE` | `default` |
| embedder | `--embedder` | `HEAPDB_MEMORY_EMBEDDER` | `none` (`hashing`, `onnx:<dir>`, `st:<model>`) |

**Claude Code**

```bash
claude mcp add heapdb-memory -e HEAPDB_MEMORY_NAMESPACE=user:me -- heapdb-memory-mcp
```

**Claude Desktop** (`claude_desktop_config.json`) or **Cursor** (`~/.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "heapdb-memory": {
      "command": "heapdb-memory-mcp",
      "env": { "HEAPDB_MEMORY_DIR": "/Users/me/.heapdb-memory", "HEAPDB_MEMORY_NAMESPACE": "user:me" }
    }
  }
}
```

If the script is not on the client's `PATH`, use the absolute path of the
venv's `heapdb-memory-mcp`, or `"command": "python", "args": ["-m", "heapdb_memory.mcp_server"]`.
The server's instructions tell the model to `recall` before answering and to
report `mark_used` afterwards.

## Framework adapters

**LangGraph**: `HeapDBStore` is a `langgraph.store.base.BaseStore`
(`put`/`get`/`search`/`delete`/`list_namespaces`). `search(query=...)` is a
HeapDB recall, and `SearchItem.score` is the HeapDB score.

```python
from heapdb_memory.adapters.langgraph import HeapDBStore
store = HeapDBStore(Memory("./memory"))
graph = builder.compile(store=store)
store.put(("user", "alice"), "diet", {"text": "Alice is vegetarian", "kind": "preference"})
hits = store.search(("user", "alice"), query="dinner ideas", limit=3)
store.mark_used([hits[0]])            # learning signal (beyond BaseStore)
```

**LlamaIndex** (`llama_index.core.memory.Memory` with memory blocks):

```python
from llama_index.core.memory import Memory as LIMemory
from heapdb_memory.adapters.llamaindex import HeapDBMemoryBlock
block = HeapDBMemoryBlock(memory=Memory("./memory"), namespace="user:alice", k=5)
li_memory = LIMemory.from_defaults(session_id="alice", memory_blocks=[block])
block.feedback(used=[...])            # ids shown by the last aget()
```

## How recall works

1. **Candidates**: the top 50 memories by BM25 (stdlib implementation) are
   unioned with the top 50 by embedding cosine, if an embedder is
   configured. Embedders are pluggable (`Embedder` protocol):
   `SentenceTransformerEmbedder`, `OnnxEmbedder` (a local ONNX export, no
   PyTorch), or the stdlib `HashingEmbedder` fallback. Nothing is ever
   downloaded implicitly.
2. **Score** (log-linear; each exponent `w` is configurable in `ScoringConfig`, and 0 turns a component off):
   - `R` relevance = `a * bm25/max_bm25 + (1-a) * minmax(cos)` over the candidates
   - `T` recency = `floor + (1-floor) * 0.5^(age/half_life)`, where `age` = time since **last use**
   - `I` importance = the memory's HeapDB dose
   - `U` usefulness = HeapDB `feature_risk(useful, features)` = noisy-OR of the learned
     `P(useful | f)` over the retrieval's features: the memory's shape features
     (`kind:`, `source:`, `len:`, `date:`, `num:`, `q:`, `ent:`, `topic:`, `tag:`), those features
     crossed with the query type (`qt:when&date:yes`), `qsrc:` (does the query name the memory's
     source?) and `mem:<id>`
   - `rules` = HeapDB rule multipliers (`superseded` x0.1)
3. **Learning**: `mark_used` / `mark_ignored` insert a HeapDB `retrieval`
   event carrying those features (no text) and confirm or reject it with
   `HeapDB.feedback`. Credit is proportional, and evidence is pooled per
   namespace plus globally, with hierarchical priors. A new namespace starts
   from what all the others have learned.

Details, defaults and design decisions: [docs/agent-memory.md](docs/agent-memory.md).

## Evaluation: LoCoMo

Evidence retrieval on [LoCoMo](https://github.com/snap-research/locomo)
(Maharana et al., ACL 2024). The dataset is CC BY-NC 4.0 and is downloaded
at run time, never committed. Reproduce with `python -m evals.locomo.run
--onnx-model DIR && python -m evals.locomo.report`; see
[evals/locomo/README.md](evals/locomo/README.md). Full results, per-category
numbers, ablations, latency and the analysis are in
[evals/locomo/RESULTS.md](evals/locomo/RESULTS.md).

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

In short: on LoCoMo, **learned usefulness is the main gain** (+0.033 MRR
over BM25 on held-out conversations). With hybrid relevance and the weights
picked by nested cross-validation on training conversations, HeapDB scoring
is +0.051 MRR over an equally tuned BM25+MiniLM hybrid. **The default
recency decay hurts on this benchmark** (-0.04 MRR). LoCoMo asks about any
point in a long history and never re-uses memories, so the untuned defaults
do not beat plain BM25. For archive-style recall, set `w_recency=0`. A
recall over ~600 memories takes about 1.6 ms (lexical) or about 3 ms (with
the ONNX query embedding) in pure Python.

## Limitations

- Single process, in-memory indexes, file persistence (JSONL). Exact vector search.
- HeapDB events are immutable. `forget(id)` removes the text and index
  entries at once, but the memory's feature-only HeapDB events stay as
  blocked tombstones until `forget(namespace=...)`. Each feedback call adds
  one small `retrieval` event, so the HeapDB log grows with feedback.
- Shape features and query types are English heuristics.
- Learning is association, not causation. The LoCoMo gains were measured
  with oracle feedback; real agent feedback is noisier.
- The benefit of recency and reinforcement for agent workloads is
  plausible but not shown by LoCoMo.

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ../heapdb-core            # or let pip fetch heapdb from GitHub
pip install -e ".[mcp,langgraph,llamaindex,onnx,test]"
pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md). License: Apache-2.0.

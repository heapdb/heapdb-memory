# LoCoMo retrieval evaluation

Evidence retrieval on **LoCoMo**: Adyasha Maharana, Dong-Ho Lee, Sergey
Tulyakov, Mohit Bansal, Francesco Barbieri and Yuwei Fang, *"Evaluating Very
Long-Term Conversational Memory of LLM Agents"*, ACL 2024,
[arXiv:2402.17753](https://arxiv.org/abs/2402.17753),
<https://github.com/snap-research/locomo>.

**License.** The dataset is released under CC BY-NC 4.0 (non-commercial).
This repository does **not** redistribute it. `run.py` downloads
`data/locomo10.json` at run time into `~/.cache/heapdb-memory/locomo/` (or
`$HEAPDB_MEMORY_CACHE`) and checks its sha256 against the version evaluated
here. `results.json` holds only conversation ids, category codes and ranks,
no dataset text.

## Task

- Every dialog turn is a memory: text `"<speaker>: <utterance>"` (plus the
  image caption for image turns), kind `episode`, source = speaker, created
  at its session's timestamp. Namespace = conversation.
- For each QA item with evidence ids, query = the question, asked one day
  after the conversation's last session. Metrics: evidence recall@1/5/10
  and MRR over the top 50, overall and per category (codes 1 multi-hop,
  2 temporal, 3 open-domain, 4 single-hop, 5 adversarial). The 4 category-3
  questions without evidence are excluded. Malformed evidence ids are parsed
  (`D8:6; D9:17`, `D:11:26`, `D30:05`).
- **Learning without leakage.** Leave-one-conversation-out over the 10
  conversations. Each fold builds a fresh store. For every QA item of the 9
  *training* conversations, it recalls the top 10, `mark_used`s the
  evidence turns among them and `mark_ignored`s the rest, in each training
  conversation's own namespace. The held-out namespace never receives
  feedback; it only borrows the global links. `+tuned` methods pick their
  weights from a small fixed grid by MRR on the training conversations
  (nested CV). A test asserts that training never touches the held-out
  conversation.

## Reproduce

```bash
pip install -e ".[onnx]"
python -m evals.locomo.run                      # lexical methods only (no embeddings)
python -m evals.locomo.run --onnx-model DIR     # + all-MiniLM-L6-v2 methods
python -m evals.locomo.report                   # results.json -> RESULTS.md
```

`DIR` must contain `tokenizer.json` and `model.onnx` (or `onnx/model.onnx`)
of the all-MiniLM-L6-v2 ONNX export. The committed results used the export
with sha256 `6fd5d72f…6452` (recorded in `results.json`). In the evaluation
environment HuggingFace was not reachable, so the model files came from an
npm redistribution of the same model (Apache-2.0). Its vectors were checked
against an independent fp16 safetensors copy through a separate NumPy BERT
forward pass (cosine 1.0000 on test sentences) and against the published
reference similarities of sentence-transformers. With HuggingFace access,
`SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2")` is
the equivalent. The full run takes about one hour on 2 vCPUs, mostly the
nested-CV tuning.

Results: [RESULTS.md](RESULTS.md).

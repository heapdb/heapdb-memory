# Contributing

Thanks for helping. This is a small, readable reference package: please keep
it that way.

- **Setup**: `python -m venv .venv && . .venv/bin/activate && pip install -e
  ".[mcp,langgraph,llamaindex,onnx,test]"`. To work against a local
  heapdb-core checkout, `pip install -e ../heapdb-core` first.
- **Tests**: `pytest`. Tests never touch the network. Framework tests skip
  if the framework is not installed, and the core SDK must pass with no
  extras installed (CI checks both).
- **Public core API only.** Use only the public `heapdb` Python API
  (`HeapDB.insert`, `feature_risk`, `feedback`, `link_report`,
  `forget_entity`, `add_rule`, `save`/`load`, ...). The core is being
  rewritten in Rust behind the same API.
- **Evaluation changes**: if you change scoring, re-run `python -m
  evals.locomo.run` and `python -m evals.locomo.report` and commit
  `results.json` and `RESULTS.md` together. Tune only on training folds.
  Report losses as well as wins.
- **Data**: never commit benchmark data or model weights; download them at
  run time.
- **Style**: type hints, docstrings on public functions, no new runtime
  dependencies in the core package (extras are fine).

By contributing you agree that your contributions are licensed under the
Apache License 2.0.

"""Pluggable text embedders and a small exact vector index.

Nothing here downloads anything. Three implementations of the
:class:`Embedder` protocol:

* :class:`HashingEmbedder` (stdlib, always available): feature-hashed,
  sub-linear TF of word stems and character 4-grams, L2-normalised. A
  lexical-ish fallback that catches morphological variants and typos that
  BM25 misses; it is **not** a semantic model.
* :class:`SentenceTransformerEmbedder` (extra ``embeddings``): any
  ``sentence-transformers`` model already present locally (or downloadable
  by the library, if your network allows it).
* :class:`OnnxEmbedder` (extra ``onnx``): a BERT-style sentence encoder from
  a local directory holding ``model.onnx`` (or ``onnx/model.onnx``) and
  ``tokenizer.json``; mean pooling over the attention mask, L2 normalised.
  With the all-MiniLM-L6-v2 export this reproduces sentence-transformers'
  vectors without PyTorch.

:class:`VectorIndex` does exact cosine search (NumPy if installed, else pure
Python). Exact search over a few thousand memories per namespace is fast
enough; swap in an ANN index behind the same methods if you need more.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Hashable, Protocol, Sequence, runtime_checkable

from .text import STOPWORDS, stem, tokenize

__all__ = [
    "Embedder",
    "HashingEmbedder",
    "OnnxEmbedder",
    "SentenceTransformerEmbedder",
    "VectorIndex",
]


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns texts into fixed-size vectors.

    ``name`` identifies the model (stored next to cached vectors so a
    different embedder never reuses them). ``embed`` returns one vector per
    text; vectors should be L2-normalised (cosine = dot product).
    """

    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def _normalise(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n > 0 else v


class HashingEmbedder:
    """Stdlib fallback: hashed TF of stems + character n-grams.

    Args:
        dim: Vector size (hash buckets).
        ngram: Character n-gram length (on ``#word#``), 0 to disable.
    """

    def __init__(self, dim: int = 512, ngram: int = 4) -> None:
        if dim < 8:
            raise ValueError("dim must be >= 8")
        self.dim = int(dim)
        self.ngram = int(ngram)
        self.name = f"hashing-{self.dim}-{self.ngram}"

    def _bucket(self, s: str) -> tuple[int, float]:
        h = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
        x = int.from_bytes(h, "little")
        return x % self.dim, (1.0 if (x >> 63) & 1 else -1.0)

    def _one(self, text: str) -> list[float]:
        counts: dict[str, float] = {}
        for tok in tokenize(text):
            if tok in STOPWORDS:
                continue
            st = stem(tok)
            counts["w:" + st] = counts.get("w:" + st, 0.0) + 1.0
            if self.ngram and len(tok) > self.ngram:
                w = f"#{tok}#"
                for i in range(len(w) - self.ngram + 1):
                    g = "c:" + w[i: i + self.ngram]
                    counts[g] = counts.get(g, 0.0) + 0.5
        v = [0.0] * self.dim
        for feat, c in counts.items():
            i, sign = self._bucket(feat)
            v[i] += sign * (1.0 + math.log(c)) if c >= 1 else sign * c
        return _normalise(v)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]


class SentenceTransformerEmbedder:
    """Adapter for ``sentence-transformers`` (``pip install heapdb-memory[embeddings]``).

    Args:
        model: Model name or local path (default ``all-MiniLM-L6-v2``).
        device: Passed to ``SentenceTransformer``.
        batch_size: Encoding batch size.
    """

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str | None = None, batch_size: int = 64) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover - depends on the extra
            raise ImportError("SentenceTransformerEmbedder needs the 'embeddings' extra: "
                              "pip install 'heapdb-memory[embeddings]'") from e
        self._model = SentenceTransformer(model, device=device)
        self.batch_size = batch_size
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.name = f"st:{Path(model).name}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vecs = self._model.encode(list(texts), batch_size=self.batch_size,
                                  normalize_embeddings=True, show_progress_bar=False)
        return [list(map(float, v)) for v in vecs]


class OnnxEmbedder:
    """Local ONNX sentence encoder (``pip install heapdb-memory[onnx]``).

    ``model_dir`` must contain ``tokenizer.json`` and ``model.onnx`` (or
    ``onnx/model.onnx``) of a BERT-style encoder whose first output is the
    last hidden state. Mean pooling over real tokens, then L2 normalisation
    (the sentence-transformers recipe for all-MiniLM-L6-v2).

    Args:
        model_dir: Directory with the files above. Nothing is downloaded.
        max_length: Truncation length in tokens (MiniLM was trained with 256).
        batch_size: Texts per ONNX run.
        name: Model name stored with cached vectors.
    """

    def __init__(self, model_dir: str | Path, max_length: int = 256, batch_size: int = 32,
                 name: str = "onnx:all-MiniLM-L6-v2", threads: int | None = None) -> None:
        try:
            import numpy as np
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as e:  # pragma: no cover - depends on the extra
            raise ImportError("OnnxEmbedder needs the 'onnx' extra: "
                              "pip install 'heapdb-memory[onnx]'") from e
        d = Path(model_dir)
        onnx_path = next((p for p in (d / "model.onnx", d / "onnx" / "model.onnx") if p.exists()), None)
        if onnx_path is None or not (d / "tokenizer.json").exists():
            raise FileNotFoundError(f"{d} needs tokenizer.json and model.onnx (or onnx/model.onnx)")
        self._np = np
        self._tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self._tok.no_padding()
        self._tok.enable_truncation(max_length)
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self._sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
        self._inputs = {i.name for i in self._sess.get_inputs()}
        self.batch_size = batch_size
        self.name = name
        self.dim = int(self.embed(["dimension probe"])[0].__len__())

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        np = self._np
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            encs = [self._tok.encode(t or " ") for t in texts[i: i + self.batch_size]]
            n = max(len(e.ids) for e in encs)
            ids = np.zeros((len(encs), n), dtype=np.int64)
            mask = np.zeros((len(encs), n), dtype=np.int64)
            for j, e in enumerate(encs):
                ids[j, : len(e.ids)] = e.ids
                mask[j, : len(e.ids)] = 1
            feeds = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
            hidden = self._sess.run(None, {k: v for k, v in feeds.items() if k in self._inputs})[0]
            m = mask[..., None].astype(hidden.dtype)
            pooled = (hidden * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
            pooled /= np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
            out.extend(pooled.astype(float).tolist())
        return out


class VectorIndex:
    """Exact cosine-similarity index over L2-normalised vectors."""

    def __init__(self) -> None:
        self._ids: list[Hashable] = []
        self._pos: dict[Hashable, int] = {}
        self._vecs: list[list[float]] = []
        self._matrix = None  # lazily built NumPy matrix

    def __len__(self) -> int:
        return len(self._ids)

    def __contains__(self, doc_id: Hashable) -> bool:
        return doc_id in self._pos

    def add(self, doc_id: Hashable, vec: Sequence[float]) -> None:
        if doc_id in self._pos:
            self._vecs[self._pos[doc_id]] = list(vec)
        else:
            self._pos[doc_id] = len(self._ids)
            self._ids.append(doc_id)
            self._vecs.append(list(vec))
        self._matrix = None

    def get(self, doc_id: Hashable) -> list[float] | None:
        i = self._pos.get(doc_id)
        return None if i is None else self._vecs[i]

    def remove(self, doc_id: Hashable) -> bool:
        i = self._pos.pop(doc_id, None)
        if i is None:
            return False
        last = len(self._ids) - 1
        if i != last:
            self._ids[i] = self._ids[last]
            self._vecs[i] = self._vecs[last]
            self._pos[self._ids[i]] = i
        self._ids.pop()
        self._vecs.pop()
        self._matrix = None
        return True

    def similarities(self, query_vec: Sequence[float]) -> dict[Hashable, float]:
        """Cosine similarity of the query to every indexed vector."""
        if not self._ids:
            return {}
        try:
            import numpy as np
        except ImportError:  # pragma: no cover - numpy is almost always there
            return {d: sum(a * b for a, b in zip(query_vec, v)) for d, v in zip(self._ids, self._vecs)}
        if self._matrix is None:
            self._matrix = np.asarray(self._vecs, dtype=np.float32)
        sims = self._matrix @ np.asarray(query_vec, dtype=np.float32)
        return dict(zip(self._ids, sims.tolist()))

    def top(self, query_vec: Sequence[float], n: int = 10) -> list[tuple[Hashable, float]]:
        sims = self.similarities(query_vec)
        return sorted(sims.items(), key=lambda kv: (-kv[1], str(kv[0])))[:n]

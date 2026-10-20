"""heapdb-memory: long-term memory for AI agents, built on HeapDB.

Recall ranks memories by ``relevance x recency x importance x learned
usefulness`` with a one-line explanation for each, and learns from which
recalled memories the agent actually used.

    >>> from heapdb_memory import Memory
    >>> mem = Memory(namespace="user:alice", clock=lambda: 1_700_000_000.0)
    >>> _ = mem.add("Alice is vegetarian and allergic to peanuts", kind="preference")
    >>> _ = mem.add("Alice's flight to Lisbon leaves on Friday", kind="fact")
    >>> mem.recall("what can Alice eat?", k=1)[0].text
    'Alice is vegetarian and allergic to peanuts'
"""

from .bm25 import BM25Index
from .embeddings import Embedder, HashingEmbedder, OnnxEmbedder, SentenceTransformerEmbedder, VectorIndex
from .memory import KINDS, Memory, MemoryItem, Recalled, RecallResult, ScoringConfig

__version__ = "0.1.0"

__all__ = [
    "BM25Index",
    "Embedder",
    "HashingEmbedder",
    "KINDS",
    "Memory",
    "MemoryItem",
    "OnnxEmbedder",
    "RecallResult",
    "Recalled",
    "ScoringConfig",
    "SentenceTransformerEmbedder",
    "VectorIndex",
]

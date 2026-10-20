"""MCP server exposing a :class:`~heapdb_memory.Memory` store as tools.

Run ``heapdb-memory-mcp`` (stdio by default). Tools:

``remember``       store a memory (text, kind, importance, tags, key)
``recall``         top-k memories for a query, each with a one-line explanation
``mark_used``      memories the agent actually used (learning: confirm + refresh)
``mark_ignored``   memories that were recalled but not used (learning: reject)
``forget``         delete memories by id
``list_memories``  newest memories, optionally filtered by kind / tag

Configuration (CLI flags override environment variables):

``HEAPDB_MEMORY_DIR``        storage directory (default ``~/.heapdb-memory``)
``HEAPDB_MEMORY_NAMESPACE``  default namespace (default ``default``)
``HEAPDB_MEMORY_EMBEDDER``   ``none`` (default), ``hashing``, ``onnx:<dir>`` or
                             ``st:<model name or path>``

Every tool takes an optional ``namespace`` so one server can serve several
users or projects. The store autosaves after every write.

Works with the official ``mcp`` Python SDK, 1.x (``FastMCP``) and 2.x
(``MCPServer``).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from .embeddings import Embedder, HashingEmbedder
from .memory import Memory

__all__ = ["build_server", "main", "make_embedder"]

INSTRUCTIONS = (
    "Long-term memory. Call `recall` with the user's request before answering when past "
    "context could matter. After answering, call `mark_used` with the ids of recalled memories "
    "you actually relied on (and optionally `mark_ignored` for ones you did not), so future "
    "recalls rank better. Call `remember` for durable facts, preferences and instructions "
    "(use `key` for facts that can change, e.g. 'user.city', so the new value supersedes the old)."
)


def _server_class():
    try:  # mcp >= 2
        from mcp.server.mcpserver import MCPServer
        return MCPServer
    except ImportError:  # mcp 1.x
        try:
            from mcp.server.fastmcp import FastMCP
        except ImportError as e:  # pragma: no cover - depends on the extra
            raise ImportError("the MCP server needs the 'mcp' extra: "
                              "pip install 'heapdb-memory[mcp]'") from e
        return FastMCP


def make_embedder(spec: str | None) -> Embedder | None:
    """``none`` | ``hashing`` | ``onnx:<dir>`` | ``st:<model>`` -> embedder."""
    if not spec or spec == "none":
        return None
    if spec == "hashing":
        return HashingEmbedder()
    if spec.startswith("onnx:"):
        from .embeddings import OnnxEmbedder
        return OnnxEmbedder(spec[5:])
    if spec.startswith("st:"):
        from .embeddings import SentenceTransformerEmbedder
        return SentenceTransformerEmbedder(spec[3:])
    raise ValueError(f"unknown embedder spec {spec!r}")


def build_server(memory: Memory, name: str = "heapdb-memory") -> Any:
    """An MCP server object (``MCPServer``/``FastMCP``) bound to ``memory``."""
    server = _server_class()(name, instructions=INSTRUCTIONS)

    def ns(namespace: str | None) -> str:
        return namespace or memory.namespace

    @server.tool()
    def remember(text: str, kind: str = "fact", importance: float | None = None,
                 tags: list[str] | None = None, key: str | None = None,
                 source: str | None = None, namespace: str | None = None) -> dict[str, Any]:
        """Store a long-term memory.

        Args:
            text: The memory, self-contained (e.g. "User prefers metric units").
            kind: fact | preference | instruction | episode | summary | tool_result | note.
            importance: 0.1-3 (1 = normal). Omit to use the kind's default.
            tags: Optional labels, e.g. ["diet", "travel"].
            key: Optional stable key (e.g. "user.city"); a new memory with the same key
                supersedes the old one.
            source: Where it came from (user, agent, tool...).
            namespace: Owner (user/agent/project). Defaults to the server's namespace.
        """
        it = memory.add(text, kind=kind, importance=importance, tags=tags or (), key=key,
                        source=source, namespace=ns(namespace))
        return {"id": it.id, "namespace": it.namespace, "kind": it.kind,
                "importance": round(it.importance, 3),
                "superseded": [m.id for m in memory.list(ns(namespace), include_superseded=True)
                               if m.superseded_by == it.id]}

    @server.tool()
    def recall(query: str, k: int = 5, kinds: list[str] | None = None,
               tags: list[str] | None = None, namespace: str | None = None) -> dict[str, Any]:
        """Retrieve the k most useful memories for a query, best first.

        Each result has an id, the text, a score and a one-line explanation
        (relevance x recency x importance x learned usefulness). Report back with
        mark_used / mark_ignored so ranking improves.
        """
        res = memory.recall(query, k=k, namespace=ns(namespace), kinds=kinds, tags=tags or ())
        return {"recall_id": res.recall_id, "namespace": res.namespace,
                "memories": [{"id": r.id, "text": r.text, "kind": r.memory.kind,
                              "score": round(r.score, 4), "why": r.explain()} for r in res]}

    @server.tool()
    def mark_used(ids: list[str], recall_id: str | None = None,
                  reject_others: bool = False, namespace: str | None = None) -> dict[str, Any]:
        """Report memories you actually used. They are reinforced (recency reset) and
        their kind of memory learns to rank higher. With reject_others=True every
        other memory shown by that recall (recall_id, default: the last one) is
        marked ignored."""
        if reject_others:
            out = memory.feedback(used=ids, recall_id=recall_id, namespace=ns(namespace))
            return {"used": len(out["used"]), "ignored": len(out["ignored"])}
        return {"used": memory.mark_used(ids)}

    @server.tool()
    def mark_ignored(ids: list[str]) -> dict[str, Any]:
        """Report recalled memories that were not useful; similar memories rank lower later."""
        return {"ignored": memory.mark_ignored(ids)}

    @server.tool()
    def forget(ids: list[str]) -> dict[str, Any]:
        """Permanently delete memories by id."""
        return memory.forget(ids)

    @server.tool()
    def list_memories(limit: int = 20, offset: int = 0, kind: str | None = None,
                      tag: str | None = None, namespace: str | None = None) -> dict[str, Any]:
        """List stored memories, newest first."""
        items = memory.list(ns(namespace), kind=kind, tags=[tag] if tag else (),
                            limit=limit, offset=offset)
        return {"namespace": ns(namespace), "total": len(memory.list(ns(namespace))),
                "memories": [{"id": m.id, "text": m.text, "kind": m.kind,
                              "created_at": m.created_at, "tags": list(m.tags),
                              "key": m.key, "uses": m.use_count} for m in items]}

    return server


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="heapdb-memory-mcp", description=__doc__.split("\n\n")[0])
    p.add_argument("--dir", default=os.environ.get("HEAPDB_MEMORY_DIR", str(Path.home() / ".heapdb-memory")))
    p.add_argument("--namespace", default=os.environ.get("HEAPDB_MEMORY_NAMESPACE", "default"))
    p.add_argument("--embedder", default=os.environ.get("HEAPDB_MEMORY_EMBEDDER", "none"))
    p.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    a = p.parse_args(argv)
    mem = Memory(a.dir, namespace=a.namespace, embedder=make_embedder(a.embedder), autosave=True)
    server = build_server(mem)
    print(f"heapdb-memory: {len(mem)} memories in {a.dir} (namespace {a.namespace})", file=sys.stderr)
    if a.transport == "stdio":
        server.run("stdio")
    else:
        try:
            server.run("streamable-http", host=a.host, port=a.port)
        except TypeError:  # mcp 1.x reads host/port from settings
            server.settings.host, server.settings.port = a.host, a.port
            server.run("streamable-http")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

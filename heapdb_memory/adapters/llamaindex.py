"""LlamaIndex memory block backed by :class:`heapdb_memory.Memory`.

Targets the current LlamaIndex memory API (``llama_index.core.memory.Memory``
with memory blocks, llama-index-core >= 0.12.35). ``HeapDBMemoryBlock`` is a
``BaseMemoryBlock[str]``:

* ``aput(messages)`` -- called when messages are flushed out of short-term
  memory -- stores each non-empty user / assistant message as an ``episode``
  memory (``source`` = the role).
* ``aget(messages)`` recalls the top-k memories for the last
  ``context_window`` messages and renders them, with ids, into the text that
  LlamaIndex inserts in the prompt.
* ``mark_used`` / ``mark_ignored`` / ``feedback`` pass usage back to
  the store (the learning signal LlamaIndex has no hook for).

::

    from llama_index.core.memory import Memory as LIMemory
    from heapdb_memory import Memory
    from heapdb_memory.adapters.llamaindex import HeapDBMemoryBlock

    block = HeapDBMemoryBlock(memory=Memory("./memory"), namespace="user:alice")
    li_memory = LIMemory.from_defaults(session_id="alice", memory_blocks=[block])
"""

from __future__ import annotations

from typing import Any, List, Optional

from llama_index.core.base.llms.types import ChatMessage, MessageRole, TextBlock
from llama_index.core.bridge.pydantic import Field
from llama_index.core.memory import BaseMemoryBlock

from ..memory import Memory

__all__ = ["HeapDBMemoryBlock"]


def _text(msg: ChatMessage) -> str:
    return " ".join(b.text for b in msg.blocks if isinstance(b, TextBlock)).strip()


class HeapDBMemoryBlock(BaseMemoryBlock[str]):
    """Long-term memory block: stores flushed messages, recalls with EXPLAIN."""

    name: str = Field(default="heapdb_memory", description="Block name.")
    description: Optional[str] = Field(
        default="Long-term memories ranked by relevance, recency, importance and learned usefulness.")
    memory: Any = Field(description="A heapdb_memory.Memory instance.")
    namespace: Optional[str] = Field(default=None, description="Memory namespace (default: the store's).")
    k: int = Field(default=5, description="Memories to recall.")
    context_window: int = Field(default=2, description="Recent messages used as the query.")
    roles: tuple = Field(default=("user", "assistant"), description="Roles stored on aput.")
    explain: bool = Field(default=False, description="Append the score explanation to each memory.")
    last_recall_ids: list = Field(default_factory=list, description="Ids shown by the last aget.")

    def _mem(self) -> Memory:
        return self.memory

    async def _aget(self, messages: Optional[List[ChatMessage]] = None, **block_kwargs: Any) -> str:
        if not messages:
            return ""
        query = " ".join(_text(m) for m in messages[-self.context_window:]).strip()
        if not query:
            return ""
        res = self._mem().recall(query, k=self.k, namespace=self.namespace, explain=self.explain)
        self.last_recall_ids = res.ids
        lines = []
        for r in res:
            line = f"- [{r.id}] {r.text}"
            if self.explain:
                line += f"  ({r.explain()})"
            lines.append(line)
        return "\n".join(lines)

    async def _aput(self, messages: List[ChatMessage]) -> None:
        mem = self._mem()
        for m in messages:
            role = m.role.value if isinstance(m.role, MessageRole) else str(m.role)
            txt = _text(m)
            if txt and role in self.roles:
                mem.add(txt, kind="episode", source=role, namespace=self.namespace)

    def mark_used(self, ids: List[str]) -> int:
        return self._mem().mark_used(ids)

    def mark_ignored(self, ids: List[str]) -> int:
        return self._mem().mark_ignored(ids)

    def feedback(self, used: List[str]) -> dict:
        """``used`` confirmed; the other ids shown by the last ``aget`` rejected."""
        shown = list(self.last_recall_ids)
        ignored = [i for i in shown if i not in set(used)]
        if used:
            self._mem().mark_used(used)
        if ignored:
            self._mem().mark_ignored(ignored)
        return {"used": list(used), "ignored": ignored}

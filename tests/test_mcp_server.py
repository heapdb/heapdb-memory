"""MCP tools through an in-process client session (no subprocess, no network)."""

from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("mcp")

from heapdb_memory import Memory  # noqa: E402
from heapdb_memory.mcp_server import build_server, main, make_embedder  # noqa: E402


def _payload(result) -> dict:
    if getattr(result, "structured_content", None) is not None:
        return result.structured_content
    if getattr(result, "structuredContent", None) is not None:  # mcp 1.x
        return result.structuredContent
    return json.loads(result.content[0].text)


async def _session(server, fn):
    try:
        from mcp import Client  # mcp >= 2
    except ImportError:  # mcp 1.x
        from mcp.shared.memory import create_connected_server_and_client_session

        async with create_connected_server_and_client_session(server._mcp_server) as s:
            return await fn(s)
    async with Client(server) as c:
        return await fn(c)


def test_tools_round_trip(tmp_path) -> None:
    mem = Memory(tmp_path, namespace="user:test", autosave=True, clock=lambda: 1_700_000_000.0)
    server = build_server(mem)

    async def scenario(c):
        tools = {t.name for t in (await c.list_tools()).tools}
        assert tools == {"remember", "recall", "mark_used", "mark_ignored", "forget", "list_memories"}
        a = _payload(await c.call_tool("remember", {"text": "User prefers metric units",
                                                    "kind": "preference", "tags": ["units"]}))
        b = _payload(await c.call_tool("remember", {"text": "User's laptop runs Linux"}))
        old = _payload(await c.call_tool("remember", {"text": "User lives in Porto", "key": "city"}))
        new = _payload(await c.call_tool("remember", {"text": "User moved to Lisbon", "key": "city"}))
        assert new["superseded"] == [old["id"]]
        r = _payload(await c.call_tool("recall", {"query": "which units does the user prefer?", "k": 2}))
        assert r["memories"][0]["id"] == a["id"]
        assert "rel" in r["memories"][0]["why"] and r["recall_id"]
        u = _payload(await c.call_tool("mark_used", {"ids": [a["id"]], "recall_id": r["recall_id"],
                                                     "reject_others": True}))
        assert u["used"] == 1
        ig = _payload(await c.call_tool("mark_ignored", {"ids": [b["id"]]}))
        assert ig == {"ignored": 1}
        lst = _payload(await c.call_tool("list_memories", {"limit": 10}))
        assert lst["total"] == 3 and {m["id"] for m in lst["memories"]} == {a["id"], b["id"], new["id"]}
        f = _payload(await c.call_tool("forget", {"ids": [b["id"]]}))
        assert f["memories"] == 1
        other = _payload(await c.call_tool("list_memories", {"namespace": "someone-else"}))
        assert other["total"] == 0
        return a["id"]

    aid = asyncio.run(_session(server, scenario))
    # autosave persisted everything, including the feedback
    back = Memory(tmp_path)
    assert back.namespace == "user:test" and back.get(aid).use_count == 1 and len(back) == 3


def test_make_embedder_specs() -> None:
    assert make_embedder("none") is None and make_embedder(None) is None
    assert make_embedder("hashing").dim == 512
    with pytest.raises(ValueError):
        make_embedder("bogus")


def test_cli_help(capsys) -> None:
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0
    assert "--namespace" in capsys.readouterr().out

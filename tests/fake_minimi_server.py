"""A fake minimi MCP server over stdio. FAKE_MINIMI_MODE selects its behaviour."""

from __future__ import annotations

import json
import os
import sys
import time

MODE = os.environ.get("FAKE_MINIMI_MODE", "ok")


def send(message: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def result_for(params: dict[str, object]) -> dict[str, object]:
    arguments = params.get("arguments", {})
    if MODE == "tool_error":
        return {"content": [{"type": "text", "text": "search failed"}], "isError": True}
    if MODE == "quota":
        return {"content": [{"type": "text", "text": "Daily quota exceeded. Try tomorrow."}]}
    echo = {"pid": os.getpid(), "tool": params.get("name"), "arguments": arguments}
    # minimi's real format: only a content key, one markdown result per match.
    text = (
        "## Search Results (1)\n\n### Result 1\n- **Source:** fake\n"
        "- **Captured:** 2026-10-02 11:20\n- **Memory:** echo\n- **Context:**\n\n"
        f"```\n{json.dumps(echo)}\n```\n"
    )
    result: dict[str, object] = {"content": [{"type": "text", "text": text}]}
    if MODE == "paged" and isinstance(arguments, dict) and arguments.get("cursor") is None:
        result["structuredContent"] = {"nextCursor": "page-2"}
    return result


for line in sys.stdin:
    request = json.loads(line)
    method = request.get("method")
    if "id" not in request:
        continue
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": request["id"], "result": {"protocolVersion": "2025-06-18"}})
        continue
    if MODE == "hang":
        time.sleep(60)
    if MODE == "die":
        sys.exit(1)
    if MODE == "noise":
        sys.stdout.write("starting up, not JSON\n")
        send({"jsonrpc": "2.0", "method": "notifications/message", "params": {}})
        send({"jsonrpc": "2.0", "id": 999, "result": {"content": []}})
        sys.stdout.write("[1, 2, 3]\n")
    if MODE == "rpc_error":
        send({"jsonrpc": "2.0", "id": request["id"], "error": {"code": -32602, "message": "bad"}})
        continue
    send({"jsonrpc": "2.0", "id": request["id"], "result": result_for(request["params"])})

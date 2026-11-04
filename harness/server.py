"""The harness behind three read-only MCP tools, over stdio.

    python -m harness.server --db runs/harness/d.db

The store's meta.now is the clock. The server rereads the store whenever another process
(the CLI adding a live message) has written to it.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field

from harness.retrieve import DEFAULT_TOKENS, Retriever
from harness.store import Store

INSTRUCTIONS = """# minimi memory

Memory of this user's messages, email and calendar, kept current: when something changes, the old
version is marked as history instead of sitting next to the new one.

- Call `minimi_get_context` first, with the user's question as written. One call usually answers it.
- `as_of` is the current time. Ignore any other clock.
- Memory text is already resolved: dates are absolute, and references like "same as last time" are
  worked out from earlier messages. The quote is the evidence; you don't need to re-derive it.
- Lines marked "(was ...)" and the "Changed recently" section are history. Use them to explain what
  changed, never as the current answer.
- If the footer shows fewer shown than the total, make the `minimi_search_memory` call it names.
- `minimi_get_source` shows the original messages behind memory ids when you need exact wording.
- If nothing answers the question, say you don't know.
"""

CONTEXT_DESC = """Answer a question about the user's life from memory in one call.

Returns a packed, ranked context: current memories as `[id] kind (owner) · time or due · text ·
"quote" · apps`, a "Changed recently" section with what was replaced or done in the last 14 days and
why, raw messages not processed yet, and a footer with how many items each filter matched versus
showed. Pass the question exactly as the user asked it."""

SEARCH_DESC = """Search or list memories with filters. An empty query lists everything matching the
filters sorted by date, with total_count, has_more and next_cursor for paging. Use it when the
`minimi_get_context` footer says more items exist than were shown."""

SOURCE_DESC = """The original messages behind memory ids: each quote, the full message, and 2
messages on each side in its conversation."""


def build_server(retriever: Retriever):
    from mcp.server.mcpserver import MCPServer
    from mcp.types import ToolAnnotations

    server = MCPServer("minimi-memory", title="minimi memory", instructions=INSTRUCTIONS)
    ro = ToolAnnotations(read_only_hint=True, idempotent_hint=True, open_world_hint=False)

    @server.tool(name="minimi_get_context", annotations=ro, description=CONTEXT_DESC)
    def get_context(
        question: Annotated[str, Field(description="The user's question, as written.")],
        time_range: Annotated[
            dict[str, str] | None,
            Field(description='Optional {"start": ISO time, "end": ISO time}. Time words in the'
                              " question are read automatically."),
        ] = None,
        people: Annotated[list[str] | None, Field(description="Optional names to focus on.")] = None,
        max_tokens: Annotated[int, Field(ge=500, le=4000, description="Context budget.")] = DEFAULT_TOKENS,
    ) -> str:
        return retriever.get_context(question, time_range, people, max_tokens)

    @server.tool(name="minimi_search_memory", annotations=ro, description=SEARCH_DESC)
    def search_memory(
        query: Annotated[str, Field(description="Words to search for. Empty means filters only.")] = "",
        kind: Annotated[
            Literal["event", "todo", "fact", "preference", "status", "episode"] | None,
            Field(description="Memory kind."),
        ] = None,
        owner: Annotated[
            Literal["you", "other"] | None,
            Field(description="For todos: who has to act."),
        ] = None,
        people: Annotated[list[str] | None, Field(description="Names; first-name match.")] = None,
        after: Annotated[str | None, Field(description="ISO time; event or said time after this.")] = None,
        before: Annotated[str | None, Field(description="ISO time; event or said time before this.")] = None,
        include_history: Annotated[bool, Field(description="Also return replaced and ended memories.")] = False,
        limit: Annotated[int, Field(ge=1, le=50)] = 25,
        cursor: Annotated[int, Field(ge=0, description="next_cursor from the previous page.")] = 0,
    ) -> dict[str, Any]:
        return retriever.search(query, kind, owner, people, after, before, include_history, limit, cursor)

    @server.tool(name="minimi_get_source", annotations=ro, description=SOURCE_DESC)
    def get_source(
        memory_ids: Annotated[list[str], Field(min_length=1, max_length=10, description='Ids like "m12".')],
    ) -> str:
        return retriever.get_source(memory_ids)

    return server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="minimi memory harness MCP server")
    parser.add_argument("--db", required=True)
    parser.add_argument("--now", help="ignored; the store's meta.now is the clock")
    args = parser.parse_args(argv)
    if not Path(args.db).exists():
        parser.error(f"database not found: {args.db}")

    from harness.text import Embedder

    build_server(Retriever(Store(args.db), Embedder())).run("stdio")


if __name__ == "__main__":
    main()

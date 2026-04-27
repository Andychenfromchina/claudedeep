"""MCP server exposing the deep_research toolkit to MCP clients (Claude Code,
Cline, Aider, etc).

Three tools surfaced:
- search_web(query, max_results)         — single search, dedup-aware
- fetch_url(url)                          — HTML or PDF body
- research(question, max_iters, lang, …)  — full agent loop, returns markdown

Run (stdio transport, the most common MCP setup):

    pip install -e ".[mcp]"
    python -m mcp_server                  # or `deep-research-mcp`

Add to a Claude Code or Cline MCP config:

    {
      "mcpServers": {
        "deep-research": {
          "command": "python",
          "args": ["-m", "mcp_server"],
          "env": {
            "DEEPSEEK_API_KEY": "sk-...",
            "TAVILY_API_KEY": "tvly-..."
          }
        }
      }
    }
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

try:
    from mcp.server import NotificationOptions, Server
    from mcp.server.models import InitializationOptions
    from mcp.server.stdio import stdio_server
    import mcp.types as types
except ImportError:
    sys.exit(
        "mcp package not installed. Install with: pip install 'mcp>=1.0' "
        "(or `pip install -e \".[mcp]\"`)."
    )

import deep_research as dr
from _log import configure_logging, log

# MCP clients communicate over stdio with JSON-RPC. Logs MUST go to stderr in
# JSON form so they don't corrupt the protocol.
configure_logging(os.environ.get("DEEP_RESEARCH_LOG", "json"))

server: Server = Server("deep-research")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="search_web",
            description=(
                "Search the public web via the configured provider "
                "(tavily/serper/bing/google_cse, auto-detected from env keys). "
                "Returns title/url/snippet, deduped against URLs already seen "
                "this session and ranked by source quality."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 5, "maximum": 10},
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="fetch_url",
            description=(
                "Fetch a URL's body text. Auto-detects PDFs and extracts text "
                "via pypdf. Body is capped at 20K chars."
            ),
            inputSchema={
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        ),
        types.Tool(
            name="research",
            description=(
                "Run a full research loop on a question and return a "
                "polished markdown report with citations. Uses DeepSeek "
                "v4-flash as planner and v4-pro as writer by default."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "max_iters": {"type": "integer", "default": 8, "maximum": 20},
                    "lang": {"type": "string", "default": "en",
                             "description": "Output language code (en, zh-CN, ja, …)"},
                    "thinking": {"type": "boolean", "default": False},
                    "planner": {"type": "string", "default": "deepseek-v4-flash"},
                    "writer": {"type": "string", "default": "deepseek-v4-pro"},
                    "token_budget": {"type": "integer"},
                },
                "required": ["question"],
            },
        ),
    ]


def _ensure_provider() -> str:
    if not dr.PROVIDER_NAME or dr.PROVIDER_NAME == "auto":
        dr.PROVIDER_NAME = dr.detect_provider()
    return dr.PROVIDER_NAME


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    log.info("mcp.tool_called", name=name, arguments=arguments)
    try:
        if name == "search_web":
            _ensure_provider()
            result = await asyncio.to_thread(
                dr.search_web,
                arguments["query"],
                int(arguments.get("max_results", 5)),
            )
            return [types.TextContent(type="text",
                                      text=json.dumps(result, ensure_ascii=False, indent=2))]

        if name == "fetch_url":
            result = await asyncio.to_thread(dr.fetch_url, arguments["url"])
            return [types.TextContent(type="text",
                                      text=json.dumps(result, ensure_ascii=False, indent=2))]

        if name == "research":
            _ensure_provider()
            from prompts import resolve_lang
            state = dr.SessionState(
                question=arguments["question"],
                provider=dr.PROVIDER_NAME,
                thinking=bool(arguments.get("thinking", False)),
                planner_model=arguments.get("planner", "deepseek-v4-flash"),
                writer_model=arguments.get("writer", "deepseek-v4-pro"),
                lang=resolve_lang(arguments.get("lang", "en")),
            )
            report = await dr.research(
                state,
                max_iters=int(arguments.get("max_iters", 8)),
                max_tokens=None,
                token_budget=arguments.get("token_budget"),
                save_path=None,
                stream=False,
            )
            return [types.TextContent(type="text", text=report)]

        return [types.TextContent(type="text", text=f"unknown tool: {name}")]
    except Exception as e:
        log.error("mcp.tool_failed", name=name, error=str(e))
        return [types.TextContent(type="text",
                                  text=f"error: {type(e).__name__}: {e}")]


async def amain() -> None:
    log.info("mcp.starting", provider=os.environ.get("SEARCH_PROVIDER", "auto"))
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="deep-research",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


def main() -> None:
    """Console-script entry for `deep-research-mcp`."""
    asyncio.run(amain())


if __name__ == "__main__":
    main()

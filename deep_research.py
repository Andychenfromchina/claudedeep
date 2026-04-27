#!/usr/bin/env python3
"""Deep research agent: DeepSeek + pluggable web search.

Features:
- AsyncOpenAI client with lazy init (--help works without keys)
- Streaming output (--stream): final report streams to stdout live
- Multi-model routing: cheap planner for tool rounds, premium writer for the
  final synthesis pass (--planner / --writer)
- Thinking mode (--thinking) — surfaces reasoning_content live to stderr
- KV-cache-friendly long stable system prompt
- Parallel tool execution via asyncio.gather
- Retry with exponential backoff on 429 / 5xx / network errors
- Hard token budget (--token-budget) and per-call output cap (--max-tokens)
- Session-wide URL dedup + domain-weighted ranking + optional semantic dedup
- Pluggable search backends: tavily | serper | bing | google_cse | auto
- JSON output mode (--format json) via schema-validated synthesis pass
- Crash-safe persistence: --save-state PATH and --resume PATH
- Local report cache with 24h TTL (--no-cache to skip; --cache-ttl to tune)
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional
from urllib.parse import urlparse, urlunparse

import httpx
from dotenv import load_dotenv
from openai import APIError, AsyncOpenAI, OpenAI, RateLimitError

import prompts as _prompts
import stats as _stats
from _log import configure_logging, log

load_dotenv()

# Optional callback signature: receives an event dict, returns awaitable or None.
EventCallback = Optional[Callable[[dict], Awaitable[None]]]

_client: Optional[AsyncOpenAI] = None


def _get_client() -> AsyncOpenAI:
    """Lazy so --help and tests don't require DEEPSEEK_API_KEY at import time."""
    global _client
    if _client is None:
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            sys.exit("Missing DEEPSEEK_API_KEY (https://platform.deepseek.com)")
        _client = AsyncOpenAI(
            api_key=key,
            base_url="https://api.deepseek.com",
            timeout=120.0,
            max_retries=2,
        )
    return _client


# ===========================================================================
# 1. Generic retry helper for blocking I/O
# ===========================================================================

def _retry_blocking(fn, *, attempts: int = 5, base_delay: float = 1.0, what: str = "request"):
    last_exc: Optional[Exception] = None
    for i in range(attempts):
        try:
            return fn()
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status != 429 and status < 500:
                raise
            last_exc = e
        except (httpx.TimeoutException, httpx.NetworkError) as e:
            last_exc = e
        if i < attempts - 1:
            wait = base_delay * (2 ** i)
            log.warning("retry.scheduled", what=what, attempt=i + 1, wait_s=wait, error=str(last_exc))
            time.sleep(wait)
    assert last_exc is not None
    raise last_exc


# ===========================================================================
# 2. URL canonicalisation, dedup, and domain-weighted ranking
# ===========================================================================

_SEEN_URLS: set[str] = set()

DOMAIN_WEIGHTS: dict[str, int] = {
    "arxiv.org": 4,
    "ietf.org": 4,
    "w3.org": 4,
    "developer.mozilla.org": 4,
    "docs.python.org": 4,
    "kernel.org": 4,
    "rust-lang.org": 4,
    "go.dev": 4,
    "github.com": 3,
    "gitlab.com": 3,
    "openai.com": 3,
    "anthropic.com": 3,
    "deepseek.com": 3,
    "api-docs.deepseek.com": 4,
    "wikipedia.org": 1,
    "reuters.com": 2,
    "bloomberg.com": 2,
    "ft.com": 2,
    "theverge.com": 1,
    "techcrunch.com": 1,
    "medium.com": -1,
    "dev.to": -1,
    "geeksforgeeks.org": -2,
    "tutorialspoint.com": -3,
    "w3schools.com": -2,
    "javatpoint.com": -3,
    "stackoverflow.com": 1,
}


def _canonical_url(url: str) -> str:
    try:
        p = urlparse(url)
    except ValueError:
        return url
    host = (p.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = p.path.rstrip("/") or "/"
    return urlunparse((p.scheme.lower() or "https", host, path, "", "", ""))


def _domain_score(url: str) -> int:
    host = (urlparse(url).hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if host.endswith(".gov") or host.endswith(".edu"):
        return 5
    for domain, weight in DOMAIN_WEIGHTS.items():
        if host == domain or host.endswith("." + domain):
            return weight
    return 0


def _dedup_and_rank(hits: list[dict], take: int) -> tuple[list[dict], int]:
    deduped: list[dict] = []
    filtered = 0
    for h in hits:
        url = h.get("url") or ""
        if not url:
            continue
        canon = _canonical_url(url)
        if canon in _SEEN_URLS:
            filtered += 1
            continue
        _SEEN_URLS.add(canon)
        deduped.append(h)
    deduped.sort(key=lambda h: -_domain_score(h.get("url", "")))
    return deduped[:take], filtered


# ===========================================================================
# 3. Optional semantic (embedding) dedup
# ===========================================================================

_SEEN_EMBEDDINGS: list[list[float]] = []
_embedding_client: Optional[OpenAI] = None
SEMANTIC_DEDUP = False
SEMANTIC_THRESHOLD = 0.85


def _ensure_embedding_client() -> OpenAI:
    global _embedding_client
    if _embedding_client is None:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY required for --semantic-dedup "
                "(uses text-embedding-3-small, ~$0.02/1M tokens)"
            )
        _embedding_client = OpenAI(api_key=key)
    return _embedding_client


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def _embed_batch(texts: list[str]) -> list[list[float]]:
    cli = _ensure_embedding_client()
    resp = cli.embeddings.create(model="text-embedding-3-small", input=texts)
    return [d.embedding for d in resp.data]


def _semantic_filter(hits: list[dict], threshold: float) -> tuple[list[dict], int]:
    if not hits:
        return [], 0
    texts = [
        ((h.get("title", "") + " " + h.get("snippet", ""))[:500]) or "(empty)"
        for h in hits
    ]
    try:
        embeddings = _embed_batch(texts)
    except Exception as e:
        log.warning("semantic_dedup.skipped", error=str(e))
        return hits, 0
    kept: list[dict] = []
    filtered = 0
    for hit, emb in zip(hits, embeddings):
        is_dup = any(_cosine(emb, prior) >= threshold for prior in _SEEN_EMBEDDINGS)
        if is_dup:
            filtered += 1
            continue
        _SEEN_EMBEDDINGS.append(emb)
        kept.append(hit)
    return kept, filtered


# ===========================================================================
# 4. Search providers — each returns a list of {title, url, snippet}
# ===========================================================================

def _search_tavily(query: str, max_results: int) -> list[dict]:
    from tavily import TavilyClient
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        raise RuntimeError("TAVILY_API_KEY missing — see https://tavily.com")
    tc = TavilyClient(api_key=key)
    resp = _retry_blocking(
        lambda: tc.search(query=query, max_results=max_results, search_depth="advanced"),
        what="tavily",
    )
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": (r.get("content") or "")[:600],
        }
        for r in resp.get("results", [])
    ]


def _search_serper(query: str, max_results: int) -> list[dict]:
    key = os.environ.get("SERPER_API_KEY")
    if not key:
        raise RuntimeError("SERPER_API_KEY missing — see https://serper.dev")

    def _call() -> dict:
        r = httpx.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": key, "Content-Type": "application/json"},
            json={"q": query, "num": max_results},
            timeout=20,
        )
        r.raise_for_status()
        return r.json()

    data = _retry_blocking(_call, what="serper")
    return [
        {
            "title": h.get("title", ""),
            "url": h.get("link", ""),
            "snippet": (h.get("snippet") or "")[:600],
        }
        for h in data.get("organic", [])[:max_results]
    ]


def _search_bing(query: str, max_results: int) -> list[dict]:
    key = os.environ.get("BING_API_KEY")
    if not key:
        raise RuntimeError("BING_API_KEY missing — get one in the Azure portal")

    def _call() -> dict:
        r = httpx.get(
            "https://api.bing.microsoft.com/v7.0/search",
            headers={"Ocp-Apim-Subscription-Key": key},
            params={"q": query, "count": max_results, "responseFilter": "Webpages"},
            timeout=20,
        )
        r.raise_for_status()
        return r.json()

    data = _retry_blocking(_call, what="bing")
    pages = data.get("webPages", {}).get("value", [])
    return [
        {
            "title": p.get("name", ""),
            "url": p.get("url", ""),
            "snippet": (p.get("snippet") or "")[:600],
        }
        for p in pages
    ]


def _search_google_cse(query: str, max_results: int) -> list[dict]:
    key = os.environ.get("GOOGLE_API_KEY")
    cx = os.environ.get("GOOGLE_CSE_ID")
    if not (key and cx):
        raise RuntimeError("GOOGLE_API_KEY and GOOGLE_CSE_ID required")

    def _call() -> dict:
        r = httpx.get(
            "https://www.googleapis.com/customsearch/v1",
            params={"key": key, "cx": cx, "q": query, "num": min(max_results, 10)},
            timeout=20,
        )
        r.raise_for_status()
        return r.json()

    data = _retry_blocking(_call, what="google_cse")
    return [
        {
            "title": i.get("title", ""),
            "url": i.get("link", ""),
            "snippet": (i.get("snippet") or "")[:600],
        }
        for i in data.get("items", [])
    ]


SEARCH_PROVIDERS = {
    "tavily": _search_tavily,
    "serper": _search_serper,
    "bing": _search_bing,
    "google_cse": _search_google_cse,
}

PROVIDER_NAME = os.environ.get("SEARCH_PROVIDER", "tavily")


def detect_provider() -> str:
    """Pick the first provider whose key(s) are set in env."""
    if os.environ.get("TAVILY_API_KEY"):
        return "tavily"
    if os.environ.get("SERPER_API_KEY"):
        return "serper"
    if os.environ.get("BING_API_KEY"):
        return "bing"
    if os.environ.get("GOOGLE_API_KEY") and os.environ.get("GOOGLE_CSE_ID"):
        return "google_cse"
    raise RuntimeError(
        "No search provider keys found. Set one of: TAVILY_API_KEY, "
        "SERPER_API_KEY, BING_API_KEY, or GOOGLE_API_KEY + GOOGLE_CSE_ID."
    )


def search_web(query: str, max_results: int) -> dict:
    fn = SEARCH_PROVIDERS.get(PROVIDER_NAME)
    if fn is None:
        raise RuntimeError(f"Unknown SEARCH_PROVIDER={PROVIDER_NAME!r}")

    take = min(max_results, 10)
    raw = fn(query, min(take * 2, 10))
    ranked, url_filtered = _dedup_and_rank(raw, take * 2)

    if SEMANTIC_DEDUP:
        ranked, sem_filtered = _semantic_filter(ranked, SEMANTIC_THRESHOLD)
    else:
        sem_filtered = 0

    ranked = ranked[:take]
    out: dict = {"results": ranked}
    notes: list[str] = []
    if url_filtered:
        notes.append(f"{url_filtered} URL duplicate(s) filtered")
    if sem_filtered:
        notes.append(f"{sem_filtered} semantic duplicate(s) filtered")
    if notes:
        out["note"] = "; ".join(notes)
    return out


def fetch_url(url: str) -> dict:
    """Fetch URL body. Auto-detects PDFs (Content-Type or .pdf suffix) and
    extracts text via pypdf so the model gets readable content instead of
    binary garbage. Cap text at 20K chars and PDF pages at 50.
    """
    def _call() -> dict:
        r = httpx.get(
            url,
            timeout=30,
            follow_redirects=True,
            headers={"User-Agent": "deep-research-agent/1.0"},
        )
        ctype = r.headers.get("content-type", "").lower()
        is_pdf = "application/pdf" in ctype or url.lower().split("?", 1)[0].endswith(".pdf")
        if is_pdf:
            return _extract_pdf_text(url, r)
        return {"url": url, "status": r.status_code,
                "content_type": ctype, "text": r.text[:20000]}

    try:
        return _retry_blocking(_call, what=f"fetch {url[:60]}")
    except Exception as e:
        return {"error": str(e), "url": url}


def _extract_pdf_text(url: str, response: httpx.Response) -> dict:
    try:
        from io import BytesIO
        from pypdf import PdfReader
    except ImportError:
        return {
            "url": url, "status": response.status_code, "is_pdf": True,
            "error": "pypdf not installed; pip install 'pypdf>=4.0' to read PDFs",
        }
    try:
        reader = PdfReader(BytesIO(response.content))
        pages = reader.pages
        n = len(pages)
        chunks = []
        for i, page in enumerate(pages[:50]):
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            if txt:
                chunks.append(f"[page {i + 1}]\n{txt}")
        body = "\n\n".join(chunks)[:20000]
        return {
            "url": url, "status": response.status_code, "is_pdf": True,
            "pages": n, "pages_extracted": min(n, 50), "text": body,
        }
    except Exception as e:
        return {"url": url, "status": response.status_code, "is_pdf": True,
                "error": f"PDF parse failed: {e}"}


# ===========================================================================
# 5. Tool schemas + prompts
# ===========================================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Search the public web. Returns a list of {title, url, snippet}. "
                "Issue multiple targeted queries instead of one broad query. "
                "When you have several independent searches, emit them all in "
                "one assistant turn — they execute in parallel. Results are "
                "deduplicated against URLs already seen in this session and "
                "ranked by source quality."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": (
                "Fetch a URL's body text (capped at 20K chars). "
                "Use only when a search snippet is too short to answer a "
                "sub-question. Multiple fetch_url calls in the same turn run "
                "in parallel."
            ),
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
]


# Prompts now live in prompts/{version}/*.md and are loaded per-session. The
# loaded dict is passed through research() / _writer_synthesis() / synthesize_json()
# so changing --prompts vN doesn't affect concurrent calls.


# ===========================================================================
# 6. Persistence — save/load session state
# ===========================================================================

@dataclass
class SessionState:
    question: str
    messages: list[dict] = field(default_factory=list)
    seen_urls: list[str] = field(default_factory=list)
    embeddings: list[list[float]] = field(default_factory=list)
    total_input: int = 0
    total_output: int = 0
    total_cached: int = 0
    iter: int = 0
    provider: str = ""
    thinking: bool = False
    semantic_dedup: bool = False
    planner_model: str = "deepseek-v4-flash"
    writer_model: str = "deepseek-v4-pro"
    prompts_version: str = "v1"
    lang: str = "en"


def save_state(path: str, state: SessionState) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(asdict(state), f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_state(path: str) -> SessionState:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return SessionState(**data)


def _restore_session_globals(state: SessionState) -> None:
    global _SEEN_URLS, _SEEN_EMBEDDINGS, PROVIDER_NAME, SEMANTIC_DEDUP
    _SEEN_URLS = set(state.seen_urls)
    _SEEN_EMBEDDINGS = list(state.embeddings)
    if state.provider:
        PROVIDER_NAME = state.provider
    SEMANTIC_DEDUP = state.semantic_dedup


# ===========================================================================
# 7. Turn execution — streaming + non-streaming, retry-wrapped
# ===========================================================================

@dataclass
class TurnResult:
    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)  # [{id, name, arguments}]
    reasoning_content: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0

    def to_assistant_message(self) -> dict:
        d: dict = {"role": "assistant"}
        if self.content:
            d["content"] = self.content
        if self.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                }
                for tc in self.tool_calls
            ]
        # DeepSeek (April 2026+) REQUIRES reasoning_content on assistant turns
        # when thinking mode is active. Older docs said the opposite. The
        # extra tokens DO eat KV-cache wins; if v4-flash adds an opt-out
        # later, gate this on `state.thinking` instead of always-include.
        if self.reasoning_content:
            d["reasoning_content"] = self.reasoning_content
        return d


async def _execute_turn_blocking(**kwargs) -> TurnResult:
    cli = _get_client()
    resp = await _create_with_retry(cli, **kwargs)
    msg = resp.choices[0].message

    tool_calls: list[dict] = []
    if msg.tool_calls:
        for tc in msg.tool_calls:
            tool_calls.append(
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments or "",
                }
            )

    cached = 0
    if resp.usage:
        details = getattr(resp.usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", 0) if details else 0
        if cached is None:
            cached = 0

    return TurnResult(
        content=msg.content or "",
        tool_calls=tool_calls,
        reasoning_content=getattr(msg, "reasoning_content", None) or "",
        prompt_tokens=resp.usage.prompt_tokens if resp.usage else 0,
        completion_tokens=resp.usage.completion_tokens if resp.usage else 0,
        cached_tokens=cached,
    )


async def _execute_turn_streaming(*, label: str, **kwargs) -> TurnResult:
    """Stream content to stdout (and reasoning to stderr) as it arrives.

    `label` is used as a header before reasoning blocks, e.g. "iter 3".
    """
    cli = _get_client()
    kwargs = {**kwargs, "stream": True, "stream_options": {"include_usage": True}}

    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls_acc: dict[int, dict] = {}
    usage = None
    reasoning_started = False

    stream = await _create_stream_with_retry(cli, **kwargs)
    async for chunk in stream:
        if chunk.choices:
            delta = chunk.choices[0].delta
            content_piece = getattr(delta, "content", None)
            if content_piece:
                content_parts.append(content_piece)
                print(content_piece, end="", flush=True)
            reasoning_piece = getattr(delta, "reasoning_content", None)
            if reasoning_piece:
                if not reasoning_started:
                    print(f"\n[{label} reasoning]\n", end="", flush=True, file=sys.stderr)
                    reasoning_started = True
                reasoning_parts.append(reasoning_piece)
                print(reasoning_piece, end="", flush=True, file=sys.stderr)
            tcd_list = getattr(delta, "tool_calls", None)
            if tcd_list:
                for tcd in tcd_list:
                    idx = getattr(tcd, "index", 0)
                    slot = tool_calls_acc.setdefault(
                        idx, {"id": "", "name": "", "arguments": ""}
                    )
                    if getattr(tcd, "id", None):
                        slot["id"] = tcd.id
                    fn = getattr(tcd, "function", None)
                    if fn is not None:
                        if getattr(fn, "name", None):
                            slot["name"] += fn.name
                        if getattr(fn, "arguments", None):
                            slot["arguments"] += fn.arguments
        if getattr(chunk, "usage", None):
            usage = chunk.usage

    if content_parts:
        print(flush=True)  # newline after streamed body
    if reasoning_started:
        print("\n[/reasoning]", file=sys.stderr, flush=True)

    cached = 0
    if usage:
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", 0) if details else 0
        if cached is None:
            cached = 0

    return TurnResult(
        content="".join(content_parts),
        tool_calls=[tool_calls_acc[k] for k in sorted(tool_calls_acc)],
        reasoning_content="".join(reasoning_parts),
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
        cached_tokens=cached,
    )


async def _create_with_retry(cli, **kwargs):
    attempts = 4
    base_delay = 2.0
    last_exc: Optional[Exception] = None
    for i in range(attempts):
        try:
            return await cli.chat.completions.create(**kwargs)
        except RateLimitError as e:
            last_exc = e
        except APIError as e:
            status = getattr(e, "status_code", None)
            if status and 500 <= status < 600:
                last_exc = e
            else:
                raise
        if i < attempts - 1:
            wait = base_delay * (2 ** i)
            log.warning("retry.scheduled", what="model", attempt=i + 1, wait_s=wait, error=str(last_exc))
            await asyncio.sleep(wait)
    assert last_exc is not None
    raise last_exc


async def _create_stream_with_retry(cli, **kwargs):
    """Same as _create_with_retry but for stream=True calls."""
    attempts = 4
    base_delay = 2.0
    last_exc: Optional[Exception] = None
    for i in range(attempts):
        try:
            return await cli.chat.completions.create(**kwargs)
        except RateLimitError as e:
            last_exc = e
        except APIError as e:
            status = getattr(e, "status_code", None)
            if status and 500 <= status < 600:
                last_exc = e
            else:
                raise
        if i < attempts - 1:
            wait = base_delay * (2 ** i)
            log.warning("retry.scheduled", what="stream", attempt=i + 1, wait_s=wait, error=str(last_exc))
            await asyncio.sleep(wait)
    assert last_exc is not None
    raise last_exc


# ===========================================================================
# 8. Async tool runner
# ===========================================================================

async def run_tool_async(name: str, args: dict) -> str:
    if name == "search_web":
        try:
            result = await asyncio.to_thread(
                search_web, args["query"], int(args.get("max_results", 5))
            )
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)
    if name == "fetch_url":
        result = await asyncio.to_thread(fetch_url, args["url"])
        return json.dumps(result, ensure_ascii=False)
    return json.dumps({"error": f"unknown tool {name}"})


# ===========================================================================
# 9. Cache (24h default TTL)
# ===========================================================================

DEFAULT_CACHE_DIR = Path(
    os.environ.get("DEEP_RESEARCH_CACHE", str(Path.home() / ".cache" / "deep-research"))
)
DEFAULT_CACHE_TTL = 24 * 3600


def _cache_key(
    question: str, planner: str, writer: str, provider: str, fmt: str,
    lang: str = "en", prompts_version: str = "v1",
) -> str:
    raw = f"{question}|{planner}|{writer}|{provider}|{fmt}|{lang}|{prompts_version}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def cache_get(key: str, ttl: int, cache_dir: Path) -> Optional[str]:
    path = cache_dir / f"{key}.json"
    if not path.exists():
        return None
    age = time.time() - path.stat().st_mtime
    if age > ttl:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("report")
    except (OSError, json.JSONDecodeError):
        return None


def cache_put(key: str, report: str, fmt: str, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.json"
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"report": report, "format": fmt, "ts": time.time()}, f)
    os.replace(tmp, path)


# ===========================================================================
# 10. Research loop — multi-model planner/writer + streaming
# ===========================================================================

async def _emit(cb: EventCallback, event: dict) -> None:
    if cb is None:
        return
    result = cb(event)
    if asyncio.iscoroutine(result):
        await result


async def research(
    state: SessionState,
    *,
    max_iters: int,
    max_tokens: Optional[int],
    token_budget: Optional[int],
    save_path: Optional[str],
    stream: bool = False,
    event_callback: EventCallback = None,
) -> str:
    """Drive the search/synthesis loop.

    Uses `state.planner_model` for tool-calling rounds. When the planner
    finishes (no more tool_calls) and `state.writer_model` differs, runs a
    final synthesis pass with the writer model.

    `event_callback` (sync or async) receives structured progress events:
        {"type": "iter.completed", "iter": 1, "tokens_in": ..., ...}
        {"type": "tool.started",   "name": "search_web", "args": {...}}
        {"type": "tool.completed", "name": "search_web", "result": {...}}
        {"type": "writer.started"}
        {"type": "done", "report": "..."}
    """
    extra_body = {"thinking": {"type": "enabled"}} if state.thinking else None
    forced_synthesis = False
    prompt_set = _prompts.load(state.prompts_version, lang=state.lang)

    await _emit(event_callback, {"type": "session.started",
                                 "question": state.question,
                                 "planner": state.planner_model,
                                 "writer": state.writer_model,
                                 "lang": state.lang,
                                 "prompts_version": state.prompts_version})

    if not state.messages:
        state.messages = [
            {"role": "system", "content": prompt_set["system"]},
            {"role": "user", "content": state.question},
        ]

    while state.iter < max_iters:
        i = state.iter
        kwargs: dict[str, Any] = dict(
            model=state.planner_model,
            messages=state.messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.2,
        )
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if extra_body:
            kwargs["extra_body"] = extra_body

        # Force synthesis if any of: budget at 85% used, or this is the last
        # allowed iter (so we never bail with the abort message when there's
        # evidence to summarise).
        if not forced_synthesis:
            used = state.total_input + state.total_output
            force_reason: Optional[str] = None
            if token_budget and used >= int(token_budget * 0.85):
                force_reason = "budget"
            elif max_iters >= 2 and state.iter == max_iters - 1:
                force_reason = "last_iter"
            if force_reason:
                log.info("iter.forcing_synthesis", reason=force_reason,
                         iter=state.iter + 1, used=used, budget=token_budget)
                await _emit(event_callback, {"type": "budget.forced",
                                             "reason": force_reason,
                                             "used": used, "budget": token_budget})
                state.messages.append({"role": "user", "content": prompt_set["force_synthesis"]})
                kwargs["tool_choice"] = "none"
                forced_synthesis = True

        if stream:
            turn = await _execute_turn_streaming(label=f"iter {i + 1}", **kwargs)
        else:
            turn = await _execute_turn_blocking(**kwargs)

        state.total_input += turn.prompt_tokens
        state.total_output += turn.completion_tokens
        state.total_cached += turn.cached_tokens

        log.info(
            "iter.completed",
            iter=i + 1,
            model=state.planner_model,
            tokens_in=turn.prompt_tokens,
            cached=turn.cached_tokens,
            tokens_out=turn.completion_tokens,
            running_total=state.total_input + state.total_output,
            budget=token_budget,
        )
        await _emit(event_callback, {
            "type": "iter.completed",
            "iter": i + 1,
            "model": state.planner_model,
            "tokens_in": turn.prompt_tokens,
            "cached": turn.cached_tokens,
            "tokens_out": turn.completion_tokens,
            "running_total": state.total_input + state.total_output,
        })

        if state.thinking and turn.reasoning_content and not stream:
            log.info("iter.reasoning", iter=i + 1, reasoning=turn.reasoning_content)

        state.messages.append(turn.to_assistant_message())
        state.iter += 1

        if save_path:
            state.seen_urls = list(_SEEN_URLS)
            state.embeddings = list(_SEEN_EMBEDDINGS)
            save_state(save_path, state)

        if not turn.tool_calls:
            if state.writer_model and state.writer_model != state.planner_model:
                return await _writer_synthesis(
                    state,
                    extra_body=extra_body,
                    max_tokens=max_tokens,
                    stream=stream,
                    event_callback=event_callback,
                    writer_nudge=prompt_set["writer_nudge"],
                )
            await _emit(event_callback, {"type": "done", "report": turn.content})
            return turn.content

        log.info("iter.dispatching", iter=i + 1, tool_calls=len(turn.tool_calls))

        async def _run_one(tc: dict) -> tuple[str, str]:
            try:
                args = json.loads(tc["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            log.info("tool.started", name=tc["name"], args=args)
            await _emit(event_callback, {"type": "tool.started",
                                         "name": tc["name"], "args": args})
            result = await run_tool_async(tc["name"], args)
            await _emit(event_callback, {"type": "tool.completed",
                                         "name": tc["name"], "result_size": len(result)})
            return tc["id"], result

        results = await asyncio.gather(*(_run_one(tc) for tc in turn.tool_calls))
        for tool_call_id, result in results:
            state.messages.append(
                {"role": "tool", "tool_call_id": tool_call_id, "content": result}
            )

        if save_path:
            state.seen_urls = list(_SEEN_URLS)
            state.embeddings = list(_SEEN_EMBEDDINGS)
            save_state(save_path, state)

        # Budget check AFTER tool messages are appended, so the message history
        # is well-formed (assistant-with-tool_calls is paired with tool results)
        # before we kick off a final synthesis pass.
        if token_budget and (state.total_input + state.total_output) >= token_budget:
            log.info("budget.post_iter_force_synthesis",
                     used=state.total_input + state.total_output)
            state.messages.append({"role": "user", "content": prompt_set["force_synthesis"]})
            forced_kwargs: dict[str, Any] = dict(
                model=state.planner_model,
                messages=state.messages,
                tool_choice="none",
                temperature=0.2,
            )
            if max_tokens:
                forced_kwargs["max_tokens"] = max_tokens
            if extra_body:
                forced_kwargs["extra_body"] = extra_body
            try:
                if stream:
                    forced_turn = await _execute_turn_streaming(
                        label="budget-synthesis", **forced_kwargs)
                else:
                    forced_turn = await _execute_turn_blocking(**forced_kwargs)
                state.total_input += forced_turn.prompt_tokens
                state.total_output += forced_turn.completion_tokens
                state.total_cached += forced_turn.cached_tokens
                final = (forced_turn.content
                         + f"\n\n[budget exhausted; synthesised from partial evidence at "
                         f"{state.total_input + state.total_output} tokens]")
            except Exception as e:
                log.warning("budget.post_iter_synth_failed", error=str(e))
                final = (turn.content
                         + f"\n\n[budget exhausted at {state.total_input + state.total_output} tokens]")
            await _emit(event_callback, {"type": "done", "report": final,
                                         "budget_exhausted": True})
            return final

    final = "[research aborted: max iterations reached]"
    await _emit(event_callback, {"type": "done", "report": final, "max_iters_hit": True})
    return final


async def _writer_synthesis(
    state: SessionState,
    *,
    extra_body: Optional[dict],
    max_tokens: Optional[int],
    stream: bool,
    event_callback: EventCallback = None,
    writer_nudge: Optional[str] = None,
) -> str:
    """Final synthesis pass using the writer model on the same evidence."""
    log.info("writer.started", model=state.writer_model)
    await _emit(event_callback, {"type": "writer.started", "model": state.writer_model})

    if writer_nudge is None:
        writer_nudge = _prompts.load(state.prompts_version, lang=state.lang)["writer_nudge"]

    base = state.messages[:-1] if state.messages and state.messages[-1].get("role") == "assistant" else list(state.messages)
    msgs = base + [{"role": "user", "content": writer_nudge}]

    kwargs: dict[str, Any] = dict(
        model=state.writer_model,
        messages=msgs,
        tool_choice="none",
        temperature=0.2,
    )
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    if extra_body:
        kwargs["extra_body"] = extra_body

    if stream:
        turn = await _execute_turn_streaming(label="writer", **kwargs)
    else:
        turn = await _execute_turn_blocking(**kwargs)

    state.total_input += turn.prompt_tokens
    state.total_output += turn.completion_tokens
    state.total_cached += turn.cached_tokens
    log.info(
        "writer.completed",
        model=state.writer_model,
        tokens_in=turn.prompt_tokens,
        cached=turn.cached_tokens,
        tokens_out=turn.completion_tokens,
    )
    await _emit(event_callback, {"type": "done", "report": turn.content})
    return turn.content


# ===========================================================================
# 11. JSON synthesis (response_format=json_object)
# ===========================================================================

async def synthesize_json(
    markdown_report: str,
    original_question: str,
    model: str,
    *,
    prompts_version: str = "v1",
) -> str:
    cli = _get_client()
    schema_prompt = _prompts.load(prompts_version, lang="en")["json_schema"]
    resp = await _create_with_retry(
        cli,
        model=model,
        messages=[
            {"role": "system", "content": schema_prompt},
            {
                "role": "user",
                "content": (
                    f"Original question: {original_question}\n\n"
                    f"Report:\n{markdown_report}"
                ),
            },
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw)
        return json.dumps(parsed, ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        print("[json synthesis returned invalid JSON; emitting raw]", file=sys.stderr)
        return raw


# ===========================================================================
# 12. CLI
# ===========================================================================

def main() -> None:
    global PROVIDER_NAME, SEMANTIC_DEDUP, SEMANTIC_THRESHOLD

    ap = argparse.ArgumentParser(
        description="Deep research agent backed by DeepSeek"
    )
    ap.add_argument(
        "question",
        nargs="?",
        help="The research question (quote it). Omit when using --resume.",
    )
    ap.add_argument("--max-iters", type=int, default=12)
    ap.add_argument(
        "--planner",
        default="deepseek-v4-flash",
        help="Model for tool-calling rounds (default: cheap flash)",
    )
    ap.add_argument(
        "--writer",
        default="deepseek-v4-pro",
        help="Model for the final synthesis pass (default: premium pro). "
             "Set equal to --planner to skip the writer pass.",
    )
    ap.add_argument(
        "--thinking",
        action="store_true",
        help="Enable thinking mode (reasoning_content surfaced to stderr)",
    )
    ap.add_argument(
        "--stream",
        action="store_true",
        help="Stream content to stdout / reasoning to stderr as it arrives",
    )
    ap.add_argument(
        "--provider",
        choices=list(SEARCH_PROVIDERS) + ["auto"],
        default=os.environ.get("SEARCH_PROVIDER", "auto"),
    )
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--token-budget", type=int, default=None)
    ap.add_argument("--semantic-dedup", action="store_true")
    ap.add_argument("--semantic-threshold", type=float, default=0.85)
    ap.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
    )
    ap.add_argument("--save-state", metavar="PATH")
    ap.add_argument("--resume", metavar="PATH")
    ap.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help=f"Cache directory (default: {DEFAULT_CACHE_DIR})",
    )
    ap.add_argument("--cache-ttl", type=int, default=DEFAULT_CACHE_TTL)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument(
        "--lang",
        default=os.environ.get("DEEP_RESEARCH_LANG", "en"),
        help="Output language for the final report. Aliases: zh / cn → zh-CN. "
             "Tool calls remain in English; only the report is translated.",
    )
    ap.add_argument(
        "--prompts",
        default=os.environ.get("DEEP_RESEARCH_PROMPTS", "v1"),
        help="Prompt version directory under prompts/ (e.g. v1, v2).",
    )
    ap.add_argument(
        "--log",
        choices=["pretty", "json", "silent"],
        default=os.environ.get("DEEP_RESEARCH_LOG", "pretty"),
        help="Log format (pretty|json|silent). Default from DEEP_RESEARCH_LOG env.",
    )
    ap.add_argument("--no-stats", action="store_true",
                    help="Skip recording session stats to the SQLite stats DB.")
    ap.add_argument("--out", help="Also write final report to this file")
    args = ap.parse_args()

    configure_logging(args.log)
    cache_dir = Path(args.cache_dir)
    lang = _prompts.resolve_lang(args.lang)
    if lang not in _prompts.LANGUAGE_NAMES and lang != "en":
        log.warning("lang.unknown", code=lang, known=list(_prompts.LANGUAGE_NAMES))
    if args.prompts not in _prompts.list_versions():
        ap.error(f"--prompts {args.prompts!r} not found; available: {_prompts.list_versions()}")

    if args.resume:
        state = load_state(args.resume)
        _restore_session_globals(state)
        log.info("session.resumed", path=args.resume, question=state.question,
                 iter=state.iter, seen_urls=len(state.seen_urls))
        if args.question and args.question != state.question:
            log.warning("resume.question_mismatch",
                        cli=args.question, saved=state.question)
        cache_lookup = False
    else:
        if not args.question:
            ap.error("question is required unless --resume is used")
        provider = args.provider
        if provider == "auto":
            provider = detect_provider()
            log.info("provider.auto", provider=provider)
        PROVIDER_NAME = provider
        SEMANTIC_DEDUP = args.semantic_dedup
        SEMANTIC_THRESHOLD = args.semantic_threshold
        if SEMANTIC_DEDUP:
            _ensure_embedding_client()
        state = SessionState(
            question=args.question,
            provider=PROVIDER_NAME,
            thinking=args.thinking,
            semantic_dedup=SEMANTIC_DEDUP,
            planner_model=args.planner,
            writer_model=args.writer,
            prompts_version=args.prompts,
            lang=lang,
        )
        cache_lookup = not args.no_cache

    if cache_lookup:
        key = _cache_key(
            state.question, state.planner_model, state.writer_model,
            PROVIDER_NAME, args.format,
            lang=state.lang, prompts_version=state.prompts_version,
        )
        cached = cache_get(key, args.cache_ttl, cache_dir)
        if cached is not None:
            log.info("cache.hit", key=key, ttl=args.cache_ttl, dir=str(cache_dir))
            print(cached)
            if args.out:
                with open(args.out, "w", encoding="utf-8") as f:
                    f.write(cached)
                log.info("output.saved", path=args.out)
            return

    log.info(
        "session.starting",
        planner=state.planner_model,
        writer=state.writer_model,
        provider=PROVIDER_NAME,
        thinking=state.thinking,
        stream=args.stream,
        semantic_dedup=SEMANTIC_DEDUP,
        max_iters=args.max_iters,
        max_tokens=args.max_tokens,
        budget=args.token_budget,
        format=args.format,
        cache=not args.no_cache,
    )

    t0 = time.time()
    error: Optional[str] = None
    try:
        markdown = asyncio.run(
            research(
                state,
                max_iters=args.max_iters,
                max_tokens=args.max_tokens,
                token_budget=args.token_budget,
                save_path=args.save_state,
                stream=args.stream,
            )
        )

        if args.format == "json":
            log.info("synthesis.json_pass")
            report = asyncio.run(
                synthesize_json(
                    markdown, state.question, state.writer_model,
                    prompts_version=state.prompts_version,
                )
            )
        else:
            report = markdown
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        log.error("session.failed", error=error)
        if not args.no_stats:
            _record_stats(state, args, t0, ok=False, error=error)
        raise

    if not (args.stream and args.format == "markdown"):
        print(report)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report)
        log.info("output.saved", path=args.out)

    if not args.no_cache and not args.resume:
        key = _cache_key(
            state.question, state.planner_model, state.writer_model,
            PROVIDER_NAME, args.format,
            lang=state.lang, prompts_version=state.prompts_version,
        )
        cache_put(key, report, args.format, cache_dir)
        log.info("cache.stored", key=key, dir=str(cache_dir))

    if not args.no_stats:
        _record_stats(state, args, t0, ok=True)


def _record_stats(state: SessionState, args, t0: float, *, ok: bool,
                  error: Optional[str] = None) -> None:
    """Persist a row to the SQLite stats DB and log a 1-line summary."""
    duration = time.time() - t0
    cost_cny = _stats.estimate_cost_cny(
        state.planner_model,
        state.total_input,
        state.total_cached,
        state.total_output,
    )
    rec = _stats.SessionRecord(
        ts=_stats.now(),
        question=state.question,
        planner=state.planner_model,
        writer=state.writer_model,
        provider=state.provider or PROVIDER_NAME,
        iters=state.iter,
        tokens_in=state.total_input,
        tokens_out=state.total_output,
        cached_tokens=state.total_cached,
        cost_cny=cost_cny,
        duration_sec=duration,
        ok=ok,
        error=error,
    )
    try:
        row_id = _stats.record_session(rec)
        log.info("stats.recorded", id=row_id, duration_s=round(duration, 2),
                 cost_cny=round(cost_cny, 4),
                 cost_usd=round(cost_cny * _stats.usd_per_cny(), 4),
                 tokens_in=state.total_input, tokens_out=state.total_output, ok=ok)
    except Exception as e:
        log.warning("stats.record_failed", error=str(e))


if __name__ == "__main__":
    main()

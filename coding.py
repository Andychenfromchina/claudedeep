"""Coding helpers exposed via the MCP server.

Two tools:

- generate_code(): general code generation / refactor / explain / fix / test.
  Routed through DeepSeek v4-flash by default (12× cheaper than v4-pro);
  caller can override `model` for hard tasks.

- fim_complete(): DeepSeek's beta /completions endpoint with prefix + suffix.
  This is *not* exposed via standard chat completions — it requires the
  beta base URL and the legacy /completions verb. Cap is 4K output tokens.
"""
from __future__ import annotations

import os
import re
from typing import Optional

from openai import AsyncOpenAI

# Lazy beta client — only constructed if fim_complete is actually called.
_fim_client: Optional[AsyncOpenAI] = None


def _get_fim_client() -> AsyncOpenAI:
    global _fim_client
    if _fim_client is None:
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError("DEEPSEEK_API_KEY missing")
        _fim_client = AsyncOpenAI(
            api_key=key,
            base_url="https://api.deepseek.com/beta",
            timeout=120.0,
            max_retries=2,
        )
    return _fim_client


_FENCE_RE = re.compile(r"^```[\w-]*\n([\s\S]*?)\n```\s*$", re.MULTILINE)


def _strip_fences(text: str) -> str:
    """If output is a single fenced block, return its body. Otherwise return as-is."""
    text = text.strip()
    m = _FENCE_RE.match(text)
    return m.group(1) if m else text


_CODE_SYSTEM_PROMPT = """You are a senior engineer. Follow the user's task strictly.

Rules:
- Output ONLY the code (and any inline comments). No prose explanations
  before or after, unless the task explicitly asks for an explanation.
- If the task says "refactor" or "fix", return the COMPLETE modified file/snippet,
  not just the changed lines.
- Use idiomatic style for the language. Match existing indentation and naming
  conventions when given existing code.
- If you genuinely can't do the task (missing context, ambiguous), say so in
  one short comment at the top of your output, then do your best attempt.
- Do not wrap output in markdown fences (```). The orchestrator handles
  formatting. If you must clarify language, prefix with a single-line comment.
"""


async def generate_code(
    task: str,
    *,
    code: str = "",
    language: str = "",
    model: str = "deepseek-v4-flash",
    max_tokens: int = 2000,
) -> str:
    """General-purpose code generation.

    `task` is the natural-language instruction (e.g. "write a Python function
    that does X", "refactor this to use async/await", "add type hints",
    "explain what this code does").

    `code` is optional existing code the task should operate on.
    `language` is an optional hint (helps when generating from scratch).
    """
    # Reuse the regular chat client (lazy-init in deep_research).
    import deep_research as dr
    cli = dr._get_client()

    parts = [f"Task: {task}"]
    if language:
        parts.append(f"Language: {language}")
    if code:
        parts.append(f"Existing code:\n{code}")
    user_msg = "\n\n".join(parts)

    resp = await cli.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _CODE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=max_tokens,
        temperature=0.2,
    )
    out = resp.choices[0].message.content or ""
    return _strip_fences(out)


async def fim_complete(
    prefix: str,
    *,
    suffix: str = "",
    max_tokens: int = 1000,
) -> str:
    """Fill-in-the-middle completion via DeepSeek's beta endpoint.

    Useful for editor-style code completion where you have text BEFORE the
    cursor (`prefix`) and optionally AFTER (`suffix`), and want the model to
    generate the middle.

    Hard limit: 4096 output tokens (DeepSeek beta constraint). The default
    of 1000 is a safer choice for most completion scenarios.
    """
    if not prefix:
        raise ValueError("prefix is required (cannot be empty)")
    cli = _get_fim_client()
    resp = await cli.completions.create(
        model="deepseek-v4-pro",
        prompt=prefix,
        suffix=suffix or None,
        max_tokens=min(max_tokens, 4096),
        temperature=0,
    )
    return resp.choices[0].text or ""

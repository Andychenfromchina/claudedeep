"""End-to-end tests for the research loop using a fake DeepSeek client.

Run:  python -m unittest test_research_loop
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import patch

os.environ.setdefault("DEEPSEEK_API_KEY", "sk-test-stub")
os.environ.setdefault("TAVILY_API_KEY", "tvly-test-stub")

import deep_research as dr  # noqa: E402


# ---------------------------------------------------------------------------
# Fake OpenAI response objects — minimal surface the research loop needs.
# ---------------------------------------------------------------------------

@dataclass
class FakeFunction:
    name: str
    arguments: str


@dataclass
class FakeToolCall:
    id: str
    function: FakeFunction
    type: str = "function"


@dataclass
class FakeMessage:
    content: Optional[str] = None
    tool_calls: Optional[list[FakeToolCall]] = None
    reasoning_content: Optional[str] = None
    role: str = "assistant"


@dataclass
class FakeUsage:
    prompt_tokens: int = 1000
    completion_tokens: int = 200
    prompt_tokens_details: Any = field(
        default_factory=lambda: SimpleNamespace(cached_tokens=0)
    )


@dataclass
class FakeChoice:
    message: FakeMessage
    finish_reason: str = "stop"
    index: int = 0


@dataclass
class FakeResponse:
    choices: list[FakeChoice]
    usage: FakeUsage = field(default_factory=FakeUsage)


def make_resp(
    *,
    content: Optional[str] = None,
    tool_calls: Optional[list[FakeToolCall]] = None,
    reasoning: Optional[str] = None,
    prompt_tokens: int = 1000,
    completion_tokens: int = 100,
    cached: int = 0,
) -> FakeResponse:
    msg = FakeMessage(content=content, tool_calls=tool_calls, reasoning_content=reasoning)
    return FakeResponse(
        choices=[FakeChoice(message=msg)],
        usage=FakeUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
        ),
    )


def make_tool_call(call_id: str, name: str, **args) -> FakeToolCall:
    return FakeToolCall(
        id=call_id,
        function=FakeFunction(name=name, arguments=json.dumps(args)),
    )


class FakeAsyncOpenAI:
    """Drop-in for AsyncOpenAI that yields scripted responses."""

    def __init__(self, responses: list[FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise RuntimeError(
                f"FakeAsyncOpenAI exhausted on call #{len(self.calls)}: "
                f"model={kwargs.get('model')}"
            )
        return self._responses.pop(0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestResearchLoopE2E(unittest.TestCase):
    def setUp(self) -> None:
        dr._SEEN_URLS.clear()
        dr._SEEN_EMBEDDINGS.clear()
        dr._client = None

    def tearDown(self) -> None:
        dr._client = None

    def _run(self, state: dr.SessionState, **kwargs) -> str:
        defaults = dict(
            max_iters=5,
            max_tokens=None,
            token_budget=None,
            save_path=None,
            stream=False,
        )
        defaults.update(kwargs)
        return asyncio.run(dr.research(state, **defaults))

    def test_simple_search_then_synthesis(self) -> None:
        fake = FakeAsyncOpenAI([
            make_resp(tool_calls=[make_tool_call("c1", "search_web", query="deepseek pricing")]),
            make_resp(content="# Final report\n\nDone."),
        ])
        dr._client = fake

        with patch.object(
            dr, "search_web",
            return_value={"results": [{"url": "https://example.com", "title": "x", "snippet": "x"}]},
        ):
            state = dr.SessionState(
                question="Q",
                planner_model="deepseek-v4-flash",
                writer_model="deepseek-v4-flash",  # same → no writer pass
            )
            result = self._run(state)

        self.assertIn("Final report", result)
        self.assertEqual(state.iter, 2)
        self.assertEqual(len(fake.calls), 2)
        # All calls used the planner model.
        self.assertEqual(fake.calls[0]["model"], "deepseek-v4-flash")
        self.assertEqual(fake.calls[1]["model"], "deepseek-v4-flash")

    def test_parallel_tool_calls_in_one_turn(self) -> None:
        fake = FakeAsyncOpenAI([
            make_resp(tool_calls=[
                make_tool_call("c1", "search_web", query="q1"),
                make_tool_call("c2", "search_web", query="q2"),
                make_tool_call("c3", "fetch_url", url="https://a.example"),
            ]),
            make_resp(content="# Report"),
        ])
        dr._client = fake

        with patch.object(dr, "search_web", return_value={"results": []}), \
             patch.object(dr, "fetch_url", return_value={"text": "..."}):
            state = dr.SessionState(
                question="Q",
                planner_model="deepseek-v4-flash",
                writer_model="deepseek-v4-flash",
            )
            result = self._run(state)

        self.assertIn("Report", result)
        # system + user + assistant(tool_calls) + 3*tool + assistant(final) = 7
        self.assertEqual(len(state.messages), 7)

    def test_writer_polish_pass_runs_when_models_differ(self) -> None:
        fake = FakeAsyncOpenAI([
            make_resp(tool_calls=[make_tool_call("c1", "search_web", query="x")]),
            make_resp(content="# Planner draft (rough)"),
            make_resp(content="# Polished by writer\n\nReal answer here."),
        ])
        dr._client = fake

        with patch.object(dr, "search_web", return_value={"results": []}):
            state = dr.SessionState(
                question="Q",
                planner_model="deepseek-v4-flash",
                writer_model="deepseek-v4-pro",
            )
            result = self._run(state)

        self.assertIn("Polished by writer", result)
        self.assertNotIn("Planner draft", result)
        self.assertEqual(len(fake.calls), 3)
        self.assertEqual(fake.calls[0]["model"], "deepseek-v4-flash")
        self.assertEqual(fake.calls[1]["model"], "deepseek-v4-flash")
        self.assertEqual(fake.calls[2]["model"], "deepseek-v4-pro")
        # Writer pass uses tool_choice=none.
        self.assertEqual(fake.calls[2]["tool_choice"], "none")

    def test_token_budget_forces_synthesis_at_85_percent(self) -> None:
        # Budget 1000, threshold 850. Iter 1 spends 900 (≥ 850 but < 1000) so
        # iter 2 starts with the synthesis nudge + tool_choice="none".
        fake = FakeAsyncOpenAI([
            make_resp(
                tool_calls=[make_tool_call("c1", "search_web", query="x")],
                prompt_tokens=800, completion_tokens=100,  # cumulative 900
            ),
            make_resp(content="# Forced final", prompt_tokens=50, completion_tokens=20),
        ])
        dr._client = fake

        with patch.object(dr, "search_web", return_value={"results": []}):
            state = dr.SessionState(
                question="Q",
                planner_model="deepseek-v4-flash",
                writer_model="deepseek-v4-flash",
            )
            result = self._run(state, token_budget=1000)

        self.assertIn("Forced final", result)
        # The 2nd call is the forced-synthesis turn.
        self.assertEqual(fake.calls[1]["tool_choice"], "none")

    def test_max_iters_cap(self) -> None:
        # Mock returns tool_calls every turn (ignoring tool_choice). The
        # last-iter force still sets tool_choice=none on the runtime side,
        # but since the fake disregards it, we end up exhausting max_iters.
        fake = FakeAsyncOpenAI([
            make_resp(tool_calls=[make_tool_call(f"c{i}", "search_web", query="x")])
            for i in range(10)
        ])
        dr._client = fake

        with patch.object(dr, "search_web", return_value={"results": []}):
            state = dr.SessionState(
                question="Q",
                planner_model="deepseek-v4-flash",
                writer_model="deepseek-v4-flash",
            )
            result = self._run(state, max_iters=3)

        self.assertIn("max iterations reached", result)
        self.assertEqual(state.iter, 3)
        # Verify the last iter (index 2 of 0..2) was sent with tool_choice=none.
        self.assertEqual(fake.calls[2]["tool_choice"], "none")

    def test_last_iter_forces_synthesis_without_budget(self) -> None:
        # max_iters=2, no budget: on iter 2 (the last) we should force
        # tool_choice=none so the model emits content instead of more tools.
        # A real API would honour that; our mock just emits content, which is
        # what we want to verify gets returned as the report.
        fake = FakeAsyncOpenAI([
            make_resp(tool_calls=[make_tool_call("c1", "search_web", query="x")]),
            make_resp(content="# Final report (last iter)"),
        ])
        dr._client = fake

        with patch.object(dr, "search_web", return_value={"results": []}):
            state = dr.SessionState(
                question="Q",
                planner_model="deepseek-v4-flash",
                writer_model="deepseek-v4-flash",
            )
            result = self._run(state, max_iters=2)

        self.assertIn("Final report (last iter)", result)
        self.assertNotIn("max iterations reached", result)
        self.assertEqual(fake.calls[1]["tool_choice"], "none")
        # Iter 1 ran normally (auto), iter 2 was forced (none).
        self.assertEqual(fake.calls[0]["tool_choice"], "auto")

    def test_post_iter_budget_runs_synthesis_pass(self) -> None:
        # Iter 1 returns tool_calls; tools blow past the budget. Instead of
        # returning the empty turn.content + "[budget exhausted]" placeholder,
        # the loop should run one extra tool_choice=none synthesis and return
        # that content with a "synthesized from partial evidence" suffix.
        fake = FakeAsyncOpenAI([
            make_resp(
                tool_calls=[make_tool_call("c1", "search_web", query="x")],
                prompt_tokens=400, completion_tokens=100,  # cumulative 500
            ),
            # iter-1 budget check at the START is fine (used=0). Tools fire,
            # totals reach 500 ≥ budget=400 → post-iter synthesis kicks in.
            make_resp(content="# Synthesised from partial evidence",
                      prompt_tokens=100, completion_tokens=50),
        ])
        dr._client = fake

        with patch.object(dr, "search_web", return_value={"results": []}):
            state = dr.SessionState(
                question="Q",
                planner_model="deepseek-v4-flash",
                writer_model="deepseek-v4-flash",
            )
            result = self._run(state, max_iters=5, token_budget=400)

        self.assertIn("Synthesised from partial evidence", result)
        self.assertIn("budget exhausted", result)
        # Two model calls total: the tool-calling one + the forced synthesis.
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(fake.calls[1]["tool_choice"], "none")

    def test_save_state_called_each_iter(self) -> None:
        fake = FakeAsyncOpenAI([
            make_resp(tool_calls=[make_tool_call("c1", "search_web", query="x")]),
            make_resp(content="# Done"),
        ])
        dr._client = fake

        with tempfile.TemporaryDirectory() as tmp:
            state_path = os.path.join(tmp, "state.json")
            with patch.object(dr, "search_web", return_value={"results": []}):
                state = dr.SessionState(
                    question="Q",
                    planner_model="deepseek-v4-flash",
                    writer_model="deepseek-v4-flash",
                )
                self._run(state, save_path=state_path)
            self.assertTrue(os.path.exists(state_path))
            loaded = dr.load_state(state_path)
            self.assertEqual(loaded.question, "Q")
            self.assertEqual(loaded.iter, 2)
            self.assertGreater(len(loaded.messages), 2)


class TestCache(unittest.TestCase):
    def test_cache_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            key = dr._cache_key("Q", "flash", "pro", "tavily", "markdown")
            self.assertIsNone(dr.cache_get(key, 86400, cache_dir))
            dr.cache_put(key, "REPORT", "markdown", cache_dir)
            self.assertEqual(dr.cache_get(key, 86400, cache_dir), "REPORT")

    def test_cache_ttl_expiry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            key = dr._cache_key("Q", "flash", "pro", "tavily", "markdown")
            dr.cache_put(key, "OLD", "markdown", cache_dir)
            old = time.time() - 7200
            os.utime(cache_dir / f"{key}.json", (old, old))
            self.assertIsNone(dr.cache_get(key, 3600, cache_dir))

    def test_cache_key_changes_with_inputs(self) -> None:
        a = dr._cache_key("Q", "flash", "pro", "tavily", "markdown")
        b = dr._cache_key("Q", "flash", "pro", "tavily", "json")  # fmt differs
        c = dr._cache_key("Q2", "flash", "pro", "tavily", "markdown")  # question differs
        self.assertNotEqual(a, b)
        self.assertNotEqual(a, c)


if __name__ == "__main__":
    unittest.main()

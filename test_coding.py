"""Tests for coding.py — fence stripping and a thin mocked path for both tools."""
from __future__ import annotations

import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("DEEPSEEK_API_KEY", "sk-test-stub")

import coding  # noqa: E402
import deep_research as dr  # noqa: E402


def _make_chat_resp(content: str):
    msg = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=10,
                              prompt_tokens_details=SimpleNamespace(cached_tokens=0)),
    )


def _make_completions_resp(text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(text=text, finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=5),
    )


class FakeChatClient:
    """Drop-in for AsyncOpenAI exposing only chat.completions.create."""

    def __init__(self, response):
        self._resp = response
        self.last_kwargs = None
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    async def _create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._resp


class FakeFimClient:
    def __init__(self, response):
        self._resp = response
        self.last_kwargs = None
        self.completions = SimpleNamespace(create=self._create)

    async def _create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._resp


class TestFenceStripping(unittest.TestCase):
    def test_strips_simple_fence(self):
        self.assertEqual(coding._strip_fences("```\nhello\n```"), "hello")

    def test_strips_language_fence(self):
        self.assertEqual(coding._strip_fences("```python\nx = 1\n```"), "x = 1")

    def test_keeps_unfenced(self):
        self.assertEqual(coding._strip_fences("def foo(): pass"), "def foo(): pass")

    def test_keeps_partial_fence(self):
        # Only the very first/last lines need to be fences; mid-text fences stay.
        text = "Use ```python``` inline"
        self.assertEqual(coding._strip_fences(text), text)


class TestGenerateCode(unittest.TestCase):
    def setUp(self):
        dr._client = None

    def tearDown(self):
        dr._client = None

    def test_generate_code_passes_task_and_strips_fences(self):
        fake = FakeChatClient(_make_chat_resp("```python\ndef add(a, b):\n    return a + b\n```"))
        dr._client = fake

        out = asyncio.run(coding.generate_code(
            "write add function", language="python", max_tokens=100,
        ))
        self.assertEqual(out, "def add(a, b):\n    return a + b")
        self.assertEqual(fake.last_kwargs["model"], "deepseek-v4-flash")
        self.assertEqual(fake.last_kwargs["max_tokens"], 100)
        # User message should mention task + language
        msgs = fake.last_kwargs["messages"]
        self.assertEqual(len(msgs), 2)
        self.assertIn("write add function", msgs[1]["content"])
        self.assertIn("python", msgs[1]["content"])

    def test_generate_code_with_existing_code(self):
        fake = FakeChatClient(_make_chat_resp("def add(a: int, b: int) -> int:\n    return a + b"))
        dr._client = fake

        out = asyncio.run(coding.generate_code(
            "add type hints",
            code="def add(a, b):\n    return a + b",
            model="deepseek-v4-pro",
        ))
        self.assertIn("int", out)
        self.assertEqual(fake.last_kwargs["model"], "deepseek-v4-pro")
        self.assertIn("def add(a, b)", fake.last_kwargs["messages"][1]["content"])


class TestFimComplete(unittest.TestCase):
    def setUp(self):
        coding._fim_client = None

    def tearDown(self):
        coding._fim_client = None

    def test_fim_complete_passes_prefix_and_suffix(self):
        fake = FakeFimClient(_make_completions_resp("    return a + b\n"))
        coding._fim_client = fake

        out = asyncio.run(coding.fim_complete(
            prefix="def add(a, b):\n",
            suffix="\n# end of file\n",
            max_tokens=50,
        ))
        self.assertEqual(out, "    return a + b\n")
        self.assertEqual(fake.last_kwargs["model"], "deepseek-v4-pro")
        self.assertEqual(fake.last_kwargs["prompt"], "def add(a, b):\n")
        self.assertEqual(fake.last_kwargs["suffix"], "\n# end of file\n")
        self.assertEqual(fake.last_kwargs["max_tokens"], 50)

    def test_fim_caps_max_tokens_at_4096(self):
        fake = FakeFimClient(_make_completions_resp(""))
        coding._fim_client = fake
        asyncio.run(coding.fim_complete(prefix="x", max_tokens=99999))
        self.assertEqual(fake.last_kwargs["max_tokens"], 4096)

    def test_fim_empty_prefix_raises(self):
        with self.assertRaises(ValueError):
            asyncio.run(coding.fim_complete(prefix=""))

    def test_fim_empty_suffix_passes_none(self):
        fake = FakeFimClient(_make_completions_resp("body"))
        coding._fim_client = fake
        asyncio.run(coding.fim_complete(prefix="prefix"))
        self.assertIsNone(fake.last_kwargs["suffix"])


if __name__ == "__main__":
    unittest.main()

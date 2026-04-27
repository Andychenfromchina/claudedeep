"""Tests for prompts loader, language directive, PDF detection, and A/B helpers."""
from __future__ import annotations

import io
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("DEEPSEEK_API_KEY", "sk-test-stub")
os.environ.setdefault("TAVILY_API_KEY", "tvly-test-stub")

import deep_research as dr  # noqa: E402
import prompts as pr  # noqa: E402
import run_ab  # noqa: E402


class TestPromptsLoader(unittest.TestCase):
    def test_v1_has_all_required_files(self) -> None:
        out = pr.load("v1")
        for k in ("system", "writer_nudge", "force_synthesis", "json_schema", "judge"):
            self.assertIn(k, out)
            self.assertGreater(len(out[k]), 50, f"{k} suspiciously short")

    def test_unknown_version_raises(self) -> None:
        with self.assertRaises(ValueError):
            pr.load("v999")

    def test_list_versions_includes_v1(self) -> None:
        self.assertIn("v1", pr.list_versions())

    def test_with_language_appends_directive(self) -> None:
        base = pr.load("v1")["system"]
        zh = pr.with_language(base, "zh-CN")
        self.assertGreater(len(zh), len(base))
        self.assertIn("Simplified Chinese", zh)
        # Original prompt content is preserved up front.
        self.assertTrue(zh.startswith(base))

    def test_load_with_lang_zh_appends(self) -> None:
        base = pr.load("v1", lang="en")["system"]
        zh = pr.load("v1", lang="zh-CN")["system"]
        self.assertGreater(len(zh), len(base))

    def test_load_with_lang_en_unchanged(self) -> None:
        en1 = pr.load("v1")["system"]
        en2 = pr.load("v1", lang="en")["system"]
        self.assertEqual(en1, en2)

    def test_resolve_lang_aliases(self) -> None:
        self.assertEqual(pr.resolve_lang("zh"), "zh-CN")
        self.assertEqual(pr.resolve_lang("CN"), "zh-CN")
        self.assertEqual(pr.resolve_lang("chinese"), "zh-CN")
        self.assertEqual(pr.resolve_lang(""), "en")
        self.assertEqual(pr.resolve_lang(None), "en")
        self.assertEqual(pr.resolve_lang("zh-CN"), "zh-CN")  # already canonical
        self.assertEqual(pr.resolve_lang("en"), "en")


class TestCacheKeyIncludesLangAndPrompts(unittest.TestCase):
    def test_lang_changes_key(self) -> None:
        a = dr._cache_key("Q", "p", "w", "tavily", "markdown", lang="en")
        b = dr._cache_key("Q", "p", "w", "tavily", "markdown", lang="zh-CN")
        self.assertNotEqual(a, b)

    def test_prompts_version_changes_key(self) -> None:
        a = dr._cache_key("Q", "p", "w", "tavily", "markdown", prompts_version="v1")
        b = dr._cache_key("Q", "p", "w", "tavily", "markdown", prompts_version="v2")
        self.assertNotEqual(a, b)


class TestPDFFetch(unittest.TestCase):
    """fetch_url should auto-detect PDFs and pull text. We monkey-patch httpx.get."""

    def test_html_passthrough(self) -> None:
        fake_resp = SimpleNamespace(
            headers={"content-type": "text/html; charset=utf-8"},
            status_code=200,
            text="<html>hello world</html>",
            content=b"<html>hello world</html>",
        )
        import httpx as _httpx
        orig = _httpx.get
        _httpx.get = lambda *a, **kw: fake_resp
        try:
            out = dr.fetch_url("https://example.com/page")
        finally:
            _httpx.get = orig
        self.assertEqual(out["status"], 200)
        self.assertIn("hello world", out["text"])
        self.assertNotIn("is_pdf", out)

    def test_pdf_via_url_suffix_extracts_text(self) -> None:
        # Build a 1-page PDF in memory using pypdf.
        try:
            from pypdf import PdfWriter
        except ImportError:
            self.skipTest("pypdf not installed")
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        buf = io.BytesIO()
        writer.write(buf)
        pdf_bytes = buf.getvalue()

        fake_resp = SimpleNamespace(
            headers={"content-type": "application/pdf"},
            status_code=200, text="", content=pdf_bytes,
        )
        import httpx as _httpx
        orig = _httpx.get
        _httpx.get = lambda *a, **kw: fake_resp
        try:
            out = dr.fetch_url("https://arxiv.org/pdf/2401.0001.pdf")
        finally:
            _httpx.get = orig
        self.assertTrue(out.get("is_pdf"))
        self.assertEqual(out["status"], 200)
        self.assertEqual(out["pages"], 1)

    def test_pdf_detected_by_query_stripped_suffix(self) -> None:
        # URL has ?download=1 after .pdf — should still detect as PDF.
        try:
            from pypdf import PdfWriter
        except ImportError:
            self.skipTest("pypdf not installed")
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        buf = io.BytesIO()
        writer.write(buf)
        fake_resp = SimpleNamespace(
            headers={"content-type": "application/octet-stream"},
            status_code=200, text="", content=buf.getvalue(),
        )
        import httpx as _httpx
        orig = _httpx.get
        _httpx.get = lambda *a, **kw: fake_resp
        try:
            out = dr.fetch_url("https://example.com/paper.pdf?download=1")
        finally:
            _httpx.get = orig
        self.assertTrue(out.get("is_pdf"))


class TestABConfigLoading(unittest.TestCase):
    def test_load_config_fills_defaults(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"name": "tiny", "planner": "deepseek-v4-flash"}')
            path = f.name
        try:
            cfg = run_ab._load_config(Path(path))
        finally:
            os.unlink(path)
        self.assertEqual(cfg["name"], "tiny")
        self.assertEqual(cfg["planner"], "deepseek-v4-flash")
        # Defaults applied:
        self.assertEqual(cfg["writer"], "deepseek-v4-pro")
        self.assertEqual(cfg["max_iters"], 8)
        self.assertEqual(cfg["prompts"], "v1")
        self.assertEqual(cfg["lang"], "en")

    def test_load_config_uses_filename_as_default_name(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False,
                                         prefix="cheap-") as f:
            f.write('{"planner": "deepseek-v4-flash"}')
            path = f.name
        try:
            cfg = run_ab._load_config(Path(path))
        finally:
            os.unlink(path)
        self.assertTrue(cfg["name"].startswith("cheap-"))


class TestSlackSignature(unittest.TestCase):
    """Test slack signature verification at the helper level."""

    def setUp(self) -> None:
        self._old = os.environ.pop("SLACK_SIGNING_SECRET", None)

    def tearDown(self) -> None:
        if self._old is not None:
            os.environ["SLACK_SIGNING_SECRET"] = self._old
        else:
            os.environ.pop("SLACK_SIGNING_SECRET", None)

    def _verify(self, body: bytes, headers: dict) -> bool:
        # Defer import so we don't need fastapi installed for the rest of tests.
        try:
            import web  # noqa: F401
        except ImportError:
            self.skipTest("fastapi not installed")
        return web._verify_slack(body, headers)

    def test_rejects_when_no_secret_configured(self) -> None:
        # No env set → reject everything.
        result = self._verify(b"x", {"x-slack-request-timestamp": "0",
                                     "x-slack-signature": "v0=abc"})
        self.assertFalse(result)

    def test_accepts_valid_signature(self) -> None:
        import hashlib as _h
        import hmac as _hm
        import time as _t
        os.environ["SLACK_SIGNING_SECRET"] = "shhh"
        ts = str(int(_t.time()))
        body = b"token=x&command=/research"
        base = f"v0:{ts}:".encode() + body
        sig = "v0=" + _hm.new(b"shhh", base, _h.sha256).hexdigest()
        headers = {"x-slack-request-timestamp": ts, "x-slack-signature": sig}
        self.assertTrue(self._verify(body, headers))

    def test_rejects_old_timestamp(self) -> None:
        os.environ["SLACK_SIGNING_SECRET"] = "shhh"
        # Timestamp 10 minutes ago → outside 5-minute replay window.
        result = self._verify(b"x", {"x-slack-request-timestamp": "1000",
                                     "x-slack-signature": "v0=abc"})
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for the pure utility functions in deep_research.

Run:  python -m unittest test_deep_research
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

# The module hard-exits if DEEPSEEK_API_KEY is missing. Stub it for tests.
os.environ.setdefault("DEEPSEEK_API_KEY", "sk-test-stub")

import deep_research as dr  # noqa: E402


class TestCanonicalURL(unittest.TestCase):
    def test_lowercases_scheme_and_host(self):
        self.assertEqual(
            dr._canonical_url("HTTP://Example.COM/Foo"),
            "http://example.com/Foo",
        )

    def test_strips_www(self):
        self.assertEqual(
            dr._canonical_url("https://www.example.com/foo"),
            "https://example.com/foo",
        )

    def test_strips_query_and_fragment(self):
        self.assertEqual(
            dr._canonical_url("https://example.com/foo?utm_source=x#section"),
            "https://example.com/foo",
        )

    def test_strips_trailing_slash(self):
        self.assertEqual(
            dr._canonical_url("https://example.com/foo/"),
            "https://example.com/foo",
        )

    def test_root_path_kept(self):
        self.assertEqual(
            dr._canonical_url("https://example.com/"),
            "https://example.com/",
        )

    def test_handles_subdomain(self):
        self.assertEqual(
            dr._canonical_url("https://docs.python.org/3/library/json.html"),
            "https://docs.python.org/3/library/json.html",
        )


class TestDomainScore(unittest.TestCase):
    def test_gov_and_edu(self):
        self.assertEqual(dr._domain_score("https://www.nih.gov/foo"), 5)
        self.assertEqual(dr._domain_score("https://csail.mit.edu/foo"), 5)

    def test_explicit_domain(self):
        self.assertEqual(dr._domain_score("https://arxiv.org/abs/2401.1"), 4)
        self.assertEqual(dr._domain_score("https://github.com/foo/bar"), 3)

    def test_subdomain_inheritance(self):
        # docs.deepseek.com → matches deepseek.com
        self.assertEqual(dr._domain_score("https://docs.deepseek.com/x"), 3)

    def test_aggregator_negative(self):
        self.assertEqual(dr._domain_score("https://medium.com/post"), -1)
        self.assertEqual(dr._domain_score("https://www.geeksforgeeks.org/x"), -2)
        self.assertEqual(dr._domain_score("https://javatpoint.com/x"), -3)

    def test_unknown_returns_zero(self):
        self.assertEqual(dr._domain_score("https://random-blog.example/x"), 0)

    def test_handles_garbage(self):
        # Should not crash on malformed URLs.
        self.assertEqual(dr._domain_score("not a url"), 0)


class TestDedupAndRank(unittest.TestCase):
    def setUp(self):
        # Each test starts with a clean session.
        dr._SEEN_URLS.clear()

    def test_dedups_and_orders_by_score(self):
        hits = [
            {"url": "https://medium.com/x", "title": "agg", "snippet": ""},
            {"url": "https://arxiv.org/abs/123", "title": "primary", "snippet": ""},
            {"url": "https://example.com/a", "title": "neutral", "snippet": ""},
        ]
        ranked, filtered = dr._dedup_and_rank(hits, take=10)
        self.assertEqual(filtered, 0)
        self.assertEqual(
            [h["url"] for h in ranked],
            [
                "https://arxiv.org/abs/123",  # +4
                "https://example.com/a",      # 0
                "https://medium.com/x",       # -1
            ],
        )

    def test_filters_already_seen_within_session(self):
        first = [{"url": "https://example.com/a", "title": "x", "snippet": ""}]
        dr._dedup_and_rank(first, take=10)
        second_same = [{"url": "https://www.example.com/a/", "title": "x", "snippet": ""}]
        ranked, filtered = dr._dedup_and_rank(second_same, take=10)
        self.assertEqual(ranked, [])
        self.assertEqual(filtered, 1)

    def test_take_truncates(self):
        hits = [
            {"url": f"https://example.com/{i}", "title": "x", "snippet": ""}
            for i in range(20)
        ]
        ranked, _ = dr._dedup_and_rank(hits, take=5)
        self.assertEqual(len(ranked), 5)


class TestCosine(unittest.TestCase):
    def test_identical(self):
        v = [1.0, 0.0, 0.0]
        self.assertAlmostEqual(dr._cosine(v, v), 1.0)

    def test_orthogonal(self):
        self.assertAlmostEqual(dr._cosine([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_opposite(self):
        self.assertAlmostEqual(dr._cosine([1.0, 0.0], [-1.0, 0.0]), -1.0)

    def test_zero_vectors(self):
        # Should not divide-by-zero; defined as 0 here.
        self.assertEqual(dr._cosine([0.0, 0.0], [1.0, 1.0]), 0.0)


class TestPersistence(unittest.TestCase):
    def test_save_and_load_roundtrip(self):
        state = dr.SessionState(
            question="What is X?",
            messages=[{"role": "user", "content": "hi"}],
            seen_urls=["https://example.com/a"],
            embeddings=[[0.1, 0.2, 0.3]],
            total_input=42,
            total_output=17,
            iter=3,
            provider="serper",
            thinking=True,
            semantic_dedup=False,
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            dr.save_state(path, state)
            loaded = dr.load_state(path)
        finally:
            os.unlink(path)
        self.assertEqual(loaded.question, state.question)
        self.assertEqual(loaded.messages, state.messages)
        self.assertEqual(loaded.seen_urls, state.seen_urls)
        self.assertEqual(loaded.embeddings, state.embeddings)
        self.assertEqual(loaded.total_input, state.total_input)
        self.assertEqual(loaded.total_output, state.total_output)
        self.assertEqual(loaded.iter, state.iter)
        self.assertEqual(loaded.provider, state.provider)
        self.assertTrue(loaded.thinking)


class TestDetectProvider(unittest.TestCase):
    def setUp(self):
        # Snapshot env so we can restore it.
        self._saved = {
            k: os.environ.pop(k, None)
            for k in (
                "TAVILY_API_KEY",
                "SERPER_API_KEY",
                "BING_API_KEY",
                "GOOGLE_API_KEY",
                "GOOGLE_CSE_ID",
            )
        }

    def tearDown(self):
        for k, v in self._saved.items():
            if v is not None:
                os.environ[k] = v

    def test_prefers_tavily(self):
        os.environ["TAVILY_API_KEY"] = "x"
        os.environ["SERPER_API_KEY"] = "y"
        self.assertEqual(dr.detect_provider(), "tavily")

    def test_falls_back_to_serper(self):
        os.environ["SERPER_API_KEY"] = "y"
        self.assertEqual(dr.detect_provider(), "serper")

    def test_google_cse_needs_both(self):
        os.environ["GOOGLE_API_KEY"] = "k"
        with self.assertRaises(RuntimeError):
            dr.detect_provider()
        os.environ["GOOGLE_CSE_ID"] = "cx"
        self.assertEqual(dr.detect_provider(), "google_cse")

    def test_no_keys_raises(self):
        with self.assertRaises(RuntimeError):
            dr.detect_provider()


if __name__ == "__main__":
    unittest.main()

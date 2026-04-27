"""Tests for stats.py persistence + run_evals.py pure helpers."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("DEEPSEEK_API_KEY", "sk-test-stub")
os.environ.setdefault("TAVILY_API_KEY", "tvly-test-stub")

import stats  # noqa: E402
import run_evals  # noqa: E402


class TestStats(unittest.TestCase):
    def test_estimate_cost_known_models(self) -> None:
        # 1M input miss, 0 cached, 1M output on flash:
        #   1*1.0 + 0*0.2 + 1*2.0 = 3.0 CNY
        cost = stats.estimate_cost_cny("deepseek-v4-flash", 1_000_000, 0, 1_000_000)
        self.assertAlmostEqual(cost, 3.0, places=4)

    def test_estimate_cost_with_cache(self) -> None:
        # 1M total in, 800K cached → 200K miss + 800K hit
        # flash: 0.2*1.0 + 0.8*0.2 + 1.0*2.0 = 0.2 + 0.16 + 2.0 = 2.36
        cost = stats.estimate_cost_cny("deepseek-v4-flash", 1_000_000, 800_000, 1_000_000)
        self.assertAlmostEqual(cost, 2.36, places=4)

    def test_estimate_cost_pro_premium(self) -> None:
        cost = stats.estimate_cost_cny("deepseek-v4-pro", 1_000_000, 0, 1_000_000)
        # 1*12 + 0 + 1*24 = 36
        self.assertAlmostEqual(cost, 36.0, places=4)

    def test_estimate_cost_unknown_model_falls_back_to_flash(self) -> None:
        cost = stats.estimate_cost_cny("never-shipped-model", 1_000_000, 0, 1_000_000)
        self.assertAlmostEqual(cost, 3.0, places=4)

    def test_record_and_list_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "stats.db"
            rec = stats.SessionRecord(
                ts=1234.5, question="Q", planner="deepseek-v4-flash",
                writer="deepseek-v4-pro", provider="tavily",
                iters=3, tokens_in=1000, tokens_out=400, cached_tokens=200,
                cost_cny=0.123, duration_sec=12.5, ok=True, error=None,
            )
            row_id = stats.record_session(rec, db_path=db)
            self.assertGreater(row_id, 0)

            rows = stats.list_sessions(limit=10, db_path=db)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["question"], "Q")
            self.assertEqual(rows[0]["iters"], 3)
            self.assertEqual(rows[0]["ok"], 1)

            agg = stats.aggregate(db_path=db)
            self.assertEqual(agg["n"], 1)
            self.assertEqual(agg["tokens_in"], 1000)
            self.assertAlmostEqual(agg["cost_cny"], 0.123, places=4)

    def test_aggregate_empty_db(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "stats.db"
            agg = stats.aggregate(db_path=db)
            self.assertEqual(agg["n"], 0)
            self.assertEqual(agg["cost_cny"], 0)


class TestEvalsHelpers(unittest.TestCase):
    def test_load_cases(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"id": "a", "question": "?", "must_mention": ["x"]}) + "\n")
            f.write("\n")  # blank line
            f.write("# comment line\n")
            f.write(json.dumps({"id": "b", "question": "??", "should_cite_domains": ["y.com"]}) + "\n")
            path = f.name
        try:
            cases = run_evals._load_cases(Path(path))
        finally:
            os.unlink(path)
        self.assertEqual([c.id for c in cases], ["a", "b"])
        self.assertEqual(cases[0].must_mention, ["x"])
        self.assertEqual(cases[1].should_cite_domains, ["y.com"])

    def test_extract_cited_domains(self) -> None:
        report = """
        See [1](https://arxiv.org/abs/2401.0001) and [2](https://www.deepseek.com/foo).
        Also (https://api-docs.deepseek.com/zh-cn/) for context.
        """
        domains = run_evals._extract_cited_domains(report)
        self.assertIn("arxiv.org", domains)
        self.assertIn("deepseek.com", domains)
        self.assertIn("api-docs.deepseek.com", domains)
        self.assertEqual(len(domains), 3)

    def test_check_mentions(self) -> None:
        report = "DeepSeek v4-pro costs ¥24 per million output tokens"
        found, missing = run_evals._check_mentions(
            report, ["DeepSeek", "v4-pro", "output", "missing-term"]
        )
        # case-insensitive substring match
        self.assertIn("DeepSeek", found)
        self.assertIn("v4-pro", found)
        self.assertIn("output", found)
        self.assertEqual(missing, ["missing-term"])

    def test_check_domain_coverage_with_subdomain(self) -> None:
        cited = ["api-docs.deepseek.com", "github.com"]
        found, missing = run_evals._check_domain_coverage(cited, ["deepseek.com", "arxiv.org"])
        self.assertIn("deepseek.com", found)  # api-docs.deepseek.com matches
        self.assertIn("arxiv.org", missing)


if __name__ == "__main__":
    unittest.main()

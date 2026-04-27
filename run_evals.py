"""Eval harness: run the agent on a JSONL of golden questions and judge each
report with an LLM rubric.

Usage:
    python run_evals.py evals/golden.jsonl
    python run_evals.py evals/golden.jsonl --planner deepseek-v4-flash --writer deepseek-v4-pro
    python run_evals.py evals/golden.jsonl --limit 2 --out evals/results-$(date +%Y%m%d).json

Each golden case is JSON like:
    {"id": "...", "question": "...",
     "must_mention": ["..."],            # optional: substrings expected in body
     "should_cite_domains": ["..."],     # optional: domains expected in sources
     "category": "..."}                   # optional: free-form bucket label

Writes one results JSON file. Each entry has the report, the deterministic
auto-checks (mention coverage, citation domain coverage), and an LLM-judge
score (0-5) with rubric breakdown.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import deep_research as dr
import prompts as _prompts
from _log import configure_logging, log


@dataclass
class GoldenCase:
    id: str
    question: str
    must_mention: list[str] = field(default_factory=list)
    should_cite_domains: list[str] = field(default_factory=list)
    category: str = ""


@dataclass
class CaseResult:
    id: str
    question: str
    category: str
    report: str
    duration_sec: float
    tokens_in: int
    tokens_out: int
    iters: int
    ok: bool
    error: Optional[str] = None
    # Deterministic auto-checks
    mentions_found: list[str] = field(default_factory=list)
    mentions_missing: list[str] = field(default_factory=list)
    cited_domains: list[str] = field(default_factory=list)
    expected_domains_found: list[str] = field(default_factory=list)
    expected_domains_missing: list[str] = field(default_factory=list)
    # LLM judge
    judge: Optional[dict] = None


def _load_cases(path: Path) -> list[GoldenCase]:
    cases: list[GoldenCase] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                sys.exit(f"{path}:{i}: invalid JSON: {e}")
            cases.append(
                GoldenCase(
                    id=d["id"],
                    question=d["question"],
                    must_mention=d.get("must_mention", []),
                    should_cite_domains=d.get("should_cite_domains", []),
                    category=d.get("category", ""),
                )
            )
    return cases


def _extract_cited_domains(report: str) -> list[str]:
    """Extract URLs from the report and return their unique hostnames."""
    urls = re.findall(r"https?://[^\s)\]]+", report)
    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        host = (urlparse(u).hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        if host and host not in seen:
            seen.add(host)
            out.append(host)
    return out


def _check_mentions(report: str, terms: list[str]) -> tuple[list[str], list[str]]:
    body = report.lower()
    found, missing = [], []
    for t in terms:
        (found if t.lower() in body else missing).append(t)
    return found, missing


def _check_domain_coverage(domains: list[str], expected: list[str]) -> tuple[list[str], list[str]]:
    found, missing = [], []
    for exp in expected:
        if any(d == exp or d.endswith("." + exp) for d in domains):
            found.append(exp)
        else:
            missing.append(exp)
    return found, missing


async def _judge(report: str, question: str, model: str, prompts_version: str = "v1") -> dict:
    cli = dr._get_client()
    judge_prompt = _prompts.load(prompts_version, lang="en")["judge"]
    resp = await dr._create_with_retry(
        cli,
        model=model,
        messages=[
            {"role": "system", "content": "You are a strict, terse evaluator. Output JSON only."},
            {"role": "user", "content": judge_prompt.format(question=question, report=report)},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "invalid judge JSON", "raw": raw}


async def run_case(case: GoldenCase, *, planner: str, writer: str,
                   max_iters: int, judge_model: str,
                   prompts_version: str = "v1", lang: str = "en") -> CaseResult:
    state = dr.SessionState(
        question=case.question,
        provider=dr.PROVIDER_NAME,
        planner_model=planner,
        writer_model=writer,
        prompts_version=prompts_version,
        lang=lang,
    )
    t0 = time.time()
    error: Optional[str] = None
    try:
        report = await dr.research(
            state, max_iters=max_iters, max_tokens=None,
            token_budget=None, save_path=None, stream=False,
        )
        ok = True
    except Exception as e:
        report = ""
        ok = False
        error = f"{type(e).__name__}: {e}"
        log.error("eval.case_failed", id=case.id, error=error)

    duration = time.time() - t0
    cited = _extract_cited_domains(report)
    found_terms, missing_terms = _check_mentions(report, case.must_mention)
    found_domains, missing_domains = _check_domain_coverage(cited, case.should_cite_domains)

    judge = None
    if ok and report:
        try:
            judge = await _judge(report, case.question, judge_model, prompts_version)
        except Exception as e:
            judge = {"error": str(e)}

    return CaseResult(
        id=case.id, question=case.question, category=case.category,
        report=report, duration_sec=duration,
        tokens_in=state.total_input, tokens_out=state.total_output,
        iters=state.iter, ok=ok, error=error,
        mentions_found=found_terms, mentions_missing=missing_terms,
        cited_domains=cited,
        expected_domains_found=found_domains,
        expected_domains_missing=missing_domains,
        judge=judge,
    )


def summarise(results: list[CaseResult]) -> dict:
    n = len(results)
    ok = sum(1 for r in results if r.ok)
    judge_totals = [r.judge.get("total", 0) for r in results if r.judge and "total" in r.judge]
    mention_rate = (
        sum(len(r.mentions_found) for r in results)
        / max(1, sum(len(r.mentions_found) + len(r.mentions_missing) for r in results))
    )
    domain_rate = (
        sum(len(r.expected_domains_found) for r in results)
        / max(1, sum(len(r.expected_domains_found) + len(r.expected_domains_missing) for r in results))
    )
    return {
        "n": n,
        "ok": ok,
        "avg_judge_total": sum(judge_totals) / len(judge_totals) if judge_totals else None,
        "mention_coverage": mention_rate,
        "expected_domain_coverage": domain_rate,
        "avg_duration_s": sum(r.duration_sec for r in results) / n if n else 0.0,
        "total_tokens_in": sum(r.tokens_in for r in results),
        "total_tokens_out": sum(r.tokens_out for r in results),
    }


async def _amain(args) -> int:
    configure_logging(args.log)

    cases = _load_cases(Path(args.golden))
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        log.error("evals.no_cases", path=args.golden)
        return 1

    if args.provider == "auto":
        dr.PROVIDER_NAME = dr.detect_provider()
    else:
        dr.PROVIDER_NAME = args.provider
    log.info("evals.starting", n=len(cases), provider=dr.PROVIDER_NAME,
             planner=args.planner, writer=args.writer)

    results: list[CaseResult] = []
    for c in cases:
        log.info("evals.case_started", id=c.id, question=c.question)
        # Each case starts with a clean dedup state so cases are independent.
        dr._SEEN_URLS.clear()
        dr._SEEN_EMBEDDINGS.clear()
        r = await run_case(
            c,
            planner=args.planner, writer=args.writer,
            max_iters=args.max_iters, judge_model=args.judge,
            prompts_version=args.prompts, lang=args.lang,
        )
        results.append(r)
        log.info(
            "evals.case_done", id=c.id, ok=r.ok,
            judge_total=(r.judge or {}).get("total"),
            duration_s=round(r.duration_sec, 1),
            mentions_missing=len(r.mentions_missing),
            domains_missing=len(r.expected_domains_missing),
        )

    summary = summarise(results)
    log.info("evals.summary", **{k: v for k, v in summary.items() if v is not None})

    out_path = Path(args.out) if args.out else Path(f"evals/results-{int(time.time())}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "config": {
                    "planner": args.planner, "writer": args.writer,
                    "judge": args.judge, "max_iters": args.max_iters,
                    "provider": dr.PROVIDER_NAME, "ts": time.time(),
                },
                "results": [asdict(r) for r in results],
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    log.info("evals.written", path=str(out_path))

    # Exit non-zero if anything failed or judge avg below threshold.
    if any(not r.ok for r in results):
        return 2
    if summary["avg_judge_total"] is not None and summary["avg_judge_total"] < args.min_judge:
        log.warning("evals.below_threshold",
                    avg=summary["avg_judge_total"], min=args.min_judge)
        return 3
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Run golden-question evals against the agent")
    ap.add_argument("golden", help="Path to evals/golden.jsonl")
    ap.add_argument("--planner", default="deepseek-v4-flash")
    ap.add_argument("--writer", default="deepseek-v4-pro")
    ap.add_argument("--judge", default="deepseek-v4-pro",
                    help="Model used for the rubric scoring pass")
    ap.add_argument("--max-iters", type=int, default=8)
    ap.add_argument("--prompts", default="v1", help="Prompt version under prompts/")
    ap.add_argument("--lang", default="en", help="Output language for reports")
    ap.add_argument("--provider", default=os.environ.get("SEARCH_PROVIDER", "auto"))
    ap.add_argument("--limit", type=int, default=None,
                    help="Run only the first N cases (smoke test)")
    ap.add_argument("--out", help="Output path (default: evals/results-<unix>.json)")
    ap.add_argument("--log", default=os.environ.get("DEEP_RESEARCH_LOG", "pretty"),
                    choices=["pretty", "json", "silent"])
    ap.add_argument("--min-judge", type=float, default=0.0,
                    help="Exit with code 3 if avg judge score is below this")
    args = ap.parse_args()
    sys.exit(asyncio.run(_amain(args)))


if __name__ == "__main__":
    main()

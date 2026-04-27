"""A/B test two configurations against the same golden eval set.

Each config is a JSON file like:

    {
      "name": "flash-only",
      "planner": "deepseek-v4-flash",
      "writer":  "deepseek-v4-flash",
      "max_iters": 8,
      "prompts": "v1",
      "lang": "en"
    }

Usage:

    python run_ab.py evals/golden.jsonl \\
        --config-a configs/flash-only.json \\
        --config-b configs/flash-pro.json \\
        --judge deepseek-v4-pro \\
        --out evals/ab-$(date +%s).json

Output: a JSON file containing per-config summaries plus a `delta` block
(B minus A) so you can see whether B is genuinely better and at what cost.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import deep_research as dr
import run_evals
import stats as _stats
from _log import configure_logging, log


def _load_config(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("name", path.stem)
    data.setdefault("planner", "deepseek-v4-flash")
    data.setdefault("writer", "deepseek-v4-pro")
    data.setdefault("max_iters", 8)
    data.setdefault("prompts", "v1")
    data.setdefault("lang", "en")
    return data


async def _run_arm(name: str, config: dict, cases: list[run_evals.GoldenCase],
                   judge_model: str) -> list[run_evals.CaseResult]:
    log.info("ab.arm_started", arm=name, config=config)
    results: list[run_evals.CaseResult] = []
    for c in cases:
        # Each case starts with a fresh dedup state to keep arms comparable.
        dr._SEEN_URLS.clear()
        dr._SEEN_EMBEDDINGS.clear()
        log.info("ab.case_started", arm=name, id=c.id)
        r = await run_evals.run_case(
            c,
            planner=config["planner"], writer=config["writer"],
            max_iters=config["max_iters"], judge_model=judge_model,
            prompts_version=config["prompts"], lang=config["lang"],
        )
        results.append(r)
        log.info("ab.case_done", arm=name, id=c.id, ok=r.ok,
                 judge_total=(r.judge or {}).get("total"),
                 tokens=r.tokens_in + r.tokens_out, duration_s=round(r.duration_sec, 1))
    return results


def _arm_summary(results: list[run_evals.CaseResult], config: dict) -> dict:
    base = run_evals.summarise(results)
    cost_cny = sum(
        _stats.estimate_cost_cny(config["planner"], r.tokens_in, 0, r.tokens_out)
        for r in results
    )
    base["cost_cny"] = cost_cny
    base["cost_usd"] = cost_cny * _stats.usd_per_cny()
    return base


async def amain(args) -> int:
    configure_logging(args.log)
    cases = run_evals._load_cases(Path(args.golden))
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        log.error("ab.no_cases")
        return 1

    config_a = _load_config(Path(args.config_a))
    config_b = _load_config(Path(args.config_b))

    if args.provider == "auto":
        dr.PROVIDER_NAME = dr.detect_provider()
    else:
        dr.PROVIDER_NAME = args.provider
    log.info("ab.starting", n_cases=len(cases), provider=dr.PROVIDER_NAME,
             a=config_a["name"], b=config_b["name"], judge=args.judge)

    results_a = await _run_arm("A:" + config_a["name"], config_a, cases, args.judge)
    results_b = await _run_arm("B:" + config_b["name"], config_b, cases, args.judge)

    summary_a = _arm_summary(results_a, config_a)
    summary_b = _arm_summary(results_b, config_b)

    delta_keys = ("avg_judge_total", "mention_coverage", "expected_domain_coverage",
                  "avg_duration_s", "cost_cny", "cost_usd",
                  "total_tokens_in", "total_tokens_out")
    delta: dict = {}
    for k in delta_keys:
        a, b = summary_a.get(k), summary_b.get(k)
        if a is None or b is None:
            continue
        delta[k] = round(b - a, 4)

    log.info("ab.summary",
             a_judge=summary_a.get("avg_judge_total"),
             b_judge=summary_b.get("avg_judge_total"),
             judge_delta=delta.get("avg_judge_total"),
             a_cost_usd=round(summary_a["cost_usd"], 4),
             b_cost_usd=round(summary_b["cost_usd"], 4),
             cost_delta_usd=round(delta.get("cost_usd", 0), 4))

    out_path = Path(args.out) if args.out else Path(f"evals/ab-{int(time.time())}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "config_a": config_a, "config_b": config_b,
        "summary_a": summary_a, "summary_b": summary_b,
        "delta_b_minus_a": delta,
        "results_a": [asdict(r) for r in results_a],
        "results_b": [asdict(r) for r in results_b],
        "ts": time.time(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("ab.written", path=str(out_path))

    # Plain-English verdict on stderr.
    print(_format_verdict(config_a, config_b, summary_a, summary_b, delta), file=sys.stderr)
    return 0


def _format_verdict(ca, cb, sa, sb, delta) -> str:
    ja, jb = sa.get("avg_judge_total"), sb.get("avg_judge_total")
    cost_a, cost_b = sa["cost_usd"], sb["cost_usd"]
    lines = [
        "=" * 60,
        f"A: {ca['name']:30s}  judge={ja}  cost=${cost_a:.4f}",
        f"B: {cb['name']:30s}  judge={jb}  cost=${cost_b:.4f}",
        "-" * 60,
    ]
    if ja is not None and jb is not None:
        verdict = "B better" if jb > ja else ("A better" if ja > jb else "tied")
        lines.append(f"Quality:  {verdict}  (Δjudge = {jb - ja:+.2f})")
    lines.append(f"Cost:     B {'cheaper' if cost_b < cost_a else 'pricier'} by "
                 f"${abs(cost_b - cost_a):.4f}  (Δ = {cost_b - cost_a:+.4f})")
    if ja and jb and ja > 0:
        per_point_a = cost_a / max(ja, 0.01)
        per_point_b = cost_b / max(jb, 0.01)
        lines.append(f"USD per judge point: A=${per_point_a:.4f}  B=${per_point_b:.4f}")
    lines.append("=" * 60)
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="A/B two configs on a golden eval set")
    ap.add_argument("golden")
    ap.add_argument("--config-a", required=True)
    ap.add_argument("--config-b", required=True)
    ap.add_argument("--judge", default="deepseek-v4-pro")
    ap.add_argument("--provider", default="auto")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out")
    ap.add_argument("--log", default="pretty",
                    choices=["pretty", "json", "silent"])
    args = ap.parse_args()
    sys.exit(asyncio.run(amain(args)))


if __name__ == "__main__":
    main()

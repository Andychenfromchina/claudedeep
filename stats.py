"""Persistent stats: per-session token use, cost, duration → SQLite.

Pricing in the table reflects DeepSeek's public April 2026 pricing in CNY per
1M tokens. We record CNY in the DB and convert to USD at request time so we
can re-cost historical rows if the FX rate moves.
"""
from __future__ import annotations

import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# CNY per 1M tokens — DeepSeek list prices, April 2026.
PRICING_CNY: dict[str, dict[str, float]] = {
    "deepseek-v4-flash": {"input_miss": 1.0,  "input_hit": 0.2, "output": 2.0},
    "deepseek-v4-pro":   {"input_miss": 12.0, "input_hit": 1.0, "output": 24.0},
    # Legacy sunset 2026-07-24 — keep for back-compat with older state files.
    "deepseek-chat":     {"input_miss": 1.0,  "input_hit": 0.1, "output": 2.0},
    "deepseek-reasoner": {"input_miss": 4.0,  "input_hit": 1.0, "output": 16.0},
}

# Approximate CNY → USD rate. Override via DEEP_RESEARCH_USD_RATE if needed.
DEFAULT_USD_PER_CNY = 1.0 / 7.2


def usd_per_cny() -> float:
    raw = os.environ.get("DEEP_RESEARCH_USD_RATE")
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return DEFAULT_USD_PER_CNY


def estimate_cost_cny(
    model: str, tokens_in: int, cached_tokens: int, tokens_out: int
) -> float:
    p = PRICING_CNY.get(model)
    if p is None:
        # Unknown model — assume flash-equivalent so we don't lie about costs.
        p = PRICING_CNY["deepseek-v4-flash"]
    miss = max(0, tokens_in - cached_tokens)
    return (
        (miss / 1_000_000) * p["input_miss"]
        + (cached_tokens / 1_000_000) * p["input_hit"]
        + (tokens_out / 1_000_000) * p["output"]
    )


def default_db_path() -> Path:
    raw = os.environ.get("DEEP_RESEARCH_STATS")
    if raw:
        return Path(raw)
    return Path.home() / ".cache" / "deep-research" / "stats.db"


@contextmanager
def _connect(db_path: Optional[Path] = None):
    path = db_path or default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        _ensure_schema(conn)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            question TEXT NOT NULL,
            planner TEXT NOT NULL,
            writer TEXT NOT NULL,
            provider TEXT NOT NULL,
            iters INTEGER NOT NULL,
            tokens_in INTEGER NOT NULL,
            tokens_out INTEGER NOT NULL,
            cached_tokens INTEGER NOT NULL,
            cost_cny REAL NOT NULL,
            duration_sec REAL NOT NULL,
            ok INTEGER NOT NULL,
            error TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_sessions_ts ON sessions(ts DESC);
        """
    )


@dataclass
class SessionRecord:
    ts: float
    question: str
    planner: str
    writer: str
    provider: str
    iters: int
    tokens_in: int
    tokens_out: int
    cached_tokens: int
    cost_cny: float
    duration_sec: float
    ok: bool
    error: Optional[str] = None


def record_session(rec: SessionRecord, db_path: Optional[Path] = None) -> int:
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO sessions
                (ts, question, planner, writer, provider, iters,
                 tokens_in, tokens_out, cached_tokens,
                 cost_cny, duration_sec, ok, error)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                rec.ts, rec.question, rec.planner, rec.writer, rec.provider,
                rec.iters, rec.tokens_in, rec.tokens_out, rec.cached_tokens,
                rec.cost_cny, rec.duration_sec, int(rec.ok), rec.error,
            ),
        )
        return int(cur.lastrowid or 0)


def list_sessions(limit: int = 50, db_path: Optional[Path] = None) -> list[dict]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def aggregate(db_path: Optional[Path] = None) -> dict:
    with _connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS n,
                COALESCE(SUM(tokens_in), 0)  AS tokens_in,
                COALESCE(SUM(tokens_out), 0) AS tokens_out,
                COALESCE(SUM(cached_tokens), 0) AS cached,
                COALESCE(SUM(cost_cny), 0)   AS cost_cny,
                COALESCE(AVG(duration_sec), 0) AS avg_duration,
                COALESCE(SUM(ok), 0)         AS ok_count
            FROM sessions
            """
        ).fetchone()
        d = dict(row)
        d["cost_usd"] = d["cost_cny"] * usd_per_cny()
        return d


def now() -> float:
    return time.time()

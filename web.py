"""FastAPI + HTMX + SSE wrapper around deep_research.

Run:
    pip install -r requirements-web.txt
    uvicorn web:app --reload --port 8000

Routes:
    GET  /                — research form
    POST /research        — blocking: returns rendered HTML fragment
    GET  /research_stream — SSE: live event feed (used by HTMX SSE extension)
    GET  /stats           — cost / usage dashboard
    GET  /healthz         — liveness check
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import html
import json
import os
import sys
import time
from typing import AsyncIterator, Optional

import httpx
from fastapi import FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

import deep_research as dr
import stats as _stats
from _log import configure_logging, log

configure_logging(os.environ.get("DEEP_RESEARCH_LOG", "json"))

app = FastAPI(title="Deep Research")


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Deep Research</title>
  <script src="https://unpkg.com/htmx.org@1.9.10"></script>
  <script src="https://unpkg.com/htmx.org@1.9.10/dist/ext/sse.js"></script>
  <style>
    * { box-sizing: border-box; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
      max-width: 880px; margin: 2em auto; padding: 0 1em;
      color: #1a1a1a; line-height: 1.5;
    }
    h1 { margin: 0 0 0.5em; }
    .sub { color: #666; margin-bottom: 1.5em; }
    nav { margin-bottom: 1em; }
    nav a { margin-right: 1em; color: #0066cc; text-decoration: none; }
    nav a:hover { text-decoration: underline; }
    form { display: flex; flex-direction: column; gap: 0.75em; }
    textarea {
      width: 100%; min-height: 90px; padding: 0.75em;
      font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
      font-size: 14px; border: 1px solid #d0d0d0; border-radius: 6px;
      resize: vertical;
    }
    textarea:focus { outline: 2px solid #0066cc; border-color: #0066cc; }
    .options { display: flex; gap: 1em; flex-wrap: wrap; align-items: center; }
    .options label { display: flex; gap: 0.4em; align-items: center; cursor: pointer; }
    .options input[type=number] { width: 4em; padding: 0.25em; }
    button {
      padding: 0.6em 1.5em; background: #0066cc; color: white;
      border: 0; border-radius: 6px; font-weight: 500; cursor: pointer;
      align-self: flex-start;
    }
    button:hover { background: #0052a3; }
    .events {
      margin-top: 1.5em; padding: 0.75em;
      background: #1f1f1f; color: #d0d0d0;
      border-radius: 6px; font-family: ui-monospace, monospace;
      font-size: 12px; max-height: 280px; overflow-y: auto;
    }
    .events:empty { display: none; }
    .events div { margin: 0.15em 0; }
    .ev-iter      { color: #6cf; }
    .ev-tool      { color: #fc6; }
    .ev-writer    { color: #c6f; }
    .ev-budget    { color: #f88; }
    .ev-done      { color: #6f6; }
    .report {
      margin-top: 1em; padding: 1.25em; background: #fafafa;
      border: 1px solid #e0e0e0; border-radius: 6px; font-size: 15px;
    }
    .report:empty { display: none; }
    .report h1, .report h2, .report h3 { margin-top: 1em; }
    .report pre { background: #1f1f1f; color: #eee; padding: 1em; border-radius: 4px; overflow-x: auto; }
    .report code { background: #eee; padding: 0.1em 0.3em; border-radius: 3px; }
    .report pre code { background: transparent; padding: 0; }
    .report a { color: #0066cc; }
    .err { color: #b00020; padding: 1em; background: #fee; border-radius: 6px; margin-top: 1em; }
    table { border-collapse: collapse; width: 100%; }
    th, td { padding: 0.5em; border-bottom: 1px solid #e0e0e0; text-align: left; font-size: 14px; }
    th { background: #f5f5f5; }
    .num { text-align: right; font-family: ui-monospace, monospace; }
    .summary { padding: 1em; background: #f0f7ff; border-radius: 6px; margin-bottom: 1em; }
    .summary span { display: inline-block; margin-right: 1.5em; }
    .summary b { color: #0066cc; }
  </style>
</head>
<body>
  <h1>Deep Research</h1>
  <nav>
    <a href="/">Research</a>
    <a href="/stats">Stats</a>
  </nav>
  <p class="sub">DeepSeek v4 + web search → cited markdown report. Live progress via SSE.</p>

  <form id="form" hx-post="/research_kick" hx-swap="none">
    <textarea name="question" placeholder="Ask a research question..." required autofocus></textarea>
    <div class="options">
      <label><input type="checkbox" name="thinking" value="1"> Thinking mode</label>
      <label><input type="checkbox" name="json_format" value="1"> JSON output</label>
      <label>Max iters: <input type="number" name="max_iters" value="8" min="1" max="20"></label>
      <label>Budget: <input type="number" name="token_budget" value="" placeholder="∞" min="1000"></label>
    </div>
    <div>
      <button type="submit">Research</button>
    </div>
  </form>

  <div id="live-area"></div>

  <script>
    // On submit, replace the live-area with an SSE-bound div that streams
    // events from /research_stream and renders them into the panel.
    document.getElementById("form").addEventListener("submit", function (e) {
      e.preventDefault();
      const data = new FormData(e.target);
      const params = new URLSearchParams();
      for (const [k, v] of data.entries()) params.append(k, v);
      const url = "/research_stream?" + params.toString();

      const live = document.getElementById("live-area");
      live.innerHTML = `
        <div class="events" id="events"></div>
        <div class="report" id="report"></div>
      `;
      const events = document.getElementById("events");
      const report = document.getElementById("report");

      const es = new EventSource(url);
      es.addEventListener("event", function (e) {
        const evt = JSON.parse(e.data);
        const div = document.createElement("div");
        div.className = "ev-" + (evt.type.split(".")[0] || "info");
        div.textContent = JSON.stringify(evt);
        events.appendChild(div);
        events.scrollTop = events.scrollHeight;

        if (evt.type === "done") {
          report.innerHTML = evt.report_html || ("<pre>" +
            (evt.report || "").replace(/&/g, "&amp;").replace(/</g, "&lt;") + "</pre>");
          es.close();
        } else if (evt.type === "error") {
          report.innerHTML = '<div class="err">' + (evt.message || "error") + '</div>';
          es.close();
        }
      });
      es.onerror = function () {
        // EventSource auto-reconnects; close manually after final event.
      };
    });
  </script>
</body>
</html>
"""


def _render_markdown(md_text: str) -> str:
    try:
        import markdown as md_lib
        return md_lib.markdown(md_text, extensions=["fenced_code", "tables", "sane_lists"])
    except ImportError:
        return f"<pre>{html.escape(md_text)}</pre>"


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return INDEX_HTML


@app.get("/healthz")
async def healthz() -> dict:
    return {"ok": True}


# --- non-streaming path (kept for clients that don't speak SSE) -------------

@app.post("/research", response_class=HTMLResponse)
async def do_research(
    question: str = Form(...),
    thinking: Optional[str] = Form(None),
    json_format: Optional[str] = Form(None),
    max_iters: int = Form(8),
    token_budget: Optional[int] = Form(None),
) -> str:
    if not question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")

    try:
        provider = dr.detect_provider()
    except RuntimeError as e:
        return f'<div class="err">Setup error: {html.escape(str(e))}</div>'

    dr.PROVIDER_NAME = provider
    fmt = "json" if json_format else "markdown"
    state = dr.SessionState(
        question=question.strip(),
        provider=provider,
        thinking=bool(thinking),
        planner_model="deepseek-v4-flash",
        writer_model="deepseek-v4-pro",
    )

    try:
        markdown_report = await dr.research(
            state, max_iters=max_iters, max_tokens=None,
            token_budget=token_budget, save_path=None, stream=False,
        )
    except Exception as e:
        log.error("web.research_failed", error=str(e))
        return f'<div class="err">Research failed: {html.escape(str(e))}</div>'

    if fmt == "json":
        try:
            json_text = await dr.synthesize_json(markdown_report, state.question, state.writer_model)
            return f"<pre><code>{html.escape(json_text)}</code></pre>"
        except Exception as e:
            return f'<div class="err">JSON synthesis failed: {html.escape(str(e))}</div>'

    return _render_markdown(markdown_report)


# --- SSE streaming path -----------------------------------------------------

async def _sse_event_stream(
    question: str, *, thinking: bool, json_format: bool,
    max_iters: int, token_budget: Optional[int],
) -> AsyncIterator[bytes]:
    """Drive research() and stream its events as SSE messages."""

    queue: asyncio.Queue = asyncio.Queue(maxsize=1024)

    async def cb(event: dict) -> None:
        # Trim the heaviest field for transport.
        if event.get("type") == "done":
            md = event.get("report", "") or ""
            event = {
                **event,
                "report": md,
                "report_html": _render_markdown(md) if not json_format else None,
            }
        await queue.put(event)

    try:
        provider = dr.detect_provider()
    except RuntimeError as e:
        yield _sse_format({"type": "error", "message": str(e)})
        return
    dr.PROVIDER_NAME = provider

    state = dr.SessionState(
        question=question.strip(),
        provider=provider,
        thinking=thinking,
        planner_model="deepseek-v4-flash",
        writer_model="deepseek-v4-pro",
    )

    async def run_research():
        try:
            md = await dr.research(
                state, max_iters=max_iters, max_tokens=None,
                token_budget=token_budget, save_path=None, stream=False,
                event_callback=cb,
            )
            if json_format:
                json_text = await dr.synthesize_json(md, state.question, state.writer_model)
                await queue.put({"type": "done", "report": json_text,
                                 "report_html": f"<pre><code>{html.escape(json_text)}</code></pre>"})
        except Exception as e:
            log.error("sse.research_failed", error=str(e))
            await queue.put({"type": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            await queue.put(None)  # sentinel

    task = asyncio.create_task(run_research())
    try:
        while True:
            evt = await queue.get()
            if evt is None:
                break
            yield _sse_format(evt)
    finally:
        if not task.done():
            task.cancel()


def _sse_format(evt: dict) -> bytes:
    payload = json.dumps(evt, ensure_ascii=False)
    return f"event: event\ndata: {payload}\n\n".encode("utf-8")


@app.post("/research_kick")
async def research_kick() -> dict:
    """Stub endpoint for HTMX form submit; the JS handles SSE itself."""
    return {"ok": True}


@app.get("/research_stream")
async def research_stream(
    question: str = Query(...),
    thinking: Optional[str] = Query(None),
    json_format: Optional[str] = Query(None),
    max_iters: int = Query(8),
    token_budget: Optional[int] = Query(None),
) -> StreamingResponse:
    if not question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")
    gen = _sse_event_stream(
        question,
        thinking=bool(thinking),
        json_format=bool(json_format),
        max_iters=max_iters,
        token_budget=token_budget,
    )
    return StreamingResponse(
        gen,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --- /stats dashboard -------------------------------------------------------

@app.get("/stats", response_class=HTMLResponse)
async def stats_page() -> str:
    rows = _stats.list_sessions(limit=50)
    summary = _stats.aggregate()
    rate = _stats.usd_per_cny()

    rows_html = "\n".join(
        f"<tr>"
        f"<td>{html.escape(_fmt_ts(r['ts']))}</td>"
        f"<td>{html.escape(r['question'][:80])}</td>"
        f"<td>{html.escape(r['planner'])}/{html.escape(r['writer'])}</td>"
        f"<td class='num'>{r['iters']}</td>"
        f"<td class='num'>{r['tokens_in']:,}</td>"
        f"<td class='num'>{r['tokens_out']:,}</td>"
        f"<td class='num'>¥{r['cost_cny']:.4f}</td>"
        f"<td class='num'>${r['cost_cny'] * rate:.4f}</td>"
        f"<td class='num'>{r['duration_sec']:.1f}s</td>"
        f"<td>{'✓' if r['ok'] else '✗'}</td>"
        f"</tr>"
        for r in rows
    )

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Stats</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
         max-width: 1100px; margin: 2em auto; padding: 0 1em; color: #1a1a1a; }}
  nav a {{ margin-right: 1em; color: #0066cc; text-decoration: none; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ padding: 0.5em; border-bottom: 1px solid #e0e0e0; text-align: left; font-size: 13px; }}
  th {{ background: #f5f5f5; }}
  .num {{ text-align: right; font-family: ui-monospace, monospace; }}
  .summary {{ padding: 1em; background: #f0f7ff; border-radius: 6px; margin-bottom: 1em; }}
  .summary span {{ display: inline-block; margin-right: 1.5em; }}
  .summary b {{ color: #0066cc; }}
</style></head>
<body>
  <h1>Stats</h1>
  <nav><a href="/">← Research</a></nav>
  <div class="summary">
    <span>Sessions: <b>{summary['n']}</b></span>
    <span>Tokens in: <b>{summary['tokens_in']:,}</b></span>
    <span>Tokens out: <b>{summary['tokens_out']:,}</b></span>
    <span>Cost: <b>¥{summary['cost_cny']:.2f}</b> / <b>${summary['cost_usd']:.2f}</b></span>
    <span>Avg duration: <b>{summary['avg_duration']:.1f}s</b></span>
  </div>
  <table>
    <thead><tr>
      <th>When</th><th>Question</th><th>Models</th><th>Iters</th>
      <th>In</th><th>Out</th><th>CNY</th><th>USD</th><th>Duration</th><th>OK</th>
    </tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</body></html>"""


def _fmt_ts(ts: float) -> str:
    import datetime as dt
    return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


# --- Slack slash command --------------------------------------------------
#
# Slack expects a 200 response within 3 seconds, so we ack immediately and
# post the actual report later via the response_url. The signing-secret check
# protects against forged requests.

def _verify_slack(req_body: bytes, headers) -> bool:
    secret = os.environ.get("SLACK_SIGNING_SECRET")
    if not secret:
        # No secret configured → reject all (don't accidentally run open).
        return False
    ts = headers.get("x-slack-request-timestamp", "")
    sig = headers.get("x-slack-signature", "")
    try:
        if abs(time.time() - float(ts)) > 60 * 5:
            return False  # replay window
    except ValueError:
        return False
    base = f"v0:{ts}:".encode() + req_body
    expected = "v0=" + hmac.new(secret.encode(), base, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, sig)


async def _slack_post(response_url: str, text: str, *, in_channel: bool = True) -> None:
    """Post a message back to Slack via the response_url they gave us."""
    payload = {
        "response_type": "in_channel" if in_channel else "ephemeral",
        "text": text,
    }
    try:
        async with httpx.AsyncClient(timeout=20) as cli:
            r = await cli.post(response_url, json=payload)
            if r.status_code >= 400:
                log.warning("slack.post_failed", status=r.status_code, body=r.text[:200])
    except Exception as e:
        log.warning("slack.post_error", error=str(e))


async def _slack_research(question: str, response_url: str, user: str) -> None:
    log.info("slack.research_started", user=user, question=question)
    try:
        provider = dr.detect_provider()
    except RuntimeError as e:
        await _slack_post(response_url, f"❌ Setup error: {e}", in_channel=False)
        return

    dr.PROVIDER_NAME = provider
    state = dr.SessionState(
        question=question, provider=provider,
        planner_model="deepseek-v4-flash", writer_model="deepseek-v4-pro",
    )
    try:
        report = await dr.research(
            state, max_iters=8, max_tokens=None,
            token_budget=None, save_path=None, stream=False,
        )
    except Exception as e:
        log.error("slack.research_failed", error=str(e))
        await _slack_post(response_url, f"❌ Research failed: {e}", in_channel=False)
        return

    # Slack message limit: 40K chars per `text`. Truncate with a marker.
    if len(report) > 38000:
        report = report[:38000] + "\n\n_…truncated; see /research --out for full report_"
    header = f"*Research from <@{user}>*: _{question}_\n\n"
    await _slack_post(response_url, header + report, in_channel=True)


@app.post("/slack")
async def slack_command(request: Request):
    body = await request.body()
    if not _verify_slack(body, request.headers):
        raise HTTPException(401, "invalid Slack signature")

    form = await request.form()
    text = (form.get("text") or "").strip()
    response_url = form.get("response_url")
    user = form.get("user_name") or form.get("user_id") or "unknown"

    if not text:
        return JSONResponse({"response_type": "ephemeral",
                             "text": "Usage: /research <your question>"})
    if not response_url:
        return JSONResponse({"response_type": "ephemeral",
                             "text": "Slack didn't include response_url; can't async-reply."})

    asyncio.create_task(_slack_research(text, response_url, user))
    return JSONResponse({
        "response_type": "ephemeral",
        "text": f"🔎 Researching: _{text}_\n_Posting results when done (30s–2min)._",
    })


def _run() -> None:
    """Console-script entry point for `deep-research-web`."""
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("web:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    _run()

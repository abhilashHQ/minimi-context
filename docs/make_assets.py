"""Build the README images from real run data.

    python -m docs.make_assets

Charts come from runs/score_d/scorecard.json. Screenshots are rendered from run 1 of each
system (never picked for looks) and from the final harness store, through headless Chrome.
"""

from __future__ import annotations

import html
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "docs" / "img"
SCORE = ROOT / "runs" / "score_d"
CHROME = Path(os.environ.get("CHROME", r"C:\Program Files\Google\Chrome\Application\chrome.exe"))

HARNESS, TODAY = "#2a78d6", "#eb6834"
HARNESS_DK, TODAY_DK = "#3987e5", "#d95926"
QUESTIONS = {
    "C1": "What time is dinner with Sam tomorrow?",
    "C2": "What do I still owe people?",
    "C3": "Do I owe Priya anything?",
    "C4": "Did I call the landlord?",
    "C5": "What did I agree to last week?",
    "C6": "Does Sam have any food restrictions?",
    "C7": "What's Sam's flight number?",
    "C8": "What's on this weekend?",
}

SVG_STYLE = f"""
  text {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; fill: #1f2328; }}
  .muted {{ fill: #59636e; }}
  .bg {{ fill: #ffffff; stroke: #d1d9e0; }}
  .grid {{ stroke: #d1d9e0; stroke-width: 1; }}
  .h {{ fill: {HARNESS}; }} .t {{ fill: {TODAY}; }}
  .hs {{ stroke: {HARNESS}; }} .ts {{ stroke: {TODAY}; }}
  .bar {{ transform-box: fill-box; transform-origin: left center; animation: grow 1.1s cubic-bezier(.2,.8,.2,1) both; }}
  .fade {{ animation: fade .5s ease-out both; }}
  @keyframes grow {{ from {{ transform: scaleX(0); }} to {{ transform: scaleX(1); }} }}
  @keyframes fade {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
  @media (prefers-color-scheme: dark) {{
    text {{ fill: #f0f6fc; }} .muted {{ fill: #9198a1; }} .grid {{ stroke: #3d444d; }}
    .bg {{ fill: #0d1117; stroke: #3d444d; }}
    .h {{ fill: {HARNESS_DK}; }} .t {{ fill: {TODAY_DK}; }}
    .hs {{ stroke: {HARNESS_DK}; }} .ts {{ stroke: {TODAY_DK}; }}
  }}
  @media (prefers-reduced-motion: reduce) {{ .bar, .fade {{ animation: none; }} }}
"""


PAD = 24


def svg_open(w: int, h: int, head: list[str]) -> list[str]:
    """An SVG that carries its own background, so text contrast holds whatever page it sits on."""
    W, H = w + 2 * PAD, h + 2 * PAD
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
            'role="img" aria-labelledby="t d">', *head, f"<style>{SVG_STYLE}</style>",
            f'<rect class="bg" x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="12"/>',
            f'<g transform="translate({PAD},{PAD})">']


def load_scorecard() -> dict:
    return json.loads((SCORE / "scorecard.json").read_text(encoding="utf-8"))


def totals(sc: dict, system: str) -> dict:
    qs = sc["systems"][system]["questions"].values()
    n = sum(q["runs"] for q in qs)
    return {
        "passed": sum(q["passed"] for q in qs),
        "judged": sum(q["judged"] for q in qs),
        "calls": sum(q["mean_tool_calls"] * q["runs"] for q in qs) / n,
        "tokens": sum(q["mean_memory_tokens"] * q["runs"] for q in qs) / n,
    }


def headline_svg(sc: dict) -> str:
    h, t = totals(sc, "harness"), totals(sc, "minimi_shape")
    metrics = [
        ("Answers judged correct", "out of 24 runs, higher is better", h["passed"], t["passed"], 24,
         lambda v: f"{v:.0f} / 24"),
        ("Tool calls per question", "lower is better", h["calls"], t["calls"], 10, lambda v: f"{v:.1f}"),
        ("Tokens read from memory per question", "lower is better", h["tokens"], t["tokens"], 9000,
         lambda v: f"{v:,.0f}"),
    ]
    W, x0, bw, row = 772, 170, 500, 26
    out = svg_open(W, 330, [
        '<title id="t">minimi memory harness vs minimi today, scenario D</title>',
        f'<desc id="d">Correct runs: harness {h["passed"]} of 24, minimi today {t["passed"]} of 24. '
        f'Tool calls: {h["calls"]:.1f} vs {t["calls"]:.1f}. Tokens read: {h["tokens"]:.0f} vs {t["tokens"]:.0f}.</desc>'])
    y = 18
    for i, (title, sub, hv, tv, mx, fmt) in enumerate(metrics):
        out.append(f'<text x="0" y="{y}" font-size="15" font-weight="600">{title}</text>')
        out.append(f'<text x="{W}" y="{y}" font-size="12" class="muted" text-anchor="end">{sub}</text>')
        y += 14
        out.append(f'<line x1="{x0}" y1="{y}" x2="{x0}" y2="{y + 2 * row + 6}" class="grid"/>')
        for j, (name, v, cls) in enumerate((("harness", hv, "h"), ("minimi today", tv, "t"))):
            yy = y + 6 + j * row
            width = max(4, bw * v / mx)
            delay = 0.15 + i * 0.35 + j * 0.12
            weight = ' font-weight="600"' if cls == "h" else ""
            out.append(f'<text x="{x0 - 10}" y="{yy + 14}" font-size="13" text-anchor="end"{weight}>{name}</text>')
            out.append(f'<rect class="bar {cls}" x="{x0}" y="{yy + 2}" width="{width:.1f}" height="16" rx="4" '
                       f'style="animation-delay:{delay:.2f}s"><title>{name}: {fmt(v)}</title></rect>')
            out.append(f'<text class="fade" x="{x0 + width + 8:.1f}" y="{yy + 15}" font-size="13" '
                       f'font-weight="600" style="animation-delay:{delay + 0.9:.2f}s">{fmt(v)}</text>')
        y += 2 * row + 52
    out.append("</g></svg>")
    return "\n".join(out)


def verdicts(system: str, qid: str) -> list[bool]:
    out = []
    for r in (1, 2, 3):
        p = SCORE / system / f"{qid}-{r}.verdict.json"
        out.append(json.loads(p.read_text(encoding="utf-8"))["pass"])
    return out


def per_question_svg(sc: dict) -> str:
    W, top, row = 772, 58, 40
    xq, xt, xh, xc = 0, 390, 500, 610
    H = top + row * (len(QUESTIONS) - 1) + 14
    out = svg_open(W, H, [
        '<title id="t">Per question: 3 runs each, filled dot = judged correct, hollow dot = judged wrong</title>',
        '<desc id="d">Each row is one question; dots are the three runs per system; bars are mean tool calls.</desc>'])
    out += [f'<text x="{xq}" y="22" font-size="13" class="muted">Question (3 runs each)</text>',
           f'<circle cx="{xt + 5}" cy="17" r="5" class="t"/>',
           f'<text x="{xt + 15}" y="22" font-size="13" font-weight="600">minimi today</text>',
           f'<circle cx="{xh + 5}" cy="17" r="5" class="h"/>',
           f'<text x="{xh + 15}" y="22" font-size="13" font-weight="600">harness</text>',
           f'<text x="{xc}" y="22" font-size="13" class="muted">tool calls per run</text>',
           f'<text x="{xt}" y="40" font-size="11" class="muted">● correct  ○ wrong</text>']
    max_calls = 10
    for i, (qid, q) in enumerate(QUESTIONS.items()):
        y = top + i * row
        d = 0.1 + i * 0.12
        out.append(f'<g class="fade" style="animation-delay:{d:.2f}s">')
        out.append(f'<line x1="0" y1="{y - 14}" x2="{W}" y2="{y - 14}" class="grid"/>')
        out.append(f'<text x="{xq}" y="{y + 5}" font-size="13"><tspan font-weight="600">{qid}</tspan>  {html.escape(q)}</text>')
        for x, system, cls in ((xt, "minimi_shape", "t"), (xh, "harness", "h")):
            for k, ok in enumerate(verdicts(system, qid)):
                cx = x + 8 + k * 22
                if ok:
                    out.append(f'<circle cx="{cx}" cy="{y}" r="7" class="{cls}"/>')
                else:
                    out.append(f'<circle cx="{cx}" cy="{y}" r="6" fill="none" stroke-width="2" class="{cls}s"/>')
        out.append("</g>")
        for j, (system, cls) in enumerate((("minimi_shape", "t"), ("harness", "h"))):
            v = sc["systems"][system]["questions"][qid]["mean_tool_calls"]
            w = max(3, 110 * v / max_calls)
            yy = y - 9 + j * 10
            out.append(f'<rect class="bar {cls}" x="{xc}" y="{yy}" width="{w:.1f}" height="8" rx="3" '
                       f'style="animation-delay:{d + 0.2:.2f}s"><title>{system}: {v:.1f} calls</title></rect>')
            out.append(f'<text class="fade muted" x="{xc + w + 6:.1f}" y="{yy + 8}" font-size="11" '
                       f'style="animation-delay:{d + 0.9:.2f}s">{v:.1f}</text>')
    out.append("</g></svg>")
    return "\n".join(out)


# screenshots

PAGE_CSS = """
* { box-sizing: border-box; }
body { margin: 0; background: #0d1117; color: #e6edf3; font: 15px/1.5 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }
.frame { padding: 22px 24px; }
.q { font-size: 13px; color: #9198a1; text-transform: uppercase; letter-spacing: .06em; margin-bottom: 4px; }
h1 { font-size: 21px; margin: 0 0 16px; font-weight: 600; }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.card { background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 14px 16px; position: relative; overflow: hidden; }
.card.clamp { max-height: var(--h); }
.card.clamp::after { content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: 60px; background: linear-gradient(transparent, #161b22); }
.who { display: flex; align-items: center; gap: 8px; font-weight: 600; margin-bottom: 10px; }
.dot { width: 10px; height: 10px; border-radius: 50%; }
.meta { margin-left: auto; font-weight: 400; font-size: 12px; color: #9198a1; }
.badge { font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 999px; }
.pass { background: #12361f; color: #56d364; } .fail { background: #3d1418; color: #f47067; }
.ans p { margin: 0 0 8px; } .ans ul { margin: 0 0 8px; padding-left: 18px; } .ans li { margin: 2px 0; }
.ans blockquote { margin: 0 0 8px; padding-left: 10px; border-left: 3px solid #30363d; color: #9198a1; }
pre { margin: 0; white-space: pre-wrap; font: 12.5px/1.55 ui-monospace, "Cascadia Code", Consolas, monospace; color: #c9d1d9; }
.hit { display: flex; gap: 8px; align-items: baseline; padding: 5px 0; border-bottom: 1px solid #21262d; font-size: 13px; }
.chip { flex: none; font-size: 11px; padding: 1px 7px; border-radius: 999px; background: #21262d; color: #9198a1; }
.chip.promo { background: #3d2a14; color: #f0a35e; }
.stale { color: #f47067; font-size: 11px; flex: none; }
.foot { margin-top: 12px; font-size: 12px; color: #9198a1; }
.arrow { text-align: center; color: #9198a1; font-size: 13px; margin: 10px 0; }
.added { color: #56d364; }
"""


def md(text: str) -> str:
    out, para, items = [], [], []

    def inline(s: str) -> str:
        s = html.escape(s)
        s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
        s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?!\w)", r"<i>\1</i>", s)
        return s

    def flush():
        if para:
            out.append(f"<p>{inline(' '.join(para))}</p>")
            para.clear()
        if items:
            out.append("<ul>" + "".join(f"<li>{inline(i)}</li>" for i in items) + "</ul>")
            items.clear()

    for line in text.splitlines():
        s = line.strip()
        if not s:
            flush()
        elif s.startswith(("- ", "* ")):
            if para:
                flush()
            items.append(s[2:])
        elif s.startswith(">"):
            flush()
            out.append(f"<blockquote>{inline(s.lstrip('> '))}</blockquote>")
        elif s.startswith("#"):
            flush()
            out.append(f"<p><b>{inline(s.lstrip('# '))}</b></p>")
        else:
            if items:
                flush()
            para.append(s)
    flush()
    return "".join(out)


def run(system: str, qid: str, n: int = 1) -> tuple[dict, dict]:
    d = json.loads((SCORE / system / f"{qid}-{n}.json").read_text(encoding="utf-8"))
    v = json.loads((SCORE / system / f"{qid}-{n}.verdict.json").read_text(encoding="utf-8"))
    return d, v


def answer_card(name: str, color: str, d: dict, v: dict, height: int) -> str:
    calls = len(d["tool_calls"])
    badge = '<span class="badge pass">JUDGED CORRECT</span>' if v["pass"] else '<span class="badge fail">JUDGED WRONG</span>'
    return (f'<div class="card clamp" style="--h:{height}px"><div class="who"><span class="dot" style="background:{color}"></span>'
            f'{name} {badge}<span class="meta">{calls} tool call{"s" if calls != 1 else ""}</span></div>'
            f'<div class="ans">{md(d["answer"])}</div></div>')


def page_c1() -> str:
    t, tv = run("minimi_shape", "C1")
    h, hv = run("harness", "C1")
    return (f'<div class="frame"><div class="q">C1 · run 1 of 3 · same month, same reader (Opus)</div>'
            f'<h1>"What time is dinner with Sam tomorrow?"</h1><div class="cols">'
            f'{answer_card("minimi today", TODAY_DK, t, tv, 470)}{answer_card("harness", HARNESS_DK, h, hv, 470)}'
            f'</div><div class="foot">Sam moved the Nov 20 dinner from 8 to 9:30 by email; on Nov 24 he asked '
            f'"Dinner Friday, same as last time?". The right answer is 9:30 PM.</div></div>')


def page_c8() -> str:
    t, _ = run("minimi_shape", "C8")
    h, _ = run("harness", "C8")
    search = next(c for c in t["tool_calls"] if c["name"].endswith("search_memory"))
    hits = []
    for r in search["result"]["results"][:10]:
        app = r["source_app"]
        promo = app == "gmail_connector"
        stale = " (Nov 7, already past)" if "2026-11-04" in r["memory"] and "brunch" in r["memory"].lower() else ""
        text = re.sub(r"\s*(on|at)\s*2026-\d\d-\d\dT[\d:+]+", "", r["memory"])
        chip = '<span class="chip promo">promo email</span>' if promo else f'<span class="chip">{app}</span>'
        flag = f'<span class="stale">{stale}</span>' if stale else ""
        hits.append(f'<div class="hit">{chip}<span>{html.escape(text)}</span>{flag}</div>')
    pack = h["tool_calls"][0]["result"]["result"]
    t_calls, h_calls = len(t["tool_calls"]), len(h["tool_calls"])
    q = html.escape(search["input"]["query"])
    return (f'<div class="frame"><div class="q">C8 · run 1 of 3 · what Claude actually reads</div>'
            f'<h1>"What\'s on this weekend?"</h1><div class="cols">'
            f'<div class="card"><div class="who"><span class="dot" style="background:{TODAY_DK}"></span>minimi today'
            f'<span class="meta">first search_memory call · top 10 hits · {t_calls} calls in this run</span></div>'
            f'{"".join(hits)}</div>'
            f'<div class="card"><div class="who"><span class="dot" style="background:{HARNESS_DK}"></span>harness'
            f'<span class="meta">minimi_get_context · the whole pack · {h_calls} call in this run</span></div>'
            f'<pre>{html.escape(pack)}</pre></div></div>'
            f'<div class="foot">Promo mail from the new Gmail connector is gated before it becomes memory; '
            f'events are filtered to Sat Nov 28 to Sun Nov 29, so the Nov 7 brunch never shows up as a peer.</div></div>')


def page_live(before: str, after: str, baseline_lines: list[str]) -> str:
    new, _, rest = after.partition("Changed recently:")
    new, rest = new.strip(), "Changed recently:" + rest if rest else ""
    base = "".join(f'<div class="hit"><span class="chip">new memory</span><span>{html.escape(x)}</span></div>'
                   for x in baseline_lines)
    return (f'<div class="frame"><div class="q">live moment · two new WhatsApp messages at 7:55 PM</div>'
            f'<h1>Sam: "Can we push tomorrow\'s dinner to 10?" · You: "Sure, 10 works"</h1>'
            f'<div class="cols"><div class="card"><div class="who"><span class="dot" style="background:{TODAY_DK}"></span>'
            f'minimi today<span class="meta">one memory per message, appended</span></div>{base}'
            f'<div class="foot">The 9:30 PM memories and calendar entry are still there as peers. '
            f'Which one is current is left to the reader.</div></div>'
            f'<div class="card"><div class="who"><span class="dot" style="background:{HARNESS_DK}"></span>harness'
            f'<span class="meta">one reconcile call, then minimi_get_context</span></div>'
            f'<pre>{html.escape(before)}</pre><div class="arrow">↓ after the two messages</div>'
            f'<pre class="added">{html.escape(new)}</pre><pre>{html.escape(rest)}</pre></div></div></div>')


def shoot(body: str, name: str, height: int, width: int = 1200) -> None:
    tmp = Path(tempfile.mkdtemp(prefix="readme-shot-"))
    page = tmp / "p.html"
    page.write_text(f"<!doctype html><meta charset=utf-8><style>{PAGE_CSS}</style>{body}", encoding="utf-8")
    out = IMG / f"{name}.png"
    subprocess.run(
        [str(CHROME), "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--user-data-dir={tmp / 'p'}",
         f"--window-size={width},{height}", "--force-device-scale-factor=2", f"--screenshot={out}", page.as_uri()],
        check=True, capture_output=True, timeout=120,
    )
    print(f"wrote {out.relative_to(ROOT)}")


def pack_lines(db: Path, question: str) -> str:
    from harness.retrieve import Retriever
    from harness.store import Store
    from harness.text import Embedder

    text = Retriever(Store(db), Embedder()).get_context(question)
    keep = [ln for ln in text.splitlines() if ln and not ln.startswith(("as_of", "Footer", "events:"))]
    return "\n".join(keep)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    IMG.mkdir(parents=True, exist_ok=True)
    sc = load_scorecard()
    (IMG / "headline.svg").write_text(headline_svg(sc), encoding="utf-8")
    (IMG / "per_question.svg").write_text(per_question_svg(sc), encoding="utf-8")
    print("wrote docs/img/headline.svg, docs/img/per_question.svg")
    shoot(page_c1(), "c1_dinner", 585)
    shoot(page_c8(), "c8_what_claude_reads", 630)
    q = "What time is dinner with Sam tomorrow?"
    live = ROOT / "runs" / "livetest"
    before = pack_lines(ROOT / "runs" / "harness" / "d.db", q)
    after = pack_lines(live / "h.db", q)
    import sqlite3

    con = sqlite3.connect(live / "b.db")
    baseline = [r[0] for r in con.execute(
        "SELECT m.text FROM memories m JOIN messages s ON s.id = m.message_id"
        " WHERE s.sent_at >= '2026-11-26T19:55' ORDER BY s.sent_at")]
    shoot(page_live(before, after, baseline), "live_moment", 470)


if __name__ == "__main__":
    main()

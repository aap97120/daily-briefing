#!/usr/bin/env python3
"""
David's Daily Morning Briefing — Anthropic API Version
Uses Claude (claude-sonnet-4-6) with built-in web search.
Four sections: Work | General AI | Personal | Data Science Topic of the Day
Plus AI Strategy Reads (Jaya Gupta, Benedict Evans, Stratechery etc.)

Two-pass link resolution:
  Pass 1: Generate story summaries and source publication names
  Pass 2: For each story, do a targeted search to resolve real article URLs

Publishes to GitHub Pages (docs/index.html) and sends ntfy push notification.
Archives dated copies to docs/archive/YYYY-MM-DD.html
"""

import os
import json
import re
import time
import datetime
import urllib.request
import urllib.error
import urllib.parse
import socket

# ── Config ────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
NTFY_TOPIC        = os.environ.get("NTFY_TOPIC", "")
PAGES_URL         = os.environ.get("PAGES_URL", "")
MODEL             = "claude-sonnet-4-6"
API_URL           = "https://api.anthropic.com/v1/messages"
# ─────────────────────────────────────────────────────────────────────────────

TODAY     = datetime.datetime.now().strftime("%A %d %B %Y")
TODAY_ISO = datetime.datetime.now().strftime("%Y-%m-%d")

# ── D&B strategic context baked into the system prompt ───────────────────────
SYSTEM_PROMPT = f"""You are the daily briefing assistant for David Anderson, Global Head of Sales & Marketing Analytics at Dun & Bradstreet (D&B).

D&B's core strategic position: positioning as the trusted, neutral data layer for agentic AI workflows — "AI Decision Intelligence." D&B's proprietary assets (Paydex, Buydex, DUNS entity resolution, corporate family trees, 600M+ business records) constitute a structural moat that makes any AI agent decisively better. The "be everywhere" / Visa analogy frames D&B's neutrality as a competitive advantage — embedding into all agentic platforms (Microsoft Copilot, Salesforce Agentforce, Google Gemini, Amazon Bedrock, ServiceNow, LangChain) rather than betting on one winner.

Key strategic concepts: Explainability First (Ground Truth), Temporal Analytics (buying windows), Entity Resolution as Foundation, six agent archetypes (Prospecting, Account Intelligence, Pipeline Risk & Revenue Protection, Territory & Market Planning, Account Expansion, Fleet & Asset).

When writing briefing sections, apply this context naturally — note when stories validate or challenge D&B's strategy, flag competitor moves, and highlight M&A/valuation implications where relevant. Write in a direct, plain-spoken style. No jargon.

CRITICAL ACCURACY RULE — READ CAREFULLY:
Today's date is {TODAY}. Treat this as a hard, non-negotiable cutoff between the past (reportable) and the future (not yet happened).
- Only report an event, score, result, or outcome if a search result EXPLICITLY confirms it has already concluded.
- If a search result is a preview, prediction, betting odds, fixture announcement, or forecast of a future or in-progress event, you must NOT report it as a completed result. Describe it as upcoming/scheduled instead, with the date if known.
- If you are not certain an event has concluded, do not guess. State that it is scheduled or ongoing rather than inventing a score, winner, or outcome.
- This applies especially to sports fixtures, elections, product launches, and earnings — anything with a specific date attached. Fabricating a result for an event that has not yet happened is a serious factual error and is not acceptable under any circumstances."""

# ── Section history logs (prevents repetition — same pattern as DS/Strategy) ──
WORK_LOG       = "docs/work_log.json"
GENERAL_AI_LOG = "docs/general_ai_log.json"
PERSONAL_LOG   = "docs/personal_log.json"

def load_recent_headlines(log_path: str, n: int = 10) -> list:
    """Reads the last n headlines from a section log, to exclude from today's prompt."""
    try:
        with open(log_path, "r") as f:
            entries = json.load(f)
        return [e["headline"] for e in entries[-n:]]
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_headlines(log_path: str, stories: list):
    """Appends today's headlines to a section log."""
    try:
        with open(log_path, "r") as f:
            entries = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        entries = []
    for s in stories:
        headline = s.get("headline", "")
        if headline:
            entries.append({"date": TODAY_ISO, "headline": headline})
    entries = entries[-80:]
    os.makedirs("docs", exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(entries, f, indent=2)

def _exclusion_clause(log_path: str, n: int = 10) -> str:
    recent = load_recent_headlines(log_path, n)
    if not recent:
        return ""
    recent_short = [h[:90] for h in recent]
    return (
        f"\n\nIMPORTANT — these stories were already covered in recent briefings, "
        f"do NOT repeat them even if they're still the top result: {'; '.join(recent_short)}. "
        f"Actively look for different, newer stories instead. Ongoing sagas are fine to "
        f"revisit only if there is a genuinely new development since the last time."
    )


def build_work_prompt() -> str:
    return f"""Today is {TODAY}. Search for the 5 most relevant and recent news stories for a senior analytics leader at Dun & Bradstreet, covering:
- Agentic AI and AI agents for enterprise/B2B use cases
- B2B data and data intelligence platforms
- Revenue intelligence, sales AI tools (ZoomInfo, Apollo, Salesforce Einstein, Gong, Clari)
- Business credit, financial risk and commercial data
- D&B, Moody's Analytics, Verisk or similar B2B data company news

Only include a story if search results confirm it as genuinely current news (published recently, not a prediction of something that hasn't happened).{_exclusion_clause(WORK_LOG)}

For each story return:
- headline: clear headline
- summary: 2-3 sentence summary with strategic framing for D&B
- source: publication name only (e.g. "Reuters", "TechCrunch") — NO URLs
- tag: category tag

Return ONLY valid JSON, no fences:
{{"stories":[{{"headline":"","summary":"","source":"","tag":""}}]}}"""


def build_general_ai_prompt() -> str:
    return f"""Today is {TODAY}. Search for the 4 most significant recent stories in the broader AI industry — funding/M&A, frontier model releases, compute/infrastructure economics, notable policy or regulatory developments, AI-native startup moves.

Only include a story if search results confirm it as genuinely current news, not a rumour, prediction, or something scheduled for the future that hasn't happened yet.{_exclusion_clause(GENERAL_AI_LOG)}

For each story return:
- headline: clear headline
- summary: 2-3 sentence summary explaining what happened and why it matters
- source: publication name only — NO URLs
- tag: category tag

Return ONLY valid JSON, no fences:
{{"stories":[{{"headline":"","summary":"","source":"","tag":""}}]}}"""


def build_personal_prompt() -> str:
    return f"""Today is {TODAY}. Search for 6 interesting recent news stories for a UK-based professional across: UK politics, US politics/international affairs, science (space/biology/physics), golf (PGA/DP World Tour), football/soccer (Premier League/World Cup), rugby (Six Nations/Premiership), plus ONE from F1/economics/culture.

SPORTS RESULTS — CRITICAL: for any match, race, or tournament, only report a final score or result if search results explicitly confirm the event has already been played and concluded as of {TODAY}. If you find a preview, prediction, or fixture announcement for an event that hasn't happened yet, report it as an upcoming fixture with the scheduled date — never invent or infer a score. Double-check the event date against today's date before stating any result.{_exclusion_clause(PERSONAL_LOG)}

For each story return:
- headline: clear headline
- summary: 2-3 sentence summary
- source: publication name only — NO URLs
- tag: category (e.g. "UK Politics", "Golf", "World Cup")

Return ONLY valid JSON, no fences:
{{"stories":[{{"headline":"","summary":"","source":"","tag":""}}]}}"""


TOPICS_LOG   = "docs/ds_topics.json"       # persists recent DS topics to avoid repetition
STRATEGY_LOG = "docs/strategy_reads.json"  # persists recent strategy pieces to avoid repetition

def load_recent_topics(n: int = 14) -> list:
    """Reads the last n data science topics covered, to pass into the prompt."""
    try:
        with open(TOPICS_LOG, "r") as f:
            entries = json.load(f)
        return [e["topic"] for e in entries[-n:]]
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_topic(topic: str):
    """Appends today's topic to the log file."""
    try:
        with open(TOPICS_LOG, "r") as f:
            entries = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        entries = []
    entries.append({"date": TODAY_ISO, "topic": topic})
    entries = entries[-60:]
    os.makedirs("docs", exist_ok=True)
    with open(TOPICS_LOG, "w") as f:
        json.dump(entries, f, indent=2)

def load_recent_strategy_reads(n: int = 7) -> list:
    """Reads the last n strategy piece headlines to exclude from tomorrow's prompt."""
    try:
        with open(STRATEGY_LOG, "r") as f:
            entries = json.load(f)
        return [e["headline"] for e in entries[-n:]]
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_strategy_reads(stories: list):
    """Saves today's strategy headlines to the log."""
    try:
        with open(STRATEGY_LOG, "r") as f:
            entries = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        entries = []
    for s in stories:
        headline = s.get("headline", "")
        if headline:
            entries.append({"date": TODAY_ISO, "headline": headline})
    entries = entries[-60:]
    os.makedirs("docs", exist_ok=True)
    with open(STRATEGY_LOG, "w") as f:
        json.dump(entries, f, indent=2)

def build_strategy_prompt() -> str:
    recent = load_recent_strategy_reads()
    exclusion = ""
    if recent:
        recent_short = [h[:80] for h in recent]
        exclusion = f"\n\nIMPORTANT — these pieces have already been featured recently, do NOT include them again: {'; '.join(recent_short)}. Find different pieces."
    return f"""Today is {TODAY}. Search for 3 high-signal analytical pieces published or widely circulated in the LAST 7 DAYS about AI strategy, market dynamics, enterprise AI adoption, AI M&A, or technology business model shifts. Prioritise content from or referencing: Jaya Gupta (Foundation Capital, @JayaGup10), Benedict Evans (@benedictevans), Ben Thompson (Stratechery, @stratechery), Matt Turck (@mattturck), Ethan Mollick (@emollick), Andrej Karpathy (@karpathy). Also include strong pieces from other credible analysts if found. Only include a piece if search results confirm it was actually published — do not infer or guess at its contents.{exclusion}

For each piece return:
- headline: title or key thesis
- summary: 3-4 sentence summary of the argument, why it matters, and how it connects to D&B's AI positioning
- source: author name and publication — NO URLs
- tag: "AI Strategy"

Return ONLY valid JSON, no fences:
{{"stories":[{{"headline":"","summary":"","source":"","tag":""}}]}}"""

def build_ds_prompt() -> str:
    recent = load_recent_topics()
    exclusion = ""
    if recent:
        exclusion = f"\n\nIMPORTANT — do NOT choose any of these recently covered topics: {', '.join(recent)}. Pick something different."
    return f"""Today is {TODAY}. Choose ONE data science topic to explain in depth, rotating broadly across: classical ML (XGBoost, Random Forests, survival analysis), deep learning (transformers, embeddings, fine-tuning), MLOps (MLflow, feature stores, model monitoring), agentic AI (LangChain, LangGraph, MCP, multi-agent orchestration), statistical methods (Bayesian inference, causal inference, SHAP, A/B testing), emerging techniques (RAG, multimodal models, RLHF), data engineering (dbt, Spark, vector databases, knowledge graphs).{exclusion}

Return ONLY valid JSON, no fences:
{{"topic":"","tagline":"","what_it_is":"","how_it_works":"","when_to_use":"","worked_example":"(concrete B2B/D&B context example)","resources":["descriptive name of resource 1","descriptive name of resource 2","descriptive name of resource 3"]}}"""

LINK_RESOLUTION_PROMPT = """Search for this specific news article and return its direct URL.

Article details:
- Headline: {headline}
- Publication: {source}
- Date: approximately {date}

Instructions:
- Search for the article using the headline and publication name
- Return ONLY the direct URL to the article on the publication's own website
- The URL must start with https://
- Do not include any explanation, punctuation, or other text
- If you find the article, return its URL (e.g. https://techcrunch.com/2026/06/27/article-name/)
- If you cannot find the exact article, return the homepage of the publication (e.g. https://techcrunch.com)"""


def call_claude(prompt: str, system: str = SYSTEM_PROMPT, max_retries: int = 3,
                 max_tokens: int = 2000, search_max_uses: int = 5) -> str:
    """Calls the Anthropic API with web search enabled. Returns raw text response.

    search_max_uses caps how many searches Claude can run per call — without
    this, a single "find 5 stories" request can trigger 10+ searches, and each
    search is billed separately ($10/1,000) on top of token costs. Main section
    calls need room to research multiple stories (default 5); URL resolution
    calls only need one targeted lookup and should pass a much lower value.
    """
    payload = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "system": system,
        "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": search_max_uses}],
        "messages": [{"role": "user", "content": prompt}],
    }
    data = json.dumps(payload).encode("utf-8")

    last_error = None
    for attempt in range(1, max_retries + 1):
        req = urllib.request.Request(
            API_URL,
            data=data,
            headers={
                "Content-Type": "application/json",
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            # Extract text from content blocks
            text = ""
            for block in result.get("content", []):
                if block.get("type") == "text":
                    text += block["text"]
            return text.strip()
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            last_error = RuntimeError(f"Anthropic API error {e.code}: {body[:300]}")
            if e.code in (429, 529) and attempt < max_retries:
                wait = 2 ** attempt
                print(f"    (attempt {attempt}/{max_retries} got {e.code}, retrying in {wait}s…)")
                time.sleep(wait)
                continue
            raise last_error
        except (socket.timeout, TimeoutError, urllib.error.URLError) as e:
            last_error = RuntimeError(f"Network error: {e}")
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"    (attempt {attempt}/{max_retries} got network error, retrying in {wait}s…)")
                time.sleep(wait)
                continue
            raise last_error
    raise last_error


def extract_json(text: str) -> dict:
    """Robustly extracts JSON from model response."""
    text = text.strip()
    # Strip markdown fences
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    text = text.strip()

    # Attempt 1: parse directly
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt 2: find balanced braces
    start = text.find("{")
    if start == -1:
        raise RuntimeError(f"No JSON found in response: {text[:200]}")

    depth = 0
    end = None
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    candidate = text[start:end + 1] if end is not None else text[start:]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Attempt 3: repair embedded unescaped quotes
    repaired = []
    in_string = False
    escape_next = False
    i = 0
    chars = list(candidate)
    n = len(chars)
    while i < n:
        c = chars[i]
        if escape_next:
            repaired.append(c)
            escape_next = False
            i += 1
            continue
        if c == "\\":
            repaired.append(c)
            escape_next = True
            i += 1
            continue
        if c == '"':
            if not in_string:
                in_string = True
                repaired.append(c)
            else:
                j = i + 1
                while j < n and chars[j] in " \t\n\r":
                    j += 1
                if j < n and chars[j] in ",:}]":
                    in_string = False
                    repaired.append(c)
                else:
                    repaired.append('\\"')
            i += 1
            continue
        repaired.append(c)
        i += 1

    return json.loads("".join(repaired))


def resolve_url(headline: str, source: str) -> str:
    """Pass 2: targeted search to find the real article URL.

    Returns the best available URL in order of preference:
    1. A real article URL found by Claude's web search
    2. A plain Google search URL (reliable fallback, never 404s)

    Deliberately capped at max_uses=2 and max_tokens=300 — this call only
    needs to return a single URL, not conduct research. Uncapped, this was
    a major driver of search-fee cost (one full API call per story, every
    story, every day).
    """
    prompt = LINK_RESOLUTION_PROMPT.format(
        headline=headline,
        source=source,
        date=TODAY,
    )
    try:
        raw = call_claude(
            prompt,
            system="You are a URL resolver. Search for the article and return only the direct URL to it. Return ONLY the URL — no explanation, no punctuation before or after, just the bare URL starting with https://",
            max_retries=2,
            max_tokens=300,
            search_max_uses=2,
        )
        raw = raw.strip().rstrip(".,;)")

        urls = re.findall(r'https?://[^\s\'"<>\]\)]+', raw)
        for url in urls:
            url = url.rstrip(".,;)")
            if (
                "news.google.com/search" not in url
                and "google.com/search" not in url
                and len(url) > 20
            ):
                return url
    except Exception as e:
        print(f"    URL resolution failed for '{headline[:50]}…': {e}")

    query = urllib.parse.quote_plus(headline[:80])
    return f"https://www.google.com/search?q={query}"


def fallback_search_link(headline: str, source: str) -> str:
    """Zero-cost link — used for sections where we skip the API-based
    resolution pass entirely (General AI, Personal) to control cost."""
    query = urllib.parse.quote_plus(f"{headline} {source}".strip())
    return f"https://www.google.com/search?q={query}"


def add_links_to_stories(stories: list, section_name: str, resolve: bool = True) -> list:
    """For each story, attach a URL. If resolve=True, spends one API call per
    story to find the real article link. If resolve=False, uses a zero-cost
    Google search link instead — used for lower-priority sections to control
    total API spend."""
    for i, story in enumerate(stories):
        headline = story.get("headline", "")
        source = story.get("source", "")
        if resolve:
            print(f"    Resolving link {i+1}/{len(stories)}: {headline[:60]}…")
            story["url"] = resolve_url(headline, source)
            time.sleep(2)
        else:
            story["url"] = fallback_search_link(headline, source)
    return stories


def make_link_html(url: str, label: str) -> str:
    """Returns a styled anchor tag."""
    return f'<a href="{url}" target="_blank" rel="noopener" class="link-btn">📰 {label}</a>'


def make_resource_link(resource_name: str) -> str:
    encoded = urllib.parse.quote_plus(resource_name)
    return f'<a href="https://www.google.com/search?q={encoded}" target="_blank" rel="noopener" class="link-btn">🔗 {resource_name}</a>'


def generate_html(data: dict, filepath: str):
    """Generates the full responsive HTML briefing page."""

    def story_card(s, badge_class):
        url = s.get("url", "#")
        link_html = make_link_html(url, s.get("source", "Source"))
        relevance = s.get("relevance", "")
        relevance_html = f'<div class="relevance"><strong>Strategy Impact:</strong> {relevance}</div>' if relevance else ""
        return (
            f'<div class="card">'
            f'<span class="badge {badge_class}">{s.get("tag","")}</span>'
            f'<h3>{s.get("headline","")}</h3>'
            f'<p>{s.get("summary","")}</p>'
            f'{relevance_html}'
            f'<div class="links-container">{link_html}</div>'
            f'</div>'
        )

    work_html     = "".join(story_card(s, "badge-work")     for s in data["work"].get("stories", []))
    ai_html       = "".join(story_card(s, "badge-ai")       for s in data["general_ai"].get("stories", []))
    personal_html = "".join(story_card(s, "badge-personal") for s in data["personal"].get("stories", []))
    strategy_html = "".join(story_card(s, "badge-strategy") for s in data["strategy"].get("stories", []))

    ds = data["data_science"]
    ds_resources = " ".join(make_resource_link(r) for r in ds.get("resources", []))
    ds_html = (
        f'<div class="ds-hero">'
        f'<h2>🎓 {ds.get("topic","")}</h2>'
        f'<p class="tagline"><em>{ds.get("tagline","")}</em></p>'
        f'<div class="ds-grid">'
        f'<div><h4>What It Is</h4><p>{ds.get("what_it_is","")}</p></div>'
        f'<div><h4>How It Works</h4><p>{ds.get("how_it_works","")}</p></div>'
        f'<div><h4>When To Use</h4><p>{ds.get("when_to_use","")}</p></div>'
        f'</div>'
        f'<div class="worked-example"><h4>🎯 Worked Example — D&amp;B Context</h4><p>{ds.get("worked_example","")}</p></div>'
        f'<div class="links-container" style="margin-top:1.5rem">{ds_resources}</div>'
        f'</div>'
    )

    css = """
    :root{--bg:#f8fafc;--surface:#fff;--text:#0f172a;--border:#e2e8f0;--work:#0ea5e9;--personal:#10b981;--ai:#8b5cf6;--strategy:#f59e0b}
    *{box-sizing:border-box;margin:0;padding:0}
    body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:var(--bg);color:var(--text);line-height:1.6;padding:2rem 1rem}
    .container{max-width:1100px;margin:0 auto}
    header{text-align:center;margin-bottom:3rem;border-bottom:2px solid var(--border);padding-bottom:1.5rem}
    h1{font-size:2.25rem;color:#1e293b;margin-bottom:0.5rem}
    .date{font-size:1.1rem;color:#64748b;font-weight:500}
    .section-title{font-size:1.5rem;margin:2.5rem 0 1rem;padding-left:0.75rem;color:#1e293b}
    .work-title{border-left:4px solid var(--work)}
    .ai-title{border-left:4px solid var(--ai)}
    .personal-title{border-left:4px solid var(--personal)}
    .strategy-title{border-left:4px solid var(--strategy)}
    .ds-title{border-left:4px solid #38bdf8}
    .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:1.5rem}
    .card{background:var(--surface);border:1px solid var(--border);padding:1.5rem;border-radius:12px;box-shadow:0 1px 3px rgba(0,0,0,.05);display:flex;flex-direction:column}
    .card h3{font-size:1.1rem;line-height:1.4;color:#1e293b;margin-bottom:0.75rem}
    .card p{color:#475569;font-size:.95rem;flex-grow:1;margin-bottom:1rem}
    .relevance{background:#f0f9ff;border:1px solid #bae6fd;padding:.6rem .75rem;border-radius:8px;font-size:.875rem;color:#0369a1;margin-bottom:.75rem}
    .badge{display:inline-block;padding:.25rem .5rem;font-size:.7rem;font-weight:700;border-radius:4px;text-transform:uppercase;margin-bottom:.75rem}
    .badge-work{background:#e0f2fe;color:#0369a1}
    .badge-personal{background:#d1fae5;color:#047857}
    .badge-ai{background:#f3e8ff;color:#6d28d9}
    .badge-strategy{background:#fef3c7;color:#92400e}
    .links-container{display:flex;flex-wrap:wrap;gap:.5rem}
    .link-btn{display:inline-block;background:#f1f5f9;color:#475569;text-decoration:none;padding:.35rem .75rem;border-radius:6px;font-size:.85rem;font-weight:500;transition:background .15s}
    .link-btn:hover{background:#e2e8f0;color:#0f172a}
    .ds-hero{background:#1e293b;color:#f8fafc;padding:2rem;border-radius:16px;margin-top:1rem}
    .ds-hero h2{color:#fff;margin-bottom:.25rem}
    .ds-hero .tagline{color:#94a3b8;margin:0 0 2rem;font-size:1.05rem}
    .ds-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:1.5rem;margin-bottom:1.5rem}
    .ds-grid h4{color:#38bdf8;font-size:.9rem;text-transform:uppercase;letter-spacing:.05em;margin-bottom:.5rem}
    .ds-grid p{color:#cbd5e1;font-size:.9rem}
    .worked-example{background:rgba(15,23,42,.6);padding:1.25rem;border-radius:8px;border-left:4px solid #38bdf8}
    .worked-example h4{color:#fff;margin-bottom:.5rem}
    .worked-example p{color:#cbd5e1;font-size:.9rem}
    .footer{text-align:center;font-size:.8rem;color:#94a3b8;margin-top:2rem;padding-top:1.5rem;border-top:1px solid var(--border)}
    @media(max-width:600px){body{padding:1rem .5rem}.ds-hero{padding:1.25rem}}
    """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
<meta http-equiv="Pragma" content="no-cache">
<meta http-equiv="Expires" content="0">
<title>David's Morning Briefing — {TODAY}</title>
<style>{css}</style>
</head>
<body>
<div class="container">
  <header>
    <h1>David's Daily Morning Briefing</h1>
    <div class="date">☀️ {data['date']}</div>
  </header>

  <h2 class="section-title work-title">💼 Work &amp; B2B Strategy</h2>
  <div class="grid">{work_html}</div>

  <h2 class="section-title ai-title">🤖 General AI</h2>
  <div class="grid">{ai_html}</div>

  <h2 class="section-title personal-title">🏠 Personal</h2>
  <div class="grid">{personal_html}</div>

  <h2 class="section-title strategy-title">📚 AI Strategy Reads</h2>
  <div class="grid">{strategy_html}</div>

  <h2 class="section-title ds-title">🧠 Data Science Topic of the Day</h2>
  {ds_html}

  <div class="footer">Generated with Claude {MODEL} · {TODAY}</div>
</div>
</body>
</html>"""

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)


def send_notification(message: str, url: str):
    if not NTFY_TOPIC:
        print("  NTFY_TOPIC not set, skipping notification")
        return
    req = urllib.request.Request(
        f"https://ntfy.sh/{NTFY_TOPIC}",
        data=message.encode("utf-8"),
        headers={"Title": "Morning briefing ready", "Click": url, "Tags": "sunny"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp.read()
    except Exception as e:
        print(f"  ntfy notification failed: {e}")


def generate_briefing():
    print(f"Starting briefing for {TODAY}…")

    print("  → Fetching Work section…")
    work = extract_json(call_claude(build_work_prompt(), search_max_uses=5))
    time.sleep(5)

    print("  → Fetching General AI section…")
    general_ai = extract_json(call_claude(build_general_ai_prompt(), search_max_uses=5))
    time.sleep(5)

    print("  → Fetching Personal section…")
    personal = extract_json(call_claude(build_personal_prompt(), search_max_uses=6))
    time.sleep(5)

    print("  → Fetching AI Strategy Reads…")
    strategy = extract_json(call_claude(build_strategy_prompt(), search_max_uses=5))
    time.sleep(5)

    print("  → Fetching Data Science topic…")
    ds = extract_json(call_claude(build_ds_prompt(), search_max_uses=3))

    # Save all section histories now that content generation succeeded —
    # done here (not deferred to the end) so a failure during link
    # resolution doesn't leave the exclusion logs out of sync with what
    # was actually generated.
    topic_name = ds.get("topic", "")
    if topic_name:
        save_topic(topic_name)
        print(f"  ✓ Logged DS topic: {topic_name}")

    strategy_stories = strategy.get("stories", [])
    if strategy_stories:
        save_strategy_reads(strategy_stories)
        print(f"  ✓ Logged {len(strategy_stories)} strategy reads")

    work_stories = work.get("stories", [])
    if work_stories:
        save_headlines(WORK_LOG, work_stories)
        print(f"  ✓ Logged {len(work_stories)} work headlines")

    general_ai_stories = general_ai.get("stories", [])
    if general_ai_stories:
        save_headlines(GENERAL_AI_LOG, general_ai_stories)
        print(f"  ✓ Logged {len(general_ai_stories)} general AI headlines")

    personal_stories = personal.get("stories", [])
    if personal_stories:
        save_headlines(PERSONAL_LOG, personal_stories)
        print(f"  ✓ Logged {len(personal_stories)} personal headlines")

    # Pass 2: resolve real URLs — only for Work and Strategy, the sections
    # most likely to be forwarded or acted on. General AI and Personal get
    # a zero-cost Google search link instead. This roughly halves the
    # number of URL-resolution API calls (and their search fees) per run.
    print("  → Resolving article URLs (Work)…")
    work["stories"] = add_links_to_stories(work_stories, "Work", resolve=True)

    print("  → Adding search links (General AI)…")
    general_ai["stories"] = add_links_to_stories(general_ai_stories, "General AI", resolve=False)

    print("  → Adding search links (Personal)…")
    personal["stories"] = add_links_to_stories(personal_stories, "Personal", resolve=False)

    print("  → Resolving article URLs (Strategy)…")
    strategy["stories"] = add_links_to_stories(strategy_stories, "Strategy", resolve=True)

    payload = {
        "date": TODAY,
        "work": work,
        "general_ai": general_ai,
        "personal": personal,
        "strategy": strategy,
        "data_science": ds,
    }

    # Save outputs
    docs_dir    = "docs"
    archive_dir = "docs/archive"
    os.makedirs(docs_dir, exist_ok=True)
    os.makedirs(archive_dir, exist_ok=True)

    with open(f"{docs_dir}/briefing.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(f"{archive_dir}/{TODAY_ISO}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    generate_html(payload, f"{docs_dir}/index.html")
    generate_html(payload, f"{archive_dir}/{TODAY_ISO}.html")
    print(f"  ✓ Saved docs/index.html and archive/{TODAY_ISO}.html")

    if NTFY_TOPIC and PAGES_URL:
        print("  → Sending notification…")
        archive_url = f"{PAGES_URL.rstrip('/')}/archive/{TODAY_ISO}.html"
        send_notification(f"☀️ Your briefing for {TODAY} is ready. Tap to view.", archive_url)
        print(f"  ✓ Notification sent")
    else:
        print("  (NTFY_TOPIC or PAGES_URL not set — skipping notification)")

    print("Done.")


if __name__ == "__main__":
    generate_briefing()

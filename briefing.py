#!/usr/bin/env python3
"""
David's Daily Morning Briefing — FREE VERSION
Uses Google Gemini's free-tier API (Flash models) instead of the paid
Anthropic API, so this runs at zero cost on top of GitHub Actions' free tier.

Sections: Work | Personal | General AI | Data Science Topic of the Day

Run manually: python briefing.py
Schedule:     See README.md for GitHub Actions setup
"""

import os
import json
import re
import time
import datetime
import urllib.request
import urllib.error
import socket

# ── Config ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY", "")    # free from aistudio.google.com
RESEND_API_KEY  = os.environ.get("RESEND_API_KEY", "")    # free from resend.com
EMAIL_FROM      = os.environ.get("BRIEFING_FROM_EMAIL", "onboarding@resend.dev")  # see README re: verified domains
EMAIL_TO        = os.environ.get("BRIEFING_TO_EMAIL", "")  # where the briefing should land
# ─────────────────────────────────────────────────────────────────────────────

TODAY = datetime.datetime.now().strftime("%A %d %B %Y")
MODEL = "gemini-2.5-flash"          # free-tier model with search grounding support
FALLBACK_MODEL = "gemini-2.0-flash" # used on final retry if the primary model is overloaded

def _gemini_url(model: str) -> str:
    return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"


WORK_PROMPT = f"""You are a professional news briefing assistant. Today is {TODAY}.

Search for and summarise the 5 most relevant and recent news stories across these topics for a senior analytics leader at Dun & Bradstreet:
- Agentic AI and AI agents for enterprise / B2B use cases
- B2B data and data intelligence platforms
- Revenue intelligence, sales AI tools (ZoomInfo, Apollo, Salesforce Einstein, Gong, Clari)
- Data science and analytics industry news
- Business credit, financial risk and commercial data
- Dun & Bradstreet, Moody's Analytics, Verisk or similar B2B data company news

For each story provide:
1. A clear headline
2. A 2-3 sentence summary in plain language
3. Why it matters for someone in B2B data and analytics strategy
4. 1-2 real source URLs

Return ONLY valid JSON, no markdown fences, no preamble.
{{
  "stories": [
    {{
      "headline": "string",
      "summary": "string",
      "relevance": "string",
      "tag": "string",
      "links": [{{"label": "string", "url": "string"}}]
    }}
  ]
}}"""

PERSONAL_PROMPT = f"""You are a personal news briefing assistant. Today is {TODAY}.

Search for and summarise the 6 most interesting recent news stories across these topics for a UK-based professional:
- UK politics and current affairs
- US politics and international affairs
- Science (space, biology, physics, technology breakthroughs)
- Golf (PGA Tour, DP World Tour, majors, Ryder Cup)
- Football / soccer (Premier League, Champions League, international)
- Rugby (Six Nations, Premiership, international)
- ONE interesting story from: Formula 1, economics, or culture

For each story provide:
1. A clear headline
2. A 2-3 sentence summary
3. 1-2 real source URLs

Return ONLY valid JSON, no markdown fences, no preamble.
{{
  "stories": [
    {{
      "headline": "string",
      "summary": "string",
      "tag": "string",
      "links": [{{"label": "string", "url": "string"}}]
    }}
  ]
}}"""

GENERAL_AI_PROMPT = f"""You are an AI industry news analyst. Today is {TODAY}.

Search for and summarise the 4 most significant recent stories in the broader AI industry — the kind of stories a well-informed AI professional would want to know about, beyond just B2B/enterprise data. Cover things like:
- Major AI company funding, acquisitions, or M&A (e.g. compute/infrastructure deals)
- Frontier model releases or major capability announcements (OpenAI, Anthropic, Google, Meta, xAI)
- AI compute, chips, and infrastructure economics (pricing changes, cost pressures, data centre deals)
- Notable AI policy, safety, or regulatory developments
- Notable moves by AI-native startups (coding agents, AI infrastructure, etc.)

For each story provide:
1. A clear headline
2. A 2-3 sentence summary explaining what happened and why it's significant
3. 1-2 real source URLs

Return ONLY valid JSON, no markdown fences, no preamble.
{{
  "stories": [
    {{
      "headline": "string",
      "summary": "string",
      "tag": "string",
      "links": [{{"label": "string", "url": "string"}}]
    }}
  ]
}}"""

DS_PROMPT = f"""You are a data science educator. Today is {TODAY}.

Choose ONE data science topic to explain in depth. Rotate broadly across:
- Classical ML (XGBoost, Random Forests, survival analysis)
- Deep learning (transformers, embeddings, fine-tuning)
- MLOps (MLflow, feature stores, model monitoring)
- Agentic AI (LangChain, LangGraph, AutoGen, MCP, multi-agent orchestration)
- Statistical methods (Bayesian inference, causal inference, SHAP, LIME, A/B testing)
- Emerging techniques (RAG, multimodal models, reasoning models, RLHF)
- Data engineering (dbt, Spark, vector databases, knowledge graphs)

Provide:
1. Topic name and one-line tagline
2. What it is — plain language explanation (3-4 sentences)
3. How it works — slightly more technical (4-5 sentences)
4. When to use it — practical guidance (3-4 sentences)
5. A concrete worked example relevant to B2B analytics or sales/marketing data
6. 2-3 real links to good resources

Return ONLY valid JSON, no markdown fences, no preamble.
{{
  "topic": "string",
  "tagline": "string",
  "what_it_is": "string",
  "how_it_works": "string",
  "when_to_use": "string",
  "worked_example": "string",
  "links": [{{"label": "string", "url": "string"}}]
}}"""


def call_gemini(prompt: str, max_retries: int = 4) -> dict:
    """Calls the free-tier Gemini API with Google Search grounding enabled.

    Retries on 503 (overloaded), 429 (rate limited), and network timeouts
    with exponential backoff, since these are transient and common on the
    free tier during busy periods or when search grounding takes a while.
    """
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "tools": [{"google_search": {}}],
        "generationConfig": {"temperature": 0.4},
    }
    data = json.dumps(payload).encode("utf-8")

    last_error = None
    for attempt in range(1, max_retries + 1):
        # On the final attempt, try the fallback model in case the primary is overloaded
        model = FALLBACK_MODEL if attempt == max_retries else MODEL
        req = urllib.request.Request(
            _gemini_url(model), data=data, headers={"Content-Type": "application/json"}
        )
        try:
            # 150s timeout (was 90s) — search-grounded calls can legitimately take a while
            with urllib.request.urlopen(req, timeout=150) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            break  # success
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            last_error = RuntimeError(f"Gemini API error {e.code}: {body[:300]}")
            if e.code in (503, 429) and attempt < max_retries:
                wait = 2 ** attempt  # 2, 4, 8, 16 seconds
                print(f"    (attempt {attempt}/{max_retries} got {e.code}, retrying in {wait}s…)")
                time.sleep(wait)
                continue
            raise last_error
        except (socket.timeout, TimeoutError, urllib.error.URLError) as e:
            last_error = RuntimeError(f"Gemini API network error: {e}")
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"    (attempt {attempt}/{max_retries} got network/timeout error, retrying in {wait}s…)")
                time.sleep(wait)
                continue
            raise last_error
    else:
        raise last_error

    text = ""
    try:
        candidates = result.get("candidates", [])
        for part in candidates[0]["content"]["parts"]:
            if "text" in part:
                text += part["text"]
    except (KeyError, IndexError):
        raise RuntimeError(f"Unexpected Gemini response shape: {json.dumps(result)[:300]}")

    # Strip markdown fences if present
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    text = text.strip()

    # Extract the JSON object even if extra text surrounds it
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)

    return json.loads(text)


def tag_color(tag: str) -> tuple:
    """Returns (bg, text) hex colours for a tag label."""
    tag_lower = tag.lower()
    if any(x in tag_lower for x in ["golf", "football", "rugby", "sport", "f1", "formula"]):
        return "#d1fae5", "#065f46"
    if any(x in tag_lower for x in ["uk politics", "us politics", "politics"]):
        return "#fee2e2", "#991b1b"
    if any(x in tag_lower for x in ["science", "space", "tech"]):
        return "#e0e7ff", "#3730a3"
    if any(x in tag_lower for x in ["funding", "compute", "infrastructure", "chip", "model release"]):
        return "#fef3c7", "#92400e"
    if any(x in tag_lower for x in ["ai", "agentic", "llm", "data", "b2b", "sales", "analytics"]):
        return "#dbeafe", "#1e40af"
    return "#f3f4f6", "#374151"


def render_links(links: list) -> str:
    if not links:
        return ""
    items = "".join(
        f'<a href="{l["url"]}" style="display:inline-block;margin:4px 6px 0 0;padding:3px 10px;'
        f'border:1px solid #cbd5e1;border-radius:6px;font-size:12px;color:#2563eb;text-decoration:none;">'
        f'↗ {l["label"]}</a>'
        for l in links
    )
    return f'<div style="margin-top:8px">{items}</div>'


def render_story_cards(stories: list) -> str:
    cards = ""
    for s in stories:
        bg, fg = tag_color(s.get("tag", ""))
        cards += f"""
        <div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:16px 20px;margin-bottom:12px;">
          <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:12px;margin-bottom:8px;">
            <div style="font-size:15px;font-weight:500;color:#0f172a;line-height:1.4">{s.get("headline","")}</div>
            <span style="flex-shrink:0;background:{bg};color:{fg};font-size:11px;font-weight:500;padding:3px 9px;border-radius:20px">{s.get("tag","")}</span>
          </div>
          <div style="font-size:13px;color:#475569;line-height:1.65">{s.get("summary","")}</div>
          {f'<div style="font-size:12px;color:#94a3b8;margin-top:6px;font-style:italic">{s.get("relevance","")}</div>' if s.get("relevance") else ""}
          {render_links(s.get("links",[]))}
        </div>"""
    return cards


def build_html(work: dict, personal: dict, general_ai: dict, ds: dict) -> str:
    work_cards = render_story_cards(work.get("stories", []))
    personal_cards = render_story_cards(personal.get("stories", []))
    ai_cards = render_story_cards(general_ai.get("stories", []))

    def ds_section(icon, title, body, bg="#f8fafc"):
        return f"""
        <div style="background:{bg};border-radius:8px;padding:14px 16px;margin-bottom:10px;">
          <div style="font-size:13px;font-weight:500;color:#1e293b;margin-bottom:6px">{icon} {title}</div>
          <div style="font-size:13px;color:#475569;line-height:1.65">{body}</div>
        </div>"""

    ds_card = f"""
    <div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:16px 20px;margin-bottom:12px;">
      <span style="background:#ede9fe;color:#5b21b6;font-size:11px;font-weight:500;padding:3px 10px;border-radius:20px">Today's topic</span>
      <div style="font-size:18px;font-weight:500;color:#0f172a;margin:10px 0 4px">{ds.get("topic","")}</div>
      <div style="font-size:13px;color:#64748b;font-style:italic;margin-bottom:14px">{ds.get("tagline","")}</div>
      {ds_section("🔍", "What it is", ds.get("what_it_is",""))}
      {ds_section("⚙️", "How it works", ds.get("how_it_works",""))}
      {ds_section("🎯", "When to use it", ds.get("when_to_use",""))}
      {ds_section("💡", "Worked example", ds.get("worked_example",""), bg="#eff6ff")}
      {render_links(ds.get("links",[]))}
    </div>"""

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:#f1f5f9;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif">
  <div style="max-width:640px;margin:0 auto;padding:24px 16px">

    <div style="background:#0f172a;border-radius:12px;padding:20px 24px;margin-bottom:20px;display:flex;align-items:center;justify-content:space-between">
      <div>
        <div style="font-size:18px;font-weight:500;color:#fff">Good morning, David</div>
        <div style="font-size:13px;color:#94a3b8;margin-top:2px">{TODAY}</div>
      </div>
      <div style="font-size:28px">☕</div>
    </div>

    <div style="font-size:13px;font-weight:500;color:#64748b;text-transform:uppercase;letter-spacing:0.05em;margin:0 0 10px">
      💼 Work
    </div>
    {work_cards}

    <div style="border-top:1px solid #e2e8f0;margin:20px 0"></div>

    <div style="font-size:13px;font-weight:500;color:#64748b;text-transform:uppercase;letter-spacing:0.05em;margin:0 0 10px">
      🤖 General AI
    </div>
    {ai_cards}

    <div style="border-top:1px solid #e2e8f0;margin:20px 0"></div>

    <div style="font-size:13px;font-weight:500;color:#64748b;text-transform:uppercase;letter-spacing:0.05em;margin:0 0 10px">
      🏠 Personal
    </div>
    {personal_cards}

    <div style="border-top:1px solid #e2e8f0;margin:20px 0"></div>

    <div style="font-size:13px;font-weight:500;color:#64748b;text-transform:uppercase;letter-spacing:0.05em;margin:0 0 10px">
      🧠 Data science topic of the day
    </div>
    {ds_card}

    <div style="text-align:center;font-size:11px;color:#94a3b8;margin-top:24px;padding-top:16px;border-top:1px solid #e2e8f0">
      Generated with Gemini (free tier) · {TODAY}
    </div>
  </div>
</body>
</html>"""


def send_email(html: str, subject: str):
    """Sends the briefing via Resend's free API instead of Gmail SMTP."""
    payload = {
        "from": f"Morning Briefing <{EMAIL_FROM}>",
        "to": [EMAIL_TO],
        "subject": subject,
        "html": html,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://api.resend.com/emails",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {RESEND_API_KEY}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Resend API error {e.code}: {body}")


def main():
    print(f"[{TODAY}] Generating briefing (free Gemini tier)…")

    print("  → Fetching work news…")
    work = call_gemini(WORK_PROMPT)

    print("  → Fetching general AI news…")
    general_ai = call_gemini(GENERAL_AI_PROMPT)

    print("  → Fetching personal news…")
    personal = call_gemini(PERSONAL_PROMPT)

    print("  → Generating data science topic…")
    ds = call_gemini(DS_PROMPT)

    print("  → Building email…")
    html = build_html(work, personal, general_ai, ds)

    subject = f"☀️ Morning briefing · {TODAY}"

    if RESEND_API_KEY and EMAIL_TO:
        print("  → Sending email…")
        send_email(html, subject)
        print(f"  ✓ Sent to {EMAIL_TO}")
    else:
        out = f"/home/claude/daily_briefing/briefing_{datetime.date.today()}.html"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            f.write(html)
        print(f"  → Email not configured — saved to {out}")

    print("Done.")


if __name__ == "__main__":
    main()

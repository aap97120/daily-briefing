#!/usr/bin/env python3
"""
David's Daily Morning Briefing — FREE VERSION (Flat Syntax Stable Release)
Uses Google Gemini's free-tier API (Flash models) with Structured JSON Outputs.
Compiles results into a local responsive HTML dashboard for GitHub Pages.

Sections: Work | Personal | General AI | Data Science Topic of the Day
"""

import os
import json
import time
import datetime
import urllib.request
import urllib.error
import socket

# ── Config ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")   # free from ://google.com
NTFY_TOPIC     = os.environ.get("NTFY_TOPIC", "")        # your unique ntfy.sh topic name
PAGES_URL      = os.environ.get("PAGES_URL", "")         # the GitHub Pages URL
# ─────────────────────────────────────────────────────────────────────────────

TODAY = datetime.datetime.now().strftime("%A %d %B %Y")
MODEL = "gemini-2.5-flash"          # free-tier model with search grounding support
FALLBACK_MODEL = "gemini-2.0-flash" # used on final retry if the primary model is overloaded

def _gemini_url(model: str) -> str:
    start = "https://" + "gen" + "erative"
    middle = "lan" + "guage" + ".google" + "apis.com"
    path = f"/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
    return start + middle + path


WORK_PROMPT = "You are a professional news briefing assistant. Today is {TODAY}.\n\nSearch for and summarise the 5 most relevant and recent news stories across these topics for a senior analytics leader at Dun & Bradstreet:\n- Agentic AI and AI agents for enterprise / B2B use cases\n- B2B data and data intelligence platforms\n- Revenue intelligence, sales AI tools (ZoomInfo, Apollo, Salesforce Einstein, Gong, Clari)\n- Data science and analytics industry news\n- Business credit, financial risk and commercial data\n- Dun & Bradstreet, Moody's Analytics, Verisk or similar B2B data company news\n\nFor each story provide:\n1. A clear headline\n2. A 2-3 sentence summary in plain language\n3. Why it matters for someone in B2B data and analytics strategy\n4. 1-2 real source URLs"

PERSONAL_PROMPT = "You are a personal news briefing assistant. Today is {TODAY}.\n\nSearch for and summarise the 6 most interesting recent news stories across these topics for a UK-based professional:\n- UK politics and current affairs\n- US politics and international affairs\n- Science (space, biology, physics, technology breakthroughs)\n- Golf (PGA Tour, DP World Tour, majors, Ryder Cup)\n- Football / soccer (Premier League, Champions League, international)\n- Rugby (Six Nations, Premiership, international)\n- ONE interesting story from: Formula 1, economics, or culture\n\nFor each story provide:\n1. A clear headline\n2. A 2-3 sentence summary\n3. 1-2 real source URLs"

GENERAL_AI_PROMPT = "You are an AI industry news analyst. Today is {TODAY}.\n\nSearch for and summarise the 4 most significant recent stories in the broader AI industry. Cover things like:\n- Major AI company funding, acquisitions, or M&A\n- Frontier model releases or major capability announcements (OpenAI, Anthropic, Google, Meta, xAI)\n- AI compute, chips, and infrastructure economics\n- Notable AI policy, safety, or regulatory developments\n- Notable moves by AI-native startups (coding agents, AI infrastructure, etc.)\n\nFor each story provide:\n1. A clear headline\n2. A 2-3 sentence summary explaining what happened and why it's significant\n3. 1-2 real source URLs"

DS_PROMPT = "You are a data science educator. Today is {TODAY}.\n\nChoose ONE data science topic to explain in depth. Rotate broadly across:\n- Classical ML (XGBoost, Random Forests, survival analysis)\n- Deep learning (transformers, embeddings, fine-tuning)\n- MLOps (MLflow, feature stores, model monitoring)\n- Agentic AI (LangChain, LangGraph, AutoGen, MCP, multi-agent orchestration)\n- Statistical methods (Bayesian inference, causal inference, SHAP, LIME, A/B testing)\n- Emerging techniques (RAG, multimodal models, reasoning models, RLHF)\n- Data engineering (dbt, Spark, vector databases, knowledge graphs)\n\nProvide:\n1. Topic name and one-line tagline\n2. What it is — plain language explanation (3-4 sentences)\n3. How it works — slightly more technical (4-5 sentences)\n4. When to use it — practical guidance (3-4 sentences)\n5. A concrete worked example relevant to B2B analytics or sales/marketing data\n6. 2-3 real links to good resources"


def call_gemini(prompt: str, response_schema: dict = None, max_retries: int = 4) -> dict:
    """Calls the Gemini API with Search grounding and strict JSON Structured Outputs."""
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "tools": [{"google_search": {}}],
        "generationConfig": {
            "temperature": 0.4,
        },
    }
    
    if response_schema:
        payload["generationConfig"]["responseSchema"] = response_schema

    data = json.dumps(payload).encode("utf-8")
    last_error = None
    
    for attempt in range(1, max_retries + 1):
        model = FALLBACK_MODEL if attempt == max_retries else MODEL
        req = urllib.request.Request(
            _gemini_url(model), data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=150) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            last_error = RuntimeError(f"Gemini API error {e.code}: {body[:300]}")
            if e.code in (503, 429) and attempt < max_retries:
                wait = 2 ** attempt
                print(f"    (attempt {attempt}/{max_retries} got {e.code}, retrying in {wait}s…)")
                time.sleep(wait)
                continue
            raise last_error
        except (socket.timeout, TimeoutError, urllib.error.URLError) as e:
            last_error = RuntimeError(f"Gemini API network error: {e}")
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"    (attempt {attempt}/{max_retries} got network error, retrying in {wait}s…)")
                time.sleep(wait)
                continue
            raise last_error
    else:
        raise last_error

    try:
        text = result["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(text.strip())
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Failed parsing response schema: {e}. Raw response: {json.dumps(result)[:300]}")

# --- PART 1 END ---
def generate_html_dashboard(data: dict, filepath: str):
    """Generates a responsive HTML page using flat string row lists to avoid editor paste truncation bugs."""
    
    def make_links(links_list):
        if not links_list: return ""
        return " ".join([f'<a href="{l.get("url", "#")}" target="_blank" class="link-btn">🔗 {l.get("label", "Source")}</a>' for l in links_list])

    work_html = "".join([f'<div class="card"><span class="badge badge-work">{s.get("tag", "B2B")}</span><h3>{s.get("headline")}</h3><p>{s.get("summary")}</p><div class="relevance"><strong>Strategy Impact:</strong> {s.get("relevance")}</div><div class="links-container">{make_links(s.get("links"))}</div></div>' for s in data['work'].get('stories', [])])
    personal_html = "".join([f'<div class="card"><span class="badge badge-personal">{s.get("tag", "News")}</span><h3>{s.get("headline")}</h3><p>{s.get("summary")}</p><div class="links-container">{make_links(s.get("links"))}</div></div>' for s in data['personal'].get('stories', [])])
    ai_html = "".join([f'<div class="card"><span class="badge badge-ai">{s.get("tag", "AI")}</span><h3>{s.get("headline")}</h3><p>{s.get("summary")}</p><div class="links-container">{make_links(s.get("links"))}</div></div>' for s in data['general_ai'].get('stories', [])])

    ds = data['data_science']
    ds_html = f'<div class="ds-hero"><h2>🎓 {ds.get("topic")}</h2><p class="tagline"><em>{ds.get("tagline")}</em></p><div class="ds-grid"><div><h4>What It Is</h4><p>{ds.get("what_it_is")}</p></div><div><h4>How It Works</h4><p>{ds.get("how_it_works")}</p></div><div><h4>When To Use</h4><p>{ds.get("when_to_use")}</p></div></div><div class="worked-example"><h4>🎯 Concrete Worked Example (B2B Context)</h4><p>{ds.get("worked_example")}</p></div><div class="links-container" style="margin-top:1.5rem;">{make_links(ds.get("links"))}</div></div>'

    rows = [
        '<!DOCTYPE html>',
        '<html lang="en">',
        '<head>',
        '    <meta charset="UTF-8">',
        '    <meta name="viewport" content="width=device-width, initial-scale=1.0">',
        "    <title>David's Morning Briefing</title>",
        '    <style>',
        '        :root { --bg: #f8fafc; --surface: #ffffff; --text: #0f172a; --border: #e2e8f0; --primary: #2563eb; --work: #0ea5e9; --personal: #10b981; --ai: #8b5cf6; }',
        '        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: var(--bg); color: var(--text); line-height: 1.5; padding: 2rem 1rem; margin: 0; }',
        '        .container { max-width: 1100px; margin: 0 auto; }',
        '        header { text-align: center; margin-bottom: 3rem; border-bottom: 2px solid var(--border); padding-bottom: 1.5rem; }',
        '        h1 { margin: 0 0 0.5rem 0; font-size: 2.25rem; color: #1e293b; }',
        '        .date { font-size: 1.1rem; color: #64748b; font-weight: 500; }',
        '        .section-title { font-size: 1.5rem; margin: 2.5rem 0 1rem; border-left: 4px solid var(--primary); padding-left: 0.75rem; color: #1e293b; }',
        '        .section-title.work-title { border-color: var(--work); }',
        '        .section-title.personal-title { border-color: var(--personal); }',
        '        .section-title.ai-title { border-color: var(--ai); }',
        '        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 1.5rem; }',
        '        .card { background: var(--surface); border: 1px solid var(--border); padding: 1.5rem; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); display: flex; flex-direction: column; }',
        '        .card h3 { margin: 0 0 0.75rem 0; font-size: 1.2rem; line-height: 1.3; color: #1e293b; }',
        '        .card p { color: #475569; font-size: 0.95rem; margin: 0 0 1rem 0; flex-grow: 1; }',
        '        .relevance { background: #f0f9ff; border: 1px solid #bae6fd; padding: 0.75rem; border-radius: 8px; font-size: 0.9rem; color: #0369a1; margin-bottom: 1rem; }',
        '        .badge { display: inline-block; padding: 0.25rem 0.5rem; font-size: 0.75rem; font-weight: 700; border-radius: 4px; text-transform: uppercase; margin-bottom: 0.75rem; width: fit-content; }',
        '        .badge-work { background: #e0f2fe; color: #0369a1; }',
        '        .badge-personal { background: #d1fae5; color: #047857; }',
        '        .badge-ai { background: #f3e8ff; color: #6d28d9; }',
        '        .links-container { display: flex; gap: 0.5rem; flex-wrap: wrap; }',
        '        .link-btn { display: inline-block; background: #f1f5f9; color: #475569; text-decoration: none; padding: 0.4rem 0.75rem; border-radius: 6px; font-size: 0.85rem; font-weight: 500; transition: background 0.2s; }',
        '        .link-btn:hover { background: #e2e8f0; color: #0f172a; }',
        '        .ds-hero { background: #1e293b; color: #f8fafc; padding: 2rem; border-radius: 16px; margin-top: 1rem; }',
        '        .ds-hero h2 { margin: 0 0 0.25rem 0; color: #fff; }',
        '        .ds-hero .tagline { color: #94a3b8; margin: 0 0 2rem 0; font-size: 1.1rem; }',
        '        .ds-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem; margin-bottom: 1.5rem; }',
        '        .ds-grid h4 { margin: 0 0 0.5rem 0; color: #38bdf8; font-size: 1rem; text-transform: uppercase; letter-spacing: 0.05em; }',
        '        .ds-grid p { margin: 0; color: #cbd5e1; font-size: 0.95rem; }',
        '        .worked-example { background: rgba(15, 23, 42, 0.6); padding: 1.25rem; border-radius: 8px; border-left: 4px solid #38bdf8; }',
        '        .worked-example h4 { margin: 0 0 0.5rem 0; color: #fff; }',
        '        .worked-example p { margin: 0; color: #cbd5e1; font-size: 0.95rem; }',
        '        @media (max-width: 600px) { body { padding: 1rem 0.5rem; } header { margin-bottom: 1.5rem; } .ds-hero { padding: 1.25rem; } }',
        '    </style>',
        '</head>',
        '<body>',
        '    <div class="container">',
        '        <header>',
        "            <h1>David's Daily Morning Briefing</h1>",
        f"            <div class=\"date\">☀️ {data['date']}</div>",
        '        </header>',
        '        <h2 class="section-title work-title">💼 Work & B2B Strategy Briefing</h2>',
        f'        <div class="grid">{work_html}</div>',
        '        <h2 class="section-title ai-title">🤖 Broader AI Industry News</h2>',
        f'        <div class="grid">{ai_html}</div>',
        '        <h2 class="section-title personal-title">🇬🇧 Personal News & Sports</h2>',
        f'        <div class="grid">{personal_html}</div>',
        '        <h2 class="section-title" style="border-color:#38bdf8;">📝 Data Science Topic of the Day</h2>',
        f'        {ds_html}',
        '    </div>',
        '</body>',
        '</html>'
    ]
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(rows))


def generate_briefing():
    """Main pipeline execution block"""
    print(f"Starting Briefing Generation for {TODAY}...")
    
    story_schema = {
        "type": "OBJECT",
        "properties": {
            "stories": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "headline": {"type": "STRING"},
                        "summary": {"type": "STRING"},
                        "tag": {"type": "STRING"},
                        "relevance": {"type": "STRING"},
                        "links": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "label": {"type": "STRING"},
                                    "url": {"type": "STRING"}
                                },
                                "required": ["label", "url"]
                            }
                        }
                    },
                    "required": ["headline", "summary", "tag", "links"]
                }
            }
        },
        "required": ["stories"]
    }
    
    ds_schema = {
        "type": "OBJECT",
        "properties": {
            "topic": {"type": "STRING"},
            "tagline": {"type": "STRING"},
            "what_it_is": {"type": "STRING"},
            "how_it_works": {"type": "STRING"},
            "when_to_use": {"type": "STRING"},
            "worked_example": {"type": "STRING"},
            "links": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "label": {"type": "STRING"},
                        "url": {"type": "STRING"}
                    },
                    "required": ["label", "url"]
                }
            }
        },
        "required": ["topic", "tagline", "what_it_is", "how_it_works", "when_to_use", "worked_example", "links"]
    }

    print("Fetching Work section...")
    work_data = call_gemini(WORK_PROMPT.format(TODAY=TODAY), response_schema=story_schema)
    time.sleep(3)
    
    print("Fetching Personal section...")
    personal_data = call_gemini(PERSONAL_PROMPT.format(TODAY=TODAY), response_schema=story_schema)
    time.sleep(3)
    
    print("Fetching General AI section...")
    general_ai_data = call_gemini(GENERAL_AI_PROMPT.format(TODAY=TODAY), response_schema=story_schema)
    time.sleep(3)
    
    print("Fetching Data Science section...")
    ds_data = call_gemini(DS_PROMPT.format(TODAY=TODAY), response_schema=ds_schema)
    
    briefing_payload = {
        "date": TODAY,
        "work": work_data,
        "personal": personal_data,
        "general_ai": general_ai_data,
        "data_science": ds_data
    }
    
    output_dir = "public"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/briefing.json", "w", encoding="utf-8") as f:
        json.dump(briefing_payload, f, indent=2)
        
    generate_html_dashboard(briefing_payload, f"{output_dir}/index.html")
    print("Saved briefing JSON and HTML dashboard successfully.")

    if NTFY_TOPIC:
        try:
            notification_text = f"Your daily briefing for {TODAY} is ready! View it here: {PAGES_URL}"
            req = urllib.request.Request(
                f"https://ntfy.sh{NTFY_TOPIC}",
                data=notification_text.encode("utf-8"),
                headers={"Title": "Daily Morning Briefing Ready ☀️"}
            )
            urllib.request.urlopen(req)
            print("Successfully sent notification via ntfy.sh!")
        except Exception as e:
            print(f"Warning: Failed to send ntfy notification: {e}")

if __name__ == "__main__":
    generate_briefing()


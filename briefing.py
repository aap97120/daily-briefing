#!/usr/bin/env python3
"""
David's Daily Morning Briefing — FREE VERSION (Unified Production Release)
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
    return f"https://googleapis.com{model}:generateContent?key={GEMINI_API_KEY}"


WORK_PROMPT = f"""You are a professional news briefing assistant. Today is {{TODAY}}.

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
4. 1-2 real source URLs"""

PERSONAL_PROMPT = f"""You are a personal news briefing assistant. Today is {{TODAY}}.

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
3. 1-2 real source URLs"""

GENERAL_AI_PROMPT = f"""You are an AI industry news analyst. Today is {{TODAY}}.

Search for and summarise the 4 most significant recent stories in the broader AI industry. Cover things like:
- Major AI company funding, acquisitions, or M&A
- Frontier model releases or major capability announcements (OpenAI, Anthropic, Google, Meta, xAI)
- AI compute, chips, and infrastructure economics
- Notable AI policy, safety, or regulatory developments
- Notable moves by AI-native startups (coding agents, AI infrastructure, etc.)

For each story provide:
1. A clear headline
2. A 2-3 sentence summary explaining what happened and why it's significant
3. 1-2 real source URLs"""

DS_PROMPT = f"""You are a data science educator. Today is {{TODAY}}.

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
6. 2-3 real links to good resources"""


def call_gemini(prompt: str, response_schema: dict = None, max_retries: int = 4) -> dict:
    """Calls the Gemini API with Search grounding and strict JSON Structured Outputs."""
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "tools": [{"google_search_retrieval": {}}],
        "generationConfig": {
            "temperature": 0.4,
            "responseMimeType": "application/json"
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
            break  # success
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


def generate_html_dashboard(data: dict, filepath: str):
    """Generates a responsive HTML page from the briefing payload."""
    
    def make_links(links_list):
        if not links_list: return ""
        return " ".join([f'<a href="{l.get("url", "#")}" target="_blank" class="link-btn">🔗 {l.get("label", "Source")}</a>' for l in links_list])

    # Build individual components
    work_html = "".join([f'<div class="card"><span class="badge badge-work">{s.get("tag", "B2B")}</span><h3>{s.get("headline")}</h3><p>{s.get("summary")}</p><div class="relevance"><strong>Strategy Impact:</strong> {s.get("relevance")}</div><div class="links-container">{make_links(s.get("links"))}</div></div>' for s in data['work'].get('stories', [])])
    personal_html = "".join([f'<div class="card"><span class="badge badge-personal">{s.get("tag", "News")}</span><h3>{s.get("headline")}</h3><p>{s.get("summary")}</p><div class="links-container">{make_links(s.get("links"))}</div></div>' for s in data['personal'].get('stories', [])])
    ai_html = "".join([f'<div class="card"><span class="badge badge-ai">{s.get("tag", "AI")}</span><h3>{s.get("headline")}</h3><p>{s.get("summary")}</p><div class="links-container">{make_links(s.get("links"))}</div></div>' for s in data['general_ai'].get('stories', [])])

    ds = data['data_science']
    ds_html = f'<div class="ds-hero"><h2>🎓 {ds.get("topic")}</h2><p class="tagline"><em>{ds.get("tagline")}</em></p><div class="ds-grid"><div><h4>What It Is</h4><p>{ds.get("what_it_is")}</p></div><div><h4>How It Works</h4><p>{ds.get("how_it_works")}</p></div><div><h4>When To Use</h4><p>{ds.get("when_to_use")}</p></div></div><div class="worked-example"><h4>🎯 Concrete Worked Example (B2B Context)</h4><p>{ds.get("worked_example")}</p></div><div class="links-container" style="margin-top:1.5rem;">{make_links(ds.get("links"))}</div></div>'

    # Safe layout structure using direct replacement strings
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>David's Morning Briefing</title>
    <style>
        :root { --bg: #f8fafc; --surface: #ffffff; --text: #0f172a; --border: #e2e8f0; --primary: #2563eb; --work: #0ea5e9; --personal: #10b981; --ai: #8b5cf6; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: var(--bg); color: var(--text); line-height: 1.5; padding: 2rem 1rem; margin: 0; }
        .container { max-width: 1100px; margin: 0 auto; }
        header { text-align: center; margin-bottom: 3rem; border-bottom: 2px solid var(--border); padding-bottom: 1.5rem; }
        h1 { margin: 0 0 0.5rem 0; font-size: 2.25rem; color: #1e293b; }
        .date { font-size: 1.1rem; color: #64748b; font-weight: 500; }
        .section-title { font-size: 1.5rem; margin: 2.5rem 0 1rem; border-left: 4px solid var(--primary); padding-left: 0.75rem; color: #1e293b; }
        .section-title.work-title { border-color: var(--work); }
        .section-title.personal-title { border-color: var(--personal); }
        .section-title.ai-title { border-color: var(--ai); }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 1.5rem; }

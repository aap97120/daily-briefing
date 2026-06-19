# Daily Morning Briefing — Free Version

Same as before, but runs entirely on free tiers: Google's free Gemini API
(with search grounding) instead of the paid Anthropic API. GitHub Actions
remains free for this usage level. **Total ongoing cost: £0.**

---

## What you get

- **Work** — 5 stories: agentic AI, B2B data, revenue intelligence, D&B/competitor news
- **General AI** — 4 stories: funding/M&A, frontier model releases, compute/infrastructure economics, AI policy
- **Personal** — 6 stories: UK & US politics, science, golf, football, rugby, plus a wildcard
- **Data science topic of the day** — deep-dive on one technique or tool, rotated daily

Each section uses Gemini's built-in Google Search grounding, so content reflects the day it runs.

---

## One-time setup (~15 minutes)

### Step 1 — Get a free Gemini API key

1. Go to https://aistudio.google.com
2. Sign in with any Google account
3. Click **Get API key** → **Create API key**
4. Copy it — no card required, no billing setup needed for the free tier

> The free tier uses `gemini-2.5-flash`, which has generous daily limits — comfortably enough for one run per day across four prompts.

### Step 2 — Get a free Resend API key (for sending the email)

Gmail's app-password system has become unreliable for many personal accounts —
Google increasingly blocks it even with 2-Step Verification on, with no clear
way to fix it from the user side. Resend is a transactional email service with
a generous free tier (100 emails/day, 3,000/month) that sidesteps this entirely.

1. Go to https://resend.com and sign up (free, no card required)
2. Go to **API Keys** → **Create API Key** → copy it
3. For the FROM address, the simplest option is to use Resend's shared testing
   domain (`onboarding@resend.dev`) — this works immediately with no setup and
   is fine for a personal briefing sent to yourself.
   - If you want the email to come from your own address instead, Resend
     supports verifying your own domain — see their docs — but this is optional.
4. Set `BRIEFING_TO_EMAIL` to wherever you want the briefing delivered (this
   can be your normal Gmail address — Resend is only used for *sending*, not
   *receiving*, so none of Gmail's app-password restrictions apply here)

### Step 3 — Create the GitHub repository

1. Create a new **private** repo (e.g. `daily-briefing`)
2. Upload these files, keeping the folder structure:
   ```
   briefing.py
   .github/workflows/daily_briefing.yml
   ```
3. Go to **Settings → Secrets and variables → Actions → New repository secret**

Add these secrets:

| Secret name               | Value                                       |
|----------------------------|---------------------------------------------|
| `GEMINI_API_KEY`          | Your free Gemini API key                     |
| `RESEND_API_KEY`          | Your free Resend API key                     |
| `BRIEFING_FROM_EMAIL`     | `onboarding@resend.dev` (or your verified domain) |
| `BRIEFING_TO_EMAIL`       | Where you want the briefing delivered        |

### Step 4 — Test it

**Actions → Daily Morning Briefing (Free) → Run workflow**.
Should complete in 60–90 seconds and land in your inbox.

---

## Scheduling

Same as before — runs at 8am UK time on weekdays and weekends, with the usual ±1 hour drift around the BST/GMT clock changes in late March and late October (GitHub Actions cron is UTC-only).

---

## Running locally

```bash
export GEMINI_API_KEY="your-free-gemini-key"
export RESEND_API_KEY="your-free-resend-key"
export BRIEFING_FROM_EMAIL="onboarding@resend.dev"
export BRIEFING_TO_EMAIL="you@gmail.com"

python briefing.py
```

If email env vars aren't set, it saves a local `.html` file instead — useful for testing.

---

## Costs

**£0.** Gemini's free tier (Flash model) covers this comfortably at one run/day.
GitHub Actions free tier covers the ~3 minutes/day of compute easily.

If Google ever tightens the free tier further (as they did with Pro models in April 2026), Flash-tier access has historically been preserved — but worth keeping an eye on `aistudio.google.com` pricing if the workflow ever starts failing with a billing error.

---

## Customising topics

Edit the prompt strings at the top of `briefing.py`:

- `WORK_PROMPT` — D&B / industry-specific topics
- `GENERAL_AI_PROMPT` — broader AI industry news (funding, models, infrastructure)
- `PERSONAL_PROMPT` — personal interest categories
- `DS_PROMPT` — data science topic rotation categories

No other code changes needed.

---

## Strategic Reading Log

Separately from the automated briefing, ask Claude in chat to add notable articles
(the kind you paste in manually, like long-form essays or analysis pieces) to your
**Strategic Reading Log** document in the project — this captures things no automated
search would find, and persists independently of chat memory.

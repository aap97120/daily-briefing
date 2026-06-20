name: Daily Morning Briefing (Free)

on:
  schedule:
    # 7:00 UTC = 8:00 UK (GMT) / 9:00 BST in summer
    - cron: '0 7 * * 1-5'   # Weekdays at 8am GMT / 9am BST
    - cron: '0 7 * * 0,6'   # Weekends, same time (kept consistent with weekdays)
  workflow_dispatch:          # Allows manual trigger from the GitHub UI

permissions:
  contents: write   # needed so the workflow can commit the published page

jobs:
  send-briefing:
    runs-on: ubuntu-latest
    steps:
      - name: Check out repo
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Run briefing script
        env:
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
          NTFY_TOPIC:     ${{ secrets.NTFY_TOPIC }}
          PAGES_URL:      ${{ secrets.PAGES_URL }}
        run: python briefing.py

      - name: Commit and publish briefing page
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add docs/index.html
          git commit -m "Update daily briefing $(date -u +%Y-%m-%d)" || echo "No changes to commit"
          git push

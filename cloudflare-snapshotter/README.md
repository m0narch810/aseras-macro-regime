# vanta-snapshotter (Cloudflare Worker)

Keeps the GEX snapshot pipeline dense and self-monitoring, without needing your
PC on. Two cron jobs:

1. **Dispatch** — every 5 min, Mon–Fri, 12:00–21:59 UTC (~08:00–17:00 ET): pokes
   the GitHub Actions workflow `gex-snapshot.yml` so it takes a snapshot.
   Cloudflare fires crons reliably, unlike GitHub's own `*/5` schedule which
   drops most ticks (the ~5-snapshots/day problem). Expect ~108/day instead.
2. **Heartbeat** — daily at 22:15 UTC (~18:15 ET): counts the day's snapshot
   commits on the `gex-snapshots` branch and alerts if the pipeline went quiet
   (dead `FF_SESSION`, expired token, etc.), so breakage surfaces within a day.

The Worker is a **thin trigger + monitor** — the real FreeFlow fetch / aggregation
/ CSV write stays in `freeflow_logger.py`, run by the workflow. Nothing to drift.

---

## One-time setup (~10 min)

### 1. Create a GitHub token
github.com → **Settings → Developer settings → Personal access tokens →
Fine-grained tokens → Generate new token**
- **Repository access:** Only select repositories → `m0narch810/vanta`
- **Permissions:**
  - **Actions → Read and write** (required — lets the Worker dispatch the workflow)
  - **Issues → Read and write** (only if you want the fallback GitHub-issue alert;
    skip if you'll use a webhook instead)
- Copy the token (`github_pat_…`).

### 2. Install wrangler + log in
```bash
cd cloudflare-snapshotter
npm install
npx wrangler login          # opens a browser, authorizes Cloudflare
```
(No Cloudflare account yet? `wrangler login` will walk you through creating the
free one.)

### 3. Add the token as a secret
```bash
npx wrangler secret put GITHUB_TOKEN
# paste the github_pat_… value when prompted
```

### 4. (Optional but recommended) wire up an alert channel
Easiest is a Discord webhook (Server Settings → Integrations → Webhooks → New →
Copy URL), then:
```bash
npx wrangler secret put HEARTBEAT_WEBHOOK
# paste the webhook URL
```
If you skip this, the heartbeat falls back to opening a GitHub issue (which emails
you via notifications) — that needs the Issues permission from step 1.

### 5. Deploy
```bash
npx wrangler deploy
```
Done. The cron jobs are now live on Cloudflare's schedule.

---

## Verify it works
After deploy, hit the Worker URL (printed by `wrangler deploy`) with a test query:
- `https://vanta-snapshotter.<your-subdomain>.workers.dev/?dispatch` → should
  return `{"action":"dispatch","status":204,"ok":true}` and you'll see a new run
  appear under the repo's **Actions → GEX 5-min snapshot**.
- `…/?heartbeat` → returns today's snapshot-commit count and health.

Give it a full trading day, then check the **gex-snapshots** branch — you should
see ~100+ snapshot commits instead of ~5.

---

## The one recurring chore this can't fix
`FF_SESSION` (the free-flow.site cookie) expires every so often. When it does,
snapshots go blank/fail no matter how reliably they're triggered. The heartbeat
will catch it within a day and alert you — at which point: grab a fresh cookie
from a logged-in free-flow.site browser session (DevTools → Application →
Cookies → `ff_session`) and update the **`FF_SESSION`** repo secret on GitHub
(Settings → Secrets and variables → Actions). That's the only upkeep.

## Config (wrangler.toml `[vars]`)
- `HEARTBEAT_MIN` — alert if fewer than this many snapshot commits land in a day
  (default 60).
- `GITHUB_REPO`, `WORKFLOW_FILE`, `WORKFLOW_REF`, `SNAPSHOT_BRANCH` — change only
  if you fork/rename.

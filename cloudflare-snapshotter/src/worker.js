/**
 * vanta-snapshotter — Cloudflare Worker
 *
 * Two cron jobs (see wrangler.toml):
 *   - every 5 min during the session → dispatch the GitHub Actions snapshot
 *     workflow (reliable trigger; replaces GitHub's flaky 5-min schedule).
 *   - once daily after the close → heartbeat: count the day's snapshot commits
 *     and alert if the pipeline went quiet.
 *
 * Deliberately a thin trigger+monitor: the actual FreeFlow fetch / aggregation /
 * CSV write stays in the proven Python logger (freeflow_logger.py) run by the
 * workflow, so there's no second implementation to drift.
 *
 * Required secret: GITHUB_TOKEN (fine-grained PAT, repo=vanta, Actions: R/W).
 * Optional secret: HEARTBEAT_WEBHOOK (Discord/Slack-style incoming webhook).
 */

const UA = "vanta-snapshotter-worker";

export default {
  async scheduled(event, env, ctx) {
    // event.cron is the exact schedule string that fired — route on it.
    const isHeartbeat = event.cron === "15 22 * * 1-5";
    ctx.waitUntil(isHeartbeat ? heartbeat(env) : dispatchSnapshot(env));
  },

  // Manual smoke-test endpoint: GET /?dispatch or /?heartbeat (handy after deploy).
  async fetch(req, env) {
    const url = new URL(req.url);
    if (url.searchParams.has("dispatch"))  return json(await dispatchSnapshot(env));
    if (url.searchParams.has("heartbeat")) return json(await heartbeat(env));
    return new Response(
      "vanta-snapshotter alive. Append ?dispatch or ?heartbeat to test.\n",
      { headers: { "content-type": "text/plain" } }
    );
  },
};

// ── Trigger one snapshot run via workflow_dispatch ────────────────────────────
async function dispatchSnapshot(env) {
  const url = `https://api.github.com/repos/${env.GITHUB_REPO}` +
              `/actions/workflows/${env.WORKFLOW_FILE}/dispatches`;
  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${env.GITHUB_TOKEN}`,
      "Accept": "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
      "User-Agent": UA,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ ref: env.WORKFLOW_REF || "master" }),
  });
  // GitHub returns 204 No Content on success.
  const ok = res.status === 204;
  const out = { action: "dispatch", status: res.status, ok };
  if (!ok) out.body = (await res.text()).slice(0, 300);
  return out;
}

// ── Daily health check: did snapshots actually land today? ─────────────────────
async function heartbeat(env) {
  // The heartbeat fires ~18 ET after the close, so a trailing 16h window cleanly
  // covers that day's full 03:00-17:00 ET session without any timezone math.
  const sinceISO = new Date(Date.now() - 16 * 3600 * 1000).toISOString();
  // Public repo → commit listing needs no auth, but send the token if present
  // (raises rate limits and works if the repo is later made private).
  const url = `https://api.github.com/repos/${env.GITHUB_REPO}/commits` +
              `?sha=${env.SNAPSHOT_BRANCH}&since=${encodeURIComponent(sinceISO)}&per_page=100`;
  const headers = { "Accept": "application/vnd.github+json", "User-Agent": UA,
                    "X-GitHub-Api-Version": "2022-11-28" };
  if (env.GITHUB_TOKEN) headers["Authorization"] = `Bearer ${env.GITHUB_TOKEN}`;

  let count = 0, err = null;
  try {
    const res = await fetch(url, { headers });
    if (res.ok) count = (await res.json()).length;
    else err = `HTTP ${res.status}`;
  } catch (e) { err = String(e); }

  const floor = parseInt(env.HEARTBEAT_MIN || "60", 10);
  const healthy = err === null && count >= floor;
  const out = { action: "heartbeat", since: sinceISO, commits: count,
                floor, healthy, err };

  if (!healthy) {
    const msg = `⚠ VANTA snapshot pipeline degraded — only ${count} snapshot ` +
                `commits on \`${env.SNAPSHOT_BRANCH}\` today (floor ${floor})` +
                (err ? ` [check error: ${err}]` : ``) +
                `. Likely a dead FF_SESSION or expired token — refresh the ` +
                `FF_SESSION repo secret.`;
    out.alerted = await alert(env, msg);
  }
  return out;
}

// ── Alerting: webhook if configured, else open/update a GitHub issue ───────────
async function alert(env, message) {
  if (env.HEARTBEAT_WEBHOOK) {
    try {
      // `content` works for Discord, `text` for Slack — send both.
      const res = await fetch(env.HEARTBEAT_WEBHOOK, {
        method: "POST",
        headers: { "Content-Type": "application/json", "User-Agent": UA },
        body: JSON.stringify({ content: message, text: message }),
      });
      return { via: "webhook", status: res.status };
    } catch (e) { return { via: "webhook", error: String(e) }; }
  }
  // Fallback: GitHub issue (emails you via notifications). Needs Issues: write.
  try {
    const res = await fetch(`https://api.github.com/repos/${env.GITHUB_REPO}/issues`, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${env.GITHUB_TOKEN}`,
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": UA, "Content-Type": "application/json",
      },
      body: JSON.stringify({ title: `⚠ snapshot pipeline degraded ${new Date().toISOString().slice(0,10)}`,
                             body: message, labels: ["pipeline-health"] }),
    });
    return { via: "issue", status: res.status };
  } catch (e) { return { via: "issue", error: String(e) }; }
}

// ── helpers ───────────────────────────────────────────────────────────────────
function json(obj) {
  return new Response(JSON.stringify(obj, null, 2),
    { headers: { "content-type": "application/json" } });
}

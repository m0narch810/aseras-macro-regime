// Shared options-flow utilities for levels.js, intraday.js, and bias.js.
// Bundled into each function by esbuild at Netlify build time.

const https = require('https');

// ── CONSTANTS ────────────────────────────────────────────────────────────────
const FILTER_PCT = 5.0;
const MIN_SCORE  = 20.0;

const VALID_USERS        = ['aseras'];
const SESSION_MAX_AGE_MS = 30 * 24 * 60 * 60 * 1000; // 30 days

// Regime-adaptive scoring weights.
// EXPANSION (high IV / vol > RV): vanna/charm matter more than raw GEX.
// CONTRACTION (low IV, pinning): GEX dominates dealer hedging.
const REGIME_WEIGHTS = {
  EXPANSION:   { gex: 0.20, vex: 0.38, charmex: 0.17, oi: 0.15, dag: 0.10 },
  NEUTRAL:     { gex: 0.32, vex: 0.28, charmex: 0.15, oi: 0.15, dag: 0.10 },
  CONTRACTION: { gex: 0.50, vex: 0.15, charmex: 0.15, oi: 0.15, dag: 0.05 },
};

const AGENT_HEADERS = {
  'Accept':             '*/*',
  'Accept-Language':    'en-US,en;q=0.9',
  'Connection':         'keep-alive',
  'Referer':            'https://www.free-flow.site/?auth=success',
  'Sec-Fetch-Dest':     'empty',
  'Sec-Fetch-Mode':     'cors',
  'Sec-Fetch-Site':     'same-origin',
  'User-Agent':         'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36',
  'sec-ch-ua':          '"Chromium";v="148", "Google Chrome";v="148", "Not/A)Brand";v="99"',
  'sec-ch-ua-mobile':   '?0',
  'sec-ch-ua-platform': '"Windows"',
};

// Base CORS + content-type headers. Each function adds its own Cache-Control.
const BASE_HEADERS = {
  'Content-Type':                 'application/json',
  'Access-Control-Allow-Origin':  '*',
  'Access-Control-Allow-Headers': 'Authorization',
};

// ── AUTH ─────────────────────────────────────────────────────────────────────
function isAuthorized(event) {
  const h = (event.headers && (event.headers.authorization || event.headers.Authorization)) || '';
  const m = h.match(/^Bearer\s+(.+)$/i);
  if (!m) return false;
  try {
    const { user, ts } = JSON.parse(Buffer.from(m[1], 'base64').toString('utf8'));
    return VALID_USERS.includes(user) && (Date.now() - ts) < SESSION_MAX_AGE_MS;
  } catch (e) {
    return false;
  }
}

// ── HTTP HELPERS ─────────────────────────────────────────────────────────────
function fetchJson(url, cookie) {
  return new Promise((resolve, reject) => {
    const req = https.get(url, {
      headers: { ...AGENT_HEADERS, Cookie: `ff_session=${cookie}` },
    }, (res) => {
      let body = '';
      res.on('data', chunk => { body += chunk; });
      res.on('end', () => {
        if (res.statusCode < 200 || res.statusCode >= 300)
          return reject(new Error(`HTTP ${res.statusCode}`));
        try { resolve(JSON.parse(body)); }
        catch (e) { reject(new Error(`JSON parse: ${e.message}`)); }
      });
    });
    req.on('error', reject);
    req.setTimeout(20000, () => { req.destroy(); reject(new Error('Timeout')); });
  });
}

function httpGetJson(url, headers, timeoutMs) {
  return new Promise((resolve, reject) => {
    const req = https.get(url, { headers: headers || {} }, (res) => {
      let body = '';
      res.on('data', chunk => { body += chunk; });
      res.on('end', () => {
        if (res.statusCode < 200 || res.statusCode >= 300)
          return reject(new Error(`HTTP ${res.statusCode}`));
        try { resolve(JSON.parse(body)); }
        catch (e) { reject(new Error(`JSON parse: ${e.message}`)); }
      });
    });
    req.on('error', reject);
    req.setTimeout(timeoutMs || 9000, () => { req.destroy(); reject(new Error('Timeout')); });
  });
}

// ── DATE ─────────────────────────────────────────────────────────────────────
function todayET() {
  return new Date().toLocaleDateString('en-CA', { timeZone: 'America/New_York' });
}

// ── OPTIONS FLOW CORE ────────────────────────────────────────────────────────

// Aggregates per-row FreeFlow data into per-strike buckets.
// Uses strike_futures when present; derives from ETF strike × ratio otherwise.
function aggregateDataset(data) {
  const rows  = data.rows  || [];
  const ratio = data.ratio || 41.14;
  const strikes = {};
  for (const row of rows) {
    const etf = row.strike_etf || 0;
    const sf  = row.strike_futures != null
      ? Math.round(row.strike_futures * 10) / 10
      : Math.round(etf * ratio * 10) / 10;
    if (!strikes[sf]) {
      strikes[sf] = { strike_etf: etf, net_gex: 0, net_vex: 0, net_charmex: 0,
                      net_dex: 0, net_vegaex: 0, net_dag: 0, total_oi: 0 };
    }
    const s = strikes[sf];
    s.net_gex     += row.gex     || 0;
    s.net_vex     += row.vex     || 0;
    s.net_charmex += row.charmex || 0;
    s.net_dex     += row.dex     || 0;
    s.net_vegaex  += row.vegaex  || 0;
    s.net_dag     += row.dag     || 0;
    s.total_oi    += row.oi      || 0;
  }
  return {
    strikes,
    futuresPrice: data.futures_price || 0,
    spotEtf:      data.etf_spot      || 0,
    ratio:        data.ratio         || 41.14,
  };
}

// Locates the gamma flip (GEX zero crossing nearest to current price) by linear
// interpolation between adjacent strikes. Falls back to the strike with minimum
// |GEX| when no clean sign crossing exists.
function computeGammaFlip(strikes, futuresPrice) {
  const sorted = Object.entries(strikes)
    .map(([sf, s]) => ({ strike: +sf, gex: s.net_gex }))
    .sort((a, b) => a.strike - b.strike);

  let bestFlip = null, bestDist = Infinity;
  for (let i = 0; i < sorted.length - 1; i++) {
    const a = sorted[i], b = sorted[i + 1];
    if ((a.gex > 0 && b.gex <= 0) || (a.gex < 0 && b.gex >= 0)) {
      const flip = a.strike + (b.strike - a.strike) * Math.abs(a.gex) / (Math.abs(a.gex) + Math.abs(b.gex));
      const dist = Math.abs(flip - futuresPrice);
      if (dist < bestDist) { bestDist = dist; bestFlip = flip; }
    }
  }
  if (bestFlip == null) {
    let minAbs = Infinity;
    for (const row of sorted)
      if (Math.abs(row.gex) < minAbs) { minAbs = Math.abs(row.gex); bestFlip = row.strike; }
  }
  return bestFlip != null ? Math.round(bestFlip * 10) / 10 : null;
}

// Min-max normalisation of absolute values for cross-metric scoring.
function normalizeAbs(values) {
  const abs = values.map(Math.abs);
  const mn = Math.min(...abs), mx = Math.max(...abs);
  if (mx === mn) return abs.map(() => 0);
  return abs.map(v => (v - mn) / (mx - mn));
}

// Scores nearby strikes using regime-adjusted weights. Returns strikes above
// MIN_SCORE sorted descending. Only strikes within FILTER_PCT of futures price.
function scoreLevels(strikes, weights, futuresPrice) {
  if (!futuresPrice) return [];
  const nearby = Object.entries(strikes)
    .map(([sf, s]) => ({ ...s, strike_futures: +sf, dist_nq: +sf - futuresPrice }))
    .filter(r => Math.abs(r.dist_nq / futuresPrice * 100) <= FILTER_PCT);
  if (!nearby.length) return [];

  const gexN = normalizeAbs(nearby.map(r => r.net_gex));
  const vexN = normalizeAbs(nearby.map(r => r.net_vex));
  const chmN = normalizeAbs(nearby.map(r => r.net_charmex));
  const oiN  = normalizeAbs(nearby.map(r => r.total_oi));
  const dagN = normalizeAbs(nearby.map(r => r.net_dag));

  return nearby
    .map((r, i) => {
      const score   = (gexN[i]*weights.gex + vexN[i]*weights.vex + chmN[i]*weights.charmex +
                       oiN[i]*weights.oi   + dagN[i]*weights.dag) * 100;
      const volSens = Math.abs(r.net_vex) / (Math.abs(r.net_gex) + 1e-9);
      const base    = r.net_gex > 0 ? 'CALL WALL' : 'PUT WALL';
      return {
        strike_futures: Math.round(r.strike_futures * 10)  / 10,
        strike_etf:     Math.round(r.strike_etf     * 100) / 100,
        dist_nq:        Math.round(r.dist_nq        * 10)  / 10,
        score:          Math.round(score            * 10)  / 10,
        type:           base + (volSens > 2.0 ? ' + VOL SENSITIVE' : ''),
        net_gex:        Math.round(r.net_gex),
        net_vex:        Math.round(r.net_vex),
        net_charmex:    Math.round(r.net_charmex),
        net_dex:        Math.round(r.net_dex),
        net_vegaex:     Math.round(r.net_vegaex),
        total_oi:       Math.round(r.total_oi),
      };
    })
    .filter(r => r.score >= MIN_SCORE)
    .sort((a, b) => b.score - a.score);
}

// Maps IV / RV:IV ratio to a vol regime name and its scoring weight set.
// Returns ['REGIME_NAME', weightsObject].
function classifyRegime(iv, rvIvRatio) {
  if (iv != null && (iv >= 30 || (rvIvRatio != null && rvIvRatio < 0.5)))
    return ['EXPANSION',   REGIME_WEIGHTS.EXPANSION];
  if (iv != null && iv >= 20)
    return ['NEUTRAL',     REGIME_WEIGHTS.NEUTRAL];
  if (iv != null)
    return ['CONTRACTION', REGIME_WEIGHTS.CONTRACTION];
  return ['NEUTRAL',       REGIME_WEIGHTS.NEUTRAL];
}

module.exports = {
  FILTER_PCT, MIN_SCORE, VALID_USERS, SESSION_MAX_AGE_MS,
  REGIME_WEIGHTS, AGENT_HEADERS, BASE_HEADERS,
  isAuthorized, fetchJson, httpGetJson, todayET,
  aggregateDataset, computeGammaFlip, normalizeAbs, scoreLevels, classifyRegime,
};

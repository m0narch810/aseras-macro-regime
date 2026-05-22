const https = require('https');

const SYMBOL     = 'QQQ';
const BASE_URL   = 'https://www.free-flow.site/api';
const FILTER_PCT = 5.0;
const MIN_SCORE  = 20.0;

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

const OUT_HEADERS = {
  'Content-Type':                'application/json',
  'Access-Control-Allow-Origin': '*',
  'Cache-Control':               'public, max-age=240',
};

// ── HTTP HELPER ───────────────────────────────────────────────
function fetchJson(url, cookie) {
  return new Promise((resolve, reject) => {
    const req = https.get(url, {
      headers: { ...AGENT_HEADERS, Cookie: `ff_session=${cookie}` },
    }, (res) => {
      let body = '';
      res.on('data', chunk => { body += chunk; });
      res.on('end', () => {
        if (res.statusCode < 200 || res.statusCode >= 300) {
          return reject(new Error(`HTTP ${res.statusCode}`));
        }
        try { resolve(JSON.parse(body)); }
        catch (e) { reject(new Error(`JSON parse: ${e.message}`)); }
      });
    });
    req.on('error', reject);
    req.setTimeout(20000, () => { req.destroy(); reject(new Error('Timeout')); });
  });
}

// ── DATE HELPERS ──────────────────────────────────────────────
function todayET() {
  // Compute current date in US/Eastern to match FreeFlow's 0DTE expiry
  return new Date().toLocaleDateString('en-CA', { timeZone: 'America/New_York' });
}

// ── AGGREGATION ───────────────────────────────────────────────
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
      strikes[sf] = { strike_etf: etf, net_gex: 0, net_vex: 0, net_charmex: 0, net_dex: 0, net_vegaex: 0, net_dag: 0, total_oi: 0 };
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
  };
}

// ── GAMMA FLIP ────────────────────────────────────────────────
// Per-strike GEX sign change: call walls (positive) → put walls (negative).
// Interpolates the exact zero crossing between adjacent strikes and returns
// the crossing nearest to the current futures price.
function computeGammaFlip(strikes, futuresPrice) {
  const sorted = Object.entries(strikes)
    .map(([sf, s]) => ({ strike: +sf, gex: s.net_gex }))
    .sort((a, b) => a.strike - b.strike); // ascending

  let bestFlip = null;
  let bestDist = Infinity;

  for (let i = 0; i < sorted.length - 1; i++) {
    const a = sorted[i], b = sorted[i + 1];
    if ((a.gex > 0 && b.gex <= 0) || (a.gex < 0 && b.gex >= 0)) {
      // Linear interpolation to exact zero between the two bracketing strikes
      const flip = a.strike + (b.strike - a.strike) * Math.abs(a.gex) / (Math.abs(a.gex) + Math.abs(b.gex));
      const dist = Math.abs(flip - futuresPrice);
      if (dist < bestDist) { bestDist = dist; bestFlip = flip; }
    }
  }

  // Fallback: strike whose per-strike GEX is closest to zero
  if (bestFlip == null) {
    let minAbs = Infinity;
    for (const row of sorted) {
      if (Math.abs(row.gex) < minAbs) { minAbs = Math.abs(row.gex); bestFlip = row.strike; }
    }
  }

  return bestFlip != null ? Math.round(bestFlip * 10) / 10 : null;
}

// ── REGIME ────────────────────────────────────────────────────
function classifyRegime(ctx) {
  const iv   = ctx.current_iv  || 37.0;
  const rviv = ctx.rv_iv_ratio || 0.46;
  if (iv >= 30 || rviv < 0.5) return ['EXPANSION',   { gex:0.20, vex:0.38, charmex:0.17, oi:0.15, dag:0.10 }];
  if (iv >= 20)                return ['NEUTRAL',     { gex:0.32, vex:0.28, charmex:0.15, oi:0.15, dag:0.10 }];
                               return ['CONTRACTION', { gex:0.50, vex:0.15, charmex:0.15, oi:0.15, dag:0.05 }];
}

// ── SCORING ───────────────────────────────────────────────────
function normalizeAbs(values) {
  const abs = values.map(Math.abs);
  const mn  = Math.min(...abs);
  const mx  = Math.max(...abs);
  if (mx === mn) return abs.map(() => 0);
  return abs.map(v => (v - mn) / (mx - mn));
}

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
      const score    = (gexN[i]*weights.gex + vexN[i]*weights.vex + chmN[i]*weights.charmex + oiN[i]*weights.oi + dagN[i]*weights.dag) * 100;
      const volSens  = Math.abs(r.net_vex) / (Math.abs(r.net_gex) + 1e-9);
      const base     = r.net_gex > 0 ? 'CALL WALL' : 'PUT WALL';
      return {
        strike_futures: Math.round(r.strike_futures * 10) / 10,
        strike_etf:     Math.round(r.strike_etf     * 100) / 100,
        dist_nq:        Math.round(r.dist_nq        * 10) / 10,
        score:          Math.round(score            * 10) / 10,
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

// ── HANDLER ───────────────────────────────────────────────────
exports.handler = async (event) => {
  if (event.httpMethod === 'OPTIONS') {
    return { statusCode: 200, headers: OUT_HEADERS, body: '' };
  }

  try {
    const cookie = process.env.FF_SESSION || '';
    const exp    = todayET();

    const data = await fetchJson(`${BASE_URL}/futures-levels?symbol=${SYMBOL}&exp=${exp}`, cookie);
    if (!data.rows || !data.rows.length) throw new Error('No rows returned from FreeFlow — FF_SESSION may be expired.');

    let ctx = { current_iv: null, rv_iv_ratio: null, hv21: null };
    try {
      const vol       = await fetchJson(`${BASE_URL}/vol/realized?symbol=${SYMBOL}`, cookie);
      ctx.current_iv  = vol.current_iv  ?? null;
      ctx.rv_iv_ratio = vol.rv_iv_ratio ?? null;
      ctx.hv21        = vol.hv21        ?? null;
    } catch (_) {}

    const { strikes, futuresPrice, spotEtf } = aggregateDataset(data);
    const [regime, weights]                  = classifyRegime(ctx);
    const levels                             = scoreLevels(strikes, weights, futuresPrice);
    const gammaFlip                          = computeGammaFlip(strikes, futuresPrice);

    const updatedET = new Date().toLocaleString('en-US', {
      timeZone: 'America/New_York',
      hour: '2-digit', minute: '2-digit', hour12: false,
    }) + ' ET';

    return {
      statusCode: 200,
      headers:    OUT_HEADERS,
      body: JSON.stringify({
        updated:     updatedET,
        nq_price:    Math.round(futuresPrice * 10) / 10,
        qqq_price:   Math.round(spotEtf      * 100) / 100,
        gamma_flip:  gammaFlip,
        regime,
        iv:          ctx.current_iv  != null ? Math.round(ctx.current_iv  * 10) / 10  : null,
        rv_iv_ratio: ctx.rv_iv_ratio != null ? Math.round(ctx.rv_iv_ratio * 1000) / 1000 : null,
        hv21:        ctx.hv21        != null ? Math.round(ctx.hv21        * 10) / 10  : null,
        levels,
      }),
    };

  } catch (err) {
    return {
      statusCode: 200,
      headers:    OUT_HEADERS,
      body: JSON.stringify({ error: true, message: err.message }),
    };
  }
};

// Shared options-flow utilities for levels.js, intraday.js, and bias.js.
// Bundled into each function by esbuild at Netlify build time.

const https = require('https');

// ── CONSTANTS ────────────────────────────────────────────────────────────────
const FILTER_PCT = 5.0;
const MIN_SCORE  = 20.0;

const VALID_USERS        = ['aseras', 'awsame303', 'pinkus'];
const SESSION_MAX_AGE_MS = 30 * 24 * 60 * 60 * 1000; // 30 days

// Two-dimensional scoring weight table: [vol_regime][gamma_regime].
//
// Vol regime (IV-driven):
//   EXPANSION   — high IV or RV/IV < 0.5: vanna/charm flows dominate dealer hedging
//   NEUTRAL     — moderate IV: balanced GEX + vanna
//   CONTRACTION — low IV, pinning: GEX is the dominant mean-reversion force
//
// Gamma regime (price vs flip):
//   POSITIVE  — price > flip + 50pts: dealers long gamma, buy dips/sell rallies → walls hold
//   NEAR_FLIP — within 50pts of flip: transitional, unstable hedging
//   NEGATIVE  — price < flip − 50pts: dealers short gamma, amplify moves → walls break
//
// Interaction logic:
//   POSITIVE gamma boosts GEX weight (pinning is real) and reduces VEX.
//   NEGATIVE gamma cuts GEX (walls less reliable) and raises VEX + DAG.
//   EXPANSION + NEGATIVE is the most dangerous combo — VEX dominates, moves are explosive.
//   CONTRACTION + POSITIVE is maximum pinning — GEX alone drives 60% of the score.
//
// All rows sum to 1.00.
// ⚠ REGIME_WEIGHTS are theoretically motivated but empirically unvalidated.
// Calibration requires ≥30 days of logged GEX snapshots with outcome labels.
// See logs/intraday_inputs_log.jsonl once freeflow_logger.py has been running.
const REGIME_WEIGHTS = {
  EXPANSION: {
    POSITIVE:  { gex: 0.22, vex: 0.38, charmex: 0.17, oi: 0.14, dag: 0.09 },
    NEAR_FLIP: { gex: 0.20, vex: 0.38, charmex: 0.17, oi: 0.15, dag: 0.10 },
    NEGATIVE:  { gex: 0.10, vex: 0.48, charmex: 0.17, oi: 0.14, dag: 0.11 },
  },
  NEUTRAL: {
    POSITIVE:  { gex: 0.42, vex: 0.22, charmex: 0.14, oi: 0.14, dag: 0.08 },
    NEAR_FLIP: { gex: 0.32, vex: 0.28, charmex: 0.15, oi: 0.15, dag: 0.10 },
    NEGATIVE:  { gex: 0.20, vex: 0.36, charmex: 0.14, oi: 0.16, dag: 0.14 },
  },
  CONTRACTION: {
    POSITIVE:  { gex: 0.60, vex: 0.10, charmex: 0.14, oi: 0.14, dag: 0.02 },
    NEAR_FLIP: { gex: 0.50, vex: 0.15, charmex: 0.15, oi: 0.15, dag: 0.05 },
    NEGATIVE:  { gex: 0.36, vex: 0.24, charmex: 0.16, oi: 0.15, dag: 0.09 },
  },
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
// timeoutMs default 8500: Netlify Functions hard-kill at 10s, so the upstream
// wait must stay under that — otherwise the function is killed mid-fetch and the
// browser gets a 502 instead of the graceful {error} envelope the frontend
// handles (preserve last render). The 0→1→2 fallback passes a shorter value so
// three sequential probes still fit the 10s budget.
function fetchJson(url, cookie, timeoutMs) {
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
    req.setTimeout(timeoutMs || 8500, () => { req.destroy(); reject(new Error('Timeout')); });
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

function currentHourET() {
  const parts = new Intl.DateTimeFormat('en-US', {
    timeZone: 'America/New_York', hour: 'numeric', minute: 'numeric', hour12: false,
  }).formatToParts(new Date());
  const h = parseInt(parts.find(p => p.type === 'hour').value,   10);
  const m = parseInt(parts.find(p => p.type === 'minute').value, 10);
  return h + m / 60;
}

// ── HOLD PROBABILITY MODEL (mechanical) ───────────────────────────────────────
// Wall-Hold Reliability R, derived purely from options-dealer mechanics — NO
// historical backtest fitting. Replaces the prior LR/XGBoost model that was
// fit on QQQ+SPY touch events (removed: that data was not predictive).
//
//   R = cap(  B_regime × P × O × A_dex × PCR × S_skew
//                       × F_term × F_vrp × F_gtbr × G_pin , 1.0)
//
// Long/short-gamma logic lives ONLY in the regime baseline. Per-strike factors
// supply the variance. Sign convention: net_gex > 0 = call-dominated (call
// wall), net_gex < 0 = put-dominated (put wall) — NOT dealer long/short gamma.
//
// ctx fields (assembled once per request by scoreLevels):
//   gammaFlip, futuresPrice, timeOfDayET — regime + GTBR + pinning gating
//   atmIv          — ATM IV % for skew comparison
//   rvIvRatio      — variance-risk-premium filter (optional → no-op)
//   hvTermRatio    — hv5/hv63 short-vs-long realized vol (optional → no-op)
//   gtbr           — expected remaining NQ range in pts (null → filter off)
//   dagDecileThr   — |net_dag| top-decile cutoff across nearby strikes (pinning)
function computeHoldProb(level, ctx) {
  ctx = ctx || {};
  const strike   = level.strike_futures;
  const spot     = ctx.futuresPrice;
  const netGex   = level.net_gex || 0;
  const absGex   = level.abs_gex || Math.abs(netGex);
  const netDex   = level.net_dex || 0;
  const netDag   = Math.abs(level.net_dag || 0);
  const distNq   = level.dist_nq != null ? level.dist_nq
                 : (strike != null && spot != null ? strike - spot : 0);

  // 1. Global gamma regime baseline — the only place long/short gamma enters.
  // Our gamma flip is computed locally (FreeFlow doesn't return one), so it can
  // be a strike or two off. A binary spot>flip test would swing every wall 5×
  // on a tiny flip error near the money. Instead use a vol-scaled NEAR_FLIP band
  // (same ±0.5×daily-1σ as levels.js): inside the band we can't trust which side
  // spot is on, so the baseline sits neutral at 0.3 rather than cliff-edging.
  let B = 0.3;
  if (ctx.gammaFlip != null && spot != null) {
    const band = ctx.atmIv > 0
      ? Math.max(30, 0.5 * spot * (ctx.atmIv / 100) / Math.sqrt(252))
      : 50;
    const diff = spot - ctx.gammaFlip;
    // Smooth ramp across the NEAR_FLIP band instead of a hard 5× cliff: linear
    // from 0.1 (deep short gamma) through 0.3 (exactly at flip) to 0.5 (deep
    // long gamma). A flip estimate a strike or two off now nudges B by a few %
    // instead of flipping the whole wall's reliability.
    B = diff >= band ? 0.5
      : diff <= -band ? 0.1
      : 0.3 + 0.2 * (diff / band);
  }

  // 2a. Protrusion (0–1) → 0.5–1.5. Dominant nodes bend local price structure.
  const prot = level.protrusion != null ? level.protrusion : 0.5;
  const P = 0.5 + prot;

  // 2b. One-sidedness |GEX|/ag ∈ [0.5,1] → 0.85–1.15. Pure directional dealer
  //     risk hedges more decisively than offsetting call/put gamma.
  const ratio = absGex > 0 ? Math.min(1, Math.abs(netGex) / absGex) : 0.5;
  const O = Math.max(0.85, Math.min(1.15, 0.85 + 0.6 * (ratio - 0.5)));

  // 2c. Hedge-polarity alignment. Counter-trend dealer delta = wall; pro-trend
  //     = acceleration zone. Above spot needs DEX>0 (sell into rallies); below
  //     spot needs DEX<0 (buy dips).
  const aboveSpot   = distNq > 0;
  const counterTrend = aboveSpot ? (netDex > 0) : (netDex < 0);
  const A_dex = counterTrend ? 1.25 : 0.5;

  // 2d. Dominant-side OI asymmetry → 0.9–1.1. Pure call/put walls > mixed-OI.
  const co = level.call_oi || 0, po = level.put_oi || 0, toi = co + po;
  const asym = toi > 0 ? Math.max(co, po) / toi : 0.5;
  const PCR = 0.9 + 0.4 * (asym - 0.5);

  // 2e. Side-aware IV skew: |strike_iv − ATM| / ATM > 0.20 → ×1.15. Absolute
  //     deviation rewards both put-skew and sharp call-side bidding. Upper
  //     sanity bound guards against garbage far-OTM iv_pct leaking in.
  let S_skew = 1.0;
  if (ctx.atmIv > 0 && level.strike_iv != null && level.strike_iv < 5 * ctx.atmIv) {
    if (Math.abs(level.strike_iv - ctx.atmIv) / ctx.atmIv > 0.20) S_skew = 1.15;
  }

  // 3a. Vol term structure: short realized spiking vs long → wall breakdown.
  const F_term = (ctx.hvTermRatio != null && ctx.hvTermRatio > 1.25) ? 0.85 : 1.0;

  // 3b. Variance risk premium: IV richly above RV → mean-reversion favored.
  const F_vrp = (ctx.rvIvRatio != null && ctx.rvIvRatio < 0.5) ? 1.15 : 1.0;

  // 3c. GTBR inelasticity guardrail: a wall beyond expected remaining range is
  //     reached only on a momentum surge that runs through it.
  const F_gtbr = (ctx.gtbr != null && Math.abs(distNq) > ctx.gtbr) ? 0.2 : 1.0;

  // 4. Late-session pinning: top-decile DAG after 14:00 ET = magnetic pull.
  const G_pin = (ctx.timeOfDayET > 14 && ctx.dagDecileThr != null
                 && netDag >= ctx.dagDecileThr) ? 1.25 : 1.0;

  const R = B * P * O * A_dex * PCR * S_skew * F_term * F_vrp * F_gtbr * G_pin;
  return Math.round(Math.min(1.0, R) * 1000) / 1000;
}

// ── OPTIONS FLOW CORE ────────────────────────────────────────────────────────

// Aggregates per-row FreeFlow data into per-strike buckets.
// Uses strike_futures when present; derives from ETF strike × ratio otherwise.
function aggregateDataset(data) {
  const rows  = data.rows  || [];
  const ratio = data.ratio || 41.14;
  const spotEtf = data.etf_spot || 0;   // QQQ price — gamma/theta are per-ETF-share
  const strikes = {};
  for (const row of rows) {
    const etf = row.strike_etf || 0;
    const sf  = row.strike_futures != null
      ? Math.round(row.strike_futures * 10) / 10
      : Math.round(etf * ratio * 10) / 10;
    if (!strikes[sf]) {
      strikes[sf] = { strike_etf: etf, net_gex: 0, abs_gex: 0, net_vex: 0, net_charmex: 0,
                      net_dex: 0, net_vegaex: 0, net_dag: 0, net_tex: 0, total_oi: 0,
                      call_oi: 0, put_oi: 0, _iv_oi_sum: 0 };
    }
    const s = strikes[sf];
    const oi = row.oi || 0;
    s.net_gex     += row.gex     || 0;
    // abs_gex (raw `ag`): gross/two-sided gamma at the strike. A strike can have
    // small net GEX but large gross gamma — relevant to wall stickiness. Falls
    // back to |gex| if the upstream row omits `ag`.
    s.abs_gex     += (row.ag != null ? row.ag : Math.abs(row.gex || 0));
    s.net_vex     += row.vex     || 0;
    s.net_charmex += row.charmex || 0;
    s.net_dex     += row.dex     || 0;
    s.net_vegaex  += row.vegaex  || 0;
    s.net_dag     += row.dag     || 0;
    s.total_oi    += oi;
    // Call/put OI split (raw `right`): each row is one side of the strike.
    // Lets downstream compute put/call ratio + dominant side per level.
    if (row.right === 'C')      s.call_oi += oi;
    else if (row.right === 'P') s.put_oi  += oi;
    // OI-weighted implied vol accumulator → strike_iv (skew vs ATM). Far-OTM
    // iv_pct is unreliable, but levels are filtered to ±FILTER_PCT (near-money)
    // where it's sane. OI-weighting damps thin strikes.
    if (row.iv_pct != null && oi > 0) s._iv_oi_sum += row.iv_pct * oi;
    // net_tex — theta exposure ($ time decay/day). FreeFlow gives no theta, so
    // derive it from the gamma-theta identity (driftless, r≈0 — exact for 0DTE):
    //   theta_annual_per_share = -½·Γ·S²·σ²    (T cancels; reuses FreeFlow's Γ)
    // → daily $ decay of this strike's OI = θ_annual/365 × 100 shares × OI.
    // Negative = the book bleeds this many $/day to decay at this strike. Garbage
    // far-OTM iv_pct yields garbage tex, but only ±FILTER_PCT strikes are surfaced.
    if (row.gamma != null && row.iv_pct != null && spotEtf > 0 && oi > 0) {
      const sig = row.iv_pct / 100;
      s.net_tex += -0.5 * row.gamma * spotEtf * spotEtf * sig * sig / 365 * 100 * oi;
    }
  }
  // Finalize OI-weighted strike IV and drop the accumulator.
  for (const sf in strikes) {
    const s = strikes[sf];
    s.strike_iv = s.total_oi > 0 ? s._iv_oi_sum / s.total_oi : null;
    delete s._iv_oi_sum;
  }
  // book_dex / book_gex: prefer the API's top-level totals, but FreeFlow's
  // futures-levels payload frequently omits `total_dex` (and sometimes
  // `total_gex`). When absent, fall back to summing the per-strike nets we
  // already accumulated above — otherwise the DEX readout is null forever even
  // though every row carried a `dex`. (This is the "gex delta doesn't work" bug.)
  let sumGex = 0, sumDex = 0;
  for (const sf in strikes) { sumGex += strikes[sf].net_gex; sumDex += strikes[sf].net_dex; }
  return {
    strikes,
    futuresPrice: data.futures_price || 0,
    spotEtf:      data.etf_spot      || 0,
    ratio:        data.ratio         || 41.14,
    bookGex:      data.total_gex     != null ? data.total_gex : sumGex,
    bookDex:      data.total_dex     != null ? data.total_dex : sumDex,
  };
}

// Smile-consistent ATM implied vol: the at-the-money strike's own OI-weighted
// iv_pct (strike_iv). Independent of the flaky /vol/realized endpoint, so it's
// used as a FALLBACK when current_iv is null/timed-out — keeping GTBR and the
// gamma-regime band alive even when the vol endpoint is down. Returns null only
// if no strike near spot carries an iv. NOTE: 0DTE smile ATM IV reads lower than
// the vol endpoint's tenor (~21 vs ~35), so vol_regime EXPANSION detection is
// conservative on this path — acceptable, since the alternative is iv=null.
function smileAtmIv(strikes, spotEtf) {
  if (!spotEtf) return null;
  let best = null, bestDist = Infinity;
  for (const sf in strikes) {
    const s = strikes[sf];
    if (s.strike_iv == null || !(s.strike_iv > 0)) continue;
    const dist = Math.abs((s.strike_etf || 0) - spotEtf);
    if (dist < bestDist) { bestDist = dist; best = s.strike_iv; }
  }
  return best != null ? Math.round(best * 10) / 10 : null;
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

// Robust normalisation: scale by the 90th-percentile magnitude (clamped to
// [0,1]) instead of the single max. A lone monster wall (e.g. a static 2.6B
// overnight strike) no longer defines the ceiling and crush every mid-tier
// intraday wall to ~0 — the "typical strong wall" sets the scale, so a genuine
// rank-5 level keeps a meaningful score.
function normalizeRobust(values) {
  const abs = values.map(Math.abs);
  if (!abs.length) return abs;
  const sorted = [...abs].sort((a, b) => a - b);
  const p90 = sorted[Math.floor(0.9 * (sorted.length - 1))] || 0;
  if (p90 <= 0) return abs.map(() => 0);
  return abs.map(v => Math.min(1, v / p90));
}

// Side-aware robust normalisation of net_gex: call walls (>0) are scaled among
// call walls, put walls (<=0) among put walls. Without this, one giant put wall
// normalises every CALL wall to ~0 (and vice-versa) even though they live on
// opposite sides of price. Index-aligned to the input array.
function normalizeGexPerSide(netGex) {
  const out = new Array(netGex.length).fill(0);
  for (const side of [1, -1]) {
    const idx = [], vals = [];
    netGex.forEach((g, i) => {
      if ((g > 0 ? 1 : -1) === side) { idx.push(i); vals.push(g); }
    });
    const norm = normalizeRobust(vals);
    idx.forEach((i, k) => { out[i] = norm[k]; });
  }
  return out;
}

// Protrusion: how much each strike's |GEX| exceeds its local neighborhood mean.
// Strikes are ordered by price (ascending) — guaranteed by Object.entries integer key ordering.
// Returns 0-1 normalized values; 1 = dominant local peak, 0 = blends into surroundings.
// Applied as a score multiplier: (0.5 + 0.5 * protrusion) so non-protruding strikes
// (shoulder/ramp walls — exactly the rungs price reverses at on a gamma ladder)
// keep at least half credit, while true isolated peaks still get full credit.
function computeProtrusion(values, windowHalf = 3) {
  const abs = values.map(Math.abs);
  const ratios = abs.map((v, i) => {
    let sum = 0, n = 0;
    for (let j = Math.max(0, i - windowHalf); j <= Math.min(abs.length - 1, i + windowHalf); j++) {
      if (j !== i) { sum += abs[j]; n++; }
    }
    const localMean = n > 0 ? sum / n : 0;
    return localMean < 1e-9 ? 1 : Math.min(v / (localMean + 1e-9), 6);
  });
  const mn = Math.min(...ratios), mx = Math.max(...ratios);
  if (mx === mn) return ratios.map(() => 1);
  return ratios.map(r => (r - mn) / (mx - mn));
}

// ── GTBR & HPS ───────────────────────────────────────────────────────────────

// Gamma-Theta Breakeven Range: expected remaining price range in NQ points.
// Full-day 1σ scaled by fraction of RTH remaining — critical for 0DTE because
// a wall 120pts away at 9:30am is unreachable by 14:00 when GTBR has halved.
// Returns null when iv or price unavailable.
function computeGTBR(futuresPrice, iv, timeOfDayET) {
  if (!futuresPrice || !iv || iv <= 0) return null;
  const daily1sd = futuresPrice * (iv / 100) / Math.sqrt(252);
  const tRemaining = Math.max(0, 16.0 - (timeOfDayET || 9.5)) / 6.5;
  return daily1sd * (tRemaining > 0 ? Math.sqrt(tRemaining) : 1);
}

// 5-condition mechanistic checklist for a single wall.
// Complements hold_prob (XGBoost, empirically trained) with transparent dealer
// mechanics reasoning. Each condition maps to a specific obligation or flow.
// protrusion: this level's gexProt score (0-1) from scoreLevels.
// Returns { score: 0-5, label: 'HIGH'|'MEDIUM'|'LOW', conditions: {…} }
function computeHPS(level, futuresPrice, iv, gammaFlip, volRegime, protrusion, timeOfDayET) {
  const isPut = (level.net_gex || 0) < 0;

  // 1. Positive gamma regime: spot above flip → dealers long gamma → walls hold
  const regime_positive = (gammaFlip != null && futuresPrice != null)
    ? futuresPrice > gammaFlip : false;

  // 2. GTBR inside: wall is within expected remaining daily range → likely to be tested today
  const gtbr = computeGTBR(futuresPrice, iv, timeOfDayET);
  const gtbr_inside = (gtbr != null)
    ? Math.abs(level.dist_nq || 0) <= gtbr : false;

  // 3. DEX aligned: dealer delta obligation is counter-trend to spot approaching wall
  //    Call wall: positive DEX → dealers net long delta → mechanically must sell into rallies
  //    Put wall:  negative DEX → dealers net short delta → mechanically must buy dips
  const dex_aligned = isPut
    ? (level.net_dex || 0) < 0
    : (level.net_dex || 0) > 0;

  // 4. Charm/Vanna structural support
  //    Charm: put wall + positive charm → OTM puts decay toward close, dealers unwind
  //           short delta = structural bid; call wall: opposite for structural sell
  //    Vanna pin: contraction vol + high vanna → IV drop reinforces pin at this strike
  const charm_aligned = isPut
    ? (level.net_charmex || 0) > 0
    : (level.net_charmex || 0) < 0;
  const vanna_pin = (volRegime === 'CONTRACTION') && (level.vex_norm || 0) > 50;
  const charm_vanna = charm_aligned || vanna_pin;

  // 5. GEX magnitude outlier: local spike vs 6-strike neighborhood
  //    protrusion > 0.6 ≈ top ~35% of the chain = dominant local peak
  const magnitude_outlier = (protrusion || 0) > 0.6;

  const conditions = { regime_positive, gtbr_inside, dex_aligned, charm_vanna, magnitude_outlier };
  const score = Object.values(conditions).filter(Boolean).length;
  const label = score >= 4 ? 'HIGH' : score === 3 ? 'MEDIUM' : 'LOW';

  return { score, label, conditions };
}

// Returns strikes within FILTER_PCT of futures price, with `strike_futures`
// and `dist_nq` attached. Used by both scoreLevels (which then filters by
// MIN_SCORE) and by callers that need the broader nearby set for aggregate
// statistics (e.g. the aggregate-bias table lookup which should see all
// proximate strikes, not just the scoring-worthy ones).
function nearbyStrikes(strikes, futuresPrice) {
  if (!futuresPrice) return [];
  return Object.entries(strikes)
    .map(([sf, s]) => ({ ...s, strike_futures: +sf, dist_nq: +sf - futuresPrice }))
    .filter(r => Math.abs(r.dist_nq / futuresPrice * 100) <= FILTER_PCT);
}

// Per-side non-maximum suppression. Walls already sorted score-desc.
// Once a wall is selected, any subsequent wall within `separation` NQ pts
// on the same side is suppressed — collapsing adjacent-strike clusters into
// the single dominant level. Returns a Set of surviving strike_futures values.
function _nmsWalls(walls, separation) {
  const survivors = [];
  for (const w of walls) {
    if (!survivors.some(s => Math.abs(s.strike_futures - w.strike_futures) < separation))
      survivors.push(w);
  }
  return new Set(survivors.map(s => s.strike_futures));
}

// Scores nearby strikes (within FILTER_PCT) using regime-adjusted weights, then
// returns those above MIN_SCORE OR in the guaranteed top-gamma-per-side set
// (surfaced_by: 'score' | 'gamma_rank'), sorted by score descending. Each level
// is tagged is_dominant (per-side NMS leader) and conviction.
// gammaFlip + iv feed the regime relevance multiplier, hold_prob, and GTBR.
// volRegime feeds HPS. volCtx (optional) = { rvIvRatio, hv5, hv63 } enables the
// VRP + term-structure filters in computeHoldProb; omitted callers no-op those.
function scoreLevels(strikes, weights, futuresPrice, volRegime, gammaFlip, iv, volCtx) {
  const nearby = nearbyStrikes(strikes, futuresPrice);
  if (!nearby.length) return [];

  const gexN = normalizeGexPerSide(nearby.map(r => r.net_gex));
  const vexN = normalizeRobust(nearby.map(r => r.net_vex));
  const chmN = normalizeRobust(nearby.map(r => r.net_charmex));
  const oiN  = normalizeRobust(nearby.map(r => r.total_oi));
  const dagN = normalizeRobust(nearby.map(r => r.net_dag));
  const gexProt = computeProtrusion(nearby.map(r => r.net_gex));

  const timeET = currentHourET();

  // Global gamma regime from spot vs flip — same vol-scaled band as levels.js
  // and hold_prob. This makes the SURFACE (not just hold_prob) respect where we
  // are relative to the flip on every calculation.
  const regimeBand = (iv > 0 && futuresPrice)
    ? Math.max(30, 0.5 * futuresPrice * (iv / 100) / Math.sqrt(252))
    : 50;
  const flipDiff = (gammaFlip != null && futuresPrice != null)
    ? futuresPrice - gammaFlip : null;
  const regimeState = flipDiff == null ? 'UNKNOWN'
    : flipDiff >  regimeBand ? 'POSITIVE'
    : flipDiff < -regimeBand ? 'NEGATIVE'
    : 'NEAR_FLIP';

  // Per-wall regime relevance. Reversal geometry = resistance above spot (call
  // wall) or support below spot (put wall) — where dealer hedging rejects price.
  // The opposite geometry is an acceleration zone. Long-gamma (POSITIVE)
  // reinforces reversals; short-gamma (NEGATIVE) lets price run through them;
  // near-flip is unstable. Gentle multipliers so nothing is over-suppressed.
  const regimeRelevance = (netGex, distNq) => {
    const reversal = (distNq > 0) ? (netGex > 0) : (netGex <= 0);
    if (regimeState === 'POSITIVE')  return reversal ? 1.15 : 0.90;
    if (regimeState === 'NEGATIVE')  return reversal ? 0.85 : 1.00;
    if (regimeState === 'NEAR_FLIP') return reversal ? 1.00 : 0.95;
    return 1.0;
  };

  // Request-level context shared by every strike's hold_prob.
  const vc      = volCtx || {};
  const hvTermRatio = (vc.hv5 != null && vc.hv63 != null && vc.hv63 > 0)
    ? vc.hv5 / vc.hv63 : null;
  const dagAbs  = nearby.map(r => Math.abs(r.net_dag || 0)).sort((a, b) => a - b);
  // 90th-percentile |DAG| cutoff for the late-session pinning boost.
  const dagDecileThr = dagAbs.length
    ? dagAbs[Math.min(dagAbs.length - 1, Math.floor(0.9 * (dagAbs.length - 1)))]
    : null;
  const holdCtx = {
    gammaFlip, futuresPrice, timeOfDayET: timeET, atmIv: iv,
    rvIvRatio:   vc.rvIvRatio != null ? vc.rvIvRatio : null,
    hvTermRatio,
    gtbr:        computeGTBR(futuresPrice, iv, timeET),
    dagDecileThr,
  };

  const scoredAll = nearby
    .map((r, i) => {
      const rawScore = (gexN[i]*weights.gex + vexN[i]*weights.vex + chmN[i]*weights.charmex +
                        oiN[i]*weights.oi   + dagN[i]*weights.dag) * 100;
      const score = rawScore * (0.5 + 0.5 * gexProt[i])
                             * regimeRelevance(r.net_gex, r.dist_nq);
      const volSens = Math.abs(r.net_vex) / (Math.abs(r.net_gex) + 1e-9);
      const base    = r.net_gex > 0 ? 'CALL WALL' : 'PUT WALL';
      const wall_reaction = classifyWallReaction(r);
      const levelWithNorm = {
        ...r,
        gex_norm:     gexN[i]  * 100,
        vex_norm:     vexN[i]  * 100,
        charmex_norm: chmN[i]  * 100,
        oi_norm:      oiN[i]   * 100,
        strike_futures: r.strike_futures,
        protrusion:   gexProt[i],
      };
      const hold_prob = computeHoldProb(levelWithNorm, holdCtx);
      const hps = computeHPS(levelWithNorm, futuresPrice, iv, gammaFlip, volRegime, gexProt[i], timeET);
      return {
        strike_futures: Math.round(r.strike_futures * 10)  / 10,
        strike_etf:     Math.round(r.strike_etf     * 100) / 100,
        dist_nq:        Math.round(r.dist_nq        * 10)  / 10,
        score:          Math.round(score            * 10)  / 10,
        hold_prob,
        hps_score:      hps.score,
        hps_label:      hps.label,
        hps_conditions: hps.conditions,
        type:           base + (volSens > 2.0 ? ' + VOL SENSITIVE' : ''),
        net_gex:        Math.round(r.net_gex),
        abs_gex:        Math.round(r.abs_gex || 0),
        call_oi:        Math.round(r.call_oi || 0),
        put_oi:         Math.round(r.put_oi  || 0),
        strike_iv:      r.strike_iv != null ? Math.round(r.strike_iv * 10) / 10 : null,
        net_vex:        Math.round(r.net_vex),
        net_charmex:    Math.round(r.net_charmex),
        net_dex:        Math.round(r.net_dex),
        net_vegaex:     Math.round(r.net_vegaex),
        net_tex:        Math.round(r.net_tex || 0),
        total_oi:       Math.round(r.total_oi),
        wall_reaction,
      };
    });

  // Guaranteed surface: the strongest gross-gamma walls per side near the money
  // are ALWAYS kept, even if the composite filtered them out — so a genuine
  // rank-5 reversal wall can never be silently dropped just because a monster
  // static wall dominates the composite. Tagged surfaced_by:'gamma_rank'.
  const NEAR_BAND_PCT = 2.5, KEEP_PER_SIDE = 5;
  const guaranteed = new Set();
  for (const callSide of [true, false]) {
    scoredAll
      .filter(l => (l.net_gex > 0) === callSide
                && Math.abs(l.dist_nq / futuresPrice * 100) <= NEAR_BAND_PCT)
      .sort((a, b) => b.abs_gex - a.abs_gex)
      .slice(0, KEEP_PER_SIDE)
      .forEach(l => guaranteed.add(l.strike_futures));
  }

  const scored = scoredAll
    .map(l => ({
      ...l,
      surfaced_by: l.score >= MIN_SCORE ? 'score'
                 : guaranteed.has(l.strike_futures) ? 'gamma_rank' : null,
    }))
    .filter(l => l.surfaced_by != null)
    .sort((a, b) => b.score - a.score);

  // NMS: ~2 QQQ strike intervals (≈0.25% of price ≈ $1.8 QQQ). Collapses only
  // immediate-neighbour strikes (and futures-rounding duplicates) into the local
  // leader — distinct dollar strikes stay separate. The old 0.6% (~4.4 QQQ pts)
  // merged genuinely different 0DTE levels (e.g. 733 vs 735) into one zone, so a
  // strike you actually reverse at got demoted to CONTEXT under its neighbour.
  const sep         = futuresPrice * 0.0025;
  const dominantSet = new Set([
    ..._nmsWalls(scored.filter(l => l.net_gex >  0), sep),
    ..._nmsWalls(scored.filter(l => l.net_gex <= 0), sep),
  ]);

  return scored.map(l => {
    const is_dominant = dominantSet.has(l.strike_futures);
    const conviction  = is_dominant && l.hps_score >= 4 ? 'STANDALONE'
                      : is_dominant && l.hps_score === 3 ? 'CONFIRM'
                      : 'CONTEXT';
    return { ...l, is_dominant, conviction };
  });
}

// Maps IV / RV:IV ratio to a vol regime name string.
function classifyVolRegime(iv, rvIvRatio) {
  if (iv != null && (iv >= 30 || (rvIvRatio != null && rvIvRatio < 0.5))) return 'EXPANSION';
  if (iv != null && iv >= 20) return 'NEUTRAL';
  if (iv != null)             return 'CONTRACTION';
  return 'NEUTRAL';
}

// Looks up the 2D weight table. gammaRegime defaults to NEAR_FLIP when unknown.
function getWeights(volRegime, gammaRegime) {
  const vr = REGIME_WEIGHTS[volRegime] || REGIME_WEIGHTS.NEUTRAL;
  return vr[gammaRegime] || vr.NEAR_FLIP;
}

// ── PRIVATE METHODOLOGY TABLES ────────────────────────────────────────────────
// Encodes reaction and bias tables from owner-only methodology sources plus
// published empirical asymmetries (Elms 2026, Dim/Eraker/Vilkov 2025). Outputs
// expected directional behavior at strikes and in aggregate — complementing the
// magnitude-only `score` field.

// Classify a single wall's expected reaction. CALL_WALL vs PUT_WALL determined
// by net_gex sign. Returns a tag.
function classifyWallReaction(level) {
  if (!level) return null;
  const isCallWall    = (level.net_gex || 0) > 0;
  const dexPositive   = (level.net_dex || 0) > 0;
  const charmPositive = (level.net_charmex || 0) > 0;
  const vannaPositive = (level.net_vex || 0) > 0;

  if (isCallWall) {
    if ( dexPositive && !charmPositive && !vannaPositive) return 'CALL_WALL_BEARISH_REJECT';
    if (!dexPositive && !charmPositive && !vannaPositive) return 'CALL_WALL_BEARISH_BREAKDOWN';
    if (!dexPositive &&  charmPositive &&  vannaPositive) return 'CALL_WALL_BULLISH_SQUEEZE';
    if ( dexPositive &&  charmPositive &&  vannaPositive) return 'CALL_WALL_BULLISH_GRIND';
    return 'CALL_WALL_MIXED';
  } else {
    if ( dexPositive &&  charmPositive &&  vannaPositive) return 'PUT_WALL_BULLISH_SUPPORT';
    if (!dexPositive && !charmPositive && !vannaPositive) return 'PUT_WALL_VULNERABLE';
    if (!dexPositive &&  charmPositive &&  vannaPositive) return 'PUT_WALL_BULLISH_REVERSAL';
    if ( dexPositive && !charmPositive && !vannaPositive) return 'PUT_WALL_WEAK_BOUNCE_FADE';
    return 'PUT_WALL_MIXED';
  }
}

// Tag-to-direction map for downstream evidence accumulation.
// Each wall_reaction tag → { dir: 'BULL'|'BEAR'|'NEUTRAL', strength: 0..2 }
// strength 2 = breakdown/expansion (highest conviction); 1 = standard reaction; 0 = mixed.
const WALL_REACTION_DIR = {
  CALL_WALL_BEARISH_REJECT:    { dir: 'BEAR', strength: 1 },
  CALL_WALL_BEARISH_BREAKDOWN: { dir: 'BEAR', strength: 2 },
  CALL_WALL_BULLISH_SQUEEZE:   { dir: 'BULL', strength: 2 },
  CALL_WALL_BULLISH_GRIND:     { dir: 'BULL', strength: 1 },
  CALL_WALL_MIXED:             { dir: 'NEUTRAL', strength: 0 },
  PUT_WALL_BULLISH_SUPPORT:    { dir: 'BULL', strength: 1 },
  PUT_WALL_VULNERABLE:         { dir: 'BEAR', strength: 2 },
  PUT_WALL_BULLISH_REVERSAL:   { dir: 'BULL', strength: 2 },
  PUT_WALL_WEAK_BOUNCE_FADE:   { dir: 'BEAR', strength: 1 },
  PUT_WALL_MIXED:              { dir: 'NEUTRAL', strength: 0 },
};

// Aggregate Greek signs across nearby strikes (feeds the bias table below).
// Returns { gex_sign, charm_sign, vanna_sign, dex_sign, totals... } or null.
function computeAggregateGreeks(levels) {
  if (!levels || !levels.length) return null;
  let total_gex = 0, total_charmex = 0, total_vex = 0, total_dex = 0;
  for (const lv of levels) {
    total_gex     += lv.net_gex     || 0;
    total_charmex += lv.net_charmex || 0;
    total_vex     += lv.net_vex     || 0;
    total_dex     += lv.net_dex     || 0;
  }
  const sgn = x => x > 0 ? 1 : x < 0 ? -1 : 0;
  return {
    gex_sign:      sgn(total_gex),
    charm_sign:    sgn(total_charmex),
    vanna_sign:    sgn(total_vex),
    dex_sign:      sgn(total_dex),
    total_gex:     Math.round(total_gex),
    total_charmex: Math.round(total_charmex),
    total_vex:     Math.round(total_vex),
    total_dex:     Math.round(total_dex),
  };
}

// Look up the aggregate-bias condition table. Picks the first matching rule.
// Trend conditions (negative gamma + concordant charm/vanna) win over chop
// conditions, which win over IV-regime conditions, which win over price-vs-flip.
function applyBiasTable(aggregates, volRegime, priceVsFlip) {
  if (!aggregates) return 'UNKNOWN';
  const { gex_sign, charm_sign, vanna_sign } = aggregates;
  if (gex_sign < 0 && charm_sign < 0 && vanna_sign < 0) return 'STRONG_BEARISH_TREND';
  if (gex_sign < 0 && charm_sign > 0 && vanna_sign > 0) return 'STRONG_BULLISH_TREND';
  if (gex_sign > 0 && charm_sign > 0 && vanna_sign > 0) return 'BULLISH_CHOP';
  if (gex_sign > 0 && charm_sign < 0 && vanna_sign < 0) return 'BEARISH_CHOP';
  if (gex_sign < 0 && volRegime === 'EXPANSION')        return 'VOLATILE_EXPANSION';
  if (gex_sign > 0 && volRegime === 'CONTRACTION')      return 'COMPRESSION';
  if (vanna_sign > 0 && volRegime === 'CONTRACTION')    return 'VANNA_BULLISH';
  if (vanna_sign < 0 && volRegime === 'EXPANSION')      return 'VANNA_BEARISH';
  if (priceVsFlip > 0)                                  return 'BULLISH_REGIME';
  if (priceVsFlip < 0)                                  return 'BEARISH_REGIME';
  return 'NEUTRAL';
}

// Tag-to-direction map for the aggregate-bias output.
const BIAS_TAG_DIR = {
  STRONG_BEARISH_TREND: { dir: 'BEAR', strength: 2 },
  STRONG_BULLISH_TREND: { dir: 'BULL', strength: 2 },
  BULLISH_CHOP:         { dir: 'BULL', strength: 1 },
  BEARISH_CHOP:         { dir: 'BEAR', strength: 1 },
  VOLATILE_EXPANSION:   { dir: 'NEUTRAL', strength: 0 },  // direction unknown
  COMPRESSION:          { dir: 'NEUTRAL', strength: 0 },
  VANNA_BULLISH:        { dir: 'BULL', strength: 1 },
  VANNA_BEARISH:        { dir: 'BEAR', strength: 1 },
  BULLISH_REGIME:       { dir: 'BULL', strength: 1 },
  BEARISH_REGIME:       { dir: 'BEAR', strength: 1 },
  NEUTRAL:              { dir: 'NEUTRAL', strength: 0 },
  UNKNOWN:              { dir: 'NEUTRAL', strength: 0 },
};

// Dim/Eraker/Vilkov 2025 asymmetry constant (2.pdf, Table 3, col 4):
// Positive MM gamma's vol-attenuation coefficient = -0.064.
// Negative MM gamma's vol-amplification coefficient = -0.022.
// The negative-gamma effect is ~65% smaller, i.e. ratio ≈ 0.022 / 0.064 ≈ 0.344.
// Use this when scaling evidence between positive- and negative-gamma signals.
const GAMMA_ASYMMETRY_RATIO = 0.344;

// Elms 2026 finding (1.pdf): modern SPX (and by inference NQ given 0DTE dominance)
// no longer exhibits pinning. High OI is associated with WIDER, not narrower, ranges
// (p=0.0003). Set this flag to true if a future per-instrument calibration shows
// NQ-specific pinning. Default false → walls are treated as amplification triggers
// when broken, not as range-bound anchors.
const PINNING_REGIME_ACTIVE = false;

module.exports = {
  FILTER_PCT, MIN_SCORE, VALID_USERS, SESSION_MAX_AGE_MS,
  REGIME_WEIGHTS, AGENT_HEADERS, BASE_HEADERS,
  isAuthorized, fetchJson, httpGetJson, todayET, currentHourET, computeHoldProb,
  aggregateDataset, computeGammaFlip, smileAtmIv,
  nearbyStrikes, scoreLevels, computeGTBR, computeHPS,
  classifyVolRegime, getWeights,
  classifyWallReaction, computeAggregateGreeks, applyBiasTable,
  WALL_REACTION_DIR, BIAS_TAG_DIR, GAMMA_ASYMMETRY_RATIO, PINNING_REGIME_ACTIVE,
};

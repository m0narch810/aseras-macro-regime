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

function currentHourET() {
  const parts = new Intl.DateTimeFormat('en-US', {
    timeZone: 'America/New_York', hour: 'numeric', minute: 'numeric', hour12: false,
  }).formatToParts(new Date());
  const h = parseInt(parts.find(p => p.type === 'hour').value,   10);
  const m = parseInt(parts.find(p => p.type === 'minute').value, 10);
  return h + m / 60;
}

// ── HOLD PROBABILITY MODEL ────────────────────────────────────────────────────
// Logistic regression fitted on 489,396 intraday touch events from 2,897
// Combined QQQ (Feb-Dec 2025) + SPY (Jan 2020-Dec 2022) dataset — 511,571 touch events.
// LR CV AUC 0.6485 ± 0.0015 vs composite score AUC 0.521. XGBoost CV AUC 0.8297.
// Multi-regime training: 2020 COVID, 2021 bull, 2022 bear, 2025 bull.
// Coefficients from models/wall_score_intraday.json — do not hand-tune.
//
// Features unavailable at inference time default to training-set means,
// zeroing out their standardised contribution:
//   - vex_over_gex, charmex_over_gex: unit-scale mismatch between OPRA and FreeFlow
//   - approach_vel, from_below: intraday touch features, not available at snapshot time
//   - pd_*/pw_* (volume profile): requires historical NQ 1m bars, not available in function
const _LR = {
  intercept: 0.003902,
  coef: {
    gex_norm:          0.001064,
    vex_norm:         -0.033479,
    charmex_norm:      0.077967,
    oi_norm:           0.033544,
    vex_over_gex:      0.000328,
    charmex_over_gex: -0.016355,
    dist_pct:          0.058991,
    is_high_vol:      -0.141109,
    is_contraction:    0.099613,
    is_put:            0.112393,
    in_neg_gamma:      0.029986,
    wall_above_flip:  -0.118096,
    confluence:        0.035603,
    time_of_day:       0.291898,
    approach_vel:     -0.017616,
    from_below:       -0.150283,
    pd_on_hvn:        -0.210247,
    pd_on_lvn:         0.005647,
    pd_in_value_area: -0.077016,
    pd_dist_to_poc:   -0.437749,
    pd_dist_to_hvn:    0.073889,
    pd_dist_to_lvn:    0.205878,
    pw_on_hvn:        -0.149423,
    pw_on_lvn:        -0.056097,
    pw_in_value_area: -0.055324,
    pw_dist_to_poc:   -0.054233,
    vp_aligned:        0.241393,
  },
  mean: {
    gex_norm:          77.233949,
    vex_norm:          65.747533,
    charmex_norm:      30.218452,
    oi_norm:           26.118006,
    vex_over_gex:      -3.684046,
    charmex_over_gex:  -1.707606,
    dist_pct:          -0.028924,
    is_high_vol:        0.737118,
    is_contraction:     0.087321,
    is_put:             0.768105,
    in_neg_gamma:       0.025246,
    wall_above_flip:    0.977366,
    confluence:         0.271563,
    time_of_day:       12.522011,
    approach_vel:       0.945891,
    from_below:         0.473461,
    pd_on_hvn:          0.105618,
    pd_on_lvn:          0.087433,
    pd_in_value_area:   0.181578,
    pd_dist_to_poc:   120.777540,
    pd_dist_to_hvn:    95.566733,
    pd_dist_to_lvn:    83.303753,
    pw_on_hvn:          0.071441,
    pw_on_lvn:          0.142000,
    pw_in_value_area:   0.277574,
    pw_dist_to_poc:   292.613343,
    vp_aligned:         0.174607,
  },
  scale: {
    gex_norm:          30.326448,
    vex_norm:          32.001922,
    charmex_norm:      31.526241,
    oi_norm:           23.179645,
    vex_over_gex:     539.318444,
    charmex_over_gex: 117.024901,
    dist_pct:           0.263279,
    is_high_vol:        0.440199,
    is_contraction:     0.282305,
    is_put:             0.422043,
    in_neg_gamma:       0.156871,
    wall_above_flip:    0.148734,
    confluence:         0.444766,
    time_of_day:        1.702099,
    approach_vel:       5.155001,
    from_below:         0.499295,
    pd_on_hvn:          0.307348,
    pd_on_lvn:          0.282468,
    pd_in_value_area:   0.385496,
    pd_dist_to_poc:   157.127889,
    pd_dist_to_hvn:   140.624823,
    pd_dist_to_lvn:   125.023897,
    pw_on_hvn:          0.257560,
    pw_on_lvn:          0.349050,
    pw_in_value_area:   0.447802,
    pw_dist_to_poc:   288.250723,
    vp_aligned:         0.379631,
  },
};

// Expected hold_prob for a wall with all-average Greek features at the given
// time of day. Used by the frontend to set a time-adjusted display threshold.
function computeTimeBaseline(timeOfDayET) {
  const z = _LR.intercept
    + _LR.coef.time_of_day * (timeOfDayET - _LR.mean.time_of_day) / _LR.scale.time_of_day;
  return 1 / (1 + Math.exp(-z));
}

// gammaFlip and futuresPrice are optional; when provided, enables in_neg_gamma
// and wall_above_flip features which meaningfully improve the prediction.
function computeHoldProb(level, volRegime, timeOfDayET, gammaFlip, futuresPrice) {
  const isPut   = (level.net_gex || 0) < 0;
  const absGex  = Math.abs(level.net_gex || 0) + 1e-9;
  const distPct = level.dist_nq != null && (level.strike_futures || futuresPrice)
    ? level.dist_nq / (level.strike_futures || futuresPrice) * 100
    : 0;

  // Start with training means for all features — unavailable features contribute zero
  // after standardisation (raw[f] == mean[f] => (raw-mean)/scale == 0).
  const raw = Object.assign({}, _LR.mean);

  // Override with computed values for features available at inference time
  raw.gex_norm         = Math.abs(level.gex_norm     || 0);
  raw.vex_norm         = Math.abs(level.vex_norm     || 0);
  raw.charmex_norm     = Math.abs(level.charmex_norm || 0);
  raw.oi_norm          = Math.abs(level.oi_norm      || 0);
  raw.dist_pct         = distPct;
  raw.is_high_vol      = volRegime === 'EXPANSION'   ? 1 : 0;
  raw.is_contraction   = volRegime === 'CONTRACTION' ? 1 : 0;
  raw.is_put           = isPut ? 1 : 0;
  raw.in_neg_gamma     = (gammaFlip != null && futuresPrice != null)
                           ? (futuresPrice < gammaFlip ? 1 : 0)
                           : _LR.mean.in_neg_gamma;
  raw.wall_above_flip  = (gammaFlip != null && level.strike_futures != null)
                           ? (level.strike_futures > gammaFlip ? 1 : 0)
                           : _LR.mean.wall_above_flip;
  raw.confluence       = (level.confluence != null) ? level.confluence : _LR.mean.confluence;
  raw.time_of_day      = timeOfDayET || _LR.mean.time_of_day;
  // vex_over_gex, charmex_over_gex, approach_vel, from_below, pd_*/pw_* stay at mean

  let z = _LR.intercept;
  for (const [feat, coef] of Object.entries(_LR.coef)) {
    z += coef * (raw[feat] - _LR.mean[feat]) / _LR.scale[feat];
  }
  return Math.round(1 / (1 + Math.exp(-z)) * 1000) / 1000;
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

// Scores nearby strikes using regime-adjusted weights. Returns strikes above
// MIN_SCORE sorted descending. Only strikes within FILTER_PCT of futures price.
// volRegime and gammaFlip are passed through to computeHoldProb.
function scoreLevels(strikes, weights, futuresPrice, volRegime, gammaFlip) {
  const nearby = nearbyStrikes(strikes, futuresPrice);
  if (!nearby.length) return [];

  const gexN = normalizeAbs(nearby.map(r => r.net_gex));
  const vexN = normalizeAbs(nearby.map(r => r.net_vex));
  const chmN = normalizeAbs(nearby.map(r => r.net_charmex));
  const oiN  = normalizeAbs(nearby.map(r => r.total_oi));
  const dagN = normalizeAbs(nearby.map(r => r.net_dag));

  const timeET = currentHourET();

  return nearby
    .map((r, i) => {
      const score   = (gexN[i]*weights.gex + vexN[i]*weights.vex + chmN[i]*weights.charmex +
                       oiN[i]*weights.oi   + dagN[i]*weights.dag) * 100;
      const volSens = Math.abs(r.net_vex) / (Math.abs(r.net_gex) + 1e-9);
      const base    = r.net_gex > 0 ? 'CALL WALL' : 'PUT WALL';
      const wall_reaction = classifyWallReaction(r);
      // Attach normalised Greek fields so computeHoldProb can read them.
      const levelWithNorm = {
        ...r,
        gex_norm:     gexN[i]  * 100,
        vex_norm:     vexN[i]  * 100,
        charmex_norm: chmN[i]  * 100,
        oi_norm:      oiN[i]   * 100,
        strike_futures: r.strike_futures,
      };
      const hold_prob = computeHoldProb(levelWithNorm, volRegime, timeET, gammaFlip, futuresPrice);
      return {
        strike_futures: Math.round(r.strike_futures * 10)  / 10,
        strike_etf:     Math.round(r.strike_etf     * 100) / 100,
        dist_nq:        Math.round(r.dist_nq        * 10)  / 10,
        score:          Math.round(score            * 10)  / 10,
        hold_prob,
        type:           base + (volSens > 2.0 ? ' + VOL SENSITIVE' : ''),
        net_gex:        Math.round(r.net_gex),
        net_vex:        Math.round(r.net_vex),
        net_charmex:    Math.round(r.net_charmex),
        net_dex:        Math.round(r.net_dex),
        net_vegaex:     Math.round(r.net_vegaex),
        total_oi:       Math.round(r.total_oi),
        wall_reaction,
      };
    })
    .filter(r => r.score >= MIN_SCORE)
    .sort((a, b) => b.score - a.score);
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
  isAuthorized, fetchJson, httpGetJson, todayET, currentHourET, computeHoldProb, computeTimeBaseline,
  aggregateDataset, computeGammaFlip, normalizeAbs,
  nearbyStrikes, scoreLevels,
  classifyVolRegime, getWeights,
  classifyWallReaction, computeAggregateGreeks, applyBiasTable,
  WALL_REACTION_DIR, BIAS_TAG_DIR, GAMMA_ASYMMETRY_RATIO, PINNING_REGIME_ACTIVE,
};

const https = require('https');

const SYMBOL     = 'QQQ';
const BASE_URL   = 'https://www.free-flow.site/api';
const FILTER_PCT = 5.0;
const MIN_SCORE  = 20.0;

// Must match 09_intraday_bias.py
const NEAR_FLIP_BUFFER     = 50.0;
const H_GEX_CONFIDENCE_CUT = 0.6;
const STRONG_WALL          = 60.0;
const EXCEPTIONAL_WALL     = 75.0;
const AIR_POCKET_PROXIMITY = 150.0;
const PROXIMITY_HALFLIFE   = 200.0;

const MACRO_BULL = new Set(['STRONG BULL', 'LEAN BULL']);
const MACRO_BEAR = new Set(['STRONG BEAR', 'LEAN BEAR']);

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

function fetchJsonPlain(url) {
  return new Promise((resolve, reject) => {
    const req = https.get(url, { headers: { 'Accept': 'application/json' } }, (res) => {
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
    req.setTimeout(10000, () => { req.destroy(); reject(new Error('Timeout')); });
  });
}

// ── DATE HELPERS ─────────────────────────────────────────────────────────────
function todayET() {
  return new Date().toLocaleDateString('en-CA', { timeZone: 'America/New_York' });
}

// ── AGGREGATION (shared with levels.js) ──────────────────────────────────────
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
    ratio:        data.ratio         || 41.14,
  };
}

// ── GAMMA FLIP (shared with levels.js) ───────────────────────────────────────
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
    for (const row of sorted) {
      if (Math.abs(row.gex) < minAbs) { minAbs = Math.abs(row.gex); bestFlip = row.strike; }
    }
  }

  return bestFlip != null ? Math.round(bestFlip * 10) / 10 : null;
}

// ── LEVEL SCORING (shared with levels.js) ────────────────────────────────────
function normalizeAbs(values) {
  const abs = values.map(Math.abs);
  const mn  = Math.min(...abs), mx = Math.max(...abs);
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
      const score   = (gexN[i]*weights.gex + vexN[i]*weights.vex + chmN[i]*weights.charmex + oiN[i]*weights.oi + dagN[i]*weights.dag) * 100;
      const volSens = Math.abs(r.net_vex) / (Math.abs(r.net_gex) + 1e-9);
      const base    = r.net_gex > 0 ? 'CALL WALL' : 'PUT WALL';
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

// ── INTRADAY-SPECIFIC FEATURES ───────────────────────────────────────────────
function computeHGEXNorm(levels) {
  if (!levels.length) return 0.5;
  const gexAbs = levels.map(l => Math.abs(l.net_gex || 0));
  const total  = gexAbs.reduce((a, b) => a + b, 0);
  if (total === 0 || gexAbs.length < 2) return 0.0;
  const probs = gexAbs.map(g => g / total).filter(p => p > 0);
  const H     = -probs.reduce((sum, p) => sum + p * Math.log2(p), 0);
  const Hmax  = Math.log2(gexAbs.length);
  return Hmax > 0 ? Math.round(H / Hmax * 10000) / 10000 : 0.0;
}

function computeGammaRegime(nqPrice, gammaFlip) {
  if (gammaFlip == null || nqPrice == null) return 'UNKNOWN';
  const diff = nqPrice - gammaFlip;
  if (diff >  NEAR_FLIP_BUFFER) return 'POSITIVE';
  if (diff < -NEAR_FLIP_BUFFER) return 'NEGATIVE';
  return 'NEAR_FLIP';
}

function computeTopWall(levels, nqPrice) {
  if (!levels.length || nqPrice == null) return null;
  let best = null, bestWScore = -1;
  for (const lv of levels) {
    const dist   = Math.abs(lv.dist_nq || 9999);
    const weight = Math.exp(-dist / PROXIMITY_HALFLIFE);
    const wscore = (lv.score || 0) * weight;
    if (wscore > bestWScore) {
      bestWScore = wscore;
      best = { ...lv, proximity_score: Math.round(wscore * 100) / 100 };
    }
  }
  return best;
}

// ── INTRADAY CLASSIFIER ──────────────────────────────────────────────────────
function classifyIntradayBias({ gammaRegime, gammaFlip, nqPrice, topWall, hGexNorm, macroBias }) {
  const topScore  = topWall ? (topWall.score  || 0) : 0;
  const topType   = topWall ? (topWall.type   || '') : '';
  const topStrike = topWall ? topWall.strike_futures : null;

  let air_pocket_watch = false;
  let air_pocket_type  = null;
  let bias, conf, reason;

  const macroBullish = MACRO_BULL.has(macroBias);
  const macroBearish = MACRO_BEAR.has(macroBias);

  if (gammaRegime === 'POSITIVE') {
    const flipClose    = nqPrice != null && gammaFlip != null &&
                         Math.abs(nqPrice - gammaFlip) < NEAR_FLIP_BUFFER + 30;
    const wallNearFlip = topStrike != null && gammaFlip != null &&
                         Math.abs(topStrike - gammaFlip) < AIR_POCKET_PROXIMITY;

    if (flipClose && wallNearFlip) {
      air_pocket_watch = true; air_pocket_type = 'FLIP_CROSS';
      bias = 'BEARISH REVERSAL WATCH'; conf = 'MODERATE';
      reason = `FLIP_CROSS: price approaching gamma flip from positive side, strong wall near flip.`;
    } else if (topScore >= STRONG_WALL) {
      if (macroBullish) {
        bias = 'BULLISH'; conf = 'HIGH';
        reason = `Positive gamma regime, strong ${topType} (score=${topScore.toFixed(0)}), macro confirms bull.`;
      } else {
        bias = 'NEUTRAL_BULLISH'; conf = 'MODERATE';
        reason = `Positive gamma regime, strong ${topType} (score=${topScore.toFixed(0)}), macro not confirming.`;
      }
    } else {
      bias = 'NEUTRAL'; conf = 'LOW';
      reason = `Positive gamma regime but no strong wall (top score=${topScore.toFixed(0)}). Range-bound likely.`;
    }

  } else if (gammaRegime === 'NEGATIVE') {
    if (topScore >= EXCEPTIONAL_WALL) {
      air_pocket_watch = true; air_pocket_type = 'EXCEPTIONAL_PUT_WALL';
    }
    bias   = 'BEARISH CONTINUATION'; conf = 'MODERATE';
    reason = `Negative gamma regime (price below flip >${NEAR_FLIP_BUFFER}pts): dealers amplify moves.`;
    if (air_pocket_watch) reason += ` Exceptional put wall (score=${topScore.toFixed(0)}) detected.`;
    if (macroBullish && !air_pocket_watch) { conf = 'LOW'; reason += ' Macro bias is bullish — conflicting signal.'; }

  } else if (gammaRegime === 'NEAR_FLIP') {
    air_pocket_watch = true; air_pocket_type = 'FLIP_CROSS';
    if (topScore >= STRONG_WALL) {
      bias = topType.toUpperCase().includes('CALL') ? 'BEARISH REVERSAL' : 'BULLISH REVERSAL';
      conf = 'MODERATE';
      reason = `NEAR_FLIP: price within ${NEAR_FLIP_BUFFER}pts of gamma flip. Strong ${topType} (score=${topScore.toFixed(0)}) nearby.`;
    } else {
      bias = 'NEUTRAL'; conf = 'LOW';
      reason = `NEAR_FLIP: price within ${NEAR_FLIP_BUFFER}pts of gamma flip. No strong wall — high uncertainty.`;
    }

  } else {
    bias = 'NEUTRAL'; conf = 'LOW';
    reason = 'Gamma regime unknown (no flip data). Cannot classify.';
  }

  // Confidence modifiers
  const CONF_ORDER = ['LOW', 'MODERATE', 'HIGH'];
  function downgrade(c, sfx) {
    const newC = CONF_ORDER[Math.max(0, CONF_ORDER.indexOf(c) - 1)];
    return [newC, sfx];
  }
  if (hGexNorm > H_GEX_CONFIDENCE_CUT) {
    [conf, reason] = downgrade(conf, reason);
    reason += ` H_GEX_norm=${hGexNorm.toFixed(2)}>0.6: GEX dispersed, confidence penalized.`;
  }
  if (!macroBullish && !macroBearish) {
    [conf, reason] = downgrade(conf, reason);
    reason += ` Macro neutral (${macroBias}): confidence penalized.`;
  }

  return { intraday_bias: bias, confidence: conf, air_pocket_watch, air_pocket_type, reason };
}

// ── HANDLER ──────────────────────────────────────────────────────────────────
exports.handler = async (event) => {
  if (event.httpMethod === 'OPTIONS') {
    return { statusCode: 200, headers: OUT_HEADERS, body: '' };
  }

  try {
    const cookie = process.env.FF_SESSION || '';
    const exp    = todayET();

    const data = await fetchJson(`${BASE_URL}/futures-levels?symbol=${SYMBOL}&exp=${exp}`, cookie);
    if (!data.rows || !data.rows.length) throw new Error('No rows — FF_SESSION may be expired.');

    const { strikes, futuresPrice, spotEtf, ratio } = aggregateDataset(data);
    const gammaFlip = computeGammaFlip(strikes, futuresPrice);

    // Vol regime
    let levelsRegime = 'UNKNOWN';
    let iv = null, rv_iv_ratio = null, hv21 = null;
    try {
      const vol = await fetchJson(`${BASE_URL}/vol/realized?symbol=${SYMBOL}`, cookie);
      iv          = vol.current_iv  ?? null;
      rv_iv_ratio = vol.rv_iv_ratio ?? null;
      hv21        = vol.hv21        ?? null;
      if (iv >= 30 || rv_iv_ratio < 0.5) levelsRegime = 'EXPANSION';
      else if (iv >= 20)                  levelsRegime = 'NEUTRAL';
      else                                levelsRegime = 'CONTRACTION';
    } catch (_) {}

    // Score levels with NEUTRAL weights as default
    const weights = { gex: 0.32, vex: 0.28, charmex: 0.15, oi: 0.15, dag: 0.10 };
    const levels  = scoreLevels(strikes, weights, futuresPrice);

    const H_GEX_norm  = computeHGEXNorm(levels);
    const gammaRegime = computeGammaRegime(futuresPrice, gammaFlip);
    const topWall     = computeTopWall(levels, futuresPrice);

    // Fetch macro bias from own deployment (best-effort)
    let macroBias = 'UNKNOWN';
    let macroRegime = {};
    const siteUrl = (process.env.URL || '').replace(/\/$/, '');
    if (siteUrl) {
      try {
        const biasData = await fetchJsonPlain(`${siteUrl}/bias_output.json`);
        macroBias   = biasData.confluence   || 'UNKNOWN';
        macroRegime = biasData.macro_regime || {};
      } catch (_) {}
    }

    const result = classifyIntradayBias({
      gammaRegime, gammaFlip, nqPrice: futuresPrice,
      topWall, hGexNorm: H_GEX_norm, macroBias,
    });

    const updatedET = new Date().toLocaleString('en-US', {
      timeZone: 'America/New_York',
      hour: '2-digit', minute: '2-digit', hour12: false,
    }) + ' ET';

    return {
      statusCode: 200,
      headers:    OUT_HEADERS,
      body: JSON.stringify({
        updated:     updatedET,
        pred_date:   todayET(),
        nq_price:    Math.round(futuresPrice * 10) / 10,
        qqq_price:   Math.round((spotEtf || 0) * 100) / 100,
        gamma_flip:  gammaFlip,
        gamma_regime:  gammaRegime,
        H_GEX_norm:    H_GEX_norm,
        levels_regime: levelsRegime,
        iv:          iv          != null ? Math.round(iv          * 10)   / 10   : null,
        rv_iv_ratio: rv_iv_ratio != null ? Math.round(rv_iv_ratio * 1000) / 1000 : null,
        hv21:        hv21        != null ? Math.round(hv21        * 10)   / 10   : null,
        top_wall:    topWall,
        levels,
        // Entropy / PCA require historical price data — not available serverless
        entropy_state: null,
        H_returns:     null,
        H_threshold:   null,
        PC1: null, PC2: null, PC3: null,
        pca_fit_date:  null,
        // MM intensification requires local log files — not available serverless
        mm_intensification: [],
        macro_bias:    macroBias,
        macro_regime:  macroRegime,
        ...result,
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

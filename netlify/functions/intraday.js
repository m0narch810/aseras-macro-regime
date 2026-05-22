const {
  BASE_HEADERS, isAuthorized, fetchJson, httpGetJson,
  AGENT_HEADERS, todayET,
  aggregateDataset, computeGammaFlip, scoreLevels, classifyRegime,
} = require('./lib/options');

const SYMBOL   = 'QQQ';
const BASE_URL = 'https://www.free-flow.site/api';

const OUT_HEADERS = { ...BASE_HEADERS, 'Cache-Control': 'public, max-age=240' };

// ── INTRADAY CONSTANTS ────────────────────────────────────────────────────────
const NEAR_FLIP_BUFFER     = 50.0;
const H_GEX_CONFIDENCE_CUT = 0.6;
const STRONG_WALL          = 60.0;
const EXCEPTIONAL_WALL     = 75.0;
const AIR_POCKET_PROXIMITY = 150.0;
const PROXIMITY_HALFLIFE   = 200.0;

// Return-entropy params
const ENTROPY_WINDOW   = 20;
const ENTROPY_BINS     = 10;
const ENTROPY_LOOKBACK = 252;
const ENTROPY_PCTILE   = 75;
const ENTROPY_MIN_BARS = 60;

// PCA params
const PCA_MIN_SAMPLES = 30;

const MACRO_BULL = new Set(['STRONG BULL', 'LEAN BULL']);
const MACRO_BEAR = new Set(['STRONG BEAR', 'LEAN BEAR']);

// Weekly macro bias — bundled at build time via esbuild require() inlining.
let macroBiasData = null;
try { macroBiasData = require('../../bias_output.json'); } catch (e) { macroBiasData = null; }

// ── YAHOO FINANCE DAILY OHLC ──────────────────────────────────────────────────
// Fetches ~2 years of daily bars. Tries NQ futures first, falls back to QQQ.
async function fetchYahooDaily() {
  const symbols = ['NQ=F', 'QQQ'];
  const hosts   = ['query1.finance.yahoo.com', 'query2.finance.yahoo.com'];
  const ua = { 'User-Agent': AGENT_HEADERS['User-Agent'], 'Accept': 'application/json' };

  for (const sym of symbols) {
    for (const host of hosts) {
      try {
        const url  = `https://${host}/v8/finance/chart/${encodeURIComponent(sym)}?range=2y&interval=1d`;
        const data = await httpGetJson(url, ua, 8000);
        const res  = data && data.chart && data.chart.result && data.chart.result[0];
        if (!res) continue;
        const ts = res.timestamp || [];
        const q  = (res.indicators && res.indicators.quote && res.indicators.quote[0]) || {};
        const bars = [];
        for (let i = 0; i < ts.length; i++) {
          const o = q.open && q.open[i], h = q.high && q.high[i];
          const l = q.low  && q.low[i],  c = q.close && q.close[i];
          if (o == null || h == null || l == null || c == null) continue;
          if (o <= 0 || c <= 0) continue;
          bars.push({ open: o, high: h, low: l, close: c });
        }
        if (bars.length >= ENTROPY_MIN_BARS) return { bars, source: sym };
      } catch (_) { /* try next combination */ }
    }
  }
  return null;
}

// ── RETURN ENTROPY ────────────────────────────────────────────────────────────
// Equal-width histogram entropy — matches numpy.histogram behaviour.
function histEntropy(values, bins) {
  if (!values.length) return 0;
  let mn = Infinity, mx = -Infinity;
  for (const v of values) { if (v < mn) mn = v; if (v > mx) mx = v; }
  if (mx === mn) return 0;
  const width  = (mx - mn) / bins;
  const counts = new Array(bins).fill(0);
  for (const v of values) {
    let idx = Math.floor((v - mn) / width);
    if (idx >= bins) idx = bins - 1;
    counts[idx]++;
  }
  const total = values.length;
  let H = 0;
  for (const c of counts) if (c > 0) { const p = c / total; H -= p * Math.log2(p); }
  return H;
}

// Linear-interpolation percentile — matches numpy.percentile default.
function percentile(sortedAsc, pct) {
  const n = sortedAsc.length;
  if (n === 0) return null;
  if (n === 1) return sortedAsc[0];
  const idx = (pct / 100) * (n - 1);
  const lo  = Math.floor(idx), hi = Math.ceil(idx);
  if (lo === hi) return sortedAsc[lo];
  return sortedAsc[lo] + (sortedAsc[hi] - sortedAsc[lo]) * (idx - lo);
}

// Shannon entropy of the most recent ENTROPY_WINDOW returns vs a backward-looking
// 75th-percentile threshold computed over ENTROPY_LOOKBACK prior windows.
// CRITICAL = disordered tape → options walls carry no directional edge.
function computeReturnEntropy(closes) {
  if (closes.length < ENTROPY_MIN_BARS + ENTROPY_WINDOW)
    return { entropy_state: 'UNKNOWN', H_returns: null, H_threshold: null };

  const logRets = [];
  for (let i = 1; i < closes.length; i++) logRets.push(Math.log(closes[i] / closes[i - 1]));

  const Hnow = histEntropy(logRets.slice(-ENTROPY_WINDOW), ENTROPY_BINS);

  const start    = Math.max(0, logRets.length - (ENTROPY_LOOKBACK + ENTROPY_WINDOW));
  const lookback = logRets.slice(start, logRets.length - ENTROPY_WINDOW);

  let Hthresh;
  if (lookback.length < ENTROPY_WINDOW) {
    Hthresh = Hnow;
  } else {
    const rollingH = [];
    for (let i = ENTROPY_WINDOW; i <= lookback.length; i++)
      rollingH.push(histEntropy(lookback.slice(i - ENTROPY_WINDOW, i), ENTROPY_BINS));
    rollingH.sort((a, b) => a - b);
    Hthresh = percentile(rollingH, ENTROPY_PCTILE);
  }

  return {
    entropy_state: Hnow > Hthresh ? 'CRITICAL' : 'STABLE',
    H_returns:     Math.round(Hnow    * 10000) / 10000,
    H_threshold:   Math.round(Hthresh * 10000) / 10000,
  };
}

// ── PCA PRICE STRUCTURE ───────────────────────────────────────────────────────
// Features: [oc_ret, hl_range, mom_5d, mom_10d, mom_20d, rvol_5d, rvol_10d, rvol_20d]
// Scaler: StandardScaler (population std, ddof=0) — matches sklearn default.
// Covariance: sample (ddof=1). Eigenvectors: cyclic Jacobi (verified correct).
// PC1 oriented so positive = upward momentum (features 2-4 are mom_5/10/20d).
function buildPCAFeatures(bars) {
  const n      = bars.length;
  const closes = bars.map(b => b.close);
  const logRet = [null];
  for (let i = 1; i < n; i++) logRet.push(Math.log(closes[i] / closes[i - 1]));

  const pctChange = (i, k) => (i - k < 0 ? null : closes[i] / closes[i - k] - 1);

  // Sample std (ddof=1) — matches pandas rolling().std() default.
  const rollStd = (i, k) => {
    if (i - k + 1 < 1) return null;
    const slice = [];
    for (let j = i - k + 1; j <= i; j++) {
      if (logRet[j] == null) return null;
      slice.push(logRet[j]);
    }
    const mean = slice.reduce((a, b) => a + b, 0) / slice.length;
    let v = 0;
    for (const x of slice) v += (x - mean) * (x - mean);
    return Math.sqrt(v / (slice.length - 1));
  };

  const rows = [];
  for (let i = 0; i < n; i++) {
    const b = bars[i];
    const f = [
      (b.close - b.open) / b.open,
      (b.high - b.low) / b.close,
      pctChange(i, 5), pctChange(i, 10), pctChange(i, 20),
      rollStd(i, 5),   rollStd(i, 10),   rollStd(i, 20),
    ];
    if (f.some(v => v == null || !isFinite(v))) continue;
    rows.push(f);
  }
  return rows;
}

// Cyclic Jacobi eigendecomposition for a symmetric n×n matrix.
// Returns eigenvalues (diagonal of transformed A) and eigenvectors (columns of V).
// Convergence: sum of squared off-diagonal elements < 1e-22.
function jacobiEigen(matrix) {
  const n = matrix.length;
  const a = matrix.map(r => r.slice());
  const v = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));

  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0;
    for (let i = 0; i < n; i++)
      for (let j = i + 1; j < n; j++) off += a[i][j] * a[i][j];
    if (off < 1e-22) break;

    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(a[p][q]) < 1e-20) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = a[k][p], akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p][k], aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = v[k][p], vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq;
          v[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }

  return {
    eigenvalues:  a.map((row, i) => row[i]),
    eigenvectors: Array.from({ length: n }, (_, m) => v.map(row => row[m])),
  };
}

function computePCA(bars) {
  const rows = buildPCAFeatures(bars);
  if (rows.length < PCA_MIN_SAMPLES)
    return { PC1: null, PC2: null, PC3: null, pca_explained: null, pca_n_samples: rows.length };

  const n = rows.length, d = rows[0].length;

  // StandardScaler: population std (ddof=0) — matches sklearn StandardScaler.
  const means = new Array(d).fill(0), stds = new Array(d).fill(0);
  for (const r of rows) for (let j = 0; j < d; j++) means[j] += r[j];
  for (let j = 0; j < d; j++) means[j] /= n;
  for (const r of rows) for (let j = 0; j < d; j++) { const dv = r[j] - means[j]; stds[j] += dv * dv; }
  for (let j = 0; j < d; j++) { stds[j] = Math.sqrt(stds[j] / n); if (!isFinite(stds[j]) || stds[j] === 0) stds[j] = 1; }
  const std = rows.map(r => r.map((x, j) => (x - means[j]) / stds[j]));

  // Sample covariance (ddof=1) — Xᵀ X / (n-1).
  const C = Array.from({ length: d }, () => new Array(d).fill(0));
  for (const r of std) for (let i = 0; i < d; i++) for (let j = i; j < d; j++) C[i][j] += r[i] * r[j];
  for (let i = 0; i < d; i++) for (let j = i; j < d; j++) { C[i][j] /= (n - 1); C[j][i] = C[i][j]; }

  const { eigenvalues, eigenvectors } = jacobiEigen(C);
  const order    = eigenvalues.map((_, i) => i).sort((x, y) => eigenvalues[y] - eigenvalues[x]);
  const totalVar = eigenvalues.reduce((a, b) => a + Math.max(0, b), 0) || 1;

  const lastRow = std[std.length - 1];
  const PC = [], explained = [];
  for (let k = 0; k < 3; k++) {
    let evec = eigenvectors[order[k]].slice();
    if (k === 0) {
      // Orient PC1 so positive = upward momentum (features 2,3,4 = mom_5/10/20d).
      if (evec[2] + evec[3] + evec[4] < 0) evec = evec.map(x => -x);
    } else {
      let mi = 0, ma = 0;
      for (let j = 0; j < evec.length; j++) if (Math.abs(evec[j]) > ma) { ma = Math.abs(evec[j]); mi = j; }
      if (evec[mi] < 0) evec = evec.map(x => -x);
    }
    let score = 0;
    for (let j = 0; j < evec.length; j++) score += lastRow[j] * evec[j];
    PC.push(Math.round(score * 10000) / 10000);
    explained.push(Math.round(Math.max(0, eigenvalues[order[k]]) / totalVar * 1000) / 10);
  }
  return { PC1: PC[0], PC2: PC[1], PC3: PC[2], pca_explained: explained, pca_n_samples: n };
}

// ── INTRADAY-SPECIFIC FEATURES ────────────────────────────────────────────────

// Shannon entropy of the |GEX| distribution across nearby strikes, normalised
// by log2(N). Values > H_GEX_CONFIDENCE_CUT indicate no dominant wall — penalises confidence.
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

// Proximity-weighted top wall: scores each level by raw score × exp(-dist/halflife).
function computeTopWall(levels, nqPrice) {
  if (!levels.length || nqPrice == null) return null;
  let best = null, bestWScore = -1;
  for (const lv of levels) {
    const wscore = (lv.score || 0) * Math.exp(-Math.abs(lv.dist_nq || 9999) / PROXIMITY_HALFLIFE);
    if (wscore > bestWScore) { bestWScore = wscore; best = { ...lv, proximity_score: Math.round(wscore * 100) / 100 }; }
  }
  return best;
}

// ── INTRADAY CLASSIFIER ───────────────────────────────────────────────────────
function classifyIntradayBias({ gammaRegime, gammaFlip, nqPrice, topWall, hGexNorm, macroBias, entropy, pca }) {
  if (entropy && entropy.entropy_state === 'CRITICAL') {
    return {
      intraday_bias:    'NO_BIAS',
      confidence:       'AVOID',
      air_pocket_watch: false,
      air_pocket_type:  null,
      reason: `CRITICAL return entropy (H=${entropy.H_returns} > 75th-pctile threshold ${entropy.H_threshold}). `
            + `Daily-return distribution is disordered — price is not respecting structure. `
            + `Options walls are unlikely to hold cleanly and carry no directional edge. Stand aside.`,
    };
  }

  const topScore  = topWall ? (topWall.score  || 0) : 0;
  const topType   = topWall ? (topWall.type   || '') : '';
  const topStrike = topWall ? topWall.strike_futures : null;

  const macroBullish = MACRO_BULL.has(macroBias);
  const macroBearish = MACRO_BEAR.has(macroBias);
  const pc1Bull      = pca && pca.PC1 != null && pca.PC1 > 0;

  let air_pocket_watch = false, air_pocket_type = null;
  let bias, conf, reason;

  if (gammaRegime === 'POSITIVE') {
    const flipClose    = nqPrice != null && gammaFlip != null &&
                         Math.abs(nqPrice - gammaFlip) < NEAR_FLIP_BUFFER + 30;
    const wallNearFlip = topStrike != null && gammaFlip != null &&
                         Math.abs(topStrike - gammaFlip) < AIR_POCKET_PROXIMITY;

    if (flipClose && wallNearFlip) {
      air_pocket_watch = true; air_pocket_type = 'FLIP_CROSS';
      bias = 'BEARISH REVERSAL WATCH'; conf = 'MODERATE';
      reason = `FLIP_CROSS setup: price hovering just above the gamma flip with a strong wall near it. `
             + `A break below flips dealers short-gamma and can void the levels fast.`;
    } else if (topScore >= STRONG_WALL) {
      if (macroBullish || pc1Bull) {
        bias = 'BULLISH'; conf = 'HIGH';
        const who = (macroBullish && pc1Bull) ? 'macro bias and PCA price structure both confirm'
                  : macroBullish ? 'macro bias confirms' : 'PCA price structure confirms';
        reason = `Positive gamma regime — dealers dampen volatility, walls tend to hold. `
               + `Strong ${topType} (score ${topScore.toFixed(0)}); ${who} upside.`;
      } else {
        bias = 'NEUTRAL_BULLISH'; conf = 'MODERATE';
        reason = `Positive gamma regime — dealers dampen volatility, walls tend to hold. `
               + `Strong ${topType} (score ${topScore.toFixed(0)}), but neither macro nor PCA confirm direction.`;
      }
    } else {
      bias = 'NEUTRAL'; conf = 'LOW';
      reason = `Positive gamma regime (dealers dampen moves) but no strong wall nearby `
             + `(top score ${topScore.toFixed(0)}). Range-bound chop likely — levels hold but offer little edge.`;
    }

  } else if (gammaRegime === 'NEGATIVE') {
    if (topScore >= EXCEPTIONAL_WALL) { air_pocket_watch = true; air_pocket_type = 'EXCEPTIONAL_PUT_WALL'; }
    bias = 'BEARISH CONTINUATION'; conf = 'MODERATE';
    reason = `Negative gamma regime — price is below the gamma flip by >${NEAR_FLIP_BUFFER} pts, `
           + `so dealers amplify directional moves. Walls are more likely to break than hold.`;
    if (air_pocket_watch) reason += ` Exceptional put wall (score ${topScore.toFixed(0)}) flagged as an air-pocket risk.`;
    if (macroBullish && !air_pocket_watch) { conf = 'LOW'; reason += ` Macro bias is bullish — conflicting signal.`; }

  } else if (gammaRegime === 'NEAR_FLIP') {
    air_pocket_watch = true; air_pocket_type = 'FLIP_CROSS';
    if (topScore >= STRONG_WALL) {
      bias = topType.toUpperCase().includes('CALL') ? 'BEARISH REVERSAL' : 'BULLISH REVERSAL';
      conf = 'MODERATE';
      reason = `NEAR_FLIP: price within ${NEAR_FLIP_BUFFER} pts of the gamma flip. `
             + `Strong ${topType} (score ${topScore.toFixed(0)}) nearby — a flip crossing can accelerate the move.`;
    } else {
      bias = 'NEUTRAL'; conf = 'LOW';
      reason = `NEAR_FLIP: price within ${NEAR_FLIP_BUFFER} pts of the gamma flip with no strong wall to anchor it. `
             + `Dealer hedging is unstable here — high uncertainty.`;
    }

  } else {
    bias = 'NEUTRAL'; conf = 'LOW';
    reason = `Gamma regime unknown (no flip data). Cannot classify.`;
  }

  // Confidence modifiers
  const CONF_ORDER = ['LOW', 'MODERATE', 'HIGH'];
  const down = c => CONF_ORDER[Math.max(0, CONF_ORDER.indexOf(c) - 1)];

  if (hGexNorm > H_GEX_CONFIDENCE_CUT) {
    conf = down(conf);
    reason += ` GEX is dispersed (H_GEX_norm ${hGexNorm.toFixed(2)} > 0.6) — no single dominant wall, confidence penalized.`;
  }
  if (!macroBullish && !macroBearish) {
    conf = down(conf);
    reason += ` Macro bias neutral (${macroBias}) — confidence penalized.`;
  }
  if (entropy && entropy.entropy_state === 'STABLE') {
    reason += ` Return entropy STABLE (H ${entropy.H_returns} < threshold ${entropy.H_threshold}) — `
            + `orderly tape, options levels carry directional edge.`;
  } else if (!entropy || entropy.entropy_state === 'UNKNOWN') {
    reason += ` (Entropy gate unavailable — historical price feed unreachable.)`;
  }

  return { intraday_bias: bias, confidence: conf, air_pocket_watch, air_pocket_type, reason };
}

// ── HANDLER ───────────────────────────────────────────────────────────────────
exports.handler = async (event) => {
  if (event.httpMethod === 'OPTIONS')
    return { statusCode: 200, headers: OUT_HEADERS, body: '' };
  if (!isAuthorized(event))
    return { statusCode: 401, headers: OUT_HEADERS, body: JSON.stringify({ error: true, message: 'unauthorized' }) };

  try {
    const cookie = process.env.FF_SESSION || '';
    const exp    = todayET();

    const [data, volData, yahoo] = await Promise.all([
      fetchJson(`${BASE_URL}/futures-levels?symbol=${SYMBOL}&exp=${exp}`, cookie),
      fetchJson(`${BASE_URL}/vol/realized?symbol=${SYMBOL}`, cookie).catch(() => null),
      fetchYahooDaily().catch(() => null),
    ]);

    if (!data.rows || !data.rows.length) throw new Error('No rows — FF_SESSION may be expired.');

    const { strikes, futuresPrice, spotEtf } = aggregateDataset(data);
    const gammaFlip = computeGammaFlip(strikes, futuresPrice);

    // Vol regime — drives scoring weights (bug fix: was always using NEUTRAL weights).
    let iv = null, rvIvRatio = null, hv21 = null;
    if (volData) {
      iv          = volData.current_iv  ?? null;
      rvIvRatio   = volData.rv_iv_ratio ?? null;
      hv21        = volData.hv21        ?? null;
    }
    const [levelsRegime, weights] = classifyRegime(iv, rvIvRatio);

    const levels      = scoreLevels(strikes, weights, futuresPrice);
    const H_GEX_norm  = computeHGEXNorm(levels);
    const gammaRegime = computeGammaRegime(futuresPrice, gammaFlip);
    const topWall     = computeTopWall(levels, futuresPrice);

    let entropy    = { entropy_state: 'UNKNOWN', H_returns: null, H_threshold: null };
    let pca        = { PC1: null, PC2: null, PC3: null, pca_explained: null, pca_n_samples: 0 };
    let priceSource = null;
    if (yahoo && yahoo.bars && yahoo.bars.length) {
      priceSource = yahoo.source;
      entropy     = computeReturnEntropy(yahoo.bars.map(b => b.close));
      pca         = computePCA(yahoo.bars);
    }

    let macroBias = 'UNKNOWN', macroRegime = {};
    if (macroBiasData) {
      macroBias   = macroBiasData.confluence   || 'UNKNOWN';
      macroRegime = macroBiasData.macro_regime || {};
    }

    const result = classifyIntradayBias({
      gammaRegime, gammaFlip, nqPrice: futuresPrice,
      topWall, hGexNorm: H_GEX_norm, macroBias, entropy, pca,
    });

    const updatedET = new Date().toLocaleString('en-US', {
      timeZone: 'America/New_York',
      hour: '2-digit', minute: '2-digit', hour12: false,
    }) + ' ET';

    return {
      statusCode: 200,
      headers:    OUT_HEADERS,
      body: JSON.stringify({
        updated:       updatedET,
        pred_date:     todayET(),
        nq_price:      Math.round(futuresPrice * 10)  / 10,
        qqq_price:     Math.round((spotEtf || 0) * 100) / 100,
        gamma_flip:    gammaFlip,
        gamma_regime:  gammaRegime,
        H_GEX_norm,
        levels_regime: levelsRegime,
        iv:            iv        != null ? Math.round(iv        * 10)   / 10   : null,
        rv_iv_ratio:   rvIvRatio != null ? Math.round(rvIvRatio * 1000) / 1000 : null,
        hv21:          hv21      != null ? Math.round(hv21      * 10)   / 10   : null,
        top_wall:      topWall,
        entropy_state: entropy.entropy_state,
        H_returns:     entropy.H_returns,
        H_threshold:   entropy.H_threshold,
        PC1:           pca.PC1,
        PC2:           pca.PC2,
        PC3:           pca.PC3,
        pca_explained: pca.pca_explained,
        pca_n_samples: pca.pca_n_samples,
        price_source:  priceSource,
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

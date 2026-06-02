const {
  BASE_HEADERS, isAuthorized, fetchJson, httpGetJson,
  AGENT_HEADERS, todayET,
  aggregateDataset, computeGammaFlip, nearbyStrikes, scoreLevels,
  classifyVolRegime, getWeights,
  computeAggregateGreeks, applyBiasTable,
  WALL_REACTION_DIR, BIAS_TAG_DIR, GAMMA_ASYMMETRY_RATIO,
} = require('./lib/options');

const SYMBOL   = 'QQQ';
const BASE_URL = 'https://www.free-flow.site/api';

const OUT_HEADERS = { ...BASE_HEADERS, 'Cache-Control': 'public, max-age=240' };

// ── CONSTANTS ─────────────────────────────────────────────────────────────────
const NEAR_FLIP_BUFFER     = 50.0;
const H_GEX_CONFIDENCE_CUT = 0.6;
const STRONG_WALL          = 60.0;
const EXCEPTIONAL_WALL     = 75.0;
const AIR_POCKET_PROXIMITY = 150.0;
const PROXIMITY_EFOLD      = 200.0;

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

// Weekly macro bias — fetched live from GitHub at runtime with 10-min cache.
// Falls back to the bundled copy if fetch fails. Keeps RTH bias factors fresh
// without needing a Netlify rebuild after each weekly action commit.
const BIAS_URL =
  'https://raw.githubusercontent.com/m0narch810/vanta/master/bias_output.json';
const BIAS_TTL_MS = 10 * 60 * 1000;
let _biasCache = null;
let _biasCacheAt = 0;
let _biasBundled = null;
try { _biasBundled = require('../../bias_output.json'); } catch (_) {}

async function getMacroBiasData() {
  const now = Date.now();
  if (_biasCache && now - _biasCacheAt < BIAS_TTL_MS) return _biasCache;
  try {
    const res = await fetch(BIAS_URL);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    _biasCache = await res.json();
    _biasCacheAt = now;
    return _biasCache;
  } catch (_) {
    return _biasCache || _biasBundled;
  }
}

// ── METHODOLOGY CONFIG LOADER ─────────────────────────────────────────────────
// Loads from METHODOLOGY_CONFIG env var (Netlify production), then falls back to
// the local file (dev). Committed code never requires the local file directly
// so esbuild does not fail when the gitignored file is absent.
let _methodologyConfig = null;
function getConfig() {
  if (_methodologyConfig) return _methodologyConfig;
  if (process.env.METHODOLOGY_CONFIG) {
    try {
      _methodologyConfig = JSON.parse(
        Buffer.from(process.env.METHODOLOGY_CONFIG, 'base64').toString('utf8')
      );
      return _methodologyConfig;
    } catch (_) {}
  }
  try {
    // String concat defeats esbuild static analysis — file is gitignored so
    // a literal require() path would fail the Netlify build.
    _methodologyConfig = require('./lib/' + 'methodology_config');
    return _methodologyConfig;
  } catch (_) {}
  _methodologyConfig = _emptyConfig();
  return _methodologyConfig;
}

function _emptyConfig() {
  const empty = (cls) => ({ label: '', cls, interp: '' });
  return {
    archetypes: {
      TYPE_A: { name: 'Down Sweep',     short: 'A · Bull',  desc: 'Manipulate price down near the gamma flip, then pump up. Delta and charm positive; put wall absorbs the sweep.', action: '', signal_keys: [] },
      TYPE_B: { name: 'Up Sweep',       short: 'B · Bear',  desc: 'Manipulate price up into call wall resistance, then dump. Delta and charm negative; overhead wall caps the sweep.', action: '', signal_keys: [] },
      TYPE_C: { name: 'Straight Bull',  short: 'C · Bull',  desc: 'Direct pump up from the open — price well above flip with call buying and room to run.', action: '', signal_keys: [] },
      TYPE_D: { name: 'Straight Bear',  short: 'D · Bear',  desc: 'Direct dump down from the open — price well below flip with put buying and room to fall.', action: '', signal_keys: [] },
    },
    rthBias: {
      BULLISH: { label: 'RTH BULLISH', cls: 'bull',  summary: '' },
      BEARISH: { label: 'RTH BEARISH', cls: 'bear',  summary: '' },
      NEUTRAL: { label: 'RTH NEUTRAL', cls: 'mixed', summary: '' },
      UNKNOWN: { label: 'RTH UNKNOWN', cls: 'ghost', summary: '' },
    },
    yieldSignals: {
      RISING_FAST: empty('bear'), RISING: empty('bear'),
      STABLE: empty('mixed'),     FALLING: empty('bull'), UNAVAILABLE: empty('ghost'),
    },
    bojSignals: {
      CARRY_UNWIND: empty('bear'), YEN_STABLE: empty('mixed'),
      YEN_WEAKENING: empty('bull'), UNAVAILABLE: empty('ghost'),
    },
    cotLabels: {
      FUMES_LONG: empty('bear'), EXTREME_SHORT: empty('bull'),
      NEUTRAL: empty('mixed'),   UNAVAILABLE: empty('ghost'),
    },
    liquidityLabels: {
      IMPROVING: empty('bull'), STABLE: empty('mixed'),
      DETERIORATING: empty('bear'), UNAVAILABLE: empty('ghost'),
    },
  };
}

// ── YAHOO FINANCE ─────────────────────────────────────────────────────────────
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
      } catch (_) {}
    }
  }
  return null;
}

// Fetches recent daily closes for a single symbol (1 month of data).
async function fetchYahooSymbol(symbol, minBars) {
  const hosts = ['query1.finance.yahoo.com', 'query2.finance.yahoo.com'];
  const ua    = { 'User-Agent': AGENT_HEADERS['User-Agent'], 'Accept': 'application/json' };
  for (const host of hosts) {
    try {
      const url  = `https://${host}/v8/finance/chart/${encodeURIComponent(symbol)}?range=1mo&interval=1d`;
      const data = await httpGetJson(url, ua, 7000);
      const res  = data && data.chart && data.chart.result && data.chart.result[0];
      if (!res) continue;
      const ts = res.timestamp || [];
      const q  = (res.indicators && res.indicators.quote && res.indicators.quote[0]) || {};
      const bars = [];
      for (let i = 0; i < ts.length; i++) {
        const c = q.close && q.close[i];
        if (c == null || c <= 0) continue;
        bars.push({ close: c });
      }
      if (bars.length >= (minBars || 4)) return bars;
    } catch (_) {}
  }
  return null;
}

// ── RETURN ENTROPY ────────────────────────────────────────────────────────────
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

function percentile(sortedAsc, pct) {
  const n = sortedAsc.length;
  if (n === 0) return null;
  if (n === 1) return sortedAsc[0];
  const idx = (pct / 100) * (n - 1);
  const lo  = Math.floor(idx), hi = Math.ceil(idx);
  if (lo === hi) return sortedAsc[lo];
  return sortedAsc[lo] + (sortedAsc[hi] - sortedAsc[lo]) * (idx - lo);
}

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
function buildPCAFeatures(bars) {
  const n      = bars.length;
  const closes = bars.map(b => b.close);
  const logRet = [null];
  for (let i = 1; i < n; i++) logRet.push(Math.log(closes[i] / closes[i - 1]));

  const pctChange = (i, k) => (i - k < 0 ? null : closes[i] / closes[i - k] - 1);

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

  const means = new Array(d).fill(0), stds = new Array(d).fill(0);
  for (const r of rows) for (let j = 0; j < d; j++) means[j] += r[j];
  for (let j = 0; j < d; j++) means[j] /= n;
  for (const r of rows) for (let j = 0; j < d; j++) { const dv = r[j] - means[j]; stds[j] += dv * dv; }
  for (let j = 0; j < d; j++) { stds[j] = Math.sqrt(stds[j] / n); if (!isFinite(stds[j]) || stds[j] === 0) stds[j] = 1; }
  const std = rows.map(r => r.map((x, j) => (x - means[j]) / stds[j]));

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

  const pc1LoadingVec  = eigenvectors[order[0]];
  const momLoadingSum  = Math.abs(pc1LoadingVec[2]) + Math.abs(pc1LoadingVec[3]) + Math.abs(pc1LoadingVec[4]);
  const pc1Valid       = explained[0] > 0 && momLoadingSum > 0.3;

  return {
    PC1: PC[0], PC2: PC[1], PC3: PC[2],
    pca_explained:         explained,
    pca_n_samples:         n,
    pc1_momentum_loadings: Math.round(momLoadingSum * 10000) / 10000,
    pc1_momentum_valid:    pc1Valid,
  };
}

// ── GEX STRUCTURE ─────────────────────────────────────────────────────────────
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

function computeTopWall(levels, nqPrice) {
  if (!levels.length || nqPrice == null) return null;
  let best = null, bestWScore = -1;
  for (const lv of levels) {
    const wscore = (lv.score || 0) * Math.exp(-Math.abs(lv.dist_nq || 9999) / PROXIMITY_EFOLD);
    if (wscore > bestWScore) { bestWScore = wscore; best = { ...lv, proximity_score: Math.round(wscore * 100) / 100 }; }
  }
  return best;
}

// ── OPTIONS-FLOW INTRADAY CLASSIFIER (existing) ───────────────────────────────
function classifyIntradayBias({ gammaRegime, volRegime, gammaFlip, nqPrice, topWall, hGexNorm, macroBias, entropy, pca,
                                pdfBiasTag, wallReactionTag, aggregateGreeks }) {
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
  const pc1Bull = pca && pca.PC1 != null && pca.pc1_momentum_valid === true && pca.PC1 > 0;

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
             + `(top score ${topScore.toFixed(0)}). Range-bound chop likely.`;
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
             + `Strong ${topType} (score ${topScore.toFixed(0)}) nearby.`;
    } else {
      bias = 'NEUTRAL'; conf = 'LOW';
      reason = `NEAR_FLIP: price within ${NEAR_FLIP_BUFFER} pts of the gamma flip with no strong wall. High uncertainty.`;
    }

  } else {
    bias = 'NEUTRAL'; conf = 'LOW';
    reason = `Gamma regime unknown (no flip data). Cannot classify.`;
  }

  const CONF_BASE = { 'LOW': -1, 'MODERATE': 0, 'HIGH': 1 };
  let evidence = CONF_BASE[conf] ?? 0;

  if (gammaRegime === 'NEGATIVE' && volRegime === 'EXPANSION') {
    reason += ` EXPANSION vol + negative gamma: dealers short and IV expanding — moves amplified.`;
    if (macroBearish) { evidence += 1; reason += ` Macro confirms — all three axes bearish.`; }
  } else if (gammaRegime === 'NEGATIVE' && volRegime === 'CONTRACTION') {
    evidence -= 1;
    reason += ` CONTRACTION vol with negative gamma is unusual — potential mean-reversion, reduce size.`;
  } else if (gammaRegime === 'POSITIVE' && volRegime === 'CONTRACTION') {
    reason += ` CONTRACTION vol + positive gamma: maximum pinning — walls highly reliable.`;
  }

  if (hGexNorm > H_GEX_CONFIDENCE_CUT) {
    evidence -= 1;
    reason += ` GEX dispersed (H_GEX_norm ${hGexNorm.toFixed(2)} > 0.6) — no dominant wall.`;
  }

  if (!macroBullish && !macroBearish) {
    evidence -= 1;
    reason += ` Macro bias neutral (${macroBias}) — confidence penalized.`;
  }

  if (entropy && entropy.entropy_state === 'STABLE') {
    reason += ` Return entropy STABLE (H ${entropy.H_returns} < threshold ${entropy.H_threshold}) — orderly tape.`;
  } else if (!entropy || entropy.entropy_state === 'UNKNOWN') {
    reason += ` (Entropy gate unavailable.)`;
  }

  const currentBull = bias.includes('BULL') && !bias.includes('BEAR');
  const currentBear = bias.includes('BEAR');

  const pdfDir = pdfBiasTag && BIAS_TAG_DIR[pdfBiasTag];
  if (pdfDir && pdfDir.dir !== 'NEUTRAL') {
    const matches   = (pdfDir.dir === 'BULL' && currentBull) || (pdfDir.dir === 'BEAR' && currentBear);
    const conflicts = (pdfDir.dir === 'BULL' && currentBear) || (pdfDir.dir === 'BEAR' && currentBull);
    if (matches)   { evidence += pdfDir.strength; reason += ` Aggregate bias table (${pdfBiasTag.replace(/_/g, ' ').toLowerCase()}) confirms.`; }
    else if (conflicts) { evidence -= pdfDir.strength; reason += ` ⚠ Aggregate bias table (${pdfBiasTag.replace(/_/g, ' ').toLowerCase()}) conflicts.`; }
  }

  const wallDir = wallReactionTag && WALL_REACTION_DIR[wallReactionTag];
  if (wallDir && wallDir.dir !== 'NEUTRAL') {
    const matches   = (wallDir.dir === 'BULL' && currentBull) || (wallDir.dir === 'BEAR' && currentBear);
    const conflicts = (wallDir.dir === 'BULL' && currentBear) || (wallDir.dir === 'BEAR' && currentBull);
    if (matches)   { evidence += wallDir.strength; reason += ` Top-wall reaction (${wallReactionTag.replace(/_/g, ' ').toLowerCase()}) confirms.`; }
    else if (conflicts) {
      evidence -= wallDir.strength;
      reason += ` ⚠ Top-wall reaction (${wallReactionTag.replace(/_/g, ' ').toLowerCase()}) conflicts.`;
      if (wallDir.strength >= 2 && !air_pocket_watch) { air_pocket_watch = true; air_pocket_type = 'WALL_BREAKDOWN'; }
    }
  }

  if (gammaRegime === 'NEGATIVE') {
    const before = evidence;
    evidence = Math.sign(evidence) * Math.abs(evidence) * GAMMA_ASYMMETRY_RATIO;
    if (Math.abs(before) > 0.01)
      reason += ` (Evidence scaled by ${GAMMA_ASYMMETRY_RATIO.toFixed(2)} — Dim/Eraker/Vilkov 2025 asymmetry.)`;
  }

  conf = evidence >= 1 ? 'HIGH' : evidence <= -1 ? 'LOW' : 'MODERATE';
  return { intraday_bias: bias, confidence: conf, air_pocket_watch, air_pocket_type, reason };
}

// ── RTH BIAS INPUTS ───────────────────────────────────────────────────────────
function classify2YSignal(bars) {
  if (!bars || bars.length < 6) return 'UNAVAILABLE';
  const closes = bars.map(b => b.close);
  const latest = closes[closes.length - 1];
  const ref5d  = closes[closes.length - 6];
  if (!ref5d || ref5d === 0) return 'UNAVAILABLE';
  const roc5 = (latest - ref5d) / ref5d;
  if (roc5 < -0.003)  return 'RISING_FAST';
  if (roc5 < -0.0008) return 'RISING';
  if (roc5 >  0.0008) return 'FALLING';
  return 'STABLE';
}

function classifyBOJSignal(bars) {
  if (!bars || bars.length < 4) return 'UNAVAILABLE';
  const closes = bars.map(b => b.close);
  const latest = closes[closes.length - 1];
  const ref3d  = closes[closes.length - 4];
  if (!ref3d || ref3d === 0) return 'UNAVAILABLE';
  const roc3 = (latest - ref3d) / ref3d;
  if (roc3 < -0.015) return 'CARRY_UNWIND';
  if (roc3 >  0.005) return 'YEN_WEAKENING';
  return 'YEN_STABLE';
}

function getLiquidityTrend(data) {
  const score = data &&
    data.macro_regime &&
    data.macro_regime.factor_scores &&
    data.macro_regime.factor_scores.net_liq_wow;
  if (score == null) return 'UNAVAILABLE';
  if (score > 0) return 'IMPROVING';
  if (score < 0) return 'DETERIORATING';
  return 'STABLE';
}

function getCotLabel(data) {
  const p = data && data.cot && data.cot.nq_lev_pctile;
  if (p == null) return 'UNAVAILABLE';
  if (p > 0.80) return 'FUMES_LONG';
  if (p < 0.20) return 'EXTREME_SHORT';
  return 'NEUTRAL';
}

function classifyRTHBias({ yieldSignal, liquidityTrend, cotLabel, bojSignal, macroConfluence }) {
  let bull = 0, bear = 0;

  if (yieldSignal === 'FALLING')          bull += 1;
  else if (yieldSignal === 'STABLE')      bull += 0.5;
  else if (yieldSignal === 'RISING')      bear += 1;
  else if (yieldSignal === 'RISING_FAST') bear += 2;

  if (liquidityTrend === 'IMPROVING')      bull += 1;
  else if (liquidityTrend === 'DETERIORATING') bear += 1;

  if (cotLabel === 'EXTREME_SHORT')        bull += 1;
  else if (cotLabel === 'FUMES_LONG')      bear += 1;

  if (bojSignal === 'CARRY_UNWIND')        bear += 2;
  else if (bojSignal === 'YEN_WEAKENING')  bull += 0.5;

  if (['STRONG BULL', 'LEAN BULL'].includes(macroConfluence))       bull += 1;
  else if (['STRONG BEAR', 'LEAN BEAR'].includes(macroConfluence))  bear += 1;

  let verdict;
  if (yieldSignal === 'UNAVAILABLE' && bojSignal === 'UNAVAILABLE') {
    verdict = 'UNKNOWN';
  } else if (bear >= 2.5) {
    verdict = 'BEARISH';
  } else if (bull >= 2.5) {
    verdict = 'BULLISH';
  } else {
    verdict = 'NEUTRAL';
  }

  return {
    verdict,
    bull_count: Math.round(bull * 2) / 2,
    bear_count: Math.round(bear * 2) / 2,
  };
}

// ── OPEN TYPE SCORING ─────────────────────────────────────────────────────────
function classifyOpenArchetype({ gammaFlip, futuresPrice, aggregateGreeks, levels, ivBand }) {
  if (!gammaFlip || !futuresPrice || !aggregateGreeks) {
    return { type: null, confidence: 0, dir: 'neutral', runner_up: null, runner_up_confidence: 0, signals: [] };
  }

  const flipDiff = futuresPrice - gammaFlip;
  const { dex_sign, vex_sign, charm_sign } = aggregateGreeks;
  const threshold = ivBand || 50;

  const callWalls = (levels || []).filter(l => l.net_gex > 0 && l.dist_nq > 0)
                                   .sort((a, b) => a.dist_nq - b.dist_nq);
  const putWalls  = (levels || []).filter(l => l.net_gex < 0 && l.dist_nq < 0)
                                   .sort((a, b) => b.dist_nq - a.dist_nq);
  const callWallDist = callWalls.length ? Math.abs(callWalls[0].dist_nq) : 999;
  const putWallDist  = putWalls.length  ? Math.abs(putWalls[0].dist_nq)  : 999;

  const scores = { TYPE_A: 0, TYPE_B: 0, TYPE_C: 0, TYPE_D: 0 };
  const sigs   = { TYPE_A: [], TYPE_B: [], TYPE_C: [], TYPE_D: [] };

  // TYPE_A
  if (flipDiff >= -threshold && flipDiff <= threshold * 0.5) { scores.TYPE_A++; sigs.TYPE_A.push('near_flip'); }
  if (dex_sign > 0)              { scores.TYPE_A++; sigs.TYPE_A.push('dex_positive'); }
  if (vex_sign > 0)              { scores.TYPE_A++; sigs.TYPE_A.push('call_bias'); }
  if (putWallDist < threshold)   { scores.TYPE_A++; sigs.TYPE_A.push('put_wall_near'); }
  if (charm_sign > 0)            { scores.TYPE_A++; sigs.TYPE_A.push('charm_positive'); }

  // TYPE_B
  if (flipDiff >= 0 && flipDiff <= threshold * 1.5) { scores.TYPE_B++; sigs.TYPE_B.push('above_flip'); }
  if (dex_sign < 0)              { scores.TYPE_B++; sigs.TYPE_B.push('dex_negative'); }
  if (vex_sign < 0)              { scores.TYPE_B++; sigs.TYPE_B.push('put_bias'); }
  if (callWallDist < threshold)  { scores.TYPE_B++; sigs.TYPE_B.push('call_wall_near'); }
  if (charm_sign < 0)            { scores.TYPE_B++; sigs.TYPE_B.push('charm_negative'); }

  // TYPE_C
  if (flipDiff > threshold)      { scores.TYPE_C++; sigs.TYPE_C.push('well_above_flip'); }
  if (flipDiff > threshold * 2)  { scores.TYPE_C++; sigs.TYPE_C.push('extended_above'); }
  if (dex_sign > 0)              { scores.TYPE_C++; sigs.TYPE_C.push('dex_positive'); }
  if (vex_sign > 0)              { scores.TYPE_C++; sigs.TYPE_C.push('call_buying'); }
  if (callWallDist > 150)        { scores.TYPE_C++; sigs.TYPE_C.push('room_to_run'); }

  // TYPE_D
  if (flipDiff < -threshold)     { scores.TYPE_D++; sigs.TYPE_D.push('well_below_flip'); }
  if (flipDiff < -threshold * 2) { scores.TYPE_D++; sigs.TYPE_D.push('extended_below'); }
  if (dex_sign < 0)              { scores.TYPE_D++; sigs.TYPE_D.push('dex_negative'); }
  if (vex_sign < 0)              { scores.TYPE_D++; sigs.TYPE_D.push('put_buying'); }
  if (putWallDist > 150)         { scores.TYPE_D++; sigs.TYPE_D.push('room_to_fall'); }

  const entries = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  const [winner, topScore] = entries[0];
  const [second, secScore] = entries[1] || ['', 0];

  // A and C are bull-resolving; B and D are bear-resolving
  const TYPE_DIRS = { TYPE_A: 'bull', TYPE_B: 'bear', TYPE_C: 'bull', TYPE_D: 'bear' };

  return {
    type:                 topScore > 0 ? winner : null,
    confidence:           topScore,
    dir:                  topScore > 0 ? (TYPE_DIRS[winner] || 'neutral') : 'neutral',
    runner_up:            secScore > 0 ? second : null,
    runner_up_confidence: secScore,
    all_scores:           scores,
    signals:              topScore > 0 ? (sigs[winner] || []) : [],
  };
}

// ── EXPIRY FALLBACK (0DTE → 1DTE → 2DTE) ────────────────────────────────────
function addDays(dateStr, n) {
  const [y, m, d] = dateStr.split('-').map(Number);
  const date = new Date(Date.UTC(y, m - 1, d));
  date.setUTCDate(date.getUTCDate() + n);
  return date.toISOString().slice(0, 10);
}

async function fetchExpiry(cookie, symbol) {
  const today = todayET();
  for (let dte = 0; dte <= 2; dte++) {
    const exp = dte === 0 ? today : addDays(today, dte);
    try {
      const data = await fetchJson(`${BASE_URL}/futures-levels?symbol=${symbol}&exp=${exp}`, cookie);
      if (data.rows && data.rows.length) return { data, exp, dte };
    } catch (_) {}
  }
  throw new Error('No options data for 0DTE, 1DTE, or 2DTE — FF_SESSION may be expired.');
}

// ── HANDLER ───────────────────────────────────────────────────────────────────
exports.handler = async (event) => {
  if (event.httpMethod === 'OPTIONS')
    return { statusCode: 200, headers: OUT_HEADERS, body: '' };
  if (!isAuthorized(event))
    return { statusCode: 401, headers: OUT_HEADERS, body: JSON.stringify({ error: true, message: 'unauthorized' }) };

  try {
    const cookie = process.env.FF_SESSION || '';

    const [{ data, exp: activeExp, dte: activeDTE }, volData, yahoo, shyBars, usdjpyBars, macroBiasData] = await Promise.all([
      fetchExpiry(cookie, SYMBOL),
      fetchJson(`${BASE_URL}/vol/realized?symbol=${SYMBOL}`, cookie).catch(() => null),
      fetchYahooDaily().catch(() => null),
      fetchYahooSymbol('SHY', 6).catch(() => null),
      fetchYahooSymbol('USDJPY=X', 4).catch(() => null),
      getMacroBiasData(),
    ]);

    const { strikes, futuresPrice, spotEtf } = aggregateDataset(data);

    const gammaFlip = computeGammaFlip(strikes, futuresPrice);
    const flipDiff  = futuresPrice != null && gammaFlip != null ? futuresPrice - gammaFlip : null;

    let iv = null, rvIvRatio = null, hv21 = null;
    if (volData) {
      iv        = volData.current_iv  ?? null;
      rvIvRatio = volData.rv_iv_ratio ?? null;
      hv21      = volData.hv21        ?? null;
    }

    const _ivBand = (iv != null && iv > 0 && futuresPrice != null)
      ? Math.max(30, 0.5 * futuresPrice * (iv / 100) / Math.sqrt(252))
      : 50;
    const gammaRegime  = flipDiff == null ? 'UNKNOWN'
      : flipDiff >  _ivBand ? 'POSITIVE'
      : flipDiff < -_ivBand ? 'NEGATIVE'
      : 'NEAR_FLIP';
    const levelsRegime = classifyVolRegime(iv, rvIvRatio);
    const weights      = getWeights(levelsRegime, gammaRegime);

    const levels     = scoreLevels(strikes, weights, futuresPrice, levelsRegime, gammaFlip, iv);
    const H_GEX_norm = computeHGEXNorm(levels);
    const topWall    = computeTopWall(levels, futuresPrice);

    const _nearbyAll      = nearbyStrikes(strikes, futuresPrice);
    const aggregateGreeks = computeAggregateGreeks(_nearbyAll);
    const wallReactionTag = topWall ? topWall.wall_reaction : null;
    const priceVsFlip     = flipDiff == null ? 0 : (flipDiff > 0 ? 1 : -1);
    const pdfBiasTag      = applyBiasTable(aggregateGreeks, levelsRegime, priceVsFlip);

    let entropy     = { entropy_state: 'UNKNOWN', H_returns: null, H_threshold: null };
    let pca         = { PC1: null, PC2: null, PC3: null, pca_explained: null, pca_n_samples: 0 };
    let priceSource = null;
    if (yahoo && yahoo.bars && yahoo.bars.length) {
      priceSource = yahoo.source;
      entropy     = computeReturnEntropy(yahoo.bars.map(b => b.close));
      pca         = computePCA(yahoo.bars);
    }

    let macroBias = 'UNKNOWN', macroRegime = {};
    if (macroBiasData) {
      macroBias   = macroBiasData.confluence    || 'UNKNOWN';
      macroRegime = macroBiasData.macro_regime  || {};
    }

    // Options-flow based intraday classifier (existing)
    const optionsResult = classifyIntradayBias({
      gammaRegime, volRegime: levelsRegime, gammaFlip, nqPrice: futuresPrice,
      topWall, hGexNorm: H_GEX_norm, macroBias, entropy, pca,
      pdfBiasTag, wallReactionTag, aggregateGreeks,
    });

    // RTH macro bias (new)
    const yieldSignal    = classify2YSignal(shyBars);
    const bojSignal      = classifyBOJSignal(usdjpyBars);
    const liquidityTrend = getLiquidityTrend(macroBiasData);
    const cotLabel       = getCotLabel(macroBiasData);
    const rthResult      = classifyRTHBias({ yieldSignal, liquidityTrend, cotLabel, bojSignal, macroConfluence: macroBias });

    // Open archetype scoring (new)
    const archResult = classifyOpenArchetype({
      gammaFlip, futuresPrice, aggregateGreeks, levels, ivBand: _ivBand,
    });

    // Enrich archetype and RTH fields with display text from methodology config
    const cfg = getConfig();
    const archCfg  = archResult.type ? (cfg.archetypes[archResult.type] || {}) : {};
    const rthCfg   = cfg.rthBias[rthResult.verdict] || {};

    const updatedET = new Date().toLocaleString('en-US', {
      timeZone: 'America/New_York',
      hour: '2-digit', minute: '2-digit', hour12: false,
    }) + ' ET';

    // Build RTH factor detail objects
    const rthFactors = {
      yield: {
        signal: yieldSignal,
        label:  (cfg.yieldSignals[yieldSignal]    || {}).label  || yieldSignal,
        interp: (cfg.yieldSignals[yieldSignal]    || {}).interp || '',
        cls:    (cfg.yieldSignals[yieldSignal]    || {}).cls    || 'ghost',
      },
      liquidity: {
        signal: liquidityTrend,
        label:  (cfg.liquidityLabels[liquidityTrend] || {}).label  || liquidityTrend,
        interp: (cfg.liquidityLabels[liquidityTrend] || {}).interp || '',
        cls:    (cfg.liquidityLabels[liquidityTrend] || {}).cls    || 'ghost',
      },
      cot: {
        signal: cotLabel,
        label:  (cfg.cotLabels[cotLabel] || {}).label  || cotLabel,
        interp: (cfg.cotLabels[cotLabel] || {}).interp || '',
        cls:    (cfg.cotLabels[cotLabel] || {}).cls    || 'ghost',
        pctile: macroBiasData && macroBiasData.cot
          ? Math.round((macroBiasData.cot.nq_lev_pctile || 0) * 100)
          : null,
      },
      boj: {
        signal: bojSignal,
        label:  (cfg.bojSignals[bojSignal] || {}).label  || bojSignal,
        interp: (cfg.bojSignals[bojSignal] || {}).interp || '',
        cls:    (cfg.bojSignals[bojSignal] || {}).cls    || 'ghost',
        usdjpy: usdjpyBars && usdjpyBars.length
          ? Math.round(usdjpyBars[usdjpyBars.length - 1].close * 100) / 100
          : null,
      },
    };

    return {
      statusCode: 200,
      headers:    OUT_HEADERS,
      body: JSON.stringify({
        updated:       updatedET,
        pred_date:     activeExp,
        expiry_dte:    activeDTE,
        nq_price:      Math.round(futuresPrice * 10)    / 10,
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
        pc1_momentum_loadings: pca.pc1_momentum_loadings,
        pc1_momentum_valid:    pca.pc1_momentum_valid,
        price_source:  priceSource,
        mm_intensification: [],
        macro_bias:    macroBias,
        macro_regime:  macroRegime,
        pdf_primary_bias:   pdfBiasTag,
        wall_reaction:      wallReactionTag,
        aggregate_greeks:   aggregateGreeks,

        // RTH macro bias
        rth_bias:          rthResult.verdict,
        rth_bias_label:    rthCfg.label   || rthResult.verdict,
        rth_bias_summary:  rthCfg.summary || '',
        rth_bias_cls:      rthCfg.cls     || 'ghost',
        rth_bull_count:    rthResult.bull_count,
        rth_bear_count:    rthResult.bear_count,
        rth_factors:       rthFactors,

        // Open archetype
        open_archetype:                    archResult.type,
        open_archetype_confidence:         archResult.confidence,
        open_archetype_dir:                archResult.dir,
        open_archetype_runner_up:          archResult.runner_up,
        open_archetype_runner_up_confidence: archResult.runner_up_confidence,
        open_archetype_all_scores:         archResult.all_scores,
        open_archetype_signals:            archResult.signals,
        open_archetype_name:               archCfg.name   || archResult.type || null,
        open_archetype_short:              archCfg.short  || archResult.type || null,
        open_archetype_desc:               archCfg.desc   || '',
        open_archetype_action:             archCfg.action || '',

        // Options-flow classifier (secondary context)
        ...optionsResult,
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

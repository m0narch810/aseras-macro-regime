const https = require('https');

const SYMBOL     = 'QQQ';
const BASE_URL   = 'https://www.free-flow.site/api';
const FILTER_PCT = 5.0;
const MIN_SCORE  = 20.0;

// Must match scripts/09_intraday_bias.py
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

// ── YAHOO FINANCE DAILY OHLC ─────────────────────────────────────────────────
// Fetches ~2 years of daily bars. Tries NQ futures, falls back to QQQ.
async function fetchYahooDaily() {
  const symbols = ['NQ=F', 'QQQ'];
  const hosts   = ['query1.finance.yahoo.com', 'query2.finance.yahoo.com'];
  const ua = { 'User-Agent': AGENT_HEADERS['User-Agent'], 'Accept': 'application/json' };

  for (const sym of symbols) {
    for (const host of hosts) {
      try {
        const url = `https://${host}/v8/finance/chart/${encodeURIComponent(sym)}?range=2y&interval=1d`;
        const data = await httpGetJson(url, ua, 8000);
        const result = data && data.chart && data.chart.result && data.chart.result[0];
        if (!result) continue;
        const ts = result.timestamp || [];
        const q  = (result.indicators && result.indicators.quote && result.indicators.quote[0]) || {};
        const bars = [];
        for (let i = 0; i < ts.length; i++) {
          const o = q.open && q.open[i], h = q.high && q.high[i];
          const l = q.low && q.low[i],   c = q.close && q.close[i];
          if (o == null || h == null || l == null || c == null) continue;
          if (o <= 0 || c <= 0) continue;
          bars.push({ open: o, high: h, low: l, close: c });
        }
        if (bars.length >= ENTROPY_MIN_BARS) return { bars, source: sym };
      } catch (_) { /* try next */ }
    }
  }
  return null;
}

// ── DATE HELPERS ─────────────────────────────────────────────────────────────
function todayET() {
  return new Date().toLocaleDateString('en-CA', { timeZone: 'America/New_York' });
}

// ── RETURN ENTROPY ───────────────────────────────────────────────────────────
function histEntropy(values, bins) {
  if (!values.length) return 0;
  let mn = Infinity, mx = -Infinity;
  for (const v of values) { if (v < mn) mn = v; if (v > mx) mx = v; }
  if (mx === mn) return 0;                       // degenerate → single bin → H=0
  const width  = (mx - mn) / bins;
  const counts = new Array(bins).fill(0);
  for (const v of values) {
    let idx = Math.floor((v - mn) / width);
    if (idx >= bins) idx = bins - 1;
    if (idx < 0)     idx = 0;
    counts[idx]++;
  }
  const total = values.length;
  let H = 0;
  for (const c of counts) {
    if (c > 0) { const p = c / total; H -= p * Math.log2(p); }
  }
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

// Shannon entropy of recent return distribution vs a backward-looking
// 75th-pctile threshold. CRITICAL = disordered tape = options levels lack edge.
function computeReturnEntropy(closes) {
  if (closes.length < ENTROPY_MIN_BARS + ENTROPY_WINDOW) {
    return { entropy_state: 'UNKNOWN', H_returns: null, H_threshold: null };
  }
  const logRets = [];
  for (let i = 1; i < closes.length; i++) logRets.push(Math.log(closes[i] / closes[i - 1]));

  const Hnow = histEntropy(logRets.slice(-ENTROPY_WINDOW), ENTROPY_BINS);

  const start = Math.max(0, logRets.length - (ENTROPY_LOOKBACK + ENTROPY_WINDOW));
  const end   = logRets.length - ENTROPY_WINDOW;
  const lookback = logRets.slice(start, end);

  let Hthresh;
  if (lookback.length < ENTROPY_WINDOW) {
    Hthresh = Hnow;
  } else {
    const rollingH = [];
    for (let i = ENTROPY_WINDOW; i <= lookback.length; i++) {
      rollingH.push(histEntropy(lookback.slice(i - ENTROPY_WINDOW, i), ENTROPY_BINS));
    }
    rollingH.sort((a, b) => a - b);
    Hthresh = percentile(rollingH, ENTROPY_PCTILE);
  }

  return {
    entropy_state: Hnow > Hthresh ? 'CRITICAL' : 'STABLE',
    H_returns:     Math.round(Hnow    * 10000) / 10000,
    H_threshold:   Math.round(Hthresh * 10000) / 10000,
  };
}

// ── PCA PRICE STRUCTURE ──────────────────────────────────────────────────────
// 8 features: [oc_ret, hl_range, mom_5/10/20d, rvol_5/10/20d]
function buildPCAFeatures(bars) {
  const n = bars.length;
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
    return Math.sqrt(v / (slice.length - 1));   // sample std (pandas ddof=1)
  };

  const rows = [];
  for (let i = 0; i < n; i++) {
    const b = bars[i];
    const ocRet   = (b.close - b.open) / b.open;
    const hlRange = (b.high - b.low) / b.close;
    const f = [ocRet, hlRange, pctChange(i, 5), pctChange(i, 10), pctChange(i, 20),
               rollStd(i, 5), rollStd(i, 10), rollStd(i, 20)];
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
        for (let k = 0; k < n; k++) {            // column rotation  a = a·J
          const akp = a[k][p], akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {            // row rotation     a = Jᵀ·a
          const apk = a[p][k], aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {            // accumulate       v = v·J
          const vkp = v[k][p], vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq;
          v[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }

  const eigenvalues  = a.map((row, i) => row[i]);
  const eigenvectors = [];
  for (let m = 0; m < n; m++) eigenvectors.push(v.map(row => row[m]));
  return { eigenvalues, eigenvectors };
}

function computePCA(bars) {
  const rows = buildPCAFeatures(bars);
  if (rows.length < PCA_MIN_SAMPLES) {
    return { PC1: null, PC2: null, PC3: null, pca_explained: null, pca_n_samples: rows.length };
  }
  const n = rows.length, d = rows[0].length;

  // StandardScaler (population std, ddof=0)
  const means = new Array(d).fill(0), stds = new Array(d).fill(0);
  for (const r of rows) for (let j = 0; j < d; j++) means[j] += r[j];
  for (let j = 0; j < d; j++) means[j] /= n;
  for (const r of rows) for (let j = 0; j < d; j++) {
    const dv = r[j] - means[j]; stds[j] += dv * dv;
  }
  for (let j = 0; j < d; j++) {
    stds[j] = Math.sqrt(stds[j] / n);
    if (!isFinite(stds[j]) || stds[j] === 0) stds[j] = 1;
  }
  const std = rows.map(r => r.map((x, j) => (x - means[j]) / stds[j]));

  // Covariance = Xᵀ X / (n-1)
  const C = Array.from({ length: d }, () => new Array(d).fill(0));
  for (const r of std)
    for (let i = 0; i < d; i++)
      for (let j = i; j < d; j++) C[i][j] += r[i] * r[j];
  for (let i = 0; i < d; i++)
    for (let j = i; j < d; j++) { C[i][j] /= (n - 1); C[j][i] = C[i][j]; }

  const { eigenvalues, eigenvectors } = jacobiEigen(C);
  const order    = eigenvalues.map((_, i) => i).sort((x, y) => eigenvalues[y] - eigenvalues[x]);
  const totalVar = eigenvalues.reduce((a, b) => a + Math.max(0, b), 0) || 1;

  const lastRow = std[std.length - 1];
  const PC = [], explained = [];
  for (let k = 0; k < 3; k++) {
    let evec = eigenvectors[order[k]].slice();
    if (k === 0) {
      // Orient PC1 so positive == upward momentum (features 2,3,4 = mom_5/10/20d)
      if (evec[2] + evec[3] + evec[4] < 0) evec = evec.map(x => -x);
    } else {
      let mi = 0, ma = 0;
      for (let j = 0; j < evec.length; j++)
        if (Math.abs(evec[j]) > ma) { ma = Math.abs(evec[j]); mi = j; }
      if (evec[mi] < 0) evec = evec.map(x => -x);
    }
    let score = 0;
    for (let j = 0; j < evec.length; j++) score += lastRow[j] * evec[j];
    PC.push(Math.round(score * 10000) / 10000);
    explained.push(Math.round(Math.max(0, eigenvalues[order[k]]) / totalVar * 1000) / 10);
  }
  return { PC1: PC[0], PC2: PC[1], PC3: PC[2], pca_explained: explained, pca_n_samples: n };
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

function normalizeAbs(values) {
  const abs = values.map(Math.abs);
  const mn = Math.min(...abs), mx = Math.max(...abs);
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
function classifyIntradayBias({ gammaRegime, gammaFlip, nqPrice, topWall, hGexNorm, macroBias, entropy, pca }) {
  // HARD GATE: disordered tape → no edge on options levels.
  if (entropy && entropy.entropy_state === 'CRITICAL') {
    return {
      intraday_bias:   'NO_BIAS',
      confidence:      'AVOID',
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

  let air_pocket_watch = false;
  let air_pocket_type  = null;
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
                  : macroBullish ? 'macro bias confirms'
                  : 'PCA price structure confirms';
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
  const down = (c) => CONF_ORDER[Math.max(0, CONF_ORDER.indexOf(c) - 1)];

  if (hGexNorm > H_GEX_CONFIDENCE_CUT) {
    conf = down(conf);
    reason += ` GEX is dispersed (H_GEX_norm ${hGexNorm.toFixed(2)} > 0.6) — no single dominant wall, confidence penalized.`;
  }
  if (!macroBullish && !macroBearish) {
    conf = down(conf);
    reason += ` Macro bias neutral (${macroBias}) — confidence penalized.`;
  }

  // Entropy context suffix
  if (entropy && entropy.entropy_state === 'STABLE') {
    reason += ` Return entropy STABLE (H ${entropy.H_returns} < threshold ${entropy.H_threshold}) — `
            + `orderly tape, options levels carry directional edge.`;
  } else if (!entropy || entropy.entropy_state === 'UNKNOWN') {
    reason += ` (Entropy gate unavailable — historical price feed unreachable.)`;
  }

  return { intraday_bias: bias, confidence: conf, air_pocket_watch, air_pocket_type, reason };
}

// ── HANDLER ──────────────────────────────────────────────────────────────────
exports.handler = async (event) => {
  if (event.httpMethod === 'OPTIONS') {
    return { statusCode: 200, headers: OUT_HEADERS, body: '' };
  }

  try {
    const cookie  = process.env.FF_SESSION || '';
    const exp     = todayET();
    const siteUrl = (process.env.URL || '').replace(/\/$/, '');

    // All upstream calls run in parallel. Only the levels call is required.
    const [data, volData, yahoo, biasData] = await Promise.all([
      fetchJson(`${BASE_URL}/futures-levels?symbol=${SYMBOL}&exp=${exp}`, cookie),
      fetchJson(`${BASE_URL}/vol/realized?symbol=${SYMBOL}`, cookie).catch(() => null),
      fetchYahooDaily().catch(() => null),
      siteUrl ? httpGetJson(`${siteUrl}/bias_output.json`, { Accept: 'application/json' }, 7000).catch(() => null)
              : Promise.resolve(null),
    ]);

    if (!data.rows || !data.rows.length) throw new Error('No rows — FF_SESSION may be expired.');

    const { strikes, futuresPrice, spotEtf } = aggregateDataset(data);
    const gammaFlip = computeGammaFlip(strikes, futuresPrice);

    // Vol regime
    let levelsRegime = 'UNKNOWN';
    let iv = null, rv_iv_ratio = null, hv21 = null;
    if (volData) {
      iv          = volData.current_iv  ?? null;
      rv_iv_ratio = volData.rv_iv_ratio ?? null;
      hv21        = volData.hv21        ?? null;
      if (iv != null && (iv >= 30 || (rv_iv_ratio != null && rv_iv_ratio < 0.5))) levelsRegime = 'EXPANSION';
      else if (iv != null && iv >= 20)                                            levelsRegime = 'NEUTRAL';
      else if (iv != null)                                                        levelsRegime = 'CONTRACTION';
    }

    const weights = { gex: 0.32, vex: 0.28, charmex: 0.15, oi: 0.15, dag: 0.10 };
    const levels  = scoreLevels(strikes, weights, futuresPrice);

    const H_GEX_norm  = computeHGEXNorm(levels);
    const gammaRegime = computeGammaRegime(futuresPrice, gammaFlip);
    const topWall     = computeTopWall(levels, futuresPrice);

    // Return entropy + PCA from Yahoo daily history
    let entropy = { entropy_state: 'UNKNOWN', H_returns: null, H_threshold: null };
    let pca     = { PC1: null, PC2: null, PC3: null, pca_explained: null, pca_n_samples: 0 };
    let priceSource = null;
    if (yahoo && yahoo.bars && yahoo.bars.length) {
      priceSource = yahoo.source;
      entropy = computeReturnEntropy(yahoo.bars.map(b => b.close));
      pca     = computePCA(yahoo.bars);
    }

    // Macro bias
    let macroBias = 'UNKNOWN', macroRegime = {};
    if (biasData) {
      macroBias   = biasData.confluence   || 'UNKNOWN';
      macroRegime = biasData.macro_regime || {};
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
        nq_price:      Math.round(futuresPrice * 10) / 10,
        qqq_price:     Math.round((spotEtf || 0) * 100) / 100,
        gamma_flip:    gammaFlip,
        gamma_regime:  gammaRegime,
        H_GEX_norm:    H_GEX_norm,
        levels_regime: levelsRegime,
        iv:            iv          != null ? Math.round(iv          * 10)   / 10   : null,
        rv_iv_ratio:   rv_iv_ratio != null ? Math.round(rv_iv_ratio * 1000) / 1000 : null,
        hv21:          hv21        != null ? Math.round(hv21        * 10)   / 10   : null,
        top_wall:      topWall,
        // Return entropy (computed server-side from Yahoo daily history)
        entropy_state: entropy.entropy_state,
        H_returns:     entropy.H_returns,
        H_threshold:   entropy.H_threshold,
        // PCA price structure (computed server-side)
        PC1:           pca.PC1,
        PC2:           pca.PC2,
        PC3:           pca.PC3,
        pca_explained: pca.pca_explained,
        pca_n_samples: pca.pca_n_samples,
        price_source:  priceSource,
        // MM intensification needs stored snapshots — not available serverless
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

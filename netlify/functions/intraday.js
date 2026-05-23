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

// ── INTRADAY CONSTANTS ────────────────────────────────────────────────────────
const NEAR_FLIP_BUFFER     = 50.0;
const H_GEX_CONFIDENCE_CUT = 0.6;
const STRONG_WALL          = 60.0;
const EXCEPTIONAL_WALL     = 75.0;
const AIR_POCKET_PROXIMITY = 150.0;
// e-folding distance: score × exp(-dist/PROXIMITY_EFOLD).
// At dist=200 pts, weight decays to 1/e ≈ 37% of peak.
// True halflife = 200 × ln(2) ≈ 139 pts.
// Previously misnamed PROXIMITY_HALFLIFE.
const PROXIMITY_EFOLD = 200.0;

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

  // Verify PC1 is actually a momentum axis before trusting its sign.
  // Features 2,3,4 (indices) are mom_5d, mom_10d, mom_20d in the loading vector.
  // If their combined absolute loading is weak (< 0.3 of unit vector magnitude),
  // PC1 may be dominated by the vol cluster — mark pc1_momentum_valid: false.
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

// Proximity-weighted top wall: scores each level by raw score × exp(-dist/halflife).
function computeTopWall(levels, nqPrice) {
  if (!levels.length || nqPrice == null) return null;
  let best = null, bestWScore = -1;
  for (const lv of levels) {
    const wscore = (lv.score || 0) * Math.exp(-Math.abs(lv.dist_nq || 9999) / PROXIMITY_EFOLD);
    if (wscore > bestWScore) { bestWScore = wscore; best = { ...lv, proximity_score: Math.round(wscore * 100) / 100 }; }
  }
  return best;
}

// ── INTRADAY CLASSIFIER ───────────────────────────────────────────────────────
// Inputs (new):
//   pdfBiasTag        — bias.pdf primary bias tag (e.g. 'BULLISH_CHOP')
//   wallReactionTag   — walls.pdf reaction tag for the top wall (e.g. 'CALL_WALL_BULLISH_GRIND')
//   aggregateGreeks   — { gex_sign, charm_sign, vanna_sign, ... } from the level set
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
  // Only use PC1 as a directional signal if the momentum loadings confirm
  // it is actually a trend/momentum axis. If pc1_momentum_valid is false,
  // PC1 is likely dominated by the vol cluster and its sign is meaningless
  // for direction.
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

  // ── CONFIDENCE: continuous score → bucket once ───────────────
  // Each modifier contributes +1 (raises evidence) or -1 (reduces evidence)
  // or 0 (neutral). We sum all modifiers, then map to LOW/MODERATE/HIGH.
  // Starting evidence is set by the initial conf assignment above.
  const CONF_BASE = { 'LOW': -1, 'MODERATE': 0, 'HIGH': 1 };
  let evidence = CONF_BASE[conf] ?? 0;

  // Vol × gamma regime interaction
  if (gammaRegime === 'NEGATIVE' && volRegime === 'EXPANSION') {
    reason += ` EXPANSION vol + negative gamma: dealers short and IV expanding — moves amplified.`;
    if (macroBearish) {
      evidence += 1;
      reason += ` Macro confirms — all three axes bearish.`;
    }
  } else if (gammaRegime === 'NEGATIVE' && volRegime === 'CONTRACTION') {
    evidence -= 1;
    reason += ` CONTRACTION vol with negative gamma is unusual — potential mean-reversion, reduce size.`;
  } else if (gammaRegime === 'POSITIVE' && volRegime === 'CONTRACTION') {
    reason += ` CONTRACTION vol + positive gamma: maximum pinning — walls highly reliable.`;
  }

  // GEX dispersion penalty
  if (hGexNorm > H_GEX_CONFIDENCE_CUT) {
    evidence -= 1;
    reason += ` GEX dispersed (H_GEX_norm ${hGexNorm.toFixed(2)} > 0.6) — no dominant wall.`;
  }

  // Macro neutrality penalty
  if (!macroBullish && !macroBearish) {
    evidence -= 1;
    reason += ` Macro bias neutral (${macroBias}) — confidence penalized.`;
  }

  // Entropy gate
  if (entropy && entropy.entropy_state === 'STABLE') {
    reason += ` Return entropy STABLE (H ${entropy.H_returns} < threshold ${entropy.H_threshold}) — orderly tape.`;
  } else if (!entropy || entropy.entropy_state === 'UNKNOWN') {
    reason += ` (Entropy gate unavailable.)`;
  }

  // ── PDF-DERIVED EVIDENCE LAYER ────────────────────────────────
  // Apply two additional signals on top of the existing accumulator:
  //   1. bias.pdf "primary bias" tag from aggregate (GEX, Charm, Vanna, IV, flip).
  //   2. walls.pdf reaction tag for the top wall.
  // Both are converted to a {dir, strength} pair via tag-direction maps in
  // lib/options.js. Strength-2 tags (breakdowns / squeezes) contribute more
  // evidence than strength-1 tags (standard reactions).
  //
  // The current `bias` variable has already been set above; we use it to
  // determine whether the PDF signals confirm or conflict with the call.
  const currentBull = bias.includes('BULL') && !bias.includes('BEAR');
  const currentBear = bias.includes('BEAR');

  const pdfDir = pdfBiasTag && BIAS_TAG_DIR[pdfBiasTag];
  if (pdfDir && pdfDir.dir !== 'NEUTRAL') {
    const matches = (pdfDir.dir === 'BULL' && currentBull) || (pdfDir.dir === 'BEAR' && currentBear);
    const conflicts = (pdfDir.dir === 'BULL' && currentBear) || (pdfDir.dir === 'BEAR' && currentBull);
    if (matches) {
      evidence += pdfDir.strength;
      reason += ` Aggregate bias table (${pdfBiasTag.replace(/_/g, ' ').toLowerCase()}) confirms.`;
    } else if (conflicts) {
      evidence -= pdfDir.strength;
      reason += ` ⚠ Aggregate bias table (${pdfBiasTag.replace(/_/g, ' ').toLowerCase()}) conflicts with directional call.`;
    }
  }

  const wallDir = wallReactionTag && WALL_REACTION_DIR[wallReactionTag];
  if (wallDir && wallDir.dir !== 'NEUTRAL') {
    const matches = (wallDir.dir === 'BULL' && currentBull) || (wallDir.dir === 'BEAR' && currentBear);
    const conflicts = (wallDir.dir === 'BULL' && currentBear) || (wallDir.dir === 'BEAR' && currentBull);
    if (matches) {
      evidence += wallDir.strength;
      reason += ` Top-wall reaction (${wallReactionTag.replace(/_/g, ' ').toLowerCase()}) confirms.`;
    } else if (conflicts) {
      evidence -= wallDir.strength;
      reason += ` ⚠ Top-wall reaction (${wallReactionTag.replace(/_/g, ' ').toLowerCase()}) conflicts.`;
      // walls.pdf strength-2 tags imply expansion/breakdown — flag air-pocket risk.
      if (wallDir.strength >= 2 && !air_pocket_watch) {
        air_pocket_watch = true;
        air_pocket_type  = 'WALL_BREAKDOWN';
      }
    }
  }

  // ── DIM/ERAKER/VILKOV ASYMMETRY ──────────────────────────────
  // 2.pdf finds positive-MM-gamma vol attenuation is ~3x stronger than
  // negative-MM-gamma vol amplification. Scale negative-gamma evidence
  // toward zero (multiply by GAMMA_ASYMMETRY_RATIO ≈ 0.34) so that signals
  // in the negative-gamma regime carry less confidence than equivalent
  // signals in the positive-gamma regime.
  if (gammaRegime === 'NEGATIVE') {
    const before = evidence;
    evidence = Math.sign(evidence) * Math.abs(evidence) * GAMMA_ASYMMETRY_RATIO;
    if (Math.abs(before) > 0.01) {
      reason += ` (Evidence scaled by ${GAMMA_ASYMMETRY_RATIO.toFixed(2)} — Dim/Eraker/Vilkov 2025 asymmetry: negative-gamma signals are weaker than positive-gamma signals.)`;
    }
  }

  // Map continuous evidence score to confidence bucket (single conversion, no clamping chain)
  conf = evidence >= 1 ? 'HIGH' : evidence <= -1 ? 'LOW' : 'MODERATE';

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

    // Compute flip + both regime axes before scoring — weights depend on both.
    const gammaFlip  = computeGammaFlip(strikes, futuresPrice);
    const flipDiff   = futuresPrice != null && gammaFlip != null ? futuresPrice - gammaFlip : null;

    let iv = null, rvIvRatio = null, hv21 = null;
    if (volData) {
      iv          = volData.current_iv  ?? null;
      rvIvRatio   = volData.rv_iv_ratio ?? null;
      hv21        = volData.hv21        ?? null;
    }

    // Vol-scaled gamma regime band: 0.5 × IV-implied daily move
    // If IV is unavailable, fall back to fixed 50 pts.
    // Derivation: daily_1sd = futuresPrice × (IV/100) / sqrt(252)
    //             band = 0.5 × daily_1sd
    // This makes NEAR_FLIP adaptive — wider when vol is high, tighter when low.
    const _ivBand = (iv != null && iv > 0 && futuresPrice != null)
      ? Math.max(30, 0.5 * futuresPrice * (iv / 100) / Math.sqrt(252))
      : 50;
    const gammaRegime = flipDiff == null ? 'UNKNOWN'
      : flipDiff >  _ivBand ? 'POSITIVE'
      : flipDiff < -_ivBand ? 'NEGATIVE'
      : 'NEAR_FLIP';
    const levelsRegime = classifyVolRegime(iv, rvIvRatio);
    const weights      = getWeights(levelsRegime, gammaRegime);

    const levels      = scoreLevels(strikes, weights, futuresPrice);
    const H_GEX_norm = computeHGEXNorm(levels);
    const topWall    = computeTopWall(levels, futuresPrice);

    // ── PDF-DERIVED CLASSIFICATIONS ──────────────────────────────────────────
    // walls.pdf reaction tag is attached per-level by scoreLevels(); pick the
    // tag for the top wall to feed the classifier.
    //
    // Aggregate Greek signs feed the bias.pdf primary-bias table. We compute
    // these on the FULL proximity-filtered set (nearbyStrikes — strikes within
    // FILTER_PCT of price) rather than on the post-MIN_SCORE levels list,
    // because bias.pdf rules describe overall dealer positioning, not just
    // the scoring-worthy walls. Low-score strikes can still tilt the
    // aggregate sign when they cluster on one side of price.
    const _nearbyAll      = nearbyStrikes(strikes, futuresPrice);
    const aggregateGreeks = computeAggregateGreeks(_nearbyAll);
    const wallReactionTag = topWall ? topWall.wall_reaction : null;
    const priceVsFlip     = flipDiff == null ? 0 : (flipDiff > 0 ? 1 : -1);
    const pdfBiasTag      = applyBiasTable(aggregateGreeks, levelsRegime, priceVsFlip);

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
      gammaRegime, volRegime: levelsRegime, gammaFlip, nqPrice: futuresPrice,
      topWall, hGexNorm: H_GEX_norm, macroBias, entropy, pca,
      pdfBiasTag, wallReactionTag, aggregateGreeks,
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
        pc1_momentum_loadings: pca.pc1_momentum_loadings,
        pc1_momentum_valid:    pca.pc1_momentum_valid,
        price_source:  priceSource,
        mm_intensification: [],
        macro_bias:    macroBias,
        macro_regime:  macroRegime,
        // PDF-derived methodology fields (bias.pdf + walls.pdf):
        pdf_primary_bias:   pdfBiasTag,
        wall_reaction:      wallReactionTag,
        aggregate_greeks:   aggregateGreeks,
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

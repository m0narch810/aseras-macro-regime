const {
  BASE_HEADERS, isAuthorized, fetchJson,
  todayET, aggregateDataset, computeGammaFlip, scoreLevels, classifyRegime,
} = require('./lib/options');

const SYMBOL   = 'QQQ';
const BASE_URL = 'https://www.free-flow.site/api';

const OUT_HEADERS = { ...BASE_HEADERS, 'Cache-Control': 'public, max-age=240' };

// ── GAMMA FLIP ────────────────────────────────────────────────────────────────
// Per-strike GEX sign change: call walls (positive) → put walls (negative).
// Interpolates the exact zero crossing between adjacent strikes and returns
// the crossing nearest to the current futures price.
// (Implementation lives in lib/options.js as computeGammaFlip)

// ── HANDLER ───────────────────────────────────────────────────────────────────
exports.handler = async (event) => {
  if (event.httpMethod === 'OPTIONS')
    return { statusCode: 200, headers: OUT_HEADERS, body: '' };
  if (!isAuthorized(event))
    return { statusCode: 401, headers: OUT_HEADERS, body: JSON.stringify({ error: true, message: 'unauthorized' }) };

  try {
    const cookie = process.env.FF_SESSION || '';
    const exp    = todayET();

    const data = await fetchJson(`${BASE_URL}/futures-levels?symbol=${SYMBOL}&exp=${exp}`, cookie);
    if (!data.rows || !data.rows.length)
      throw new Error('No rows returned from FreeFlow — FF_SESSION may be expired.');

    let iv = null, rvIvRatio = null, hv21 = null;
    try {
      const vol = await fetchJson(`${BASE_URL}/vol/realized?symbol=${SYMBOL}`, cookie);
      iv          = vol.current_iv  ?? null;
      rvIvRatio   = vol.rv_iv_ratio ?? null;
      hv21        = vol.hv21        ?? null;
    } catch (_) {}

    const { strikes, futuresPrice, spotEtf, ratio } = aggregateDataset(data);
    const [regime, weights] = classifyRegime(iv, rvIvRatio);
    const levels            = scoreLevels(strikes, weights, futuresPrice);
    const gammaFlip         = computeGammaFlip(strikes, futuresPrice);

    const updatedET = new Date().toLocaleString('en-US', {
      timeZone: 'America/New_York',
      hour: '2-digit', minute: '2-digit', hour12: false,
    }) + ' ET';

    return {
      statusCode: 200,
      headers:    OUT_HEADERS,
      body: JSON.stringify({
        updated:     updatedET,
        nq_price:    Math.round(futuresPrice * 10)  / 10,
        qqq_price:   Math.round(spotEtf      * 100) / 100,
        ratio:       Math.round(ratio        * 100) / 100,
        gamma_flip:  gammaFlip,
        regime,
        iv:          iv          != null ? Math.round(iv          * 10)   / 10   : null,
        rv_iv_ratio: rvIvRatio   != null ? Math.round(rvIvRatio   * 1000) / 1000 : null,
        hv21:        hv21        != null ? Math.round(hv21        * 10)   / 10   : null,
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

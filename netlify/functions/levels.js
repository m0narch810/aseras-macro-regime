const {
  BASE_HEADERS, isAuthorized, fetchJson,
  todayET, currentHourET, aggregateDataset, computeGammaFlip, scoreLevels,
  classifyVolRegime, getWeights, computeGTBR,
} = require('./lib/options');

const SYMBOL   = 'QQQ';
const BASE_URL = 'https://www.free-flow.site/api';

const OUT_HEADERS = { ...BASE_HEADERS, 'Cache-Control': 'public, max-age=240' };

function addDays(dateStr, n) {
  const [y, m, d] = dateStr.split('-').map(Number);
  const date = new Date(Date.UTC(y, m - 1, d));
  date.setUTCDate(date.getUTCDate() + n);
  return date.toISOString().slice(0, 10);
}

// ── HANDLER ───────────────────────────────────────────────────────────────────
exports.handler = async (event) => {
  if (event.httpMethod === 'OPTIONS')
    return { statusCode: 200, headers: OUT_HEADERS, body: '' };
  if (!isAuthorized(event))
    return { statusCode: 401, headers: OUT_HEADERS, body: JSON.stringify({ error: true, message: 'unauthorized' }) };

  try {
    const cookie   = process.env.FF_SESSION || '';
    const params   = event.queryStringParameters || {};
    const today    = todayET();

    // Kick off the vol fetch concurrently — it's independent of the levels lookup,
    // so overlapping the two upstream round-trips shaves ~one RTT off every refresh.
    const volPromise = fetchJson(`${BASE_URL}/vol/realized?symbol=${SYMBOL}`, cookie)
      .then(v => ({ v }))
      .catch(e => ({ err: e.message }));

    // If caller passes ?dte=N use that specific offset; otherwise auto-fallback 0→1→2.
    let data, activeExp, activeDTE;
    const requestedDte = params.dte != null ? parseInt(params.dte, 10) : null;

    if (requestedDte != null && requestedDte >= 0 && requestedDte <= 2) {
      activeExp = requestedDte === 0 ? today : addDays(today, requestedDte);
      activeDTE = requestedDte;
      data = await fetchJson(`${BASE_URL}/futures-levels?symbol=${SYMBOL}&exp=${activeExp}`, cookie);
      if (!data.rows || !data.rows.length)
        throw new Error(`No data for ${activeDTE}DTE (${activeExp}) — markets may be closed.`);
    } else {
      // Auto: try 0DTE → 1DTE → 2DTE
      let found = false;
      for (let dte = 0; dte <= 2; dte++) {
        const exp = dte === 0 ? today : addDays(today, dte);
        try {
          // Shorter per-probe timeout: three sequential tries must fit Netlify's 10s budget.
          const d = await fetchJson(`${BASE_URL}/futures-levels?symbol=${SYMBOL}&exp=${exp}`, cookie, 4500);
          if (d.rows && d.rows.length) { data = d; activeExp = exp; activeDTE = dte; found = true; break; }
        } catch (_) {}
      }
      if (!found) throw new Error('No options data for 0DTE, 1DTE, or 2DTE — FF_SESSION may be expired.');
    }

    let iv = null, rvIvRatio = null, hv21 = null, volError = null;
    let hv5 = null, hv10 = null, hv63 = null;
    const volRes = await volPromise;
    if (volRes.err) {
      volError = volRes.err;
    } else {
      iv          = volRes.v.current_iv  ?? null;
      rvIvRatio   = volRes.v.rv_iv_ratio ?? null;
      hv5         = volRes.v.hv5         ?? null;
      hv10        = volRes.v.hv10        ?? null;
      hv21        = volRes.v.hv21        ?? null;
      hv63        = volRes.v.hv63        ?? null;
    }

    const { strikes, futuresPrice, spotEtf, ratio, bookGex, bookDex } = aggregateDataset(data);

    // Compute flip + gamma regime first so weights can use both axes.
    const gammaFlip   = computeGammaFlip(strikes, futuresPrice);
    const diff        = futuresPrice != null && gammaFlip != null ? futuresPrice - gammaFlip : null;

    // Vol-scaled gamma regime band: 0.5 × IV-implied daily move
    // If IV is unavailable, fall back to fixed 50 pts.
    // Derivation: daily_1sd = futuresPrice × (IV/100) / sqrt(252)
    //             band = 0.5 × daily_1sd
    // This makes NEAR_FLIP adaptive — wider when vol is high, tighter when low.
    const _ivBand = (iv != null && iv > 0 && futuresPrice != null)
      ? Math.max(30, 0.5 * futuresPrice * (iv / 100) / Math.sqrt(252))
      : 50;
    const gammaRegime = diff == null ? 'UNKNOWN'
      : diff >  _ivBand ? 'POSITIVE'
      : diff < -_ivBand ? 'NEGATIVE'
      : 'NEAR_FLIP';

    const volRegime = classifyVolRegime(iv, rvIvRatio);
    const weights   = getWeights(volRegime, gammaRegime);
    const levels    = scoreLevels(strikes, weights, futuresPrice, volRegime, gammaFlip, iv,
                                  { rvIvRatio, hv5, hv63 });

    const updatedET = new Date().toLocaleString('en-US', {
      timeZone: 'America/New_York',
      hour: '2-digit', minute: '2-digit', hour12: false,
    }) + ' ET';

    return {
      statusCode: 200,
      headers:    OUT_HEADERS,
      body: JSON.stringify({
        updated:     updatedET,
        expiry_date: activeExp,
        expiry_dte:  activeDTE,
        nq_price:    Math.round(futuresPrice * 10)  / 10,
        qqq_price:   Math.round(spotEtf      * 100) / 100,
        ratio:       Math.round(ratio        * 100) / 100,
        gamma_flip:    gammaFlip,
        gamma_regime:  gammaRegime,
        book_gex:      bookGex != null ? Math.round(bookGex) : null,
        book_dex:      bookDex != null ? Math.round(bookDex) : null,
        gtbr_pts:      (function() { const g = computeGTBR(futuresPrice, iv, currentHourET()); return g != null ? Math.round(g) : null; })(),
        vol_regime:    volRegime,
        iv:          iv          != null ? Math.round(iv          * 10)   / 10   : null,
        rv_iv_ratio: rvIvRatio   != null ? Math.round(rvIvRatio   * 1000) / 1000 : null,
        hv5:         hv5         != null ? Math.round(hv5         * 10)   / 10   : null,
        hv10:        hv10        != null ? Math.round(hv10        * 10)   / 10   : null,
        hv21:        hv21        != null ? Math.round(hv21        * 10)   / 10   : null,
        hv63:        hv63        != null ? Math.round(hv63        * 10)   / 10   : null,
        vol_error:   volError,
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

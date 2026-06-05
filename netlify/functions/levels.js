const {
  BASE_HEADERS, isAuthorized, fetchJson,
  todayET, currentHourET, aggregateDataset, computeGammaFlip, scoreLevels,
  classifyVolRegime, getWeights, computeGTBR, smileAtmIv,
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
    // /vol/realized is a heavier, flakier endpoint than /futures-levels, so give it
    // ONE quick retry with a tight per-attempt timeout: recovers most transient
    // blips while staying inside Netlify's 10s budget (2×4000ms, concurrent with the
    // levels probe). If it's genuinely down, both fail fast → smile-IV fallback.
    const VOL_URL = `${BASE_URL}/vol/realized?symbol=${SYMBOL}`;
    const volPromise = fetchJson(VOL_URL, cookie, 4000)
      .catch(() => fetchJson(VOL_URL, cookie, 4000))
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

    // IV fallback: the /vol/realized endpoint times out (or returns a null
    // current_iv) often enough that GTBR and the gamma-regime band silently die
    // on those ticks — the readout shows "vol fetch failed" + NEUTRAL regime +
    // no GTBR. The per-strike smile is always present, so when the endpoint IV
    // is missing, fall back to the ATM strike's own iv_pct. Strictly better than
    // null (which forced GTBR=null and vol_regime=NEUTRAL). `iv_source` lets the
    // dashboard show "smile" instead of an alarming timeout when it degrades.
    let ivSource = iv != null ? 'endpoint' : null;
    if (iv == null) {
      const smileIv = smileAtmIv(strikes, spotEtf);
      if (smileIv != null) { iv = smileIv; ivSource = 'smile'; }
    }

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

    // Vol regime. When IV came from the smile fallback AND we have no rv/iv ratio
    // either (full endpoint outage), the absolute smile IV is a different/lower
    // tenor than the EXPANSION/CONTRACTION thresholds were calibrated for — it
    // would mislabel the same market (endpoint EXPANSION → smile CONTRACTION) and
    // wrongly hide the HIGH-VOL banner during a dump. Instead of trusting that,
    // derive the regime from gamma position (spot vs flip), which needs no IV and
    // maps to vol behavior: negative gamma → dealers amplify → expansion; positive
    // → dampened → contraction (Dim/Eraker/Vilkov 2025, Garmash 2024). Safe in a
    // dump: spot below flip → EXPANSION → HIGH-VOL banner shows. The rv/iv ratio,
    // when present, IS tenor-independent, so keep trusting it over the gamma proxy.
    let volRegime = classifyVolRegime(iv, rvIvRatio);
    if (ivSource === 'smile' && rvIvRatio == null) {
      volRegime = gammaRegime === 'NEGATIVE' ? 'EXPANSION'
                : gammaRegime === 'POSITIVE' ? 'CONTRACTION'
                : 'NEUTRAL';
    }
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
        iv_source:   ivSource,
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

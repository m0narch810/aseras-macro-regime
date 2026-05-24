// Serves the weekly macro bias.
// Fetches bias_output.json directly from GitHub at runtime so data is always
// current without requiring a Netlify rebuild after each weekly action commit.
// Falls back to the bundled version (last deploy) if the fetch fails.

const { BASE_HEADERS, isAuthorized } = require('./lib/options');

const BIAS_URL =
  'https://raw.githubusercontent.com/m0narch810/vanta/master/bias_output.json';
const CACHE_TTL_MS = 10 * 60 * 1000; // 10 min

let _cache = null;
let _cacheAt = 0;

let _bundled = null;
try { _bundled = require('../../bias_output.json'); } catch (_) {}

async function getBiasData() {
  const now = Date.now();
  if (_cache && now - _cacheAt < CACHE_TTL_MS) return _cache;
  try {
    const res = await fetch(BIAS_URL);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    _cache = await res.json();
    _cacheAt = now;
    return _cache;
  } catch (_) {
    return _cache || _bundled;
  }
}

const OUT_HEADERS = { ...BASE_HEADERS, 'Cache-Control': 'public, max-age=600' };

exports.handler = async (event) => {
  if (event.httpMethod === 'OPTIONS')
    return { statusCode: 200, headers: OUT_HEADERS, body: '' };
  if (!isAuthorized(event))
    return { statusCode: 401, headers: OUT_HEADERS, body: JSON.stringify({ error: true, message: 'unauthorized' }) };
  const biasData = await getBiasData();
  if (!biasData)
    return { statusCode: 200, headers: OUT_HEADERS, body: JSON.stringify({ error: true, message: 'bias_output.json unavailable' }) };
  return { statusCode: 200, headers: OUT_HEADERS, body: JSON.stringify(biasData) };
};

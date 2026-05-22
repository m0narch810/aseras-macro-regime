// Serves the weekly macro bias.
// bias_output.json is bundled at Netlify build time via esbuild require() inlining,
// so this endpoint works regardless of static-file serving.

const { BASE_HEADERS, isAuthorized } = require('./lib/options');

let biasData = null;
try { biasData = require('../../bias_output.json'); } catch (e) { biasData = null; }

const OUT_HEADERS = { ...BASE_HEADERS, 'Cache-Control': 'public, max-age=600' };

exports.handler = async (event) => {
  if (event.httpMethod === 'OPTIONS')
    return { statusCode: 200, headers: OUT_HEADERS, body: '' };
  if (!isAuthorized(event))
    return { statusCode: 401, headers: OUT_HEADERS, body: JSON.stringify({ error: true, message: 'unauthorized' }) };
  if (!biasData)
    return { statusCode: 200, headers: OUT_HEADERS, body: JSON.stringify({ error: true, message: 'bias_output.json unavailable' }) };
  return { statusCode: 200, headers: OUT_HEADERS, body: JSON.stringify(biasData) };
};

// Serves the weekly macro bias.
// bias_output.json is bundled into the function at build time (esbuild inlines
// the require), so this endpoint works regardless of static-file serving — it
// is what the Macro Bias tab fetches. Requires a valid session bearer token.

let biasData = null;
try {
  biasData = require('../../bias_output.json');
} catch (e) {
  biasData = null;
}

const VALID_USERS = ['aseras'];
const SESSION_MAX_AGE_MS = 30 * 24 * 60 * 60 * 1000;

const OUT_HEADERS = {
  'Content-Type':                 'application/json',
  'Access-Control-Allow-Origin':  '*',
  'Access-Control-Allow-Headers': 'Authorization',
  'Cache-Control':                'public, max-age=600',
};

// Verifies the Authorization: Bearer <vanta_session> header.
function isAuthorized(event) {
  const h = (event.headers && (event.headers.authorization || event.headers.Authorization)) || '';
  const m = h.match(/^Bearer\s+(.+)$/i);
  if (!m) return false;
  try {
    const { user, ts } = JSON.parse(Buffer.from(m[1], 'base64').toString('utf8'));
    return VALID_USERS.includes(user) && (Date.now() - ts) < SESSION_MAX_AGE_MS;
  } catch (e) {
    return false;
  }
}

exports.handler = async (event) => {
  if (event.httpMethod === 'OPTIONS') {
    return { statusCode: 200, headers: OUT_HEADERS, body: '' };
  }
  if (!isAuthorized(event)) {
    return { statusCode: 401, headers: OUT_HEADERS, body: JSON.stringify({ error: true, message: 'unauthorized' }) };
  }
  if (!biasData) {
    return { statusCode: 200, headers: OUT_HEADERS, body: JSON.stringify({ error: true, message: 'bias_output.json unavailable' }) };
  }
  return { statusCode: 200, headers: OUT_HEADERS, body: JSON.stringify(biasData) };
};

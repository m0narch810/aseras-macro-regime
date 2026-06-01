// VANTA access control.
// Passwords are stored as SHA-256 hashes — the plaintext never appears here.
const VALID_USERS = {
  // aseras
  "aseras": "1df6106046101d8351881262131311974c170c7e12195cf3bbee3d210fded14e",
  // awsame303
  "awsame303": "9ec0dbe01c3bfdb683df0e31db6fb99e033996fa15fe39f4f469950db816e0ee",
  // pinkus
  "pinkus": "f9f274527ee483268b16d0f42476b9f5b852f61d834a1116f45e5ae87ac383fb"
};

const SESSION_KEY = "vanta_session";
const SESSION_MAX_AGE_MS = 30 * 24 * 60 * 60 * 1000; // 30 days

// SHA-256 hex digest of a string (browser-native, no dependencies).
async function sha256Hex(str) {
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(str));
  return Array.from(new Uint8Array(buf))
    .map(b => b.toString(16).padStart(2, "0"))
    .join("");
}

// Returns the logged-in username if a valid, unexpired session exists.
// Any stale, expired, or old-format token is purged so it cannot trigger
// a redirect loop.
function activeSession() {
  const raw = localStorage.getItem(SESSION_KEY);
  if (!raw) return null;
  try {
    const { user, ts } = JSON.parse(atob(raw));
    if (user && VALID_USERS[user] && (Date.now() - ts) <= SESSION_MAX_AGE_MS) {
      return user;
    }
  } catch (e) {
    /* malformed token — fall through to purge */
  }
  localStorage.removeItem(SESSION_KEY);
  return null;
}

// Redirects between login and dashboard based on session state.
function checkAuth() {
  const isLoginPage = window.location.pathname.includes("login.html");
  const session = activeSession();
  if (!session && !isLoginPage) {
    window.location.replace("login.html");
  } else if (session && isLoginPage) {
    window.location.replace("index.html");
  }
}

// Verifies credentials and, on success, opens a session.
async function vantaLogin(username, password) {
  const stored = VALID_USERS[username];
  if (!stored) return false;
  const hash = await sha256Hex(password);
  if (hash !== stored) return false;
  localStorage.setItem(SESSION_KEY, btoa(JSON.stringify({ user: username, ts: Date.now() })));
  window.location.replace("index.html");
  return true;
}

function vantaLogout() {
  localStorage.removeItem(SESSION_KEY);
  window.location.replace("login.html");
}

// Gate the page the moment this script loads.
checkAuth();

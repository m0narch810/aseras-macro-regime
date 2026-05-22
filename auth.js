// ─────────────────────────────────────────────────────────────
//  CONFIG — must match the values in login.html exactly
// ─────────────────────────────────────────────────────────────
const SUPABASE_URL      = "PASTE_YOUR_SUPABASE_URL_HERE";
const SUPABASE_ANON_KEY = "PASTE_YOUR_SUPABASE_ANON_KEY_HERE";
// ─────────────────────────────────────────────────────────────

(function () {
  const { createClient } = supabase;
  const sb = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

  // Block page immediately if no valid session.
  // Hides content until auth check resolves.
  document.documentElement.style.visibility = "hidden";

  sb.auth.getSession().then(function ({ data }) {
    if (!data.session) {
      window.location.replace("login.html");
    } else {
      document.documentElement.style.visibility = "";
    }
  });

  // Also listen for session expiry mid-session
  sb.auth.onAuthStateChange(function (event) {
    if (event === "SIGNED_OUT") {
      window.location.replace("login.html");
    }
  });

  // Exposed globally so the logout button in index.html can call it
  window.vantaLogout = async function () {
    await sb.auth.signOut();
    window.location.replace("login.html");
  };
})();

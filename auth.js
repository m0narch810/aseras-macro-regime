// Hardcoded users for you and your friends
const VALID_USERS = {
  "awsame303": "pt$67143",
  "friend2": "password456",
  "friend3": "password789"
};

// Simple helper to check if someone is logged in
function checkAuth() {
  const session = localStorage.getItem("vanta_session");
  const isLoginPage = window.location.pathname.includes("login.html");

  if (!session && !isLoginPage) {
    // Not logged in and trying to view index.html -> send to login
    window.location.replace("login.html");
  } else if (session && isLoginPage) {
    // Already logged in and trying to view login page -> send to index
    window.location.replace("index.html");
  }
}

// Handle the login form submission
function vantaLogin(username, password) {
  if (VALID_USERS[username] && VALID_USERS[username] === password) {
    // Create a simple dummy session token
    localStorage.setItem("vanta_session", btoa(username));
    window.location.replace("index.html");
    return true;
  } else {
    return false;
  }
}

// Handle logging out
function vantaLogout() {
  localStorage.removeItem("vanta_session");
  window.location.replace("login.html");
}

// Run the auth check automatically when the script loads
checkAuth();
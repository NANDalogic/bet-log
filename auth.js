// Shared sign-in for the bets.nandalogic.com pages (index.html, live.html, plan.html).
// No libraries: talks to Supabase Auth over fetch. Email one-time code, sign-ups are off (only Josh's user exists).
// Auth.fetch(url, init) is a drop-in for fetch(): adds the apikey + the signed-in user's token, refreshes the
// token when it is about to expire, and shows the sign-in screen (the call waits) when there is no session.
(() => {
  const SB_URL = "https://pqkteawpnislalfxxtvc.supabase.co";
  const SB_KEY = "sb_publishable_M6Bh-PooIWoAAsrPiWJCsQ_y8QOspTG";   // public by design; it only identifies the project
  const SLOT = "nlAuth", EMAIL_SLOT = "nlAuthEmail";
  let mem = null;   // fallback when localStorage is blocked (private window): sign in again on reload

  function load() { try { return JSON.parse(localStorage.getItem(SLOT)) || mem; } catch { return mem; } }
  function save(s) { mem = s; try { s ? localStorage.setItem(SLOT, JSON.stringify(s)) : localStorage.removeItem(SLOT); } catch {} }
  const fromGrant = (d) => ({ access: d.access_token, refresh: d.refresh_token, exp: Date.now() + (d.expires_in || 3600) * 1000 });

  async function authCall(path, body, token) {
    const r = await fetch(`${SB_URL}/auth/v1/${path}`, { method: "POST", cache: "no-store",
      headers: { apikey: SB_KEY, "Content-Type": "application/json", ...(token ? { Authorization: "Bearer " + token } : {}) },
      body: JSON.stringify(body) });
    const d = await r.json().catch(() => ({}));
    return { ok: r.ok, status: r.status, d };
  }

  // One refresh at a time; a rejected refresh token (revoked, or signed out elsewhere) means sign in again.
  let refreshing = null;
  function refresh(s) {
    if (!refreshing) refreshing = (async () => {
      const r = await authCall("token?grant_type=refresh_token", { refresh_token: s.refresh });
      if (r.status === 400 || r.status === 401 || r.status === 403) { save(null); return null; }
      if (!r.ok) throw new Error("sign-in service HTTP " + r.status);
      const n = fromGrant(r.d); save(n); return n;
    })().finally(() => { refreshing = null; });
    return refreshing;
  }

  let gate = null;
  function signInGate() {
    if (!gate) gate = new Promise(showOverlay).finally(() => { gate = null; });
    return gate;
  }

  async function session() {
    let s = load();
    if (s && s.exp - Date.now() < 60000) s = await refresh(s);
    return s || signInGate();
  }

  async function authFetch(url, init = {}) {
    for (let attempt = 0; ; attempt++) {
      const s = await session();
      const h = new Headers(init.headers || {});
      h.set("apikey", SB_KEY); h.set("Authorization", "Bearer " + s.access);
      const r = await fetch(url, { ...init, headers: h });
      if (r.status !== 401 || attempt) return r;
      const c = load(); if (c) { c.exp = 0; save(c); }   // token rejected: force one refresh and retry
    }
  }

  async function signOut() {
    const s = load();
    if (s) await authCall("logout", {}, s.access).catch(() => {});
    save(null); location.reload();
  }

  // ---------- sign-in screen ----------
  function showOverlay(resolve) {
    const build = () => {
      const el = document.createElement("div");
      el.id = "nlAuth";
      el.style.cssText = "position:fixed;inset:0;z-index:99999;display:flex;align-items:center;justify-content:center;background:rgba(8,10,14,.94);padding:16px;font:16px/1.4 -apple-system,system-ui,sans-serif";
      el.innerHTML =
        '<form style="background:#fff;color:#111;max-width:340px;width:100%;border-radius:14px;padding:22px;box-shadow:0 8px 40px rgba(0,0,0,.5)">' +
        '<h2 style="margin:0 0 4px;font-size:20px">Sign in</h2>' +
        '<p data-msg style="margin:0 0 14px;color:#555;font-size:14px"></p>' +
        '<input data-email type="email" name="email" autocomplete="email" placeholder="Email" required style="box-sizing:border-box;width:100%;padding:12px;font-size:16px;border:1px solid #bbb;border-radius:8px;margin-bottom:10px">' +
        '<input data-code type="text" name="code" inputmode="numeric" autocomplete="one-time-code" maxlength="6" placeholder="6-digit code" hidden style="box-sizing:border-box;width:100%;padding:12px;font-size:22px;letter-spacing:6px;text-align:center;border:1px solid #bbb;border-radius:8px;margin-bottom:10px">' +
        '<button data-go type="submit" style="width:100%;padding:12px;font-size:16px;font-weight:600;border:0;border-radius:8px;background:#111;color:#fff">Send code</button>' +
        '<p style="margin:12px 0 0;text-align:center"><a data-back href="#" hidden style="color:#555;font-size:14px">Use a different email or resend</a></p>' +
        '</form>';
      document.body.appendChild(el);
      const $ = (q) => el.querySelector(q);
      const form = $("form"), msg = $("[data-msg]"), email = $("[data-email]"), code = $("[data-code]"), go = $("[data-go]"), back = $("[data-back]");
      let step = "email";
      const say = (t, bad) => { msg.textContent = t; msg.style.color = bad ? "#b00020" : "#555"; };
      const setStep = (s) => {
        step = s;
        email.hidden = s !== "email"; code.hidden = s !== "code"; back.hidden = s !== "code";
        go.textContent = s === "email" ? "Send code" : "Sign in";
        (s === "email" ? email : code).focus();
      };
      say("We'll email you a 6-digit code. You only do this once per device.");
      try { email.value = localStorage.getItem(EMAIL_SLOT) || ""; } catch {}
      setStep("email");
      back.addEventListener("click", (e) => { e.preventDefault(); code.value = ""; say("We'll email you a 6-digit code.", false); setStep("email"); });
      form.addEventListener("submit", async (e) => {
        e.preventDefault(); go.disabled = true;
        try {
          const addr = email.value.trim().toLowerCase();
          if (step === "email") {
            const r = await authCall("otp", { email: addr, create_user: false });
            if (!r.ok) say(r.status === 429 ? "Too many emails, wait a few minutes and try again." : (r.d.msg || r.d.error_description || "Could not send a code."), true);
            else { try { localStorage.setItem(EMAIL_SLOT, addr); } catch {} say("Code sent to " + addr + ". It can take a minute."); setStep("code"); }
          } else {
            const r = await authCall("verify", { type: "email", email: addr, token: code.value.replace(/\D/g, "") });
            if (!r.ok) say("That code didn't work. Check the email, or resend.", true);
            else { save(fromGrant(r.d)); el.remove(); resolve(load()); }
          }
        } catch { say("No connection. Try again.", true); }
        go.disabled = false;
      });
    };
    document.body ? build() : document.addEventListener("DOMContentLoaded", build);
  }

  window.Auth = { fetch: authFetch, signOut, signedIn: () => !!load() };
})();

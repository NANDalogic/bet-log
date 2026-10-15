// live-core.js — shared by live.html (browser) and the Supabase Edge Function live-snapshot (Deno).
// One source of truth for the odds model and for turning bets + ESPN data into model inputs, so the page and the
// 10-minute history snapshots compute the same numbers. Edit HERE, then redeploy live-snapshot (deploy-live-snapshot.sh
// copies this file into the function). No DOM; sets globalThis.LiveModel and globalThis.LiveCore.

// ===================== model =====================
// Pure simulation model (no DOM). Port of bet-sim's football and hockey score models (betsim.py fit_football/simulate,
// fit_hockey/simulate_hockey/goals_mask), plus live conditioning on score/clock (and ESPN win probability for football).
// Other sports are win/lose only. Exposed as window.LiveModel.
(function (root) {
  function erf(x) {   // Abramowitz–Stegun 7.1.26 (error < 1.5e-7)
    const s = x < 0 ? -1 : 1; x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return s * y;
  }
  const ncdf = (z) => 0.5 * (1 + erf(z / Math.SQRT2));
  function ninv(p) { let lo = -10, hi = 10; for (let i = 0; i < 100; i++) { const m = (lo + hi) / 2; if (ncdf(m) < p) lo = m; else hi = m; } return (lo + hi) / 2; }
  const clamp = (v, a, z) => Math.min(z, Math.max(a, v));
  function rng(seed) {   // mulberry32 + Box–Muller; fixed seed so numbers don't jitter between refreshes
    let a = seed >>> 0, spare = null;
    const u = () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
    const n = () => { if (spare != null) { const s = spare; spare = null; return s; } let x, y, r; do { x = 2 * u() - 1; y = 2 * u() - 1; r = x * x + y * y; } while (!r || r >= 1); const f = Math.sqrt(-2 * Math.log(r) / r); spare = y * f; return x * f; };
    return { u, n };
  }

  // Fit margin ~ N(mu_m, sd_m) to ML + spread, total ~ N(mu_t, sd_t) to the over price (same as betsim fit_football).
  function fitFootball(L, sdT) {
    const out = { sd_t: sdT };
    const s0 = L.spread != null ? -L.spread : null;
    const z1 = L.spread != null ? ninv(1 - (L.spread_p ?? 0.5)) : null;
    if (L.home_ml_p != null) {
      const z0 = ninv(1 - clamp(L.home_ml_p, 0.005, 0.995));
      out.sd_m = s0 != null && Math.abs(s0) >= 1 && (z1 - z0) * s0 > 0.05 ? s0 / (z1 - z0) : 13.5;
      out.sd_m = clamp(out.sd_m, 6, 25);
      out.mu_m = -z0 * out.sd_m;
    } else if (s0 != null) { out.sd_m = 13.5; out.mu_m = s0 - z1 * 13.5; }
    else { out.sd_m = 13.5; out.mu_m = 0; }
    out.mu_t = L.total != null ? L.total - sdT * ninv(1 - (L.over_p ?? 0.5)) : (sdT > 14 ? 55 : 44);
    return out;
  }

  // Final scores for one football game. live: {state, home, away, secLeft, wp}. Pregame f = 1.
  function simFootball(fit, live, n, R) {
    const H = new Float32Array(n), A = new Float32Array(n);
    if (live.state === "post") { H.fill(live.home); A.fill(live.away); return { H, A, f: 0 }; }
    const pre = live.state !== "in";
    const curH = pre ? 0 : live.home, curA = pre ? 0 : live.away, curM = curH - curA;
    const f = pre ? 1 : clamp(live.secLeft / 3600, 0.003, 1);
    const sdR = fit.sd_m * Math.sqrt(f), sdTR = fit.sd_t * Math.sqrt(f), muTR = fit.mu_t * f;
    let muR = fit.mu_m * f;
    // Live: centre the final margin so P(home wins) matches ESPN's win probability (it knows possession and field position).
    if (!pre && live.wp != null) muR = sdR * ninv(clamp(live.wp, 0.002, 0.998)) - curM;
    for (let i = 0; i < n; i++) {
      const rm = muR + sdR * R.n();
      const rt = Math.max(Math.abs(rm), muTR + sdTR * R.n(), 0);
      let h = curH + Math.max(0, Math.round((rt + rm) / 2)), a = curA + Math.max(0, Math.round((rt - rm) / 2));
      if (h === a) { if (R.u() < 0.5) h += 3; else a += 3; }   // no ties: overtime field goal
      H[i] = h; A[i] = a;
    }
    return { H, A, f };
  }

  // ---------- hockey (NHL): port of betsim fit_hockey / simulate_hockey / goals_mask ----------
  // Regulation goals per team ~ Poisson(rate × F), F ~ Gamma(1/disp, disp) a shared scoring factor for the night.
  // A regulation tie goes to an OT goal (OT_P, split by the teams' rates) or a shootout (winner +1 on the final score,
  // but no player gets a goal). Regular-season rules (5-min OT, shootout); playoff OT isn't modelled.
  const OT_P = 0.6, DISP = 0.05;
  function gammaDraw(R, k, theta) {   // Marsaglia–Tsang (shape k >= 1)
    const d = k - 1 / 3, c = 1 / Math.sqrt(9 * d);
    for (;;) {
      let x, v; do { x = R.n(); v = 1 + c * x; } while (v <= 0);
      v = v * v * v; const u = R.u();
      if (u < 1 - 0.0331 * x * x * x * x || Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v * theta;
    }
  }
  function poisDraw(R, lam) { if (!(lam > 0)) return 0; const u = R.u(); let k = 0, p = Math.exp(-lam), c = p; while (u > c && k < 60) { k++; p *= lam / k; c += p; } return k; }
  function binomDraw(R, T, s) { let k = 0; for (let j = 0; j < T; j++) if (R.u() < s) k++; return k; }
  let hockeyF = null;   // 400 evenly spaced quantiles of the scoring factor, mean exactly 1 (as betsim)
  function factorQuantiles() {
    if (hockeyF) return hockeyF;
    const R = rng(12345), d = Array.from({ length: 40000 }, () => gammaDraw(R, 1 / DISP, DISP)).sort((a, b) => a - b);
    const q = []; for (let i = 50; i < d.length; i += 100) q.push(d[i]);
    const m = q.reduce((a, b) => a + b, 0) / q.length;
    return (hockeyF = q.map((x) => x / m));
  }
  function poisPmf(lam, kmax) { const p = new Float64Array(kmax + 1); p[0] = Math.exp(-lam); for (let k = 1; k <= kmax; k++) p[k] = p[k - 1] * lam / k; return p; }
  // P(home wins incl. OT/shootout), P(final total > line), averaged over the scoring factor.
  function hockeyProbs(lh, la, line, F, otP) {
    const qh = otP * lh / (lh + la) + (1 - otP) * 0.5, K = 16;
    let pHome = 0, pOver = 0;
    for (const f of F) {
      const ph = poisPmf(lh * f, K), pa = poisPmf(la * f, K);
      for (let i = 0; i <= K; i++) for (let j = 0; j <= K; j++) {
        const x = ph[i] * pa[j];
        if (i > j) pHome += x; else if (i === j) pHome += qh * x;
        if (i + j + (i === j ? 1 : 0) > line) pOver += x;   // OT goal / shootout winner adds 1 to the total
      }
    }
    return [pHome / F.length, pOver / F.length];
  }
  // Regulation goal rates (home, away) that reproduce the 2-way ML and the total's over price. No line → league average.
  const hockeyFits = new Map();
  function fitHockey(L) {
    const ml = clamp(L.home_ml_p ?? 0.5, 0.03, 0.97), line = L.total ?? 6, over = clamp(L.total != null ? L.over_p ?? 0.5 : 0.5, 0.03, 0.97);
    const key = `${ml}|${line}|${over}`;
    if (hockeyFits.has(key)) return hockeyFits.get(key);
    const F = factorQuantiles();
    let mu = line, r = 0.5;
    for (let it = 0; it < 30; it++) {
      const mu0 = mu, r0 = r;
      let lo = 0.05, hi = 0.95;
      for (let k = 0; k < 40; k++) { r = (lo + hi) / 2; if (hockeyProbs(mu * r, mu * (1 - r), line, F, OT_P)[0] < ml) lo = r; else hi = r; }
      lo = 1; hi = 14;
      for (let k = 0; k < 40; k++) { mu = (lo + hi) / 2; if (hockeyProbs(mu * r, mu * (1 - r), line, F, OT_P)[1] < over) lo = mu; else hi = mu; }
      if (Math.abs(mu - mu0) < 1e-5 && Math.abs(r - r0) < 1e-6) break;
    }
    const fit = { lh: mu * r, la: mu * (1 - r), disp: DISP, otP: OT_P, noLine: L.home_ml_p == null && L.total == null };
    hockeyFits.set(key, fit);
    return fit;
  }
  // Final scores (H/A, shootout +1 included) and player-creditable goals (GH/GA: regulation + OT) for one NHL game.
  // live: {state, home, away, secLeft, period}. In regulation the rest of the game is Poisson × the time left; in OT
  // (period 4) it's sudden death over the OT seconds left; in the shootout (period 5) a coin flip.
  function simHockey(fit, live, n, R) {
    const H = new Float32Array(n), A = new Float32Array(n), GH = new Float32Array(n), GA = new Float32Array(n);
    if (live.state === "post") { H.fill(live.home); A.fill(live.away); GH.fill(live.home); GA.fill(live.away); return { H, A, GH, GA, f: 0, hockey: true }; }
    const pre = live.state !== "in", per = pre ? 0 : live.period || 1, secLeft = live.secLeft ?? 0;
    const curH = pre ? 0 : live.home, curA = pre ? 0 : live.away;
    const fReg = pre ? 1 : per <= 3 ? clamp(secLeft / 3600, 0, 1) : 0;
    const otLeft = pre || per <= 3 ? 1 : per === 4 ? clamp(secLeft / 300, 0, 1) : 0;
    const pOT = 1 - Math.pow(1 - fit.otP, otLeft), shareH = fit.lh / (fit.lh + fit.la);
    for (let i = 0; i < n; i++) {
      const F = fit.disp > 0 ? gammaDraw(R, 1 / fit.disp, fit.disp) : 1;
      let gh = curH + poisDraw(R, fit.lh * F * fReg), ga = curA + poisDraw(R, fit.la * F * fReg), h = gh, a = ga;
      if (gh === ga) {
        if (R.u() < pOT) { if (R.u() < shareH) gh++; else ga++; h = gh; a = ga; }
        else if (R.u() < 0.5) h = gh + 1; else a = ga + 1;
      }
      H[i] = h; A[i] = a; GH[i] = gh; GA[i] = ga;
    }
    return { H, A, GH, GA, f: pre ? 1 : clamp(secLeft / 3600, 0, 1), hockey: true, cur: { home: curH, away: curA } };
  }
  // Player goal share s with mean over sims of P(Binomial(T, s) > line) = pOver; T = the team's creditable goals.
  function goalShare(T, line, pOver) {
    const hist = new Map(); for (let i = 0; i < T.length; i++) hist.set(T[i], (hist.get(T[i]) || 0) + 1);
    const k = Math.floor(line) + 1;
    const pAtLeast = (s) => { let tot = 0; for (const [t, c] of hist) { let pmf = Math.pow(1 - s, t), cdf = 0; for (let j = 0; j < k && j <= t; j++) { cdf += pmf; pmf *= (t - j) / (j + 1) * s / (1 - s); } tot += c * (1 - cdf); } return tot / T.length; };
    let lo = 1e-6, hi = 0.95;
    for (let it = 0; it < 50; it++) { const s = (lo + hi) / 2; if (pAtLeast(s) < pOver) lo = s; else hi = s; }
    return (lo + hi) / 2;
  }

  function poisAtLeast(lam, k) { if (k <= 0) return 1; let term = Math.exp(-lam), cdf = term; for (let j = 1; j < k; j++) { term *= lam / j; cdf += term; } return clamp(1 - cdf, 0, 1); }
  function poisLam(pOver, line) {   // lambda with P(N > line) = pOver
    const k = Math.floor(line) + 1; let lo = 1e-4, hi = 20;
    for (let i = 0; i < 60; i++) { const m = (lo + hi) / 2; if (poisAtLeast(m, k) < pOver) lo = m; else hi = m; }
    return (lo + hi) / 2;
  }
  const COUNT_STATS = new Set(["td", "pass_tds", "goals", "assists", "points"]);
  // P(prop leg wins) given the stat so far x and the share of the game left f. Pregame: the leg's own fair p.
  function propProb(leg, x, f, post) {
    const line = Number(leg.line ?? 0.5), over = (leg.ou || "over") === "over";
    if (post) return (over ? x > line : x < line) ? 1 : x === line ? 0.5 : 0;
    const p0 = leg.p ?? 0.5, pOver0 = over ? p0 : 1 - p0;
    let pOver;
    if (COUNT_STATS.has(leg.stat)) {
      const need = Math.floor(line) + 1 - x;
      pOver = need <= 0 ? 1 : poisAtLeast(poisLam(pOver0, line) * f, need);
    } else {
      const yards = /yds/.test(leg.stat || "");
      const sd0 = yards ? Math.max(12, 0.5 * line) : Math.max(1.3, 0.35 * line + 0.8);
      const mu0 = line + sd0 * ninv(clamp(pOver0, 0.01, 0.99));
      if (x > line) pOver = 1;
      else if (f <= 0.003) pOver = 0;
      else pOver = 1 - ncdf((line - x - mu0 * f) / (sd0 * Math.sqrt(f)));
    }
    return over ? pOver : 1 - pOver;
  }

  // positions: { games: {key: G}, bets: [...], n, seed, bonusValue }
  //   G: {kind:"football"|"hockey"|"binary"|"none", fit, live:{state,home,away,secLeft,period,wp}, homeP, maxLead:{home,away},
//      stats:{player: {stat: x}}, playerTeam:{player: team abbreviation}, g: ESPN game}
  //   bet: {id, cash, payout, refund, fixedPL (settled) | legs:[{g, leg}] | flatP}
  // Each leg maps to a shared "outcome" (e.g. home covers -9.5), so bets on the same thing share one draw.
  function outcomeOf(g, leg, betId, i) {
    const k = leg.kind, side = leg.side;
    if (k === "ml") return { key: `${g}:ml`, type: "ml", g, want: side === "home" };
    if (k === "spread") { const hl = side === "home" ? Number(leg.line) : -Number(leg.line); return { key: `${g}:spr:${hl}`, type: "spr", g, homeLine: hl, want: side === "home" }; }
    if (k === "total") return { key: `${g}:tot:${leg.line}`, type: "tot", g, line: Number(leg.line), want: side === "over" };
    if (k === "team_total") return { key: `${g}:tt:${side}:${leg.line}`, type: "tt", g, side, line: Number(leg.line), want: (leg.ou || "over") === "over" };
    if (k === "prop") return { key: `${g}:prop:${(leg.player || "").toLowerCase()}:${leg.stat}:${leg.line}`, type: "prop", g, leg: { ...leg, ou: "over", p: leg.p == null ? null : (leg.ou || "over") === "over" ? leg.p : 1 - leg.p }, want: (leg.ou || "over") === "over" };
    return { key: `bet:${betId}:${i}`, type: "flat", g, want: true, label: leg.desc || "other leg" };
  }
  function legOutcome(g, leg, betId, i) {
    if (leg.kind === "ml" && leg.lead_by) return { key: `${g}:lead:${leg.side}:${leg.lead_by}`, type: "lead", g, side: leg.side, N: Number(leg.lead_by), want: true };
    return outcomeOf(g, leg, betId, i);
  }

  function run(pos) {
    const n = pos.n || 20000, R = rng(pos.seed || 7), bv = pos.bonusValue ?? 0.7;
    const sims = {};
    for (const [key, G] of Object.entries(pos.games)) {
      if (G.kind === "football") sims[key] = simFootball(G.fit, G.live, n, R);
      else if (G.kind === "hockey") sims[key] = simHockey(G.fit, G.live, n, R);
      else if (G.live && G.live.state === "post") sims[key] = { H: new Float32Array(n).fill(G.live.home), A: new Float32Array(n).fill(G.live.away), f: 0 };
      else if (G.kind === "binary" && G.homeP != null) { const hw = new Uint8Array(n); for (let i = 0; i < n; i++) hw[i] = R.u() < G.homeP ? 1 : 0; sims[key] = { hw }; }
      else sims[key] = {};
    }
    const outcomes = new Map();   // key -> {spec, v: Int8Array (1 yes, 0 no, 2 push), p, est}
    const cmpArr = (fn) => { const v = new Int8Array(n); for (let i = 0; i < n; i++) { const d = fn(i); v[i] = d > 0 ? 1 : d === 0 ? 2 : 0; } return v; };
    const drawArr = (p) => { const v = new Int8Array(n); for (let i = 0; i < n; i++) v[i] = R.u() < p ? 1 : 0; return v; };
    // NHL player goals (betsim goals_mask): the player's goals = box score so far + a Binomial share of his team's
    // creditable goals still to come, drawn one player after another from what's left (each goal to at most one player).
    // The share is calibrated on the pregame game so the pregame P(over) = the leg's fair p. Team from ESPN's box score;
    // unknown (pregame, replay) → a share of both teams' goals.
    function hockeyGoals(spec, S, G) {
      const nm = (spec.leg.player || "").toLowerCase(), line = Number(spec.leg.line ?? 0.5);
      const tm = (G.playerTeam || {})[nm], side = G.g && tm === G.g.home ? "home" : G.g && tm === G.g.away ? "away" : "both";
      if (!S.pre) S.pre = G.live && G.live.state === "in" ? simHockey(G.fit, { state: "pre" }, n, rng(11)) : S;
      const pool = (X, i) => (side === "home" ? X.GH[i] : side === "away" ? X.GA[i] : X.GH[i] + X.GA[i]);
      const Tpre = new Float32Array(n); for (let i = 0; i < n; i++) Tpre[i] = pool(S.pre, i);
      const alloc = S.alloc || (S.alloc = {});
      if (!alloc[side]) {
        const c = S.cur || { home: 0, away: 0 }, c0 = side === "home" ? c.home : side === "away" ? c.away : c.home + c.away;
        const left = new Int16Array(n); for (let i = 0; i < n; i++) left[i] = Math.max(0, pool(S, i) - c0);
        alloc[side] = { left, share: 1 };
      }
      const team = alloc[side], sh = Math.min(goalShare(Tpre, line, spec.leg.p ?? 0.5), team.share * 0.999);
      const x0 = ((G.stats || {})[nm] || {}).goals ?? 0, v = new Int8Array(n);
      for (let i = 0; i < n; i++) {
        const c = binomDraw(R, team.left[i], sh / team.share), x = x0 + c;
        team.left[i] -= c; v[i] = x > line ? 1 : x === line ? 2 : 0;
      }
      team.share -= sh;
      return v;
    }
    function getOutcome(spec, pYes) {
      if (outcomes.has(spec.key)) return outcomes.get(spec.key);
      const S = sims[spec.g] || {}, G = pos.games[spec.g] || {};
      let v = null, est = false;
      if (S.H) {
        const H = S.H, A = S.A;
        if (spec.type === "ml") v = cmpArr((i) => H[i] - A[i]);
        else if (spec.type === "spr") v = cmpArr((i) => H[i] - A[i] + spec.homeLine);
        else if (spec.type === "tot") v = cmpArr((i) => H[i] + A[i] - spec.line);
        else if (spec.type === "tt") v = cmpArr((i) => (spec.side === "home" ? H[i] : A[i]) - spec.line);
        else if (spec.type === "lead") {
          const led = G.maxLead && G.maxLead[spec.side] >= spec.N;
          // betsim's football heuristic for "led by N, then lost", scaled by time left. Hockey: winners only.
          const scale = S.hockey ? 0 : 0.42 * Math.exp(-(spec.N - 7) / 6) * (S.f ?? 1);
          v = new Int8Array(n);
          for (let i = 0; i < n; i++) {
            const m = spec.side === "home" ? H[i] - A[i] : A[i] - H[i];
            v[i] = led || m > 0 || R.u() < scale * Math.exp(-Math.abs(m) / 9) ? 1 : 0;
          }
          est = !led;
        }
      } else if (S.hw && spec.type === "ml") { v = new Int8Array(n); for (let i = 0; i < n; i++) v[i] = S.hw[i]; }
      if (spec.type === "prop" && spec.leg.stat === "goals" && S.hockey && G.live.state !== "post") { v = hockeyGoals(spec, S, G); est = true; }
      else if (spec.type === "prop") {
        const st =(G.stats || {})[(spec.leg.player || "").toLowerCase()] || {};
        const post = G.live && G.live.state === "post";
        const f = G.live && G.live.state === "in" ? (S.f ?? clamp((G.live.secLeft ?? 3600) / 3600, 0, 1)) : 1;
        const p = propProb(spec.leg, st[spec.leg.stat] ?? 0, f, post);
        v = drawArr(p); est = !post;
      }
      if (!v) { v = drawArr(pYes ?? 0.5); est = true; }
      let yes = 0, push = 0; for (let i = 0; i < n; i++) { if (v[i] === 1) yes++; else if (v[i] === 2) push++; }
      const o = { spec, v, p: yes / n, push: push / n, est };
      outcomes.set(spec.key, o);
      return o;
    }

    const pl = new Float64Array(n), refunds = new Float64Array(n);
    let fixed = 0;
    const perBet = [];
    for (const b of pos.bets) {
      if (b.fixedPL != null) { fixed += b.fixedPL; perBet.push({ id: b.id, settled: true }); continue; }
      const legsV = [];
      if (b.legs && b.legs.length) {
        b.legs.forEach(({ g, leg }, i) => {
          const spec = legOutcome(g, leg, b.id, i);
          const pYes = leg.p == null ? null : spec.want ? leg.p : 1 - leg.p;
          const o = getOutcome(spec, pYes);
          legsV.push({ o, want: spec.want });
        });
      } else {
        const o = getOutcome({ key: `bet:${b.id}`, type: "flat", g: null, want: true, label: b.label || "bet" }, b.flatP ?? 0.5);
        o.est = true; legsV.push({ o, want: true });
      }
      let wins = 0, pushes = 0;
      const win = new Uint8Array(n);
      for (let i = 0; i < n; i++) {
        let ok = true, allPush = true;
        for (const { o, want } of legsV) {
          const x = o.v[i];
          if (x === 2) continue;
          allPush = false;
          if ((x === 1) !== want) { ok = false; break; }
        }
        if (ok && allPush) { pushes++; continue; }   // straight bet pushed: stake back
        if (ok) { win[i] = 1; wins++; pl[i] += b.payout - b.cash; }
        else { pl[i] -= b.cash; refunds[i] += (b.refund || 0) * bv; }
      }
      perBet.push({ id: b.id, winP: wins / n, pushP: pushes / n, est: legsV.some((x) => x.o.est), win,
        cash: b.cash, payout: b.payout, legs: legsV.map(({ o, want }) => ({ key: o.spec.key, want })) });
    }
    for (let i = 0; i < n; i++) pl[i] += fixed;
    return { n, pl, refunds, fixed, perBet, outcomes, sims };
  }

  // Summary stats of the simulated P/L.
  function summarize(res) {
    const { n, pl, refunds } = res;
    const sorted = Float64Array.from(pl).sort();
    const q = (p) => sorted[Math.min(n - 1, Math.max(0, Math.floor(p * (n - 1))))];
    let black = 0, red = 0, sum = 0, sumR = 0, blackR = 0;
    for (let i = 0; i < n; i++) { const x = pl[i]; sum += x; if (x > 0.005) black++; else if (x < -0.005) red++; const y = x + refunds[i]; sumR += y; if (y > 0.005) blackR++; }
    return { pBlack: black / n, pRed: red / n, ev: sum / n, evR: sumR / n, pBlackR: blackR / n,
      p5: q(0.05), p25: q(0.25), median: q(0.5), p75: q(0.75), p95: q(0.95), min: sorted[0], max: sorted[n - 1], sorted };
  }

  // Best and worst final P/L over combinations of results that can actually happen together (conflicting bets can't
  // both win). Results on one game move together: each game's possible result combos are the distinct ones seen in the
  // sims; props and unmodelled legs vary on their own. Starts from the best (worst) sim, then switches one game at a
  // time to its best (worst) combo until nothing improves. Exact when no parlay spans games; never worse than the sims.
  // Chance of that P/L: the share of sims that land on it, or, when it's too rare for the sims to hit often, the product
  // over games of the chance each game's results give it (games are independent).
  function extremes(res) {
    const { n, pl, outcomes, perBet, fixed } = res;
    const bets = perBet.filter((b) => b.legs);
    if (!bets.length) return { best: fixed, worst: fixed, pBest: 1, pWorst: 1 };
    const groups = new Map();
    for (const o of outcomes.values()) {
      const own = o.spec.type === "prop" || o.spec.type === "flat" || !o.spec.g;
      const gk = own ? "o:" + o.spec.key : "g:" + o.spec.g;
      (groups.get(gk) || groups.set(gk, []).get(gk)).push(o);
    }
    const G = [...groups.values()].map((os) => {
      const combos = new Map();
      for (let i = 0; i < n; i++) { const t = os.map((o) => o.v[i]); const k = t.join(""); const c = combos.get(k); if (c) c.c++; else combos.set(k, { t, c: 1 }); }
      const list = [...combos.values()];
      return { keys: os.map((o) => o.spec.key), combos: list.map((x) => x.t), pr: list.map((x) => x.c / n) };
    });
    const total = (A) => {
      let s = fixed;
      for (const b of bets) {
        let ok = true, allPush = true;
        for (const { key, want } of b.legs) { const x = A.get(key); if (x === 2) continue; allPush = false; if ((x === 1) !== want) { ok = false; break; } }
        if (ok && !allPush) s += b.payout - b.cash; else if (!ok) s -= b.cash;
      }
      return s;
    };
    function search(better) {
      let i0 = 0;
      for (let i = 1; i < n; i++) if (better(pl[i], pl[i0])) i0 = i;
      const A = new Map([...outcomes.values()].map((o) => [o.spec.key, o.v[i0]]));
      let cur = total(A);
      for (let pass = 0; pass < 20; pass++) {
        let moved = false;
        for (const g of G) {
          const was = g.keys.map((k) => A.get(k));
          let bestT = was, bestV = cur;
          for (const t of g.combos) {
            g.keys.forEach((k, j) => A.set(k, t[j]));
            const v = total(A);
            if (better(v, bestV)) { bestV = v; bestT = t; }
          }
          g.keys.forEach((k, j) => A.set(k, bestT[j]));
          if (bestT !== was) { cur = bestV; moved = true; }
        }
        if (!moved) break;
      }
      let p = 1;
      for (const g of G) {
        const was = g.keys.map((k) => A.get(k));
        let pg = 0;
        g.combos.forEach((t, j) => { g.keys.forEach((k, i) => A.set(k, t[i])); if (Math.abs(total(A) - cur) < 0.005) pg += g.pr[j]; });
        g.keys.forEach((k, i) => A.set(k, was[i]));
        p *= pg;
      }
      let hit = 0;
      for (let i = 0; i < n; i++) if (Math.abs(pl[i] - cur) < 0.005) hit++;
      return { v: cur, p: Math.max(p, hit / n) };
    }
    const B = search((a, b) => a > b + 1e-9), W = search((a, b) => a < b - 1e-9);
    return { best: B.v, worst: W.v, pBest: B.p, pWorst: W.p };
  }

  // Cheer list: each undecided outcome, oriented so "it happens" helps; ranked by swing x uncertainty.
  function cheer(res, extraKeys) {
    const { n, pl } = res, rows = [];
    for (const o of res.outcomes.values()) rows.push(o);
    for (const o of extraKeys || []) if (!res.outcomes.has(o.spec.key)) rows.push(o);
    const out = [];
    for (const o of rows) {
      let sy = 0, ny = 0, sn = 0, nn = 0;
      for (let i = 0; i < n; i++) { if (o.v[i] === 1) { sy += pl[i]; ny++; } else if (o.v[i] === 0) { sn += pl[i]; nn++; } }
      if (ny < n * 0.002 || nn < n * 0.002) continue;   // already decided
      const ifYes = sy / ny, ifNo = sn / nn, swing = ifYes - ifNo;
      if (Math.abs(swing) < 0.5) continue;
      const p = ny / (ny + nn), flip = swing < 0;
      out.push({ o, yes: !flip, p: flip ? 1 - p : p, ifYes: flip ? ifNo : ifYes, ifNo: flip ? ifYes : ifNo, swing: Math.abs(swing),
        score: Math.abs(swing) * Math.sqrt(p * (1 - p)) });
    }
    return out.sort((a, b) => b.score - a.score);
  }

  root.LiveModel = { ncdf, ninv, fitFootball, simFootball, fitHockey, simHockey, hockeyProbs, propProb, run, summarize, extremes, cheer, rng, legOutcome };
})(typeof window !== "undefined" ? window : globalThis);

// ===================== bets + ESPN -> model inputs =====================
(function (root) {
  const M = root.LiveModel;
  const ESPN = { NFL: "football/nfl", CFB: "football/college-football", MLB: "baseball/mlb", NHL: "hockey/nhl", NBA: "basketball/nba", NCAAB: "basketball/mens-college-basketball" };
  const ESPN_GROUPS = { CFB: "&groups=80", NCAAB: "&groups=50" };
  const FOOTBALL = new Set(["NFL", "CFB"]);
  const SCORED = new Set([...FOOTBALL, "NHL"]);   // leagues with a score model (the rest are win/lose)
  // Betting weeks run Tuesday–Monday, counted from each NFL season's opening Tuesday.
  const WEEK_ANCHORS = ["2026-09-08", "2027-09-07", "2028-09-05"];
  function num(v) { if (v == null || v === "") return null; const n = Number(v); return Number.isFinite(n) ? n : null; }
  const scoreboardUrl = (league, date) => `https://site.api.espn.com/apis/site/v2/sports/${ESPN[league]}/scoreboard?dates=${date.replaceAll("-", "")}&limit=300${ESPN_GROUPS[league] || ""}`;
  // A bet counts for a league if the bet or any of its legs is in it ("Mixed" parlays show under each of their leagues).
  const betLeagues = (b) => new Set([b.league, ...(b.legs || []).map((l) => l.league)].filter((x) => x && x !== "Mixed"));
  const summaryUrl = (g) => `https://site.api.espn.com/apis/site/v2/sports/${ESPN[g.league]}/summary?event=${g.id}`;

  function parseEvent(e, league) {
    const c = e.competitions[0], stt = c.status || {};
    const home = c.competitors.find((x) => x.homeAway === "home"), away = c.competitors.find((x) => x.homeAway === "away");
    const state = stt.type?.state || "pre";
    const period = stt.period || 0, clk = Number(stt.clock || 0);
    let secLeft = null;
    if (FOOTBALL.has(league) && state === "in") secLeft = period <= 4 ? (4 - period) * 900 + clk : 0;
    // NHL scoreboard clock counts down; OT counts only its own minutes, a shootout (period 5) is over for props.
    if (league === "NHL" && state === "in") secLeft = period <= 3 ? (3 - period) * 1200 + clk : period === 4 ? clk : 0;
    const prob = c.situation?.lastPlay?.probability;
    return {
      id: e.id, league, date: e.date, state, detail: stt.type?.shortDetail || "",
      home: home.team.abbreviation, away: away.team.abbreviation,
      homeName: home.team.displayName, awayName: away.team.displayName,
      homeScore: num(home.score) ?? 0, awayScore: num(away.score) ?? 0, secLeft, period: state === "in" ? period : null,
      wp: state === "in" && prob && prob.homeWinPercentage != null ? prob.homeWinPercentage : null,
    };
  }

  // ESPN game summary -> box-score stats and team by lower-case player name, biggest lead each side reached, ESPN's own line.
  function parseSummary(d) {
    const stats = {}, playerTeam = {};
    for (const team of d.boxscore?.players || []) for (const grp of team.statistics || []) {
      const keys = grp.keys || [];
      for (const a of grp.athletes || []) {
        const nm = (a.athlete?.displayName || "").toLowerCase(), o = stats[nm] || (stats[nm] = {});
        if (team.team?.abbreviation) playerTeam[nm] = team.team.abbreviation;
        keys.forEach((k, i) => { const v = Number(a.stats?.[i]); if (Number.isFinite(v)) o[k] = v; });
      }
    }
    for (const o of Object.values(stats)) {
      o.pass_yds = o.passingYards ?? 0; o.rush_yds = o.rushingYards ?? 0; o.rec_yds = o.receivingYards ?? 0;
      o.rush_rec_yds = o.rush_yds + o.rec_yds; o.receptions = o.receptions ?? 0; o.pass_tds = o.passingTouchdowns ?? 0;
      o.td = (o.rushingTouchdowns ?? 0) + (o.receivingTouchdowns ?? 0);
      if (o.goals != null) o.points = o.goals + (o.assists ?? 0);   // NHL skaters
    }
    const maxLead = { home: 0, away: 0 };
    for (const p of d.scoringPlays || []) { maxLead.home = Math.max(maxLead.home, p.homeScore - p.awayScore); maxLead.away = Math.max(maxLead.away, p.awayScore - p.homeScore); }
    const pc = (d.pickcenter || [])[0];
    let espnLine = null;
    if (pc) {
      const hm = num(pc.homeTeamOdds?.moneyLine), am = num(pc.awayTeamOdds?.moneyLine);
      const imp = (o) => (o > 0 ? 100 / (o + 100) : -o / (-o + 100));
      espnLine = { spread: num(pc.spread), total: num(pc.overUnder), spread_p: 0.5, over_p: 0.5,
        home_ml_p: hm != null && am != null ? imp(hm) / (imp(hm) + imp(am)) : null, source: "ESPN (DraftKings)" };
    }
    return { stats, playerTeam, maxLead, espnLine };
  }

  // ---------- point in time (ESPN play-by-play) ----------
  // ESPN game summary -> [{t, home, away, secLeft, wp, detail}] per play, in time order (wallclock). NFL/CFB/MLB carry ESPN's
  // win probability per play; NHL has scores only (the hockey model works from score, period and clock).
  // NHL play clocks count UP (time elapsed in the period), unlike football's.
  function parseTimeline(d, league) {
    const plays = (d.drives?.previous || []).flatMap((dr) => dr.plays || []).concat(d.plays || []);
    const wpById = new Map((d.winprobability || []).map((w) => [String(w.playId), w.homeWinPercentage]));
    const out = [];
    let wp = null;
    for (const p of plays) {
      if (!p.wallclock) continue;
      if (wpById.has(String(p.id))) wp = wpById.get(String(p.id));
      const period = p.period?.number || 0, [mm, ss] = String(p.clock?.displayValue || "0:00").split(":").map(Number);
      const el = (mm || 0) * 60 + (ss || 0);
      const secLeft = FOOTBALL.has(league) ? (period <= 4 ? (4 - period) * 900 + el : 0)
        : league === "NHL" ? (period <= 3 ? (3 - period) * 1200 + Math.max(0, 1200 - el) : period === 4 ? Math.max(0, 300 - el) : 0) : null;
      const detail = league === "MLB" ? `${p.period?.type || ""} ${period}`.trim() : `${league === "NHL" ? "P" : "Q"}${period} ${p.clock?.displayValue || ""}`.trim();
      out.push({ t: Date.parse(p.wallclock), home: num(p.homeScore) ?? 0, away: num(p.awayScore) ?? 0, secLeft, period, wp, detail });
    }
    return out.sort((a, b) => a.t - b.t);
  }
  // A copy of st as it stood at time T: each game pre / in (score, clock, ESPN win % at the last play before T) / post.
  // timelines: Map(game id -> parseTimeline). Box-score stats count only once a game is final (ESPN has no per-play
  // stat lines), so props stay at their in-game estimate until the end; Early Win leads use the plays up to T.
  function atTime(st, timelines, T) {
    const games = new Map(), summaries = new Map();
    for (const [id, g] of st.games) {
      const tl = timelines.get(id) || [], start = tl.length ? tl[0].t : Date.parse(g.date), end = tl.length ? tl[tl.length - 1].t : Date.parse(g.date) + 3.5 * 3600e3;
      let x;
      if (T < start) x = { ...g, state: "pre", homeScore: 0, awayScore: 0, secLeft: null, period: null, wp: null };
      else if (g.state === "post" && T >= end) x = g;
      else {
        let k = -1; while (k + 1 < tl.length && tl[k + 1].t <= T) k++;
        const p = tl[k] || { home: 0, away: 0, secLeft: FOOTBALL.has(g.league) || g.league === "NHL" ? 3600 : null, period: 1, wp: null };
        x = { ...g, state: "in", homeScore: p.home, awayScore: p.away, secLeft: p.secLeft, period: p.period, wp: p.wp, detail: p.detail || "In progress" };
      }
      games.set(id, x);
      const sum = st.summaries.get(id);
      if (sum) {
        const maxLead = { home: 0, away: 0 };
        for (const p of tl) { if (p.t > T) break; maxLead.home = Math.max(maxLead.home, p.home - p.away); maxLead.away = Math.max(maxLead.away, p.away - p.home); }
        summaries.set(id, { ...sum, stats: x.state === "post" ? sum.stats : {}, maxLead });
      }
    }
    return { ...st, games, summaries };
  }

  // One history row per league ('all' + each) for a scope's bets: the chance numbers, plus `bets` = {bet id: final P/L
  // once decided (settled, or certain in the sims), else null}. The page diffs `bets` between consecutive rows to find
  // which bets a game decided (the callout cards).
  function snapshotRows(C, view, scope, period) {
    const M = root.LiveModel, rows = [], leagues = new Set();
    for (const b of view) for (const lg of betLeagues(b)) leagues.add(lg);
    for (const lg of ["all", ...leagues]) {
      const P = C.positions(lg === "all" ? view : view.filter((b) => betLeagues(b).has(lg)));
      if (!P.bets.length) continue;
      const res = M.run({ games: P.games, bets: P.bets, n: 20000, seed: 7 }), S = M.summarize(res);
      const pb = new Map(res.perBet.map((x) => [x.id, x])), bets = {};
      for (const b of P.bets) {
        const x = pb.get(b.id);
        const pl = b.fixedPL != null ? b.fixedPL : !x || x.settled ? null : x.winP >= 0.9999 ? b.payout - b.cash : x.pushP >= 0.9999 ? 0 : x.winP + x.pushP <= 0.0001 ? -b.cash : null;
        bets[b.id] = pl == null ? null : Math.round(pl * 100) / 100;
      }
      rows.push({ scope, period_start: period, league: lg, p_black: S.pBlack, p_red: S.pRed, ev: S.ev, fixed: res.fixed,
        n_open: Object.values(bets).filter((v) => v == null).length, bets });
    }
    return rows;
  }

  // st: { games: Map(id -> parsed game, with .day), summaries: Map(id -> parseSummary), lines: live_lines rows }.
  // replay: a YYYY-MM-DD string (every bet except voids treated as open at kickoff) or null.
  function make(st, REPLAY) {
    function findGame(l) {
      if (l.espn_id && st.games.has(String(l.espn_id))) return st.games.get(String(l.espn_id));
      for (const g of st.games.values()) if (g.league === l.league && g.home === l.home && g.away === l.away && (!l.date || g.day === l.date)) return g;
      return null;
    }
    const normName = (x) => (x || "").toLowerCase().replace(/[^a-z0-9 ]/g, "").trim();
    function lineFor(g) {
      const hn = normName(g.homeName), an = normName(g.awayName);
      return st.lines.find((r) => r.league === g.league && normName(r.home) === hn && normName(r.away) === an)
        || st.lines.find((r) => r.league === g.league && Math.abs(Date.parse(r.commence_time) - Date.parse(g.date)) < 3 * 3600e3 &&
             (normName(r.home).endsWith(hn.split(" ").pop()) || hn.endsWith(normName(r.home).split(" ").pop())));
    }
    function gameLine(g, legsOnGame) {
      // Field by field: Odds API line, else the bets' own fair prices, else ESPN's DraftKings line.
      const L = {}, src = new Set();
      const odds = REPLAY ? null : lineFor(g);
      // From the bets: several spread/total bets on one game may be on alt lines, so use the one priced closest to 50/50 (the main line).
      const fromLegs = {}, near = (l) => Math.abs(l.p - 0.5);
      const priced = legsOnGame.filter((l) => l.p != null);
      const ml = priced.find((l) => l.kind === "ml" && !l.lead_by);
      if (ml) fromLegs.home_ml_p = ml.side === "home" ? ml.p : 1 - ml.p;
      const sp = priced.filter((l) => l.kind === "spread").sort((x, y) => near(x) - near(y))[0];
      if (sp) { fromLegs.spread = sp.side === "home" ? Number(sp.line) : -Number(sp.line); fromLegs.spread_p = sp.side === "home" ? sp.p : 1 - sp.p; }
      const to = priced.filter((l) => l.kind === "total").sort((x, y) => near(x) - near(y))[0];
      if (to) { fromLegs.total = Number(to.line); fromLegs.over_p = to.side === "over" ? to.p : 1 - to.p; }
      const espn = st.summaries.get(g.id)?.espnLine;
      for (const [src_, o] of [["Odds API", odds], ["your bets' fair prices", fromLegs], ["ESPN (DraftKings)", espn]]) {
        if (!o) continue;
        if (L.home_ml_p == null && o.home_ml_p != null) { L.home_ml_p = num(o.home_ml_p); src.add(src_); }
        if (L.spread == null && o.spread != null) { L.spread = num(o.spread); L.spread_p = num(o.spread_p) ?? 0.5; src.add(src_); }
        if (L.total == null && o.total != null) { L.total = num(o.total); L.over_p = num(o.over_p) ?? 0.5; src.add(src_); }
      }
      return { L, src: [...src] };
    }
    function toPay(b) {
      const tp = num(b.to_pay); if (tp != null) return tp;
      const o = num(b.odds), stake = num(b.stake) || 0, cash = num(b.cash_risked) ?? stake;
      if (o == null) return cash;
      const profit = stake * (o > 0 ? o / 100 : 100 / -o) * (1 + (num(b.boost_pct) || 0));
      return cash === 0 ? profit : stake + profit;
    }
    // view: bet rows (already filtered to the window/league) -> model positions.
    function positions(view) {
      const games = {}, bets = [], meta = new Map(), gameBets = new Map(), lineSrc = {};
      const legsByGame = new Map();
      for (const b of view) for (const l of b.legs || []) { const g = findGame(l); if (g) (legsByGame.get(g.id) || legsByGame.set(g.id, []).get(g.id)).push(l); }
      for (const b of view) {
        const pending = b.result === "Pending" || (REPLAY && b.result !== "Void");
        const cash = num(b.cash_risked) ?? num(b.stake) ?? 0;
        const item = { id: b.id, label: b.bet, cash, payout: toPay(b), refund: num(b.refund_on_loss) || 0 };
        if (!pending) {
          let ret = num(b.returned);
          if (ret == null) ret = b.result === "Push" || b.result === "Void" ? cash : 0;
          item.fixedPL = ret - cash;
        } else if (b.legs && b.legs.length) {
          item.legs = b.legs.map((l) => {
            const g = findGame(l);
            const key = g ? g.id : `${l.league}:${l.away}@${l.home}`;
            if (!games[key]) {
              const live = g ? (REPLAY ? { state: "pre" } : { state: g.state, home: g.homeScore, away: g.awayScore, secLeft: g.secLeft, period: g.period, wp: g.wp }) : { state: "pre" };
              const sum = REPLAY ? null : g && st.summaries.get(g.id);
              const G = { live, maxLead: sum?.maxLead, stats: sum?.stats, playerTeam: sum?.playerTeam, kind: "none", g };
              if (g) {
                const { L, src } = gameLine(g, legsByGame.get(g.id) || []);
                lineSrc[key] = src;
                if (FOOTBALL.has(g.league)) { G.kind = "football"; G.fit = M.fitFootball(L, g.league === "CFB" ? 16 : 13); }
                else if (g.league === "NHL") { G.kind = "hockey"; G.fit = M.fitHockey(L); }
                else { G.kind = "binary"; G.homeP = live.state === "in" && g.wp != null ? g.wp : L.home_ml_p ?? null; }
              }
              games[key] = G;
            }
            return { g: key, leg: l };
          });
        } else {
          const o = num(b.odds);
          item.flatP = num(b.fair_win_pct) ?? (o == null ? 0.5 : o > 0 ? 100 / (o + 100) : -o / (-o + 100));
        }
        bets.push(item); meta.set(b.id, b);
        const keys = new Set((b.legs || []).map((l) => { const g = findGame(l); return g ? g.id : `${l.league}:${l.away}@${l.home}`; }));
        const gk = keys.size === 1 ? [...keys][0] : keys.size > 1 ? "multi" : "none";
        (gameBets.get(gk) || gameBets.set(gk, []).get(gk)).push(b.id);
      }
      return { games, bets, meta, gameBets, lineSrc, view };
    }
    return { findGame, lineFor, gameLine, toPay, positions };
  }

  // ---------- closing line value (since 10/15/2026) ----------
  // Fair chance each bet had at kickoff: the bet priced pregame by the score models, from the Odds API line cached just
  // before its game started (live_lines rows freeze at kickoff). Gradeable legs: ML, spread, total, team total in
  // NFL/CFB/NHL (score models), ML elsewhere. Anything else (props, Early Win = a heuristic not a market price, futures,
  // no Odds API line) -> src "none".
  // st: { games: Map (parseEvent rows with .day, for name matching), lines: live_lines rows }. bets: rows with legs.
  // today: US date (YYYY-MM-DD); a leg with no ESPN game counts as ungradeable only once its date is past.
  // Straight bets at the line's own number (ML, main spread, main total) take the line's fair price as is; the rest are
  // simulated. A live bet is "none" (no pregame price to beat): its text says "live", or it was logged over
  // LIVE_GRACE_MIN after the first start (Odds API commence time, else ESPN's) (logging lags placing by a few minutes, and puck drop / kickoff
  // runs a few minutes past the scheduled time, so a bet logged just after the start is usually pregame).
  // Returns [{id, p, src: "fresh"|"stale"|"none", age (min from line fetch to kickoff, worst leg), why}]; bets with a
  // game not started yet are left out (a later run grades them).
  const CLOSE_KINDS = new Set(["ml", "spread", "total", "team_total"]), LIVE_GRACE_MIN = 15;
  function closeProbs(st, bets, now, today, freshMin = 20, n = 100000) {
    const preGames = new Map([...st.games].map(([id, g]) => [id, { ...g, state: "pre", homeScore: 0, awayScore: 0, secLeft: null, period: null, wp: null }]));
    const C = make({ games: preGames, summaries: new Map(), lines: st.lines || [] }, null);
    const out = [], ok = [], ages = new Map();
    for (const b of bets) {
      let why = null, wait = false, age = 0, first = Infinity, direct = null;
      for (const l of b.legs || []) {
        const g = C.findGame(l);
        if (!g) { if (l.date && l.date < today) why = why || "no ESPN game"; else wait = true; continue; }
        // Started = past both ESPN's scheduled time and the Odds API commence time (the line freezes at the latter).
        const r = C.lineFor(g), start = Math.max(Date.parse(g.date), r ? Date.parse(r.commence_time) : 0);
        if (start > now) { wait = true; continue; }
        first = Math.min(first, start);
        if (!CLOSE_KINDS.has(l.kind)) { why = why || `${l.kind} leg`; continue; }
        if (l.lead_by) { why = why || "Early Win leg"; continue; }
        if (!SCORED.has(g.league) && l.kind !== "ml") { why = why || `${g.league} ${l.kind} (no score model)`; continue; }
        const need = FOOTBALL.has(g.league) ? ["home_ml_p", "spread", "spread_p", "total", "over_p"] : g.league === "NHL" ? ["home_ml_p", "total", "over_p"] : ["home_ml_p"];
        if (!r || need.some((k) => r[k] == null)) { why = why || "no Odds API line"; continue; }
        age = Math.max(age, (Date.parse(r.commence_time) - Date.parse(r.fetched_at)) / 60000);
        if (b.legs.length === 1) {
          const home = l.side === "home", hl = home ? Number(l.line) : -Number(l.line);
          if (l.kind === "ml") direct = home ? num(r.home_ml_p) : 1 - num(r.home_ml_p);
          else if (l.kind === "spread" && r.spread != null && hl === num(r.spread)) direct = home ? num(r.spread_p) : 1 - num(r.spread_p);
          else if (l.kind === "total" && r.total != null && Number(l.line) === num(r.total)) direct = l.side === "over" ? num(r.over_p) : 1 - num(r.over_p);
        }
      }
      if (!(b.legs || []).length) why = "no legs";
      if (!why && (/\blive\b/i.test(b.bet || "") || (b.created_at && Date.parse(b.created_at) > first + LIVE_GRACE_MIN * 60000))) why = "live bet";
      if (wait && !why) continue;
      if (why) { out.push({ id: b.id, p: null, src: "none", age: null, why }); continue; }
      const a = Math.round(age);
      if (direct != null) { out.push({ id: b.id, p: Math.round(direct * 10000) / 10000, src: a <= freshMin ? "fresh" : "stale", age: a, why: null }); continue; }
      ok.push({ ...b, result: "Pending" }); ages.set(b.id, Math.round(age));
    }
    if (ok.length) {
      const P = C.positions(ok), res = M.run({ games: P.games, bets: P.bets, n, seed: 7 });
      for (const x of res.perBet) {
        const age = ages.get(x.id);
        out.push({ id: x.id, p: Math.round(x.winP * 10000) / 10000, src: age <= freshMin ? "fresh" : "stale", age, why: null });
      }
    }
    return out;
  }

  root.LiveCore = { ESPN, ESPN_GROUPS, FOOTBALL, SCORED, WEEK_ANCHORS, num, betLeagues, scoreboardUrl, summaryUrl, parseEvent, parseSummary, parseTimeline, atTime, snapshotRows, make, closeProbs };
})(typeof window !== "undefined" ? window : globalThis);

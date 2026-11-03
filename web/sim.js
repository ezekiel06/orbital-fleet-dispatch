// Browser port of fleet.py's simulate(): 12 satellites, minute by minute,
// LOCAL (each satellite alone) vs FLEET (shared queue, best-fit packing,
// radiation-zone handoffs over the laser link). Same physics and rules as the
// Python version; only the random-number generator differs.
(function (root) {
  const SIGMA = 5.670e-8;
  const JOB_TYPES = [[15, 10, false], [30, 20, true], [60, 40, true]]; // kW, minutes, error-sensitive
  const WEATHER_MULT = { LOW: 1, ELEVATED: 3, HIGH: 10 };
  const ERR = (() => {               // radiation.py, calibrated on Google TPU tests
    const avg = 12.4 / 1.275 / 525600;
    return { outside: avg / (1 - 0.14 + 5 * 0.14), saaMult: 5 };
  })();
  const AGING_ORDER = [0, 4, 8, 11, 2, 6];

  function rng(seed) {               // mulberry32
    let a = seed >>> 0;
    return () => {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function poisson(lam, r) {
    const L = Math.exp(-lam); let k = 0, p = 1;
    do { k++; p *= r(); } while (p > L);
    return k - 1;
  }
  function lookahead(lit) {
    const N = lit.length, toSh = new Float64Array(N), shLen = new Float64Array(N), left = new Float64Array(N);
    let nxt = N, nlen = 35, run = 0;
    for (let i = N - 1; i >= 0; i--) {
      if (!lit[i]) { run++; left[i] = run; nxt = i; nlen = run; }
      else { run = 0; toSh[i] = nxt - i; }
      shLen[i] = nlen;
    }
    return { toSh, shLen, left };
  }

  function makeSat(p, lit, saa) {
    const la = lookahead(lit);
    return { p, lit, saa, ...la, batt: p.battery_kwh * p.battery_start, tempK: p.temp_start_c + 273.15,
             running: [], minLow: 0, minBatt: 100, peak: 0 };
  }
  function safeBudget(s, i) {
    const p = s.p, floor = p.battery_kwh * p.battery_floor;
    let b;
    if (s.lit[i]) {
      const need = floor + s.shLen[i] * (p.bus_kw + 15) / 60;
      const rate = Math.max(0, need - s.batt) / Math.max(s.toSh[i], 1) * 60 / p.charge_eff;
      b = p.solar_kw - p.bus_kw - rate;
    } else {
      b = (s.batt - floor) / (Math.max(s.left[i], 1) / 60) - p.bus_kw;
    }
    const tc = s.tempK - 273.15;
    if (tc > p.throttle_at_c - 4) b *= Math.max(0, (p.throttle_at_c - tc) / 4);
    return Math.max(0, Math.min(b, p.gpu_max_kw));
  }
  function physics(s, i, gpu) {
    const p = s.p, lit = s.lit[i];
    const gen = lit ? p.solar_kw : 0, use = gpu + p.bus_kw, net = gen - use;
    s.batt = Math.min(Math.max(s.batt + (net > 0 ? net * p.charge_eff : net) / 60, 0), p.battery_kwh);
    const heatIn = use * 1000 + (lit ? p.sun_heat_kw * 1000 : 0);
    const heatOut = p.emissivity * SIGMA * p.radiator_m2 * (Math.pow(s.tempK, 4) - 81);
    s.tempK += (heatIn - heatOut) * 60 / p.heat_capacity_j_per_k;
    const soc = s.batt / p.battery_kwh * 100;
    if (soc < p.battery_floor * 100) s.minLow++;
    s.minBatt = Math.min(s.minBatt, soc);
    s.peak = Math.max(s.peak, s.tempK - 273.15);
  }
  function makeJobs(minutes, rate, r) {
    const jobs = [];
    for (let t = 0; t < minutes; t++) {
      const n = poisson(rate, r);
      for (let k = 0; k < n; k++) {
        const [kw, mins, sens] = JOB_TYPES[Math.floor(r() * 3)];
        jobs.push({ kw, mins, sens, arrive: t, done: 0, state: "future", sat: null, hold: 0, prev: null, stall: 0, finish: null });
      }
    }
    return jobs;
  }

  // spec: satellite numbers; orbit: {lit:[bool[]], saa:[bool[]]}; aging: {count, battery, solar}
  function simulate(mode, spec, orbit, weather, seed, aging, wantLog) {
    const minutes = orbit.lit[0].length, n = orbit.lit.length;
    const r = rng(seed), er = rng(seed + 99);
    const jobs = makeJobs(minutes, 1.3 * spec.gpu_max_kw / 100, r);
    const worn = new Set(AGING_ORDER.slice(0, aging ? aging.count : 0));
    const sats = orbit.lit.map((lit, k) => {
      const p = { ...spec };
      if (worn.has(k)) { p.battery_kwh *= aging.battery; p.solar_kw *= aging.solar; }
      return makeSat(p, lit, orbit.saa[k]);
    });
    const wm = WEATHER_MULT[weather];
    let restarts = 0, moves = 0, wasted = 0, nxt = 0, rr = 0;
    const log = wantLog ? [] : null;
    const usedKw = (s) => s.running.reduce((a, j) => a + j.kw, 0);

    for (let i = 0; i < minutes; i++) {
      while (nxt < jobs.length && jobs[nxt].arrive <= i) {
        const j = jobs[nxt++]; j.state = "queued";
        if (mode === "local") { j.sat = rr % n; rr++; }
      }
      const budgets = sats.map((s) => safeBudget(s, i));
      sats.forEach((s, si) => {
        const keep = []; let used = 0;
        s.running.slice().sort((a, b) => a.kw - b.kw).forEach((j) => {
          if (used + j.kw <= budgets[si]) { keep.push(j); used += j.kw; }
          else { j.state = "queued"; if (mode === "fleet") { j.prev = si; j.sat = null; j.hold = 1; } }
        });
        s.running = keep;
      });
      const queued = jobs.filter((j) => j.state === "queued").sort((a, b) => a.arrive - b.arrive);
      for (const j of queued) {
        if (j.hold > 0) { j.hold--; continue; }
        let tiers;
        if (mode === "local") tiers = [[j.sat]];
        else {
          const all = [...Array(n).keys()], clean = all.filter((c) => !sats[c].saa[i]);
          tiers = j.sens ? [clean, all] : [all];
        }
        let best = null;
        for (const cands of tiers) {
          let bs = null;
          for (const c of cands) {
            const spare = budgets[c] - usedKw(sats[c]);
            if (spare >= j.kw && (bs === null || spare < bs)) { best = c; bs = spare; }
          }
          if (best !== null) break;
        }
        if (best !== null) {
          if (j.prev !== null && j.prev !== best) moves++;
          j.prev = null; j.sat = best; j.state = "running"; sats[best].running.push(j);
        }
      }
      if (mode === "fleet") {
        sats.forEach((s, si) => {
          if (!s.saa[i]) return;
          for (const j of s.running.filter((x) => x.sens)) {
            let tgt = null, ts = null;
            sats.forEach((t, c) => {
              if (t.saa[i] || c === si) return;
              const spare = budgets[c] - usedKw(t);
              if (spare >= j.kw && (ts === null || spare < ts)) { tgt = c; ts = spare; }
            });
            if (tgt !== null) {
              s.running.splice(s.running.indexOf(j), 1); sats[tgt].running.push(j);
              j.sat = tgt; j.stall = 1; moves++;
            }
          }
        });
      }
      let total = 0;
      for (const s of sats) {
        let speed = s.tempK - 273.15 > s.p.temp_limit_c ? 0.3 : 1;
        if (s.batt <= 0.5) speed = 0;
        for (const j of s.running.slice()) {
          if (j.stall) { j.stall--; continue; }
          const pe = j.kw * ERR.outside * (s.saa[i] ? ERR.saaMult : 1) * wm;
          if (j.sens && speed > 0 && er() < pe) { wasted += j.kw * j.done / 60; j.done = 0; restarts++; continue; }
          j.done += speed;
          if (j.done >= j.mins) { j.state = "done"; j.finish = i; s.running.splice(s.running.indexOf(j), 1); }
        }
        const gpu = s.running.reduce((a, j) => a + (j.stall ? 0 : j.kw), 0) * speed;
        total += gpu;
        physics(s, i, gpu);
      }
      if (log) log.push({ gpu: total, lit: orbit.lit.reduce((a, l) => a + (l[i] ? 1 : 0), 0) });
    }
    const done = jobs.filter((j) => j.state === "done");
    const waits = done.map((j) => j.finish - j.arrive - j.mins);
    return {
      work: done.reduce((a, j) => a + j.kw * j.mins, 0) / 60,
      jobs: done.length, arrived: jobs.length, restarts, wasted, moves,
      delay: waits.length ? waits.reduce((a, b) => a + b, 0) / waits.length : 0,
      lowMin: sats.reduce((a, s) => a + s.minLow, 0),
      minBatt: Math.min(...sats.map((s) => s.minBatt)),
      peak: Math.max(...sats.map((s) => s.peak)),
      log,
    };
  }

  function settleTempC(heatKw, areaM2, emis) {
    return Math.pow(heatKw * 1000 / (emis * SIGMA * areaM2), 0.25) - 273.15;
  }

  const api = { simulate, settleTempC, WEATHER_MULT };
  if (typeof module !== "undefined") module.exports = api; else root.OrbitSim = api;
})(this);

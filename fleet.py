"""Step 6: A fleet of orbital compute satellites that share work.

The SpaceX-style setup: by default 12 satellites in Starlink-style orbits (53 degree
tilt, 550 km up), 3 orbital planes x 4 satellites; SAT_PROFILE and SAT_FLEET change
the orbit and size (up to 96 tested). They're linked by lasers, so a job
can run on whichever satellite is best placed right now.

Two new real-world effects:
  - Shadows are staggered: while some satellites are in Earth's shadow, others
    are in full sun. A fleet can move work to the sunlit ones.
  - The South Atlantic Anomaly (SAA): a region over South America and the South
    Atlantic where Earth's radiation belt dips low. Satellites passing through it
    get far more bit flips. A fleet can keep error-sensitive jobs off satellites
    that are in it.

Compares two dispatchers on identical orbits, jobs and weather:
  LOCAL  each satellite gets its own share of jobs and runs the step-4 rules alone
  FLEET  one shared queue; each minute, jobs are packed onto the satellite whose
         safe spare power fits them best. Sensitive jobs prefer satellites outside
         the SAA, and are handed off over the laser link (progress kept, 1 minute
         in transit) when their satellite enters it, but only if another has room

Assumptions (see walkthrough): Starlink-style orbit is a stand-in, not confirmed
for SpaceX data centers; satellite specs reuse step 2's; the SAA is a simple box
with a x5 bit-flip rate (rates from radiation.py, calibrated on Google TPU tests); moving a job between satellites costs 1 minute.

Usage:  python fleet.py [--hours 12] [--weather LOW|ELEVATED|HIGH] [--seed 1]
Saves:  fleet_results.json
"""
import argparse, json, math
import numpy as np
from sgp4.api import Satrec, WGS72
from skyfield.api import load, EarthSatellite, wgs84
from power_thermal import SAT, SIGMA
from space_weather import ERROR_MULT

R_EARTH = 6378.137
ALT_KM, INC_DEG = 550.0, 53.0
import sat_profile as _sp
PLANES, PER_PLANE = _sp.PLANES_N, _sp.PER_PLANE_N   # SAT_FLEET, default 3x4
SAA = {"lat": (-50, 0), "lon": (-90, 40), "mult": 5}   # rough box, assumed
from radiation import p_error  # v3: calibrated from Google TPU radiation tests
JOB_TYPES = [("Image tagging", 15, 10, False), ("Translation batch", 30, 20, True),
             ("Video analysis", 60, 40, True)]
MOVE_DELAY_MIN = 1

# Laser and ground links (SAT_LINK). "none" = the original model: any number of
# moves per minute, each costing 1 minute. The others limit how much data a
# satellite can send per minute and make big jobs take longer to move.
#   today: 3 lasers x 100 Gbps sustained (Starlink figure, SPIE Photonics West;
#          3 lasers per satellite is our assumption), 100 Gbps to the ground
#   v3:    6 lasers x 400 Gbps, 1 Tbps to the ground (reported Starlink V3 specs)
#   slow:  1 laser x 10 Gbps, 10 Gbps to the ground (top of the 1-10 Gbps range
#          older commercial laser links offer, per the Suncatcher paper)
import os as _os
LINKS = {"none": None,
         "today": {"lasers": 3, "gbps": 100, "ground_gbps": 100},
         "v3": {"lasers": 6, "gbps": 400, "ground_gbps": 1000},
         "slow": {"lasers": 1, "gbps": 10, "ground_gbps": 10},
         "nomove": {"lasers": 0, "gbps": 1, "ground_gbps": 1e9}}  # shared queue, but running jobs never move
LINK = LINKS[_os.environ.get("SAT_LINK", "none")]
# Data a 20 kW job must carry when it moves (its working state: model, batch,
# progress), in GB, scaled by job size. SAT_JOB_IN_GB = input each new job must
# upload from the ground before it can start (0 = ignore the ground link).
JOB_GB = float(_os.environ.get("SAT_JOB_GB", "10"))
JOB_IN_GB = float(_os.environ.get("SAT_JOB_IN_GB", "0"))
# SAT_MOVE=smart (default): with link limits on, only move a running job when the
# transfer is quicker than waiting for its own satellite and short next to the
# work it has left. SAT_MOVE=always: move whenever another satellite has room.
SMART_MOVES = _os.environ.get("SAT_MOVE", "smart") == "smart"


# Equinox (Erol & Mahajan, Georgia Tech, arXiv:2604.19958, Apr 2026), rebuilt for comparison.
# Each satellite prices its own work: P = 1.40 x f_batt x f_thermal x f_queue, with
#   f_batt  = 1 + 0.001 / (SoC - SoC_c)^2        (SoC_c = our 70% floor, the level never to reach)
#   f_therm = 1 + g_T (T - 50)^2 / ((85 - T)^2 + 1)   (g_T not stated in the paper; we use 0.1)
#   f_queue = 1 + 0.01 x (jobs waiting on that satellite)
# A satellite runs work only while the job's value beats P (value = EQX_VALUE, default 4.0,
# the middle of the paper's task values), and never draws its battery below SoC_c. A job that
# can't run where it arrived hops once to the satellite with the lowest P_phys (no queue term)
# if P_n + 0.05 P_local + 0.10 < P_local. Choices that favor Equinox: every satellite counts as
# a neighbor (the paper uses those within 5,000 km), and waiting jobs never expire (the paper
# drops them after 300 s); the queue term counts only jobs waiting 5 minutes or less, matching
# the paper's 300-second queue, and a job that hasn't started hops with no transfer delay.
EQX_VALUE = float(_os.environ.get("EQX_VALUE", "4.0"))
EQX_GAMMA_T = float(_os.environ.get("EQX_GAMMA_T", "0.1"))


def eqx_cost(sat, queue_len=0):
    p = sat.p
    soc = sat.batt / p["battery_kwh"]
    gap = max(soc - p["battery_floor"], 1e-3)
    t = sat.temp_k - 273.15
    f_b = 1 + 0.001 / gap ** 2
    f_t = 1 + EQX_GAMMA_T * (t - 50) ** 2 / ((85 - t) ** 2 + 1) if t > 50 else 1.0
    return 1.40 * f_b * f_t * (1 + 0.01 * queue_len)


def eqx_budget(sat, i, queue_len):
    """GPU power Equinox lets this satellite use this minute."""
    p = sat.p
    if eqx_cost(sat, queue_len) >= EQX_VALUE:
        return 0.0
    gen = p["solar_kw"] if sat.lit[i] else 0.0
    spare_kwh = sat.batt - p["battery_kwh"] * p["battery_floor"]
    return max(0.0, min(p["gpu_max_kw"], spare_kwh * 60 + gen - p["bus_kw"]))


def _worth_moving(j, sat, i, transfer):
    """Is moving this job cheaper than letting it wait?"""
    if not LINK or not SMART_MOVES:
        return True
    left = j["mins"] - j["done_min"]
    wait = sat.sh_left[i] if not sat.lit[i] else 10  # shadow: wait it out; sunlit but saving battery: ~10 min
    return transfer < wait and transfer <= 0.25 * left


def _state_gb(j):
    return JOB_GB * j["kw"] / 20


def _transfer_min(gb):
    """Minutes a move takes over one laser (at least 1, as before)."""
    return max(1, math.ceil(gb * 8 / LINK["gbps"] / 60)) if LINK else MOVE_DELAY_MIN


def sun_dir(jd):
    n = jd - 2451545.0
    L = np.radians((280.460 + 0.9856474 * n) % 360)
    g = np.radians((357.528 + 0.9856003 * n) % 360)
    lam = L + np.radians(1.915) * np.sin(g) + np.radians(0.020) * np.sin(2 * g)
    e = np.radians(23.439 - 0.0000004 * n)
    return np.stack([np.cos(lam), np.cos(e) * np.sin(lam), np.sin(e) * np.sin(lam)], axis=-1)


def build_fleet(minutes, start=None, offset_days=0):
    """Sunlight and in-SAA flags for every satellite, every minute.
    Uses the AI1 orbit when SAT_PROFILE picks one (see sat_profile.py)."""
    import sat_profile
    o = sat_profile.ORBIT
    alt, inc = (o["alt"], o["inc"]) if o else (ALT_KM, INC_DEG)
    start = start or (o["start"] if o and o["start"] else (2026, 11, 3))
    ts = load.timescale()
    t0 = ts.tt_jd(ts.utc(*start).tt + offset_days)
    a = R_EARTH + alt
    n_rad_min = math.sqrt(398600.4418 / a ** 3) * 60
    m = np.arange(minutes)
    t = ts.tt_jd(t0.tt + m / 1440)
    sun = sun_dir(t.ut1)
    if o and o["sso"]:  # dawn-dusk: orbit plane rides the day/night line (node at 6 pm local time)
        s0 = sun_dir(np.array([t0.ut1]))[0]
        term = math.degrees(math.atan2(s0[1], s0[0])) + 90
        raans = [term + (q - (PLANES - 1) / 2) * 5 for q in range(PLANES)]
    else:
        raans = [p * 360 / PLANES for p in range(PLANES)]
    lit, saa, names = [], [], []
    for p in range(PLANES):
        for k in range(PER_PLANE):
            s = Satrec()
            s.sgp4init(WGS72, "i", 90000 + p * 100 + k, t0.ut1 - 2433281.5, 0.0, 0.0, 0.0,
                       0.0001, 0.0, math.radians(inc),
                       math.radians(k * 360 / PER_PLANE + p * 360 / (PER_PLANE * PLANES)),  # stagger satellites
                       n_rad_min, math.radians(raans[p]))
            l, z = _flags(EarthSatellite.from_satrec(s, ts), t, sun)
            lit.append(l); saa.append(z)
            names.append(f"{'AI1-' if o else ''}P{p + 1}-S{k + 1}")
    return np.array(lit), np.array(saa), names


def _flags(sat, t, sun):
    """In sunlight? Inside the SAA box? For each time in t."""
    geo = sat.at(t)
    pos = geo.position.km.T
    along = (pos * sun).sum(1)
    perp = (pos ** 2).sum(1) - along ** 2
    lit = (along >= 0) | (perp > R_EARTH ** 2)
    lat, lon = (x.degrees for x in wgs84.latlon_of(geo))
    saa = ((lat > SAA["lat"][0]) & (lat < SAA["lat"][1]) &
           (lon > SAA["lon"][0]) & (lon < SAA["lon"][1]))
    return lit, saa


def build_fleet_real(minutes, tle_path="starlink_fleet12.tle"):
    """Same flags, but for REAL satellites from a CelesTrak TLE file, starting at
    the data's own date. starlink_fleet12.tle = 12 real Starlink satellites from
    the 53-degree shell (~463 km), 3 orbital planes x 4.
    With an AI1 profile there are no real satellites to use, so this returns the
    profile's orbits for days 7-9, which training never sees."""
    import sat_profile
    if sat_profile.IS_AI1:
        return build_fleet(minutes, offset_days=7)
    ts = load.timescale()
    rows = [l.rstrip() for l in open(tle_path) if l.strip()]
    sats = [EarthSatellite(rows[k + 1], rows[k + 2], rows[k].strip(), ts) for k in range(0, len(rows), 3)]
    t0 = max(s.epoch.tt for s in sats)
    t = ts.tt_jd(t0 + np.arange(minutes) / 1440)
    sun = sun_dir(t.ut1)
    lit, saa = zip(*(_flags(s, t, sun) for s in sats))
    return np.array(lit), np.array(saa), [s.name for s in sats]


def shadow_lookahead(lit_row):
    N = len(lit_row)
    to_sh, sh_len, left = np.zeros(N), np.zeros(N), np.zeros(N)
    nxt, nlen, run = N, 35, 0
    for i in range(N - 1, -1, -1):
        if not lit_row[i]:
            run += 1; left[i] = run; nxt, nlen = i, run
        else:
            run = 0; to_sh[i] = nxt - i
        sh_len[i] = nlen
    return to_sh, sh_len, left


class Sat:
    def __init__(self, name, lit, saa, p=SAT):
        self.name, self.lit, self.saa, self.p = name, lit, saa, p
        self.to_sh, self.sh_len, self.sh_left = shadow_lookahead(lit)
        self.batt = p["battery_kwh"] * p["battery_start"]
        self.temp_k = p["temp_start_c"] + 273.15
        self.running = []
        self.min_low, self.min_batt, self.peak = 0, 100.0, 0.0

    def safe_budget(self, i):
        """Step-4 rules: save battery for the next shadow, stay under the throttle point."""
        p = self.p
        floor = p["battery_kwh"] * p["battery_floor"]
        if self.lit[i]:
            need = floor + self.sh_len[i] * (p["bus_kw"] + 15) / 60
            rate = max(0.0, need - self.batt) / max(self.to_sh[i], 1) * 60 / p["charge_eff"]
            b = p["solar_kw"] - p["bus_kw"] - rate
        else:
            b = (self.batt - floor) / (max(self.sh_left[i], 1) / 60) - p["bus_kw"]
        tc = self.temp_k - 273.15
        if tc > p["throttle_at_c"] - 4:
            b *= max(0.0, (p["throttle_at_c"] - tc) / 4)
        return max(0.0, min(b, p["gpu_max_kw"]))

    def physics(self, i, gpu_kw):
        p = self.p
        lit = self.lit[i]
        gen = p["solar_kw"] if lit else 0.0
        use = gpu_kw + p["bus_kw"]
        net = gen - use
        self.batt = min(max(self.batt + (net * p["charge_eff"] if net > 0 else net) / 60, 0.0), p["battery_kwh"])
        heat_in = use * 1000 + (p["sun_heat_kw"] * 1000 if lit else 0)
        heat_out = p["emissivity"] * SIGMA * p["radiator_m2"] * (self.temp_k ** 4 - 3.0 ** 4)
        self.temp_k += (heat_in - heat_out) * 60 / p["heat_capacity_j_per_k"]
        soc = self.batt / p["battery_kwh"] * 100
        self.min_low += soc < p["battery_floor"] * 100
        self.min_batt = min(self.min_batt, soc)
        self.peak = max(self.peak, self.temp_k - 273.15)


def make_jobs(minutes, rate_per_min, rng):
    jobs, jid = [], 0
    for t in range(minutes):
        for _ in range(rng.poisson(rate_per_min)):
            name, kw, mins, sens = JOB_TYPES[rng.integers(3)]
            jobs.append({"id": jid, "type": name, "kw": kw, "mins": mins, "sensitive": sens,
                         "arrive": t, "done_min": 0.0, "state": "future", "sat": None,
                         "hold": 0, "finish": None})
            jid += 1
    return jobs


def simulate(mode, lit, saa, names, weather, minutes, seed):
    rng = np.random.default_rng(seed)
    import sat_profile
    if _os.environ.get("SAT_JOBS") == "helios":  # real AI jobs from the Helios GPU trace
        import workload
        jobs = workload.helios_jobs(minutes, len(names), SAT["gpu_max_kw"], rng)
    else:
        jobs = make_jobs(minutes, 1.3 * sat_profile.job_rate_scale(SAT), rng)
    err_rng = np.random.default_rng(seed + 99)
    import sat_profile
    sats = [Sat(n, lit[s], saa[s], sat_profile.sat_params(SAT, s)) for s, n in enumerate(names)]
    restarts, moves, energy_wasted = 0, 0, 0.0
    delivered = 0.0  # GPU work actually done, finished or not (long real jobs outlast the run)
    by_arrival = sorted(jobs, key=lambda j: j["arrive"])
    nxt = 0
    rr = 0  # round-robin pointer for LOCAL
    fleet_log = []

    busy = [[] for _ in names]  # per satellite: minute each busy laser frees up
    ground_cap = len(names) * LINK["ground_gbps"] * 60 / 8 if LINK else None
    blocked = 0
    for i in range(minutes):
        for b in busy:
            b[:] = [e for e in b if e > i]
        def can_send(src, gb):
            """A free laser on the sending satellite? A big job ties one up for several minutes."""
            return not LINK or src is None or len(busy[src]) < LINK["lasers"]
        def send(src, gb):
            m = _transfer_min(gb)
            if LINK:
                busy[src].append(i + m)
            return m
        # 1. arrivals (inputs come up from the ground, shared fleet-wide link)
        up = 0.0
        while nxt < len(by_arrival) and by_arrival[nxt]["arrive"] <= i:
            if LINK and JOB_IN_GB > 0:
                gb = JOB_IN_GB * by_arrival[nxt]["kw"] / 20
                if up + gb > ground_cap:
                    break  # waits for next minute's uplink
                up += gb
            j = by_arrival[nxt]; nxt += 1
            j["state"] = "queued"
            if mode in ("local", "equinox"):
                j["sat"] = rr % len(sats); rr += 1
        mult_weather = ERROR_MULT[weather]

        # 2. each satellite's safe budget; drop running jobs that no longer fit
        if mode == "equinox":
            qlen = [0] * len(sats)
            for j in jobs:  # the paper's queue holds work deferred within the last 300 s
                if j["state"] == "queued" and j["sat"] is not None and i - j["arrive"] <= 5:
                    qlen[j["sat"]] += 1
            budgets = [eqx_budget(s, i, qlen[si]) for si, s in enumerate(sats)]
        else:
            budgets = [s.safe_budget(i) for s in sats]
        for si, s in enumerate(sats):
            keep, used = [], 0
            for j in sorted(s.running, key=lambda x: x["kw"]):
                if used + j["kw"] <= budgets[si]:
                    keep.append(j); used += j["kw"]
                else:
                    j["state"] = "queued"
                    if mode == "fleet":  # hand it over the laser link, progress kept
                        j["prev"] = si; j["sat"] = None; j["hold"] = 0 if LINK else MOVE_DELAY_MIN
            s.running = keep

        # 3. assign queued jobs
        queued = sorted((j for j in jobs if j["state"] == "queued"), key=lambda j: j["arrive"])
        for j in queued:
            if j["hold"] > 0:
                j["hold"] -= 1
                continue
            if mode in ("local", "equinox"):
                tiers = [[j["sat"]]]
            else:
                allc = list(range(len(sats)))
                clean = [c for c in allc if not sats[c].saa[i]]
                tiers = [clean, allc] if j["sensitive"] else [allc]
            best = None
            for cands in tiers:
                best_spare = None
                for c in cands:
                    src = j.get("prev")
                    if src is not None and c != src and (not can_send(src, _state_gb(j)) or
                            not _worth_moving(j, sats[src], i, _transfer_min(_state_gb(j)))):
                        continue  # sender's lasers are full, or moving costs more than waiting
                    spare = budgets[c] - sum(x["kw"] for x in sats[c].running)
                    # best fit: the satellite whose spare power is closest to the job's size
                    if spare >= j["kw"] and (best_spare is None or spare < best_spare):
                        best, best_spare = c, spare
                if best is not None:
                    break
            if best is not None:
                if j.get("prev") is not None and j["prev"] != best:
                    moves += 1
                    if LINK:
                        j["stall"] = send(j["prev"], _state_gb(j))
                elif j.get("prev") is not None:
                    pass
                j["prev"] = None
                j["sat"] = best; j["state"] = "running"
                sats[best].running.append(j)
            elif mode == "equinox" and not j.get("hopped"):
                # can't run where it is: one hop to the cheapest neighbor, if worth the link cost
                here = j["sat"]
                p_local = eqx_cost(sats[here])
                tgt, p_best = None, None
                for c, t in enumerate(sats):
                    if c == here or budgets[c] - sum(x["kw"] for x in t.running) < j["kw"]:
                        continue
                    p_n = eqx_cost(t)
                    if p_n + 0.05 * p_local + 0.10 < p_local and p_n < EQX_VALUE and (p_best is None or p_n < p_best):
                        tgt, p_best = c, p_n
                if tgt is not None and can_send(here, _state_gb(j)):
                    moves += 1; j["hopped"] = True
                    # a job that hasn't started has nothing to carry over (as for our first placement)
                    if j["done_min"] > 0:
                        j["stall"] = send(here, _state_gb(j)) if LINK else MOVE_DELAY_MIN
                    j["sat"] = tgt; j["state"] = "running"
                    sats[tgt].running.append(j)

        # 3b. FLEET: hand sensitive jobs off satellites inside the SAA, if a clean one has room
        if mode == "fleet":
            for si, s in enumerate(sats):
                if not s.saa[i]:
                    continue
                for j in [x for x in s.running if x["sensitive"]]:
                    target, tspare = None, None
                    for c, t in enumerate(sats):
                        if t.saa[i] or c == si:
                            continue
                        spare = budgets[c] - sum(x["kw"] for x in t.running)
                        if spare >= j["kw"] and (tspare is None or spare < tspare):
                            target, tspare = c, spare
                    if target is not None and not can_send(si, _state_gb(j)):
                        blocked += 1; target = None
                    if target is not None and LINK and SMART_MOVES and _transfer_min(_state_gb(j)) > 1:
                        target = None  # a radiation dodge is only worth a quick move
                    if target is not None:
                        s.running.remove(j); sats[target].running.append(j)
                        j["sat"] = target; j["stall"] = send(si, _state_gb(j)); moves += 1

        # 4. run a minute: progress, bit flips, physics
        total_gpu = 0.0
        for s in sats:
            speed = 0.3 if s.temp_k - 273.15 > s.p["temp_limit_c"] else 1.0
            if s.batt <= 0.5:
                speed = 0.0
            for j in list(s.running):
                if j.get("stall"):  # in transit over the laser link this minute
                    j["stall"] -= 1
                    continue
                if j["sensitive"] and speed > 0 and err_rng.random() < p_error(j["kw"], mult_weather, bool(s.saa[i])):
                    energy_wasted += j["kw"] * j["done_min"] / 60
                    j["done_min"] = 0.0; restarts += 1
                    continue
                j["done_min"] += speed
                delivered += j["kw"] * speed / 60
                if j["done_min"] >= j["mins"]:
                    j["state"] = "done"; j["finish"] = i
                    s.running.remove(j)
            gpu = sum(j["kw"] for j in s.running if not j.get("stall")) * speed
            total_gpu += gpu
            s.physics(i, gpu)
        fleet_log.append({"lit": int(lit[:, i].sum()), "saa": int(saa[:, i].sum()),
                          "gpu_kw": round(total_gpu, 1),
                          "min_batt": round(min(s.batt / s.p["battery_kwh"] * 100 for s in sats), 1)})

    done = [j for j in jobs if j["state"] == "done"]
    waits = [j["finish"] - j["arrive"] - j["mins"] for j in done]
    return {
        "mode": mode,
        "jobs_arrived": len(jobs),
        "jobs_done": len(done),
        "work_kwh": round(sum(j["kw"] * j["mins"] for j in done) / 60, 1),
        "delivered_kwh": round(delivered - energy_wasted, 1),
        "restarts": restarts,
        "wasted_kwh": round(energy_wasted, 1),
        "moves": moves,
        "moves_blocked": blocked,
        "avg_delay_min": round(float(np.mean(waits)), 1) if waits else None,
        "sat_minutes_battery_low": int(sum(s.min_low for s in sats)),
        "lowest_battery_pct": round(min(s.min_batt for s in sats), 1),
        "peak_temp_c": round(max(s.peak for s in sats), 1),
        "log": fleet_log,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=12)
    ap.add_argument("--weather", default="LOW", choices=list(ERROR_MULT))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--real", action="store_true", help="use 12 real Starlink satellites (starlink_fleet12.tle)")
    a = ap.parse_args()
    minutes = int(a.hours * 60)
    if a.real:
        lit, saa, names = build_fleet_real(minutes)
        print(f"{len(names)} REAL Starlink satellites (53 deg shell) | {a.hours} h | weather {a.weather}")
    else:
        lit, saa, names = build_fleet(minutes)
        print(f"{len(names)} satellites, {INC_DEG:.0f} deg / {ALT_KM:.0f} km | {a.hours} h | weather {a.weather}")
    print(f"Avg satellites in sun: {lit.sum(0).mean():.1f}/{len(names)} | in SAA: {saa.sum(0).mean():.1f}\n")
    res = [simulate(m, lit, saa, names, a.weather, minutes, a.seed) for m in ("local", "fleet")]
    keys = ["jobs_arrived", "jobs_done", "work_kwh", "restarts", "wasted_kwh", "moves",
            "avg_delay_min", "sat_minutes_battery_low", "lowest_battery_pct", "peak_temp_c"]
    print(f"{'':26}{'LOCAL':>10}{'FLEET':>10}")
    for k in keys:
        print(f"{k:26}{str(res[0][k]):>10}{str(res[1][k]):>10}")
    with open("fleet_results.json", "w") as f:
        json.dump({"weather": a.weather, "hours": a.hours, "satellites": names, "results": res}, f)
    print("\nSaved fleet_results.json")


if __name__ == "__main__":
    main()

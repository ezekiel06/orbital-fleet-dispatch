"""Step 4: The scheduler, the "brain" that decides which jobs run when.

Compares two approaches on the same orbit, jobs and space weather:
  BASELINE  run everything as fast as possible; throttle only in an emergency
            (that's what step 2 did).
  SMART     plans ahead with what the satellite knows:
            - minutes until the next shadow and how long it lasts (step 1)
            - battery and temperature (step 2)
            - storm risk (step 3)
            It saves enough battery for the shadow, keeps GPUs below the
            throttle point, and during storms sends error-sensitive jobs
            to the ground.

Rules-based for now. A learning version (AI trained on many simulated orbits)
can replace pick_jobs() later without touching anything else.

Usage:  python scheduler.py [--hours 6] [--fake-storm] [--offline]
"""
import argparse, copy, json, random
from power_thermal import SAT, SIGMA, sunlight_timeline
import space_weather

JOB_TYPES = [  # name, GPU kW, minutes, error-sensitive?
    ("Image tagging", 15, 10, False),
    ("Translation batch", 30, 20, True),
    ("Video analysis", 60, 40, True),
]
from radiation import p_error  # v3: calibrated from Google TPU radiation tests


def make_jobs(n=150, seed=7):
    rnd = random.Random(seed)
    jobs = []
    for i in range(n):
        name, kw, mins, sens = rnd.choice(JOB_TYPES)
        jobs.append({"id": f"J-{100+i}", "type": name, "kw": kw, "mins": mins,
                     "sensitive": sens, "done_min": 0.0, "where": "queued"})
    return jobs


def shadow_info(timeline, i):
    """Minutes until next shadow starts, and how long that shadow lasts."""
    n = len(timeline)
    j = i
    while j < n and timeline[j][1]:
        j += 1
    k = j
    while k < n and not timeline[k][1]:
        k += 1
    return j - i, max(k - j, 35 if j >= n else 0)


def pick_jobs(mode, jobs, running, budget_kw, risk, p):
    """Return the list of jobs to run this minute."""
    if mode == "baseline":
        chosen, used = [], 0
        for j in running + [x for x in jobs if x["where"] == "queued"]:
            if used + j["kw"] <= p["gpu_max_kw"]:
                chosen.append(j); used += j["kw"]
        return chosen
    # SMART
    chosen, used = [], 0
    # 1. Storm: send sensitive work to the ground instead of risking errors.
    if risk == "HIGH":
        for j in jobs:
            if j["sensitive"] and j["where"] in ("queued", "running", "paused"):
                j["where"] = "ground"
    # 2. Keep running jobs going first (finishing beats starting), smallest first.
    for j in sorted(running, key=lambda x: x["kw"]):
        if j["where"] != "ground" and used + j["kw"] <= budget_kw:
            chosen.append(j); used += j["kw"]
    # 3. Fill leftover budget. When storm risk is ELEVATED, skip new big sensitive jobs.
    for j in [x for x in jobs if x["where"] in ("queued", "paused") and x not in chosen]:
        if risk == "ELEVATED" and j["sensitive"] and j["kw"] >= 60:
            continue
        if used + j["kw"] <= budget_kw:
            chosen.append(j); used += j["kw"]
    return chosen


def run(mode, timeline, weather, p=SAT, seed=7):
    rnd = random.Random(seed + 1)
    jobs = make_jobs(seed=seed)
    batt = p["battery_kwh"] * p["battery_start"]
    floor = p["battery_kwh"] * p["battery_floor"]
    temp_k = p["temp_start_c"] + 273.15
    running, restarts, minutes_over, min_batt, max_temp = [], 0, 0, 100.0, 0.0
    risk, mult = weather["risk"], weather["error_mult"]
    log = []

    for i, (t, lit) in enumerate(timeline):
        temp_c = temp_k - 273.15
        gen = p["solar_kw"] if lit else 0.0

        # --- How much GPU power can we afford this minute? (SMART only)
        to_shadow, shadow_len = shadow_info(timeline, i)
        if lit:
            need_at_shadow = floor + shadow_len * (p["bus_kw"] + 15) / 60  # keep light work alive in shadow
            short = max(0.0, need_at_shadow - batt)
            charge_rate = short / max(to_shadow, 1) * 60 / p["charge_eff"]
            budget = gen - p["bus_kw"] - charge_rate
        else:
            left = 1
            while i + left < len(timeline) and not timeline[i + left][1]:
                left += 1
            budget = (batt - floor) / (left / 60) - p["bus_kw"]
        if temp_c > p["throttle_at_c"] - 4:  # stay under the throttle point
            budget *= max(0.0, (p["throttle_at_c"] - temp_c) / 4)
        budget = max(0.0, min(budget, p["gpu_max_kw"]))

        chosen = pick_jobs(mode, jobs, running, budget, risk, p)
        for j in jobs:
            if j in running and j not in chosen and j["where"] == "running":
                j["where"] = "paused"
        running = chosen

        # BASELINE emergency throttle (same rule as step 2)
        speed = 1.0
        if mode == "baseline":
            soc = batt / p["battery_kwh"]
            if temp_c > p["throttle_at_c"] or (not lit and soc < p["battery_floor"]):
                speed = 0.3
            if batt <= 0.5:
                speed = 0.0
        gpu_kw = sum(j["kw"] for j in running) * speed

        # --- Progress + radiation errors
        for j in running:
            j["where"] = "running"
            if j["sensitive"] and rnd.random() < p_error(j["kw"], mult):
                j["done_min"] = 0.0; restarts += 1  # bit flip: start over
                continue
            j["done_min"] += speed
            if j["done_min"] >= j["mins"]:
                j["where"] = "done"
        running = [j for j in running if j["where"] == "running"]

        # --- Physics (same as step 2)
        use = gpu_kw + p["bus_kw"]
        net = gen - use
        batt = min(max(batt + (net * p["charge_eff"] if net > 0 else net) / 60, 0.0), p["battery_kwh"])
        heat_in = use * 1000 + (p["sun_heat_kw"] * 1000 if lit else 0)
        heat_out = p["emissivity"] * SIGMA * p["radiator_m2"] * (temp_k ** 4 - 3.0 ** 4)
        temp_k += (heat_in - heat_out) * 60 / p["heat_capacity_j_per_k"]

        min_batt = min(min_batt, batt / p["battery_kwh"] * 100)
        max_temp = max(max_temp, temp_k - 273.15)
        minutes_over += (temp_k - 273.15) > p["temp_limit_c"]
        log.append({"t": t.utc_strftime("%H:%M"), "lit": bool(lit), "gpu_kw": round(gpu_kw, 1),
                    "batt": round(batt / p["battery_kwh"] * 100, 1), "temp": round(temp_k - 273.15, 1)})

    done = [j for j in jobs if j["where"] == "done"]
    return {
        "mode": mode,
        "jobs_done": len(done),
        "work_done_kwh": round(sum(j["kw"] * j["mins"] for j in done) / 60, 1),
        "sent_to_ground": sum(j["where"] == "ground" for j in jobs),
        "error_restarts": restarts,
        "lowest_battery_pct": round(min_batt, 1),
        "peak_temp_c": round(max_temp, 1),
        "minutes_over_limit": int(minutes_over),
        "jobs": jobs, "log": log,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=6)
    ap.add_argument("--fake-storm", action="store_true")
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    weather = space_weather.current(a.fake_storm)
    name, tl = sunlight_timeline(a.hours, a.offline)
    print(f"Orbit: {name} | {a.hours} h | space weather: {weather['risk']} ({weather['source']})\n")
    results = [run("baseline", tl, weather), run("smart", tl, weather)]
    keys = ["jobs_done", "work_done_kwh", "sent_to_ground", "error_restarts",
            "lowest_battery_pct", "peak_temp_c", "minutes_over_limit"]
    print(f"{'':22}{'BASELINE':>10}{'SMART':>10}")
    for k in keys:
        print(f"{k:22}{results[0][k]:>10}{results[1][k]:>10}")
    out = {"weather": weather, "orbit": name,
           "results": [{k: r[k] for k in keys + ['mode', 'log']} |
                       {"jobs": [{x: j[x] for x in ('id', 'type', 'where')} for j in r["jobs"]]}
                       for r in results]}
    with open("scheduler_results.json", "w") as f:
        json.dump(out, f)
    print("\nSaved full run to scheduler_results.json (the dashboard will read this).")


if __name__ == "__main__":
    main()

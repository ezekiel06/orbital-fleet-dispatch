"""Step 25: Does Earth's own heat change the results?

In low orbit, radiators don't only see cold space: Earth's infrared glow
(~234 W/m2, NASA/TM-2001-211221) and reflected sunlight land on them too.
power_thermal.earth_heat_w adds that load; by default the radiators are
edge-on to Earth (each face sees ~24% Earth at 650 km).

This reruns key cases with Earth's heat off (SAT_EARTH_VIEW=0), on by default
(0.24), and the worst case of one face pointed straight down (0.41 averaged
over both faces), and compares the sharing gain, work delivered and peak
temperature. AI1 specs, 53-degree orbit, 6 runs per weather, 12 hours each.

Usage:  python earth_heat_test.py     (~15-25 minutes)
Saves:  earth_heat_results.json
"""
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

VIEWS = {"off": "0", "edge-on (default)": "0.24", "face down": "0.41"}
CASES = [
    ("made-up · 12 sats", {"SAT_FLEET": "3x4"}),
    ("made-up · 12 sats · bursting to 150 kW", {"SAT_FLEET": "3x4", "SAT_SCENARIO": "burst150"}),
    ("made-up · 48 sats", {"SAT_FLEET": "6x8"}),
    ("real jobs · 12 sats", {"SAT_FLEET": "3x4", "SAT_JOBS": "helios", "SAT_LOAD": "1.0"}),
]
WEATHER = ["LOW", "ELEVATED", "HIGH"]
SEEDS = [1, 2, 3, 4, 5, 6]

WORKER = r'''
import json, numpy as np, fleet
lit, saa, names = fleet.build_fleet_real(720)
out = {}
for w in %s:
    L = [fleet.simulate("local", lit, saa, names, w, 720, s) for s in %s]
    F = [fleet.simulate("fleet", lit, saa, names, w, 720, s) for s in %s]
    out[w] = {"gain_pct": float(np.mean([f["delivered_kwh"] / l["delivered_kwh"] - 1 for l, f in zip(L, F)])) * 100,
              "delivered_fleet": float(np.mean([f["delivered_kwh"] for f in F])),
              "peak_temp_c": float(max(r["peak_temp_c"] for r in L + F)),
              "low_batt_min": int(sum(r["sat_minutes_battery_low"] for r in L + F))}
print(json.dumps(out))
''' % (WEATHER, SEEDS, SEEDS)


def run(job):
    (name, extra), (vname, v) = job
    env = dict(os.environ, SAT_PROFILE="ai1_53", OMP_NUM_THREADS="1", SAT_EARTH_VIEW=v, **extra)
    p = subprocess.run([sys.executable, "-c", WORKER], capture_output=True, text=True, env=env)
    if p.returncode:
        sys.exit(f"{name}/{vname}: {p.stderr[-2000:]}")
    return json.loads(p.stdout.strip().splitlines()[-1])


def main():
    jobs = [(c, v) for c in CASES for v in VIEWS.items()]
    results = {}
    with ThreadPoolExecutor(max_workers=2) as ex:
        for ((name, _), (vname, _)), r in zip(jobs, ex.map(run, jobs)):
            results.setdefault(name, {})[vname] = r
            print(f"{name:40} Earth heat {vname:18} " + " | ".join(
                f"{w}: sharing {x['gain_pct']:+.1f}%, work {x['delivered_fleet']:.0f} kWh, peak {x['peak_temp_c']:.1f} C"
                for w, x in r.items()), flush=True)
    json.dump(results, open("earth_heat_results.json", "w"), indent=1)
    print("Saved earth_heat_results.json")


if __name__ == "__main__":
    main()

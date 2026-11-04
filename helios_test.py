"""Step 20: Does sharing still pay on REAL AI workloads?

Replays real deep-learning jobs from the Helios GPU trace (see workload.py)
instead of the made-up job types: real arrival bursts, real GPU counts (1 to
94 GPUs here), real run times (seconds to days).

Compares satellites working alone vs. sharing, on identical jobs and orbits,
for AI1 specs on the 53-degree orbit:
  - 12 and 48 satellites
  - fleet offered 80%, 100% and 120% of its full GPU power
  - all three weather levels
  - 6 different 12-hour slices of the trace per case
  - plus a laser-limited version: Starlink-today links, every moved job
    carrying 80 GB per GPU (a full H100's memory), smart moves on

Measure: GPU work actually delivered (kWh), counting unfinished long jobs'
progress and subtracting work lost to bit-flip restarts.

Usage:  python helios_test.py      (~15-25 minutes)
Saves:  helios_results.json
"""
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

SIZES = {"12": "3x4", "48": "6x8"}
LOADS = [0.8, 1.0, 1.2]
WEATHER = ["LOW", "ELEVATED", "HIGH"]
SEEDS = [1, 2, 3, 4, 5, 6]
GB_PER_20KW = 80 * 20 / 1.275   # 80 GB per GPU, ~15.7 GPUs per 20 kW

WORKER = r'''
import json, numpy as np, fleet, workload
lit, saa, names = fleet.build_fleet_real(720)
out = {}
for w in %s:
    L = [fleet.simulate("local", lit, saa, names, w, 720, s) for s in %s]
    F = [fleet.simulate("fleet", lit, saa, names, w, 720, s) for s in %s]
    g = [f["delivered_kwh"] / l["delivered_kwh"] - 1 for l, f in zip(L, F)]
    out[w] = {"gain_pct": float(np.mean(g)) * 100, "min_pct": float(min(g)) * 100, "max_pct": float(max(g)) * 100,
              "wins": int(sum(x > 0 for x in g)), "runs": len(g),
              "delay_local": float(np.mean([l["avg_delay_min"] for l in L])), "delay_fleet": float(np.mean([f["avg_delay_min"] for f in F])),
              "low_batt_min": int(max(sum(r["sat_minutes_battery_low"] for r in L), sum(r["sat_minutes_battery_low"] for r in F)))}
out["_stats"] = workload.STATS
print(json.dumps(out))
''' % (WEATHER, SEEDS, SEEDS)


def run(size, load, link):
    env = dict(os.environ, SAT_PROFILE="ai1_53", SAT_FLEET=SIZES[size], SAT_JOBS="helios", SAT_LOAD=str(load),
               SAT_LINK=link, SAT_JOB_GB=str(GB_PER_20KW), SAT_MOVE="smart", OMP_NUM_THREADS="1")
    p = subprocess.run([sys.executable, "-c", WORKER], capture_output=True, text=True, env=env)
    if p.returncode:
        sys.exit(p.stderr[-2000:])
    return json.loads(p.stdout.strip().splitlines()[-1])


def main():
    cases = [(n, ld, "none") for n in SIZES for ld in LOADS] + [(n, 1.0, "today") for n in SIZES]
    results = {}
    with ThreadPoolExecutor(max_workers=2) as ex:
        for (n, ld, lk), r in zip(cases, ex.map(lambda a: run(*a), cases)):
            results[f"{n}|{ld}|{lk}"] = r
            print(f"{n:>3} sats  load {ld:.0%}  links {lk:5}  " + " | ".join(
                f"{w}: {v['gain_pct']:+.1f}% ({v['min_pct']:+.1f} to {v['max_pct']:+.1f}, won {v['wins']}/{v['runs']})"
                for w, v in r.items() if not w.startswith("_")), flush=True)
    json.dump(results, open("helios_results.json", "w"), indent=1)
    print("Saved helios_results.json")


if __name__ == "__main__":
    main()

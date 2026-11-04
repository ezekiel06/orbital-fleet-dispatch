"""Step 24: Our dispatcher vs. Equinox, a published orbital scheduler.

Equinox (Erol & Mahajan, Georgia Tech, arXiv:2604.19958, April 2026) prices
each satellite's work from its battery, temperature and queue, runs work only
when it's worth that price, and hops leftover jobs to a cheaper neighbor over
a laser link. It's rebuilt in fleet.py (mode "equinox"), with every unclear
choice set in its favor (see the comment there).

Same orbits, same jobs, same weather for all three:
  local    each satellite alone, with our battery-saving rules
  ours     shared queue, best fit, SAA dodging (fleet.py mode "fleet")
  equinox  Equinox's cost-based admission + one-hop offloading

Cases (AI1 specs, 53-degree orbit, 6 runs per weather, 12 hours each):
  made-up jobs: 12 and 48 satellites, comfortable and aging fleets
  real AI jobs (Helios trace, demand = 100% of GPU power): 12 and 48 satellites
  sensitivity: Equinox's job value at 2 and 8 instead of 4 (12 satellites)

Measure: GPU work actually delivered (kWh), minus work lost to restarts.

Usage:  python equinox_test.py     (~15-30 minutes; real-job cases need helios_jobs.npz)
Saves:  equinox_results.json
"""
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

WEATHER = ["LOW", "ELEVATED", "HIGH"]
SEEDS = [1, 2, 3, 4, 5, 6]
CASES = [  # name, env
    ("made-up · 12 sats", {"SAT_FLEET": "3x4"}),
    ("made-up · 48 sats", {"SAT_FLEET": "6x8"}),
    ("made-up · 12 sats · aging", {"SAT_FLEET": "3x4", "SAT_SCENARIO": "aging"}),
    ("made-up · 48 sats · aging", {"SAT_FLEET": "6x8", "SAT_SCENARIO": "aging"}),
    ("real jobs · 12 sats", {"SAT_FLEET": "3x4", "SAT_JOBS": "helios", "SAT_LOAD": "1.0"}),
    ("real jobs · 48 sats", {"SAT_FLEET": "6x8", "SAT_JOBS": "helios", "SAT_LOAD": "1.0"}),
    ("made-up · 12 sats · Equinox value 2", {"SAT_FLEET": "3x4", "EQX_VALUE": "2"}),
    ("made-up · 12 sats · Equinox value 8", {"SAT_FLEET": "3x4", "EQX_VALUE": "8"}),
]

WORKER = r'''
import json, numpy as np, fleet
lit, saa, names = fleet.build_fleet_real(720)
out = {}
for w in %s:
    R = {m: [fleet.simulate(m, lit, saa, names, w, 720, s) for s in %s] for m in ("local", "fleet", "equinox")}
    d = {m: [r["delivered_kwh"] for r in v] for m, v in R.items()}
    g = lambda a, b: [x / y - 1 for x, y in zip(d[a], d[b])]
    out[w] = {"ours_vs_local": float(np.mean(g("fleet", "local"))) * 100,
              "equinox_vs_local": float(np.mean(g("equinox", "local"))) * 100,
              "ours_vs_equinox": float(np.mean(g("fleet", "equinox"))) * 100,
              "ours_wins": int(sum(x > 0 for x in g("fleet", "equinox"))), "runs": len(d["fleet"]),
              "low_batt_min": {m: int(sum(r["sat_minutes_battery_low"] for r in v)) for m, v in R.items()},
              "delay": {m: float(np.mean([r["avg_delay_min"] for r in v])) for m, v in R.items()},
              "moves": {m: float(np.mean([r["moves"] for r in v])) for m, v in R.items()}}
print(json.dumps(out))
''' % (WEATHER, SEEDS)


def run(case):
    name, extra = case
    env = dict(os.environ, SAT_PROFILE="ai1_53", OMP_NUM_THREADS="1", **extra)
    p = subprocess.run([sys.executable, "-c", WORKER], capture_output=True, text=True, env=env)
    if p.returncode:
        sys.exit(f"{name}: {p.stderr[-2000:]}")
    return json.loads(p.stdout.strip().splitlines()[-1])


def main():
    results = {}
    with ThreadPoolExecutor(max_workers=2) as ex:
        for (name, _), r in zip(CASES, ex.map(run, CASES)):
            results[name] = r
            print(f"{name:38} " + " | ".join(
                f"{w}: ours {v['ours_vs_local']:+.1f}%, Equinox {v['equinox_vs_local']:+.1f}% vs alone; "
                f"ours vs Equinox {v['ours_vs_equinox']:+.1f}% ({v['ours_wins']}/{v['runs']})" for w, v in r.items()), flush=True)
    json.dump(results, open("equinox_results.json", "w"), indent=1)
    print("Saved equinox_results.json")


if __name__ == "__main__":
    main()

"""Step 18: Does sharing still pay at bigger fleet sizes?

Runs the AI1 53-degree setup at 12, 48, and 96 satellites (same orbit shell,
more planes and more satellites per plane), with demand scaled to fleet size,
and compares satellites working alone vs. sharing over laser links. Covers all
three space-weather levels, on comfortable hardware and on an aging fleet
(one in three satellites worn: battery at 60%, solar at 85%).

Usage:  python scale_test.py          (takes ~20-30 minutes; 96 satellites is slow)
Saves:  scale_results.json
"""
import json, os, subprocess, sys

SIZES = {"12": "3x4", "48": "6x8", "96": "8x12"}
WEATHER = ["LOW", "ELEVATED", "HIGH"]
SEEDS = [1, 2, 3, 4]

WORKER = r'''
import json, sys, numpy as np, fleet
lit, saa, names = fleet.build_fleet_real(720)
out = {}
for w in %s:
    L = [fleet.simulate("local", lit, saa, names, w, 720, s) for s in %s]
    F = [fleet.simulate("fleet", lit, saa, names, w, 720, s) for s in %s]
    m = lambda R, k: float(np.mean([r[k] for r in R]))
    out[w] = {"local_work": m(L, "work_kwh"), "fleet_work": m(F, "work_kwh"),
              "local_delay": m(L, "avg_delay_min"), "fleet_delay": m(F, "avg_delay_min"),
              "gains": [f["work_kwh"] / l["work_kwh"] - 1 for l, f in zip(L, F)],
              "low_batt_min": max(m(L, "sat_minutes_battery_low"), m(F, "sat_minutes_battery_low"))}
print(json.dumps(out))
''' % (WEATHER, SEEDS, SEEDS)


def main():
    results = {}
    for setup in ("", "aging"):
        for n, fleet_shape in SIZES.items():
            env = dict(os.environ, SAT_PROFILE="ai1_53", SAT_FLEET=fleet_shape, SAT_SCENARIO=setup)
            p = subprocess.run([sys.executable, "-c", WORKER], capture_output=True, text=True, env=env)
            if p.returncode:
                sys.exit(p.stderr[-2000:])
            r = json.loads(p.stdout.strip().splitlines()[-1])
            results[f"{setup or 'comfortable'}_{n}"] = r
            line = " | ".join(f"{w}: {(v['fleet_work'] / v['local_work'] - 1) * 100:+.1f}% "
                              f"(delay {v['local_delay']:.0f}->{v['fleet_delay']:.0f} min)" for w, v in r.items())
            print(f"{setup or 'comfortable':12} {n:>3} satellites  {line}", flush=True)
    json.dump(results, open("scale_results.json", "w"), indent=1)
    print("Saved scale_results.json")


if __name__ == "__main__":
    main()

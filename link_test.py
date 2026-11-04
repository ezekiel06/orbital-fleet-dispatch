"""Step 19: Does sharing still pay once laser links have real speed limits?

Moving a job between satellites means sending its working state (model, batch,
progress) over a laser link. This test limits each satellite's lasers to real
speeds and makes jobs carry more and more data, then checks how much of the
sharing gain survives.

Link speeds (see fleet.py, SAT_LINK):
  today  3 lasers x 100 Gbps (Starlink's reported sustained rate per link)
  v3     6 lasers x 400 Gbps (reported Starlink V3)
  slow   1 laser  x 10 Gbps  (older commercial laser links)
Data per 20 kW job moved (SAT_JOB_GB): 10 GB (inference batch, model already
on board) up to 5,000 GB (a full training checkpoint).

AI1 specs, 53-degree orbit, 12 and 48 satellites, all three weather levels.

Usage:  python link_test.py          (~20-40 minutes)
Saves:  link_results.json
"""
import json, os, subprocess, sys

SIZES = {"12": "3x4", "48": "6x8"}
CASES = [("none", 10), ("nomove", 10)] + [(l, gb) for l in ("v3", "today", "slow") for gb in (10, 100, 1000, 5000)]
WEATHER = ["LOW", "ELEVATED", "HIGH"]
SEEDS = [1, 2, 3]

WORKER = r'''
import json, numpy as np, fleet
lit, saa, names = fleet.build_fleet_real(720)
out = {}
for w in %s:
    L = [fleet.simulate("local", lit, saa, names, w, 720, s)["work_kwh"] for s in %s]
    F = [fleet.simulate("fleet", lit, saa, names, w, 720, s) for s in %s]
    out[w] = {"gain_pct": float(np.mean([f["work_kwh"] / l - 1 for l, f in zip(L, F)]) * 100),
              "moves": float(np.mean([f["moves"] for f in F])),
              "blocked": float(np.mean([f["moves_blocked"] for f in F])),
              "delay": float(np.mean([f["avg_delay_min"] for f in F]))}
print(json.dumps(out))
''' % (WEATHER, SEEDS, SEEDS)


def run(size, link, gb):
    env = dict(os.environ, SAT_PROFILE="ai1_53", SAT_FLEET=SIZES[size], SAT_LINK=link, SAT_JOB_GB=str(gb),
               OMP_NUM_THREADS="1")
    p = subprocess.run([sys.executable, "-c", WORKER], capture_output=True, text=True, env=env)
    if p.returncode:
        sys.exit(p.stderr[-2000:])
    return json.loads(p.stdout.strip().splitlines()[-1])


def main():
    from concurrent.futures import ThreadPoolExecutor
    jobs = [(n, l, gb) for n in SIZES for l, gb in CASES]
    results = {}
    with ThreadPoolExecutor(max_workers=2) as ex:
        for (n, l, gb), r in zip(jobs, ex.map(lambda a: run(*a), jobs)):
            results[f"{n}|{l}|{gb}"] = r
            print(f"{n:>3} satellites  link {l:5}  {gb:>5} GB/job  " +
                  " | ".join(f"{w}: {v['gain_pct']:+.1f}% (moves {v['moves']:.0f}, blocked {v['blocked']:.0f})" for w, v in r.items()),
                  flush=True)
    json.dump(results, open("link_results.json", "w"), indent=1)
    print("Saved link_results.json")


if __name__ == "__main__":
    main()

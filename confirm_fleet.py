"""Step 11: Is the fleet AI's edge real, or luck?

Plays every dispatcher on the SAME runs (same orbit slice, same jobs, same
bit-flip dice) on the 12 real Starlink satellites, then compares them run by
run ("paired" comparison). For each weather it reports the AI's average edge
over each rule with a 95% range (bootstrap: resample the runs 10,000 times).
If that range stays above zero, the edge is unlikely to be luck.

It also tests all three trained AIs (seeds 0, 1, 2), not just the one we
picked, so one lucky model can't fake the result.

Rules compared:
  fleet    plain fleet dispatcher
  strict   always pull sensitive jobs out of the radiation zone
  switch   the simple weather rule: fleet normally, strict in a big storm

Usage:  python confirm_fleet.py [--episodes 40]
Saves:  fleet_confirm.json
"""
import argparse, json, os
import numpy as np
from stable_baselines3 import PPO
import fleet
from fleet_rl import FleetEnv, play, EP_MIN, RISKS


def ci(diffs, n=10_000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(diffs)
    boots = rng.choice(d, size=(n, len(d)), replace=True).mean(1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    a = ap.parse_args()
    os.environ.setdefault("SAT_JOBS_PART", "test")  # real-job runs: test on weeks training never saw
    lit, saa, names = fleet.build_fleet_real(2 * 1440 + EP_MIN + 61)
    starts = np.random.default_rng(2026).integers(0, 2 * 1440, size=a.episodes)
    models = {"ai": PPO.load("fleet_ai")}  # the run picked on validation (day 6)
    models.update({f"ai_seed{s}": PPO.load(f"fleet_ai_seed{s}") for s in (0, 1, 2)})
    env = FleetEnv(lit, saa, names, "train", mode="ai")
    out = {}
    for w in RISKS:
        per = {}
        for p in ["fleet", "strict"]:
            per[p] = [play(env, p, None, int(s), w, 5000 + k) for k, s in enumerate(starts)]
        per["switch"] = per["strict"] if w == "HIGH" else per["fleet"]
        for m, mdl in models.items():
            per[m] = [play(env, "ai", mdl, int(s), w, 5000 + k) for k, s in enumerate(starts)]
        res = {"mean": {p: {k: round(float(np.mean([r[k] for r in v])), 1) for k in ("score", "work_kwh")}
                        for p, v in per.items()}, "edge": {}}
        for m in models:
            for rule in ("fleet", "strict", "switch"):
                for k in ("score", "work_kwh"):
                    diffs = [x[k] - y[k] for x, y in zip(per[m], per[rule])]
                    mean, lo, hi = ci(diffs)
                    res["edge"][f"{m}_vs_{rule}_{k}"] = {
                        "mean": round(mean, 1), "low": round(lo, 1), "high": round(hi, 1),
                        "ai_wins_runs": int(sum(d > 0 for d in diffs)), "runs": len(diffs)}
        out[w] = res
        print(f"\n=== {w}: {a.episodes} paired runs on {__import__('sat_profile').LABEL if __import__('sat_profile').IS_AI1 else '12 real Starlink satellites'} ===")
        print("avg score: " + ", ".join(f"{p} {v['score']}" for p, v in res["mean"].items()))
        for m in models:
            e = res["edge"][f"{m}_vs_switch_score"]
            ew = res["edge"][f"{m}_vs_switch_work_kwh"]
            verdict = "REAL edge" if e["low"] > 0 else ("REAL loss" if e["high"] < 0 else "too close to call")
            print(f"  {m} vs weather-switch rule: score {e['mean']:+.1f} (95% range {e['low']:+.1f} to {e['high']:+.1f}), "
                  f"won {e['ai_wins_runs']}/{e['runs']} runs, work {ew['mean']:+.0f} kWh -> {verdict}")
    with open("fleet_confirm.json", "w") as f:
        json.dump({"episodes": a.episodes, "orbit": "real Starlink, 53 deg shell", "results": out}, f, indent=1)
    print("\nSaved fleet_confirm.json")


if __name__ == "__main__":
    main()

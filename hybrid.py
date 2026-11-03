"""Step 12: The hybrid dispatcher. Use each tool where it proved best.

v3: the plan is no longer hand-picked. For each weather level it uses the AI
only if Step 11 (confirm_fleet.py) showed a REAL edge over the best simple rule
(whole 95% range above zero); otherwise it keeps the rule.

The weather level comes from step 3 (live NOAA data), which a real operator has.

To avoid fooling ourselves, this is tested on FRESH runs (different orbit
slices, jobs and bit-flip dice) than the ones used to design it, against the
best simple rule (fleet normally, strict in a big storm).

Usage:  python hybrid.py [--episodes 40]
Saves:  hybrid_results.json
"""
import argparse, json
import numpy as np
from stable_baselines3 import PPO
import fleet
from fleet_rl import FleetEnv, play, EP_MIN, RISKS
from confirm_fleet import ci

AI_MODEL = "fleet_ai"  # the run picked on validation


CONF = json.load(open("fleet_confirm.json"))["results"]


def rule_for(weather):
    """The best simple rule for this weather, picked on the Step 11 runs:
    whichever of plain fleet or strict scored higher there."""
    m = CONF[weather]["mean"]
    return "strict" if m["strict"]["score"] > m["fleet"]["score"] else "fleet"


def build_plan():
    """Use the AI only in weather where Step 11 showed a REAL edge over that
    best rule (whole 95% range above zero). Everywhere else, keep the rule."""
    return {w: ("ai" if CONF[w]["edge"][f"ai_vs_{rule_for(w)}_score"]["low"] > 0 else rule_for(w)) for w in RISKS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    a = ap.parse_args()
    lit, saa, names = fleet.build_fleet_real(2 * 1440 + EP_MIN + 61)
    starts = np.random.default_rng(777).integers(0, 2 * 1440, size=a.episodes)  # fresh, not Step 11's
    model = PPO.load(AI_MODEL)
    plan = build_plan()
    print("Plan from the confirmation test:", plan)
    env = FleetEnv(lit, saa, names, "train", mode="ai")
    env_local = FleetEnv(lit, saa, names, "train", mode="local")
    out, tot = {}, {"hybrid": [], "switch": [], "local": []}
    for w in RISKS:
        runs = {}
        runs["local"] = [play(env_local, "local", None, int(s), w, 8000 + k) for k, s in enumerate(starts)]
        runs["switch"] = [play(env, rule_for(w), None, int(s), w, 8000 + k)
                          for k, s in enumerate(starts)]
        hp = plan[w]
        runs["hybrid"] = [play(env, hp, model if hp == "ai" else None, int(s), w, 8000 + k)
                          for k, s in enumerate(starts)]
        res = {"policy_used": hp, "mean": {}, "edge_vs_switch": {}, "edge_vs_local": {}}
        for p, v in runs.items():
            res["mean"][p] = {k: round(float(np.mean([r[k] for r in v])), 1)
                              for k in ("score", "work_kwh", "avg_delay_min", "restarts", "low_batt_min")}
            tot[p] += v
        for k in ("score", "work_kwh"):
            for base, key in (("switch", "edge_vs_switch"), ("local", "edge_vs_local")):
                d = [x[k] - y[k] for x, y in zip(runs["hybrid"], runs[base])]
                m, lo, hi = ci(d)
                res[key][k] = {"mean": round(m, 1), "low": round(lo, 1), "high": round(hi, 1),
                               "wins": int(sum(x > 0 for x in d)), "ties": int(sum(x == 0 for x in d)), "runs": len(d)}
        out[w] = res
        e, ew = res["edge_vs_switch"]["score"], res["edge_vs_switch"]["work_kwh"]
        print(f"{w:9} hybrid uses {hp:6} | vs best rule: score {e['mean']:+6.1f} ({e['low']:+.1f} to {e['high']:+.1f}), "
              f"work {ew['mean']:+5.0f} kWh, won {e['wins']}/{e['runs']} (tied {e['ties']}) | battery under floor: "
              f"{res['mean']['hybrid']['low_batt_min']} min")
    # whole-year mix: assume weather is calm 50%, moderate 25%, big storm 25% of the time (the training mix)
    mix = {"LOW": 0.5, "ELEVATED": 0.25, "HIGH": 0.25}
    blend = lambda p, k: sum(mix[w] * out[w]["mean"][p][k] for w in RISKS)
    summary = {p: {k: round(blend(p, k), 1) for k in ("score", "work_kwh")} for p in ("local", "switch", "hybrid")}
    gain_rule = (summary["hybrid"]["work_kwh"] / summary["switch"]["work_kwh"] - 1) * 100
    gain_local = (summary["hybrid"]["work_kwh"] / summary["local"]["work_kwh"] - 1) * 100
    print(f"\nWeather mix (50% calm / 25% moderate / 25% big storm), work per 6 h for 12 satellites:")
    print(f"  local {summary['local']['work_kwh']:.0f} kWh | best rule {summary['switch']['work_kwh']:.0f} | hybrid "
          f"{summary['hybrid']['work_kwh']:.0f}  ->  +{gain_rule:.1f}% vs best rule, +{gain_local:.1f}% vs no sharing")
    with open("hybrid_results.json", "w") as f:
        json.dump({"episodes": a.episodes, "ai_model": AI_MODEL, "plan": plan, "results": out, "weather_mix": mix,
                   "blended": summary, "gain_vs_rule_pct": round(gain_rule, 2),
                   "gain_vs_local_pct": round(gain_local, 2)}, f, indent=1)
    print("Saved hybrid_results.json")


if __name__ == "__main__":
    main()

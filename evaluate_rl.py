"""Step 5c: Test the trained AI(s) against the baseline and the step-4 rules.

All controllers get the exact same held-out orbits (days 11-13, never seen in
training), the same job mixes and the same storms.

Controllers:
  baseline  always full power
  rules     the step-4 hand-written planner
  ai        the learning AI (sat_scheduler_ppo.zip)
  ai_safe   the learning AI with a hard safety limit (sat_scheduler_ppo_safe.zip),
            included automatically if that file exists

Usage:  python evaluate_rl.py [--episodes 20]
Saves:  rl_results.json  and rebuilds dashboard.html (open it in any browser)
"""
import argparse, json, os
import numpy as np
from stable_baselines3 import PPO
from rl_env import OrbitalComputeEnv, baseline_action, rules_action, RISKS, EPISODE_MIN
from power_thermal import SAT
from train_rl import get_sunlight

METRICS = ["work_kwh", "jobs_done", "restarts", "ground", "min_low_batt", "min_batt", "peak_temp", "score"]
LABELS = {"work_kwh": "Work done (kWh)", "jobs_done": "Jobs finished", "restarts": "Bit-flip restarts",
          "ground": "Sent to ground", "min_low_batt": "Min. under floor", "min_batt": "Lowest battery %",
          "peak_temp": "Peak temp C", "score": "Overall score"}


def play(env, controller, models, start, risk, seed):
    env.safety_cap = controller == "ai_safe"
    obs, _ = env.reset(seed=seed, options={"start": start, "risk_schedule": [risk] * EPISODE_MIN})
    score, done = 0.0, False
    while not done:
        if controller in models:
            act, _ = models[controller].predict(obs, deterministic=True)
        else:
            act = (baseline_action if controller == "baseline" else rules_action)(env)
        obs, r, done, _, _ = env.step(act)
        score += r
    return dict(env.stats, score=score), env.log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20, help="per weather level")
    a = ap.parse_args()
    lit = get_sunlight()
    env = OrbitalComputeEnv(lit, split="test")
    models = {}
    if os.path.exists("sat_scheduler_ppo.zip"):
        models["ai"] = PPO.load("sat_scheduler_ppo")
    if os.path.exists("sat_scheduler_ppo_safe.zip"):
        models["ai_safe"] = PPO.load("sat_scheduler_ppo_safe")
    controllers = ["baseline", "rules"] + list(models)
    rng = np.random.default_rng(123)
    starts = rng.integers(*env.range, size=a.episodes)

    results, samples = {}, {}
    for risk in RISKS:
        results[risk] = {}
        for c in controllers:
            runs = []
            for k, s in enumerate(starts):
                stats, log = play(env, c, models, int(s), risk, seed=1000 + k)
                runs.append(stats)
                if k == 0:
                    samples[f"{c}_{risk}"] = log
            results[risk][c] = {m: round(float(np.mean([r[m] for r in runs])), 1) for m in METRICS}

    for risk in RISKS:
        print(f"\n=== Space weather: {risk}  (avg of {a.episodes} unseen 6-hour runs) ===")
        print(f"{'':22}" + "".join(f"{c.upper():>10}" for c in controllers))
        for m in METRICS:
            print(f"{LABELS[m]:22}" + "".join(f"{results[risk][c][m]:>10}" for c in controllers))
    with open("rl_results.json", "w") as f:
        json.dump({"averages": results, "sample_runs": samples, "episodes_per_level": a.episodes,
                   "controllers": controllers, "battery_floor_pct": round(SAT["battery_floor"] * 100)}, f)
    print("\nSaved rl_results.json")

    import make_dashboard
    make_dashboard.build()


if __name__ == "__main__":
    main()

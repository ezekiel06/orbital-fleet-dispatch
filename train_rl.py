"""Step 5b: Train the learning AI scheduler.

Uses PPO (a standard reinforcement-learning method) from Stable-Baselines3.
The "brain" is a small neural network: 11 inputs -> two layers of 64 -> actions.
It plays thousands of random 6-hour episodes and keeps what scores well.

Training is a bit random: two runs can learn different habits. So this trains
several runs (different "seeds", i.e. random starting points), scores each one
on validation orbits (days 9-10, never trained on), and keeps the best.
The final test in evaluate_rl.py uses yet another set of orbits (days 11-13).

Usage:  pip install stable-baselines3 gymnasium skyfield
        python train_rl.py                 # 3 runs, keeps the best (~15-25 min)
        python train_rl.py --seeds 1       # quick single run
        python train_rl.py --safe          # safety-limited version
Saves:  sat_scheduler_ppo.zip  (or sat_scheduler_ppo_safe.zip with --safe)
"""
import argparse, os, shutil
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from rl_env import OrbitalComputeEnv, build_sunlight, RISKS, EPISODE_MIN


def get_sunlight():
    if os.path.exists("sunlight_14d.npy"):
        return np.load("sunlight_14d.npy")
    lit = build_sunlight()
    np.save("sunlight_14d.npy", lit)
    return lit


def train_one(lit, seed, steps, out, safe=False):
    env = make_vec_env(lambda: OrbitalComputeEnv(lit, split="train", safety_cap=safe), n_envs=4, seed=seed)
    model = PPO("MlpPolicy", env, n_steps=720, batch_size=360, gamma=0.995,
                learning_rate=3e-4, ent_coef=0.01, verbose=0, seed=seed)
    chunk = 100_000
    for done in range(0, steps, chunk):
        model.learn(chunk, reset_num_timesteps=False)
        ep = [e["r"] for e in model.ep_info_buffer]
        print(f"  run {seed}: {done + chunk:>8} steps | avg training score {np.mean(ep):7.1f}", flush=True)
    model.save(out)
    return out


def validate(lit, path, episodes=12, safe=False):
    """Average score on validation orbits across all three weather levels."""
    model = PPO.load(path)
    env = OrbitalComputeEnv(lit, split="val", safety_cap=safe)
    rng = np.random.default_rng(99)
    starts = rng.integers(*env.range, size=episodes)
    total = []
    for risk in RISKS:
        for k, s in enumerate(starts):
            obs, _ = env.reset(seed=500 + k, options={"start": int(s), "risk_schedule": [risk] * EPISODE_MIN})
            score, done = 0.0, False
            while not done:
                act, _ = model.predict(obs, deterministic=True)
                obs, r, done, _, _ = env.step(act)
                score += r
            total.append(score)
    return float(np.mean(total))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600_000, help="per run")
    ap.add_argument("--seeds", type=int, default=3, help="how many runs to try")
    ap.add_argument("--seed", type=int, help="train only this one seed (no picking)")
    ap.add_argument("--safe", action="store_true",
                    help="safety-limited AI: never above the rules' safe power level")
    a = ap.parse_args()
    lit = get_sunlight()
    tag = "_safe" if a.safe else ""
    final = f"sat_scheduler_ppo{tag}.zip"

    if a.seed is not None:
        train_one(lit, a.seed, a.steps, f"run{tag}_seed{a.seed}", a.safe)
        print(f"validation score: {validate(lit, f'run{tag}_seed{a.seed}.zip', safe=a.safe):.1f}")
        return

    scores = {}
    for s in range(a.seeds):
        print(f"Training run {s + 1} of {a.seeds}")
        path = train_one(lit, s, a.steps, f"run{tag}_seed{s}", a.safe) + ".zip"
        scores[path] = validate(lit, path, safe=a.safe)
        print(f"  run {s}: validation score {scores[path]:.1f}\n")
    best = max(scores, key=scores.get)
    shutil.copy(best, final)
    print(f"Best run: {best} ({scores[best]:.1f}) -> saved as {final}")


if __name__ == "__main__":
    main()

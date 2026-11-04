"""Step 7: Train an AI to run the fleet's dispatcher.

The step-6 fleet faces a trade-off: use every kW of power right now, or keep
some room free on "clean" satellites (outside the South Atlantic Anomaly) so
sensitive jobs can be moved out of the radiation zone. The right answer depends
on the weather and how much sensitive work is running. So the AI decides,
every minute, three dispatcher settings:

  1. SAA placement: may new sensitive jobs start on satellites inside the SAA? (yes/no)
  2. Handoff reserve: how much power (0 / 60 / 120 / 240 kW) to keep free on
     clean satellites for jobs fleeing the SAA
  3. Forced exit: when a satellite enters the SAA and no clean one has room,
     pause its sensitive jobs anyway (yes/no)

The packing itself (best fit) and each satellite's safety limit are the same
as step 6. The AI only steers the dispatcher, it can't override battery safety.

Fixed settings to compare against:
  local    no sharing (step 6 LOCAL)
  fleet    always [allow, 0 kW, no]    (step 6 FLEET)
  strict   always [never, 0 kW, yes]   (forced SAA exit, the "cautious" version)

9 days of orbits: days 0-5 train, day 6 picks the best run, days 7-8 final test.

Usage:  python fleet_rl.py train [--steps 300000] [--seed 0]
        python fleet_rl.py eval          # writes fleet_eval.json, rebuilds dashboard
        python fleet_rl.py eval --real   # same, on 12 real Starlink satellites
"""
import argparse, json, os, sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import fleet
from fleet import Sat, SAA, MOVE_DELAY_MIN, JOB_TYPES
from radiation import p_error
from space_weather import ERROR_MULT

DAYS, EP_MIN = 9, 360
RESERVES = [0, 60, 120, 240]
RISKS = ["LOW", "ELEVATED", "HIGH"]
SPLITS = {"train": (0, 6), "val": (6, 7), "test": (7, 9)}
FIXED = {"fleet": (0, 0, 0), "strict": (1, 0, 1)}
LOOKAHEAD = 10  # minutes


def get_orbits():
    if os.path.exists("fleet_orbits_9d.npz"):
        d = np.load("fleet_orbits_9d.npz")
        return d["lit"], d["saa"], list(d["names"])
    lit, saa, names = fleet.build_fleet(DAYS * 1440)
    np.savez_compressed("fleet_orbits_9d.npz", lit=lit, saa=saa, names=np.array(names))
    return lit, saa, names


class FleetEnv(gym.Env):
    def __init__(self, lit, saa, names, split="train", mode="ai"):
        super().__init__()
        self.lit, self.saa, self.names, self.mode = lit, saa, names, mode
        lo, hi = SPLITS[split]
        self.range = (lo * 1440, hi * 1440 - EP_MIN - 1)
        # SAA "entering soon" lookahead, per satellite per minute
        n = saa.shape[1]
        soon = np.zeros_like(saa)
        for k in range(1, LOOKAHEAD + 1):
            soon[:, :n - k] |= saa[:, k:]
        self.soon = soon & ~saa
        self.action_space = spaces.MultiDiscrete([2, len(RESERVES), 2])
        self.observation_space = spaces.Box(-1.0, 5.0, shape=(12,), dtype=np.float32)

    # ---------- setup
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        o = options or {}
        self.i0 = int(o.get("start", self.np_random.integers(*self.range)))
        self.weather = o.get("weather") or RISKS[self.np_random.choice(3, p=[0.5, 0.25, 0.25])]
        rng = np.random.default_rng(int(self.np_random.integers(1 << 30)))
        import sat_profile
        self.jobs = fleet.make_jobs(EP_MIN, 1.3 * sat_profile.job_rate_scale(fleet.SAT), rng)
        self.err_rng = np.random.default_rng(int(self.np_random.integers(1 << 30)))
        self.sats = []
        for s, nm in enumerate(self.names):
            sat = Sat(nm, self.lit[s, self.i0:self.i0 + EP_MIN + 60], self.saa[s, self.i0:self.i0 + EP_MIN + 60],
                      __import__("sat_profile").sat_params(fleet.SAT, s))
            self.sats.append(sat)
        self.future = sorted(self.jobs, key=lambda j: j["arrive"])
        self.queue, self.nxt, self.rr, self.t = [], 0, 0, 0
        self.stats = {"work_kwh": 0.0, "jobs_done": 0, "restarts": 0, "wasted_kwh": 0.0, "moves": 0,
                      "delays": [], "low_batt_min": 0}
        self._prepare()
        return self._obs(), {}

    def _prepare(self):
        """Arrivals + each satellite's safe budget for the coming minute."""
        i = self.t
        while self.nxt < len(self.future) and self.future[self.nxt]["arrive"] <= i:
            j = self.future[self.nxt]; self.nxt += 1
            j["state"] = "queued"
            if self.mode == "local":
                j["sat"] = self.rr % len(self.sats); self.rr += 1
            self.queue.append(j)
        self.budgets = [s.safe_budget(i) for s in self.sats]

    def _spare(self, c):
        return self.budgets[c] - sum(x["kw"] for x in self.sats[c].running)

    def _obs(self):
        i, g = self.t, self.i0 + self.t
        in_saa = np.array([s.saa[i] for s in self.sats])
        soon = self.soon[:, g]
        sens_saa = sum(j["kw"] for s in self.sats if s.saa[i] for j in s.running if j["sensitive"])
        sens_soon = sum(j["kw"] for c, s in enumerate(self.sats) if soon[c] for j in s.running if j["sensitive"])
        q_sens = sum(j["kw"] for j in self.queue if j["sensitive"])
        q_all = sum(j["kw"] for j in self.queue)
        spare = sum(max(0, self._spare(c)) for c in range(len(self.sats)))
        clean_spare = sum(max(0, self._spare(c)) for c in range(len(self.sats)) if not in_saa[c])
        batt = np.mean([s.batt / s.p["battery_kwh"] for s in self.sats])
        w = self.weather
        f = len(self.sats) / 12   # inputs scaled to fleet size (1.0 for 12 satellites)
        return np.array([w == "LOW", w == "ELEVATED", w == "HIGH",
                         in_saa.sum() / (4 * f), soon.sum() / (4 * f),
                         sens_saa / (200 * f), sens_soon / (200 * f),
                         min(q_sens / (1000 * f), 5), min(q_all / (1000 * f), 5),
                         spare / (400 * f), clean_spare / (400 * f), batt], dtype=np.float32)

    # ---------- one minute
    def step(self, action):
        a_saa, a_res, a_force = (int(x) for x in action)
        i, sats, st = self.t, self.sats, self.stats
        local = self.mode == "local"
        reserve = RESERVES[a_res]
        reward = 0.0

        # drop running jobs that no longer fit a satellite's safe budget
        for si, s in enumerate(sats):
            keep, used = [], 0
            for j in sorted(s.running, key=lambda x: x["kw"]):
                if used + j["kw"] <= self.budgets[si]:
                    keep.append(j); used += j["kw"]
                else:
                    j["state"] = "queued"
                    if not local:
                        j["prev"] = si; j["sat"] = None; j["hold"] = MOVE_DELAY_MIN
                    self.queue.append(j)
            s.running = keep

        # forced SAA exit (AI/strict only)
        if not local and a_force:
            for si, s in enumerate(sats):
                if s.saa[i]:
                    for j in [x for x in s.running if x["sensitive"]]:
                        s.running.remove(j); j["state"] = "queued"
                        j["prev"] = si; j["sat"] = None; j["hold"] = MOVE_DELAY_MIN
                        self.queue.append(j)

        # handoffs out of the SAA where a clean satellite has room (uses the reserve)
        if not local:
            for si, s in enumerate(sats):
                if not s.saa[i]:
                    continue
                for j in [x for x in s.running if x["sensitive"]]:
                    tgt, tsp = None, None
                    for c, t in enumerate(sats):
                        if t.saa[i] or c == si:
                            continue
                        sp = self._spare(c)
                        if sp >= j["kw"] and (tsp is None or sp < tsp):
                            tgt, tsp = c, sp
                    if tgt is not None:
                        s.running.remove(j); sats[tgt].running.append(j)
                        j["sat"] = tgt; j["stall"] = MOVE_DELAY_MIN; st["moves"] += 1

        # assign queued jobs (oldest first, best fit)
        self.queue.sort(key=lambda j: j["arrive"])
        clean_spare = sum(max(0, self._spare(c)) for c in range(len(sats)) if not sats[c].saa[i])
        still = []
        for j in self.queue:
            if j["state"] != "queued":
                continue
            if j.get("hold", 0) > 0:
                j["hold"] -= 1; still.append(j); continue
            if local:
                tiers = [[j["sat"]]]
            else:
                allc = list(range(len(sats)))
                clean = [c for c in allc if not sats[c].saa[i]]
                dirty = [c for c in allc if sats[c].saa[i]]
                if j["sensitive"]:
                    tiers = [clean] if a_saa else [clean, dirty]
                else:
                    tiers = [dirty, clean]  # non-sensitive work soaks up SAA satellites first
            best = None
            for cands in tiers:
                bsp = None
                for c in cands:
                    sp = self._spare(c)
                    if sp < j["kw"]:
                        continue
                    # keep the handoff reserve free on clean satellites (for new placements)
                    if not local and not sats[c].saa[i] and clean_spare - j["kw"] < reserve:
                        continue
                    if bsp is None or sp < bsp:
                        best, bsp = c, sp
                if best is not None:
                    break
            if best is None:
                still.append(j); continue
            if j.get("prev") is not None and j["prev"] != best:
                st["moves"] += 1
            j["prev"] = None; j["sat"] = best; j["state"] = "running"
            sats[best].running.append(j)
            if not sats[best].saa[i]:
                clean_spare -= j["kw"]
        self.queue = still

        # run the minute
        wm = ERROR_MULT[self.weather]
        for s in sats:
            speed = 0.3 if s.temp_k - 273.15 > s.p["temp_limit_c"] else 1.0
            if s.batt <= 0.5:
                speed = 0.0
            for j in list(s.running):
                if j.get("stall"):
                    j["stall"] -= 1; continue
                if j["sensitive"] and speed > 0 and self.err_rng.random() < p_error(j["kw"], wm, bool(s.saa[i])):
                    lost = j["kw"] * j["done_min"] / 60
                    st["wasted_kwh"] += lost; st["restarts"] += 1
                    reward -= 1.5 * lost / 10
                    j["done_min"] = 0.0; continue
                j["done_min"] += speed
                reward += j["kw"] * speed / 60 / 10
                if j["done_min"] >= j["mins"]:
                    j["state"] = "done"; s.running.remove(j)
                    st["jobs_done"] += 1; st["work_kwh"] += j["kw"] * j["mins"] / 60
                    st["delays"].append(i - j["arrive"] - j["mins"])
            gpu = sum(j["kw"] for j in s.running if not j.get("stall")) * speed
            s.physics(i, gpu)
            soc = s.batt / s.p["battery_kwh"]
            if soc < s.p["battery_floor"]:
                st["low_batt_min"] += 1
                reward -= 0.3 + (s.p["battery_floor"] - soc) * 20

        self.t += 1
        done = self.t >= EP_MIN
        if not done:
            self._prepare()
            obs = self._obs()
        else:
            obs = np.zeros(12, dtype=np.float32)
        return obs, float(reward), done, False, {}


def play(env, policy, model, start, weather, seed):
    obs, _ = env.reset(seed=seed, options={"start": start, "weather": weather})
    score, done = 0.0, False
    while not done:
        if policy == "ai":
            act, _ = model.predict(obs, deterministic=True)
        else:
            act = FIXED.get(policy, (0, 0, 0))
        obs, r, done, _, _ = env.step(act)
        score += r
    s = dict(env.stats)
    s["avg_delay_min"] = float(np.mean(s.pop("delays"))) if env.stats["delays"] else 0.0
    s["score"] = score
    return s


def train(seed, steps):
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    lit, saa, names = get_orbits()
    env = make_vec_env(lambda: FleetEnv(lit, saa, names, "train"), n_envs=4, seed=seed)
    ckpt, ckpt_steps = f"fleet_ai_seed{seed}_ckpt", f"fleet_ai_seed{seed}_ckpt.steps"
    start = 0
    if os.path.exists(ckpt + ".zip") and os.path.exists(ckpt_steps):   # resume after an interruption
        model = PPO.load(ckpt, env=env)
        start = int(open(ckpt_steps).read())
        print(f"  seed {seed}: resuming from {start} steps", flush=True)
    else:
        model = PPO("MlpPolicy", env, n_steps=720, batch_size=360, gamma=0.995,
                    learning_rate=3e-4, ent_coef=0.01, verbose=0, seed=seed)
    chunk = 10_000
    for done in range(start, steps, chunk):
        model.learn(chunk, reset_num_timesteps=False)
        model.save(ckpt)
        open(ckpt_steps, "w").write(str(done + chunk))
        print(f"  seed {seed}: {done + chunk:>7} steps | avg training score "
              f"{np.mean([e['r'] for e in model.ep_info_buffer] or [0]):7.1f}", flush=True)
    out = f"fleet_ai_seed{seed}"
    model.save(out)
    # validation score (day 6)
    venv = FleetEnv(lit, saa, names, "val")
    rng = np.random.default_rng(7)
    vals = [play(venv, "ai", model, int(s), w, 300 + k)["score"]
            for w in RISKS for k, s in enumerate(rng.integers(*venv.range, size=6))]
    print(f"validation score: {np.mean(vals):.1f}")


def evaluate(episodes=8, model_path="fleet_ai.zip", real=False):
    """real=True: test on 12 REAL Starlink satellites (2 days from today's orbit data).
    The AI was trained only on the stand-in orbits, so this also checks it generalizes."""
    from stable_baselines3 import PPO
    lit, saa, names = fleet.build_fleet_real(2 * 1440 + EP_MIN + 61) if real else get_orbits()
    model = PPO.load(model_path) if os.path.exists(model_path) else None
    policies = ["local", "fleet", "strict"] + (["ai"] if model else [])
    rng = np.random.default_rng(321)
    out = {}
    for w in RISKS:
        out[w] = {}
        test_range = (0, 2 * 1440) if real else FleetEnv(lit, saa, names, "test").range
        starts = np.random.default_rng(321).integers(*test_range, size=episodes)
        for p in policies:
            env = FleetEnv(lit, saa, names, "train" if real else "test", mode="local" if p == "local" else "ai")
            runs = [play(env, p, model, int(s), w, 900 + k) for k, s in enumerate(starts)]
            out[w][p] = {k: round(float(np.mean([r[k] for r in runs])), 1) for k in runs[0]}
    keys = ["work_kwh", "jobs_done", "avg_delay_min", "restarts", "wasted_kwh", "moves", "low_batt_min", "score"]
    for w in RISKS:
        print(f"\n=== {w}  (avg of {episodes} unseen 6-hour runs, 12 satellites) ===")
        print(f"{'':16}" + "".join(f"{p.upper():>10}" for p in policies))
        for k in keys:
            print(f"{k:16}" + "".join(f"{out[w][p][k]:>10}" for p in policies))
    fname = "fleet_eval_real.json" if real else "fleet_eval.json"
    with open(fname, "w") as f:
        json.dump({"averages": out, "policies": policies, "episodes": episodes, "satellites": names,
                   "orbit": __import__("sat_profile").LABEL if __import__("sat_profile").IS_AI1
                   else "real Starlink, 53 deg shell, ~463 km" if real
                   else f"stand-in {fleet.INC_DEG:.0f} deg / {fleet.ALT_KM:.0f} km"}, f)
    print(f"\nSaved {fname}")
    if os.path.exists("rl_results.json"):
        import make_dashboard
        make_dashboard.build()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval"])
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--real", action="store_true", help="eval on real Starlink orbits")
    a = ap.parse_args()
    if a.cmd == "train":
        train(a.seed, a.steps)
    else:
        evaluate(a.episodes, real=a.real)

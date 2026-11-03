"""Step 5a: The simulator as a training "game" for the learning AI.

Each step = 1 simulated minute. The AI sees the satellite's situation and picks:
  1. a GPU power budget: 0%, 25%, 50%, 75% or 100% of max
  2. storm mode on/off: when on, any error-sensitive job that would START
     this minute is sent to the ground instead (costs a small penalty)
A simple filler then packs jobs into that budget (finish running jobs first).

Score (reward): points for useful work done; penalties for low battery,
overheating, bit-flip restarts (wasted work) and jobs sent to the ground.

Every episode is a random 6-hour slice of a 14-day orbit with random storms
and a random job mix, so the AI can't just memorize one orbit.
Days 0-8 are for training, days 9-10 for picking the best training run,
and days 11-13 are held back for the final test.
"""
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from skyfield.api import EarthSatellite, load

from power_thermal import SAT, SIGMA
from orbit_sun import SAMPLE_TLE, fetch_tle, EARTH_RADIUS_KM
from space_weather import ERROR_MULT

# Recent ISS orbit (Nov 2026), used if a live download isn't possible.
CACHED_TLE = (
    "ISS (ZARYA)",
    "1 25544U 98067A   26306.43427755  .00005374  00000+0  10508-3 0  9999",
    "2 25544  51.6332 335.9757 0007139 339.3739  20.6961 15.49150046593183",
)
JOB_TYPES = [("Image tagging", 15, 10, False),
             ("Translation batch", 30, 20, True),
             ("Video analysis", 60, 40, True)]
BUDGETS = [0.0, 0.25, 0.5, 0.75, 1.0]
RISKS = ["LOW", "ELEVATED", "HIGH"]
from radiation import p_error  # v3: calibrated from Google TPU radiation tests
EPISODE_MIN = 360
DAYS = 14
TRAIN_DAYS = 11


def build_sunlight(days=DAYS, live=True):
    """Minute-by-minute sunlight (True/False) for `days` days, computed in one go."""
    import sat_profile
    if sat_profile.IS_AI1:  # one satellite on the AI1 orbit
        import fleet
        return fleet.build_fleet(days * 1440)[0][0]
    tle = CACHED_TLE
    if live:
        try:
            tle = fetch_tle(25544)
        except Exception:
            pass
    ts = load.timescale()
    sat = EarthSatellite(tle[1], tle[2], tle[0], ts)
    start = sat.epoch
    mins = np.arange(days * 1440)
    times = ts.tt_jd(start.tt + mins / 1440.0)
    pos = sat.at(times).position.km.T  # shape (N, 3)
    # Sun direction for every minute (same low-precision formula as step 1, vectorized)
    jd = times.ut1
    n = jd - 2451545.0
    L = np.radians((280.460 + 0.9856474 * n) % 360)
    g = np.radians((357.528 + 0.9856003 * n) % 360)
    lam = L + np.radians(1.915) * np.sin(g) + np.radians(0.020) * np.sin(2 * g)
    eps = np.radians(23.439 - 0.0000004 * n)
    sun = np.stack([np.cos(lam), np.cos(eps) * np.sin(lam), np.sin(eps) * np.sin(lam)], axis=1)
    along = (pos * sun).sum(axis=1)
    perp_sq = (pos ** 2).sum(axis=1) - along ** 2
    return (along >= 0) | (perp_sq > EARTH_RADIUS_KM ** 2)


def shadow_lookahead(lit):
    """For every minute: minutes until next shadow, its length, and shadow minutes left."""
    N = len(lit)
    to_shadow = np.zeros(N); shadow_len = np.zeros(N); left = np.zeros(N)
    nxt, nxt_len, run = N, 35, 0
    for i in range(N - 1, -1, -1):
        if not lit[i]:
            run += 1
            left[i] = run
            nxt, nxt_len = i, run
            to_shadow[i] = 0
        else:
            run = 0
            to_shadow[i] = nxt - i
        shadow_len[i] = nxt_len
    return to_shadow, shadow_len, left


class OrbitalComputeEnv(gym.Env):
    metadata = {}

    def __init__(self, lit, split="train", storm_mix=(0.5, 0.25, 0.25), p=SAT, seed=None,
                 safety_cap=False):
        super().__init__()
        self.lit = lit
        # safety_cap: the AI's power choice can never exceed the rules' safe level.
        self.safety_cap = safety_cap
        self.to_shadow, self.shadow_len, self.shadow_left = shadow_lookahead(lit)
        # train = days 0-8, val = days 9-10 (used to pick the best run), test = days 11-13
        lo, hi = {"train": (0, 9 * 1440), "val": (9 * 1440, TRAIN_DAYS * 1440),
                  "test": (TRAIN_DAYS * 1440, DAYS * 1440)}[split]
        self.range = (lo, hi - EPISODE_MIN - 1)
        self.storm_mix = storm_mix
        self.p = p
        self.action_space = spaces.MultiDiscrete([len(BUDGETS), 2])
        self.observation_space = spaces.Box(-1.0, 3.0, shape=(11,), dtype=np.float32)

    # ---------- helpers
    def _new_jobs(self, n=150):
        jobs = []
        for i in range(n):
            name, kw, mins, sens = JOB_TYPES[self.np_random.integers(len(JOB_TYPES))]
            jobs.append({"id": f"J-{100+i}", "type": name, "kw": kw, "mins": mins,
                         "sensitive": sens, "done_min": 0.0, "where": "queued"})
        return jobs

    def _risk_schedule(self):
        """Weather for the episode: one level, sometimes switching once mid-way."""
        a = self.np_random.choice(3, p=self.storm_mix)
        sched = [RISKS[a]] * EPISODE_MIN
        if self.np_random.random() < 0.4:
            cut = int(self.np_random.integers(60, EPISODE_MIN - 60))
            b = self.np_random.choice(3, p=self.storm_mix)
            sched[cut:] = [RISKS[b]] * (EPISODE_MIN - cut)
        return sched

    def _obs(self):
        i = self.i0 + self.t
        r = self.risk[min(self.t, EPISODE_MIN - 1)]
        run_kw = sum(j["kw"] for j in self.running)
        sens = sum(j["kw"] for j in self.running if j["sensitive"])
        return np.array([
            float(self.lit[i]),
            min(self.to_shadow[i], 120) / 60,
            self.shadow_len[i] / 60,
            self.shadow_left[i] / 60,
            self.batt / self.p["battery_kwh"],
            (self.temp_k - 273.15 - 40) / 50,
            r == "LOW", r == "ELEVATED", r == "HIGH",
            run_kw / self.p["gpu_max_kw"],
            sens / max(run_kw, 1),
        ], dtype=np.float32)

    # ---------- gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        opts = options or {}
        self.i0 = int(opts.get("start", self.np_random.integers(*self.range)))
        self.risk = opts.get("risk_schedule") or self._risk_schedule()
        self.jobs = self._new_jobs()
        self.running = []
        self.batt = self.p["battery_kwh"] * self.p["battery_start"]
        self.temp_k = self.p["temp_start_c"] + 273.15
        self.t = 0
        self.stats = {"work_kwh": 0.0, "restarts": 0, "ground": 0, "min_batt": 100.0,
                      "peak_temp": 0.0, "min_over_temp": 0, "min_low_batt": 0, "jobs_done": 0}
        self.log = []
        return self._obs(), {}

    def step(self, action):
        p, i = self.p, self.i0 + self.t
        lit = bool(self.lit[i])
        risk = self.risk[self.t]
        budget = BUDGETS[int(action[0])] * p["gpu_max_kw"]
        if self.safety_cap:
            budget = min(budget, rules_budget_frac(self) * p["gpu_max_kw"])
        storm_mode = bool(action[1])
        reward = 0.0

        # Fill the budget: keep running jobs (smallest first), then start new ones.
        chosen, used = [], 0
        for j in sorted(self.running, key=lambda x: x["kw"]):
            if used + j["kw"] <= budget:
                chosen.append(j); used += j["kw"]
        for j in self.jobs:
            if j["where"] not in ("queued", "paused") or j in chosen:
                continue
            if used + j["kw"] > budget:
                continue
            if storm_mode and j["sensitive"] and j["where"] == "queued":
                j["where"] = "ground"; self.stats["ground"] += 1
                reward -= 0.5 * j["kw"] * j["mins"] / 60 / 10  # half the job's value
                continue
            chosen.append(j); used += j["kw"]
        for j in self.running:
            if j not in chosen:
                j["where"] = "paused"
        self.running = chosen

        # Hardware protection every policy gets (not a decision): hot or dead battery.
        temp_c = self.temp_k - 273.15
        speed = 1.0
        if temp_c > p["temp_limit_c"]:
            speed = 0.3
        if self.batt <= 0.5:
            speed = 0.0

        mult = ERROR_MULT[risk]
        for j in self.running:
            j["where"] = "running"
            if j["sensitive"] and speed > 0 and self.np_random.random() < p_error(j["kw"], mult):
                lost = j["kw"] * j["done_min"] / 60
                reward -= 1.5 * lost / 10
                j["done_min"] = 0.0; self.stats["restarts"] += 1
                continue
            j["done_min"] += speed
            reward += j["kw"] * speed / 60 / 10  # progress points
            if j["done_min"] >= j["mins"]:
                j["where"] = "done"; self.stats["jobs_done"] += 1
                self.stats["work_kwh"] += j["kw"] * j["mins"] / 60
        gpu_kw = sum(j["kw"] for j in self.running) * speed
        self.running = [j for j in self.running if j["where"] == "running"]

        # Physics (same model as step 2)
        gen = p["solar_kw"] if lit else 0.0
        use = gpu_kw + p["bus_kw"]
        net = gen - use
        self.batt = min(max(self.batt + (net * p["charge_eff"] if net > 0 else net) / 60, 0.0), p["battery_kwh"])
        heat_in = use * 1000 + (p["sun_heat_kw"] * 1000 if lit else 0)
        heat_out = p["emissivity"] * SIGMA * p["radiator_m2"] * (self.temp_k ** 4 - 3.0 ** 4)
        self.temp_k += (heat_in - heat_out) * 60 / p["heat_capacity_j_per_k"]

        # Penalties
        soc = self.batt / p["battery_kwh"]
        temp_c = self.temp_k - 273.15
        if soc < p["battery_floor"]:
            # v2: much stronger. A flat cost per minute below the floor plus more the lower it goes.
            reward -= 0.3 + (p["battery_floor"] - soc) * 20
            self.stats["min_low_batt"] += 1
        if self.batt <= 0.5:
            reward -= 2
        if temp_c > p["throttle_at_c"]:
            reward -= (temp_c - p["throttle_at_c"]) * 0.2
        if temp_c > p["temp_limit_c"]:
            reward -= 2; self.stats["min_over_temp"] += 1
        self.stats["min_batt"] = min(self.stats["min_batt"], soc * 100)
        self.stats["peak_temp"] = max(self.stats["peak_temp"], temp_c)
        self.log.append({"lit": lit, "risk": risk, "gpu_kw": round(gpu_kw, 1),
                         "batt": round(soc * 100, 1), "temp": round(temp_c, 1),
                         "budget": BUDGETS[int(action[0])], "storm_mode": storm_mode})

        self.t += 1
        done = self.t >= EPISODE_MIN
        obs = self._obs() if not done else np.zeros(11, dtype=np.float32)
        return obs, float(reward), done, False, {}


# ---------- the two hand-written controllers, expressed as actions in this game
def baseline_action(env):
    return np.array([4, 0])  # always full power, never offload


def rules_action(env):
    """The step-4 'smart' rules, mapped onto the same action choices."""
    frac = rules_budget_frac(env)
    level = max(k for k, b in enumerate(BUDGETS) if b <= frac + 1e-9)
    return np.array([level, int(env.risk[env.t] == "HIGH")])


def rules_budget_frac(env):
    """Safe GPU power (0-1 of max) by the step-4 rules: save enough battery for
    the coming shadow and stay under the throttle temperature."""
    p, i = env.p, env.i0 + env.t
    lit = env.lit[i]
    floor = p["battery_kwh"] * p["battery_floor"]
    if lit:
        need = floor + env.shadow_len[i] * (p["bus_kw"] + 15) / 60
        short = max(0.0, need - env.batt)
        rate = short / max(env.to_shadow[i], 1) * 60 / p["charge_eff"]
        budget = p["solar_kw"] - p["bus_kw"] - rate
    else:
        budget = (env.batt - floor) / (max(env.shadow_left[i], 1) / 60) - p["bus_kw"]
    temp_c = env.temp_k - 273.15
    if temp_c > p["throttle_at_c"] - 4:
        budget *= max(0.0, (p["throttle_at_c"] - temp_c) / 4)
    return max(0.0, min(budget / p["gpu_max_kw"], 1.0))

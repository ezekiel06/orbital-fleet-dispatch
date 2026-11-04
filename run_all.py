"""Run the whole SAT-01 pipeline in order, with one command.

    python run_all.py              QUICK: reuse the trained AIs, rerun every test (~10 min)
    python run_all.py --train      FULL: also retrain both AIs, 3 tries each, keep the best (~30-40 min)
    python run_all.py --refresh    also download today's Starlink orbits first (needs internet)
    python run_all.py --telemetry  also rerun the real-telemetry steps (ESA problem detector,
                                   CANVAS analysis), if their data files are present
    python run_all.py --extended   also run the bigger tests: 12/48/96 satellites, laser-link
                                   limits, and real AI jobs from the Helios trace (~1 hour;
                                   downloads the Helios trace from GitHub the first time)

It stops at the first step that fails and says which one. At the end it prints
a summary and rebuilds dashboard.html.

Training notes: the 3 tries run at the same time, one CPU thread each. The best
try is picked on VALIDATION orbits (never the final test), and saved as
sat_scheduler_ppo_safe.zip (single satellite) and fleet_ai.zip (fleet).
"""
import argparse, json, os, re, subprocess, sys, time

PY = sys.executable
ENV1 = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")  # one thread per training run
LOG = []


def run(label, args, required=True, timeout=None):
    t = time.time()
    print(f"\n=== {label} ===", flush=True)
    try:
        p = subprocess.run([PY] + args, capture_output=True, text=True, env=ENV1, timeout=timeout)
    except subprocess.TimeoutExpired:
        p = None
    secs = time.time() - t
    ok = p is not None and p.returncode == 0
    out = p.stdout if p else ""
    tail = "\n".join(l for l in out.strip().splitlines() if "warn" not in l.lower())[-1500:]
    if tail:
        print(tail)
    LOG.append((label, "ok" if ok else "FAILED", secs))
    if not ok:
        err = (p.stderr[-2000:] if p else f"timed out after {timeout}s")
        print(f"\n!! {label} failed:\n{err}")
        if required:
            summary()
            sys.exit(f"Stopped at: {label}")
    return out


def train_parallel(label, cmd_for_seed, model_for_seed, final_name, seeds=(0, 1, 2)):
    """Start one training run per seed at the same time, wait, keep the best on validation."""
    t = time.time()
    print(f"\n=== {label}: {len(seeds)} tries in parallel ===", flush=True)
    procs = {s: subprocess.Popen([PY] + cmd_for_seed(s), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, env=ENV1) for s in seeds}
    scores = {}
    for s, p in procs.items():
        out, _ = p.communicate()
        m = re.findall(r"validation score: (-?[\d.]+)", out)
        if p.returncode != 0 or not m:
            print(out[-1500:])
            LOG.append((label, "FAILED", time.time() - t)); summary()
            sys.exit(f"Stopped at: {label} (try {s})")
        scores[s] = float(m[-1])
        print(f"  try {s}: validation score {scores[s]:.1f}")
    best = max(scores, key=scores.get)
    import shutil
    shutil.copy(model_for_seed(best), final_name)
    print(f"  kept try {best} -> {final_name}")
    LOG.append((label, "ok", time.time() - t))


def summary():
    print("\n" + "=" * 60 + "\nSUMMARY")
    for label, status, secs in LOG:
        print(f"  {status:7} {secs:6.0f}s  {label}")
    try:
        h = json.load(open("hybrid_results.json"))
        v = json.load(open("value_results.json"))["mixes"]["realistic"]
        ns, br = v["vs no sharing"], v["vs best simple rule"]
        lo = min(x[0] for x in ns["usd_per_sat_year"].values()); hi = max(x[1] for x in ns["usd_per_sat_year"].values())
        print(f"\nHybrid plan: {h.get('plan')}")
        print(f"Extra computing (realistic weather): +{ns['extra_pct']}% vs no sharing, "
              f"+{br['extra_pct']}% vs best simple rule")
        print(f"Worth about ${lo * 1000 / 1e6:.1f}M-${hi * 1000 / 1e6:.1f}M per year per 1,000 satellites")
    except (FileNotFoundError, KeyError):
        pass
    print("Open dashboard.html to see everything.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="retrain both AIs")
    ap.add_argument("--refresh", action="store_true", help="download today's Starlink orbits first")
    ap.add_argument("--telemetry", action="store_true", help="also rerun the real-telemetry steps")
    ap.add_argument("--extended", action="store_true", help="also run fleet-size, laser-link and real-job tests")
    ap.add_argument("--steps", type=int, default=None, help="training steps per try (default: each script's own)")
    ap.add_argument("--episodes", type=int, default=40, help="test runs per weather for the fleet checks")
    a = ap.parse_args()
    start = time.time()

    if a.refresh:
        run("0. Download today's Starlink orbits", ["fetch_starlink.py"])
    run("1. Orbit + sunlight (live ISS orbit)", ["orbit_sun.py", "--hours", "3"], required=False)
    run("2. Space weather (live NOAA)", ["space_weather.py"], required=False)
    run("3. Heat + power model", ["power_thermal.py", "--hours", "6"])
    run("4. Rules scheduler", ["scheduler.py"])

    if a.train:
        st = ["--steps", str(a.steps)] if a.steps else []
        train_parallel("5. Train single-satellite AI (with safety limit)",
                       lambda s: ["train_rl.py", "--safe", "--seed", str(s)] + st,
                       lambda s: f"run_safe_seed{s}.zip", "sat_scheduler_ppo_safe.zip")
    run("6. Test single-satellite schedulers", ["evaluate_rl.py"])

    if a.train:
        st = ["--steps", str(a.steps)] if a.steps else []
        train_parallel("7. Train fleet AI",
                       lambda s: ["fleet_rl.py", "train", "--seed", str(s)] + st,
                       lambda s: f"fleet_ai_seed{s}.zip", "fleet_ai.zip")
    run("8. Test fleet (stand-in orbits)", ["fleet_rl.py", "eval", "--episodes", "8"])
    run("9. Test fleet (real Starlink orbits)", ["fleet_rl.py", "eval", "--real", "--episodes", str(a.episodes)])
    run("10. Confirm the AI's edge (paired runs)", ["confirm_fleet.py", "--episodes", str(a.episodes)])
    run("11. Hybrid on fresh runs", ["hybrid.py", "--episodes", str(a.episodes)])
    run("12. Dollar value", ["value.py"])

    if a.telemetry:
        if os.path.exists("opssat/dataset.csv"):
            run("13. Problem detector on ESA OPS-SAT data", ["telemetry_check.py"], required=False)
        if os.path.exists("satnogs_68635.jsonl"):
            run("14. CANVAS battery/temperature analysis", ["canvas_analysis.py"], required=False)

    if a.extended:
        run("15. Bigger fleets (12 / 48 / 96 satellites)", ["scale_test.py"])
        run("16. Laser-link speed limits", ["link_test.py"])
        if not os.path.exists("helios_jobs.npz"):
            run("17a. Download real AI jobs (Helios trace)", ["workload.py"])
        run("17. Real AI jobs", ["helios_test.py"])

    run("18. Rebuild dashboard", ["make_dashboard.py"])
    print(f"\nDone in {(time.time() - start) / 60:.1f} minutes.")
    summary()


if __name__ == "__main__":
    main()

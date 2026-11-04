"""Real AI workloads for the fleet simulator (SAT_JOBS=helios).

Instead of the three made-up job types, replay real deep-learning jobs from
SenseTime's Helios GPU datacenter trace (4 clusters, Mar-Sep 2020, ~1.6 million
GPU jobs; https://github.com/S-Lab-System-Group/HeliosData, SC '21 paper
"Characterization and Prediction of Deep Learning Workloads in Large-Scale GPU
Datacenters"). Each job keeps its real arrival pattern, GPU count and run time.

Mapping (assumptions):
  - power = GPUs x 1.275 kW (one H100-class GPU plus its share of the server,
    the same figure value.py uses for prices)
  - jobs bigger than one satellite's GPU power are left out and counted; they
    would need several satellites working as one machine, which isn't modeled
  - jobs running over 30 minutes count as error-sensitive (a bit flip forces a
    restart); shorter ones just rerun
  - load: whole 12-hour slices of a random cluster are stacked until the fleet
    is offered SAT_LOAD x its full GPU power (default 1.0), then thinned at
    random to hit it exactly, so bursts and quiet spells stay real

Build the job file once:  python workload.py   (needs git; writes helios_jobs.npz)
"""
import math, os, subprocess, sys, zipfile
import numpy as np

KW_PER_GPU = 1.275
CLUSTERS = ["Venus", "Earth", "Saturn", "Uranus"]
NPZ = "helios_jobs.npz"
STATS = {"too_big": 0, "kept": 0}
_cache = None


def build(src_dir="helios_src"):
    """Download the Helios trace (GitHub) and keep only what the sim needs."""
    import pandas as pd
    if not os.path.isdir(src_dir):
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/S-Lab-System-Group/HeliosData", src_dir], check=True)
    with zipfile.ZipFile(os.path.join(src_dir, "data.zip")) as z:
        out = {}
        for c in CLUSTERS:
            d = pd.read_csv(z.open(f"data/{c}/cluster_log.csv"))
            d = d[(d.gpu_num > 0) & (d.duration > 0)]
            t = pd.to_datetime(d.submit_time)
            out[f"{c}_submit"] = ((t - t.min()).dt.total_seconds()).to_numpy(np.int64)
            out[f"{c}_gpu"] = d.gpu_num.to_numpy(np.int32)
            out[f"{c}_dur"] = d.duration.to_numpy(np.int64)
            print(f"{c}: {len(d):,} GPU jobs")
    np.savez_compressed(NPZ, **out)
    print(f"Saved {NPZ}")


def _load():
    global _cache
    if _cache is None:
        if not os.path.exists(NPZ):
            sys.exit(f"{NPZ} missing: run `python workload.py` first")
        _cache = dict(np.load(NPZ))
    return _cache


def helios_jobs(minutes, n_sats, gpu_max_kw, rng, load=None):
    load = float(os.environ.get("SAT_LOAD", "1.0")) if load is None else load
    d = _load()
    cl = CLUSTERS[rng.integers(len(CLUSTERS))]
    sub, gpu, dur = d[f"{cl}_submit"], d[f"{cl}_gpu"], d[f"{cl}_dur"]
    win = minutes * 60
    target = load * n_sats * gpu_max_kw * minutes  # kW-minutes offered over the run
    picked, offered, too_big = [], 0.0, 0
    while offered < target:
        t0 = rng.integers(0, int(sub.max()) - win)
        idx = np.nonzero((sub >= t0) & (sub < t0 + win))[0]
        for k in idx:
            kw = gpu[k] * KW_PER_GPU
            if kw > gpu_max_kw:
                too_big += 1
                continue
            arrive = int((sub[k] - t0) // 60)
            mins = max(1, math.ceil(dur[k] / 60))
            picked.append((arrive, kw, mins))
            offered += kw * min(mins, minutes - arrive)
    # thin at random down to the target load
    order = rng.permutation(len(picked))
    jobs, total = [], 0.0
    for k in order:
        a, kw, mins = picked[k]
        if total >= target:
            break
        total += kw * min(mins, minutes - a)
        jobs.append((a, kw, mins))
    jobs.sort()
    STATS["too_big"] += too_big; STATS["kept"] += len(jobs)
    return [{"id": i, "type": f"{cl} job", "kw": kw, "mins": mins, "sensitive": mins > 30,
             "arrive": a, "done_min": 0.0, "state": "future", "sat": None, "hold": 0, "finish": None}
            for i, (a, kw, mins) in enumerate(jobs)]


if __name__ == "__main__":
    build()

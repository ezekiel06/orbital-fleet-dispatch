"""Quick checks that the simulator's key behaviors still hold. Run: pytest -q"""
import math
import numpy as np
import pytest

import fleet, radiation, workload


@pytest.fixture(scope="module")
def orbits():
    return fleet.build_fleet(360)  # 12 satellites, 6 hours, built offline from orbital elements


def test_bit_flip_rate_matches_tpu_calibration():
    # Google's TPU tests: ~12.4 job-breaking errors per chip per year at ~1.275 kW per chip
    assert radiation.ORBIT_AVG_PER_KW_MIN == pytest.approx(1.85e-5, rel=0.01)
    assert radiation.p_error(20, 1, in_saa=True) > radiation.p_error(20, 1, in_saa=False)
    assert radiation.p_error(20, 10) > radiation.p_error(20, 1)


def test_orbits_have_shadow_and_saa(orbits):
    lit, saa, names = orbits
    assert lit.shape == (12, 360) and len(names) == 12
    assert 0.55 < lit.mean() < 0.75   # roughly a third of each lap in Earth's shadow at 53 degrees
    assert saa.any()


@pytest.mark.parametrize("weather", ["LOW", "HIGH"])
def test_sharing_beats_working_alone(orbits, weather):
    lit, saa, names = orbits
    local = fleet.simulate("local", lit, saa, names, weather, 360, 1)
    shared = fleet.simulate("fleet", lit, saa, names, weather, 360, 1)
    assert shared["work_kwh"] > local["work_kwh"]


def test_battery_never_below_floor_when_sharing(orbits):
    lit, saa, names = orbits
    r = fleet.simulate("fleet", lit, saa, names, "ELEVATED", 360, 2)
    assert r["sat_minutes_battery_low"] == 0


def test_runs_are_repeatable(orbits):
    lit, saa, names = orbits
    a = fleet.simulate("fleet", lit, saa, names, "LOW", 180, 7)
    b = fleet.simulate("fleet", lit, saa, names, "LOW", 180, 7)
    assert a["work_kwh"] == b["work_kwh"] and a["moves"] == b["moves"]


def test_fleet_ai_environment_plays_an_episode(orbits):
    import fleet_rl
    lit, saa, names = fleet.build_fleet(2 * 1440 + fleet_rl.EP_MIN + 61)
    env = fleet_rl.FleetEnv(lit, saa, names, "train")
    r = fleet_rl.play(env, "fleet", None, 100, "LOW", 1)
    assert math.isfinite(r["score"]) and r["jobs_done"] > 0


def test_real_job_sampler_hits_target_load(tmp_path, monkeypatch):
    # A tiny stand-in for the Helios trace: 1-8 GPU jobs arriving over 10 days
    rng = np.random.default_rng(0)
    n = 20000
    data = {}
    for c in workload.CLUSTERS:
        data[f"{c}_submit"] = np.sort(rng.integers(0, 10 * 86400, n))
        data[f"{c}_gpu"] = rng.integers(1, 9, n)
        data[f"{c}_dur"] = rng.integers(60, 7200, n)
    monkeypatch.setattr(workload, "_cache", data)
    jobs = workload.helios_jobs(720, 12, 100.0, np.random.default_rng(1), load=1.0, part="test")
    offered = sum(j["kw"] * min(j["mins"], 720 - j["arrive"]) for j in jobs)
    assert offered == pytest.approx(12 * 100.0 * 720, rel=0.02)
    assert all(j["kw"] <= 100.0 for j in jobs)

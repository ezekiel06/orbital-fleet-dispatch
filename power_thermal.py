"""Step 2: Heat + power model for one orbital compute satellite.

Uses step 1 (orbit_sun.py) to know when the satellite is in sunlight, then
simulates minute by minute:
  - Power: solar panels charge the battery in sun; the battery alone runs
    everything in Earth's shadow.
  - Heat: the GPUs turn nearly all their power into heat; radiators get rid
    of it by glowing it off as infrared (the only way to shed heat in a vacuum).
  - Safety throttle: GPUs slow down if they run too hot or the battery runs low.
    (Step 4 replaces this simple rule with the real scheduler.)

All satellite numbers below are made-up but realistic-scale assumptions.
Tweak them in SAT to experiment.

Usage:  python power_thermal.py [--hours 6] [--offline] [--csv out.csv]
"""
import argparse, csv, math
from datetime import timedelta
from skyfield.api import EarthSatellite, load
from orbit_sun import fetch_tle, SAMPLE_TLE, sun_direction, in_sunlight

SIGMA = 5.670e-8  # physics constant for how strongly hot things radiate heat

# Earth's thermal environment: NASA/TM-2001-211221 (Anderson et al.) orbit-average Earth IR for
# 30-60 degree orbits (234 W/m2; global average 239), albedo 0.30. Radiator coating: white
# paint at beginning of life (alpha ~0.1-0.2, NASA TM-104335). Absorptivity/emissivity are for a white or silvered radiator coating.
EARTH_HEAT = {"earth_ir_wm2": 234.0, "albedo": 0.30, "solar_wm2": 1361.0,
              "rad_absorptivity": 0.15, "albedo_sunlit_avg": 0.5}


def earth_heat_w(p, lit):
    """Heat Earth adds to the radiators (W): its infrared glow always, reflected sunlight
    on the day side. Scales with how much of Earth the radiator faces see."""
    f = p.get("earth_view", 0.0)
    if f <= 0:
        return 0.0
    e = EARTH_HEAT
    area = p["radiator_m2"]
    ir = p["emissivity"] * e["earth_ir_wm2"] * f * area
    alb = (e["rad_absorptivity"] * e["albedo"] * e["solar_wm2"] * e["albedo_sunlit_avg"] * f * area) if lit else 0.0
    return ir + alb


SAT = {
    "solar_kw": 140.0,         # panel output in full sunlight
    "gpu_max_kw": 100.0,       # GPU draw at 100% load
    "bus_kw": 5.0,             # always-on systems (radio, computer, heaters)
    "battery_kwh": 120.0,      # battery size
    "battery_start": 0.95,     # starting charge (95%, like the real CANVAS data)
    "charge_eff": 0.95,        # charging loses ~5%
    "radiator_m2": 150.0,      # radiator area
    "emissivity": 0.90,        # how well radiators radiate (1.0 = perfect)
    "sun_heat_kw": 10.0,       # extra heat from sunlight hitting the craft
    # Earth's own heat (infrared glow) and reflected sunlight (albedo) landing on the radiators.
    # Values filled in from sources in EARTH_HEAT below; earth_view = share of each radiator
    # face's view taken up by Earth (0 = facing deep space, ~0.3 = edge-on to Earth at 650 km).
    "earth_view": 0.24,        # radiators edge-on to Earth at ~650 km (vertical plate: 0.24 per face)
    "heat_capacity_j_per_k": 1.6e6,  # thermal mass: higher = temps change slower
    "temp_start_c": 40.0,
    "temp_limit_c": 85.0,      # hard limit for the GPUs
    "throttle_at_c": 80.0,     # start slowing down here
    "battery_floor": 0.70,     # v2: keep drains shallow (30% max), like real satellites.
                               # Real CANVAS telemetry stayed ~95%; deep drains wear
                               # batteries out over ~5,500 orbits a year. (v1 used 20%.)
}
import sat_profile  # SAT_PROFILE=ai1_53 / ai1_sso swaps in AI1-style specs
sat_profile.apply(SAT)


def sunlight_timeline(hours, offline):
    ts = load.timescale()
    if offline:
        name, l1, l2 = SAMPLE_TLE
    else:
        try:
            name, l1, l2 = fetch_tle(25544)
        except Exception as e:
            print(f"Live download failed ({e}); using sample orbit.")
            name, l1, l2 = SAMPLE_TLE
    sat = EarthSatellite(l1, l2, name, ts)
    t0 = sat.epoch if (name, l1, l2) == SAMPLE_TLE else ts.now()
    out = []
    for m in range(int(hours * 60)):
        t = t0 + timedelta(minutes=m)
        out.append((t, in_sunlight(sat.at(t).position.km, sun_direction(t.utc_datetime()))))
    return name, out


def simulate(timeline, p=SAT):
    batt = p["battery_kwh"] * p["battery_start"]
    temp_k = p["temp_start_c"] + 273.15
    load_frac = 1.0
    rows = []
    for t, lit in timeline:
        temp_c = temp_k - 273.15
        soc = batt / p["battery_kwh"]
        # Safety throttle: ease off when hot or battery low, recover slowly otherwise.
        if temp_c > p["throttle_at_c"] or (not lit and soc < p["battery_floor"]):
            load_frac = max(0.2, load_frac - 0.10)
        else:
            load_frac = min(1.0, load_frac + 0.05)
        gpu_kw = p["gpu_max_kw"] * load_frac
        use_kw = gpu_kw + p["bus_kw"]
        gen_kw = p["solar_kw"] if lit else 0.0

        # Battery: surplus charges it, shortfall drains it (1-minute step).
        net = gen_kw - use_kw
        batt += (net * p["charge_eff"] if net > 0 else net) / 60.0
        batt = min(max(batt, 0.0), p["battery_kwh"])

        # Heat: in = electronics + sunlight; out = radiated to space.
        heat_in_w = use_kw * 1000 + (p["sun_heat_kw"] * 1000 if lit else 0) + earth_heat_w(p, lit)
        heat_out_w = p["emissivity"] * SIGMA * p["radiator_m2"] * (temp_k ** 4 - 3.0 ** 4)
        temp_k += (heat_in_w - heat_out_w) * 60 / p["heat_capacity_j_per_k"]

        rows.append({
            "time_utc": t.utc_strftime("%H:%M"),
            "state": "SUN" if lit else "SHADOW",
            "gpu_load_pct": round(load_frac * 100),
            "solar_kw": gen_kw,
            "use_kw": round(use_kw, 1),
            "battery_pct": round(batt / p["battery_kwh"] * 100, 1),
            "gpu_temp_c": round(temp_k - 273.15, 1),
            "over_limit": (temp_k - 273.15) > p["temp_limit_c"],
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=6)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--csv", default="sim_output.csv")
    a = ap.parse_args()
    name, tl = sunlight_timeline(a.hours, a.offline)
    rows = simulate(tl)
    print(f"Simulating an orbital compute satellite on {name}'s orbit, {a.hours} h\n")
    print("Time  State   Load  Battery  GPU temp")
    for r in rows[::10]:
        print(f"{r['time_utc']} {r['state']:<7} {r['gpu_load_pct']:>3}%   {r['battery_pct']:>5}%   {r['gpu_temp_c']:>5} C")
    temps = [r["gpu_temp_c"] for r in rows]; batts = [r["battery_pct"] for r in rows]
    print(f"\nTemp range {min(temps)}-{max(temps)} C | battery low {min(batts)}% | "
          f"minutes over limit: {sum(r['over_limit'] for r in rows)} | "
          f"avg GPU load {sum(r['gpu_load_pct'] for r in rows)/len(rows):.0f}%")
    with open(a.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f"Saved minute-by-minute data to {a.csv}")


if __name__ == "__main__":
    main()

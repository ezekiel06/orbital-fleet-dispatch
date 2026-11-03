"""Step 1: Orbit + sunlight tracker for the orbital compute sim.

Tracks a real satellite (default: the ISS) and reports, minute by minute,
whether it is in sunlight or in Earth's shadow, plus a summary of each
sun/shadow period. Later steps feed this into the heat + power model.

Usage:
    pip install skyfield
    python orbit_sun.py                  # ISS, next 3 hours
    python orbit_sun.py --norad 25544 --hours 6
    python orbit_sun.py --offline        # use the bundled sample orbit
"""
import argparse
import math
import urllib.request
from datetime import timedelta

from skyfield.api import EarthSatellite, load

EARTH_RADIUS_KM = 6378.137

# Sample orbit data (ISS, Jan 2014) used only when the live download fails.
# Orbit data goes stale within days, so real runs should always fetch live.
SAMPLE_TLE = (
    "ISS (ZARYA)",
    "1 25544U 98067A   14020.93268519  .00009878  00000-0  18200-3 0  5082",
    "2 25544  51.6498 109.4756 0003572  55.9686 274.8005 15.49815350868473",
)


def fetch_tle(norad_id):
    """Download the latest orbit data (a TLE) from CelesTrak."""
    url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=tle"
    with urllib.request.urlopen(url, timeout=15) as r:
        lines = [l.strip() for l in r.read().decode().splitlines() if l.strip()]
    if len(lines) < 3:
        raise ValueError(f"CelesTrak returned no orbit for NORAD {norad_id}")
    return lines[0], lines[1], lines[2]


def sun_direction(dt):
    """Unit vector from Earth toward the Sun (low-precision formula, ~0.01 deg).

    Avoids downloading a large planet-position file just to know where the Sun is.
    """
    jd = dt.timestamp() / 86400.0 + 2440587.5
    n = jd - 2451545.0
    mean_long = math.radians((280.460 + 0.9856474 * n) % 360)
    mean_anom = math.radians((357.528 + 0.9856003 * n) % 360)
    ecl_long = mean_long + math.radians(1.915) * math.sin(mean_anom) + math.radians(0.020) * math.sin(2 * mean_anom)
    tilt = math.radians(23.439 - 0.0000004 * n)
    return (math.cos(ecl_long), math.cos(tilt) * math.sin(ecl_long), math.sin(tilt) * math.sin(ecl_long))


def in_sunlight(sat_pos_km, sun_dir):
    """Simple cylinder shadow model: shadowed if behind Earth and inside its radius."""
    along = sum(p * s for p, s in zip(sat_pos_km, sun_dir))
    if along >= 0:
        return True  # on the sunny side of Earth
    perp_sq = sum(p * p for p in sat_pos_km) - along * along
    return perp_sq > EARTH_RADIUS_KM ** 2


def run(name, l1, l2, hours, step_min, start=None):
    ts = load.timescale()
    sat = EarthSatellite(l1, l2, name, ts)
    t0 = start if start is not None else ts.now()
    age_days = abs(t0 - sat.epoch)
    print(f"Satellite: {name}   orbit data age: {age_days:.1f} days")
    if age_days > 3:
        print("  WARNING: orbit data is more than 3 days old; positions drift.")

    periods, rows = [], []
    current, period_start = None, None
    steps = int(hours * 60 / step_min) + 1
    for i in range(steps):
        t = t0 + timedelta(minutes=i * step_min)
        pos = sat.at(t).position.km  # Earth-centered position
        lit = in_sunlight(pos, sun_direction(t.utc_datetime()))
        alt = math.sqrt(sum(p * p for p in pos)) - EARTH_RADIUS_KM
        rows.append((t.utc_strftime("%H:%M"), "SUN" if lit else "SHADOW", alt))
        if lit != current:
            if current is not None:
                periods.append((current, period_start, t))
            current, period_start = lit, t
    periods.append((current, period_start, t0 + timedelta(minutes=(steps - 1) * step_min)))

    print("\nTime(UTC)  State    Altitude")
    for r in rows[:: max(1, 10 // step_min)]:
        print(f"{r[0]}      {r[1]:<7}  {r[2]:6.0f} km")

    print("\nSun/shadow periods:")
    for lit, a, b in periods:
        mins = (b - a) * 24 * 60
        print(f"  {'SUN   ' if lit else 'SHADOW'}  {a.utc_strftime('%H:%M')} -> {b.utc_strftime('%H:%M')}  ({mins:.0f} min)")
    return periods


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--norad", type=int, default=25544, help="satellite ID number (ISS = 25544)")
    ap.add_argument("--hours", type=float, default=3)
    ap.add_argument("--step", type=int, default=1, help="minutes between checks")
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    start = None
    if args.offline:
        name, l1, l2 = SAMPLE_TLE
    else:
        try:
            name, l1, l2 = fetch_tle(args.norad)
        except Exception as e:
            print(f"Live download failed ({e}); using bundled sample orbit.")
            name, l1, l2 = SAMPLE_TLE
    if (name, l1, l2) == SAMPLE_TLE:
        ts = load.timescale()
        start = EarthSatellite(l1, l2, name, ts).epoch  # run at the sample's own date
    run(name, l1, l2, args.hours, args.step, start)


if __name__ == "__main__":
    main()

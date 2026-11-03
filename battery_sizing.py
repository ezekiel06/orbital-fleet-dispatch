"""Step 15b: How small can the battery be?

Batteries are heavy, and every kg costs launch money. With the 70% safety
floor only 30% of the battery is usable, so the question is: how much computing
do you lose with a smaller battery?

Runs the 12-satellite fleet (plain sharing dispatcher) on the chosen
SAT_PROFILE's orbit for 12 hours, for several battery sizes, averaged over a
few random job mixes. Reports work done as a share of the biggest battery.

Usage:  set SAT_PROFILE=ai1_53   (or ai1_sso)   then   python battery_sizing.py
Saves:  battery_sizing_<profile>.json
"""
import json
import numpy as np
import fleet, sat_profile
from power_thermal import SAT

SIZES = [30, 60, 90, 120, 180, 240, 360]   # kWh
SEEDS = range(1, 5)
MIN = 720


def main():
    lit, saa, names = fleet.build_fleet(MIN, offset_days=0)
    rows = []
    for kwh in SIZES:
        SAT["battery_kwh"] = float(kwh)   # Sat objects read this shared dict
        runs = [fleet.simulate("fleet", lit, saa, names, "LOW", MIN, s) for s in SEEDS]
        rows.append({"kwh": kwh, "work_kwh": float(np.mean([r["work_kwh"] for r in runs])),
                     "lowest_batt": float(min(r["lowest_battery_pct"] for r in runs)),
                     "low_min": int(sum(r["sat_minutes_battery_low"] for r in runs))})
    top = rows[-1]["work_kwh"]
    print(f"{sat_profile.LABEL}\n{'battery':>8} {'usable':>7} {'work':>8} {'vs biggest':>10} {'lowest':>7}")
    for r in rows:
        r["share"] = r["work_kwh"] / top
        print(f"{r['kwh']:>6} kWh {r['kwh'] * 0.3:>5.0f} kWh {r['work_kwh']:>8.0f} {r['share'] * 100:>9.1f}% {r['lowest_batt']:>6.0f}%")
    pick = next(r for r in rows if r["share"] >= 0.98)
    print(f"Smallest battery within 2% of the biggest: {pick['kwh']} kWh")
    json.dump({"profile": sat_profile.PROFILE, "rows": rows, "pick_kwh": pick["kwh"]},
              open(f"battery_sizing_{sat_profile.PROFILE}.json", "w"), indent=1)


if __name__ == "__main__":
    main()

"""Step 17: Money SAVED, not just made. How much battery (and launch mass) does
fleet sharing let you leave on the ground?

Method:
  1. Run the 12-satellite fleet on the chosen SAT_PROFILE (default ai1_53) with
     several battery sizes, with and without sharing (calm weather, 4 job mixes).
  2. Find the battery a SHARING fleet needs to match a NON-sharing fleet that
     carries the sized battery (240 kWh). The difference is battery you don't launch.
  3. Battery kWh -> kg using real space battery packs:
       ~170 Wh/kg  Airbus LEO constellation pack (OneWeb heritage), optimistic
       ~120 Wh/kg  GS Yuasa modular pack (by its listed mass), conservative
  4. kg -> launch dollars. Starship has no published price yet, so scenarios:
       $100/kg (Musk's stated near-term target), $1,000/kg (illustrative early
       commercial price), and ~$3,250/kg (Falcon 9 list price / max payload).

This is a ONE-TIME saving at launch, per satellite. Battery hardware cost, and
the extra space/structure a bigger battery needs, are not counted, so it's a floor.

Usage:  set SAT_PROFILE=ai1_53   then   python savings.py
Saves:  savings_results.json
"""
import json
import numpy as np
import fleet, sat_profile
from power_thermal import SAT

SIZES = [120, 150, 180, 200, 240, 300, 360]
SEEDS = range(1, 5)
WH_PER_KG = {"optimistic (170 Wh/kg)": 170, "conservative (120 Wh/kg)": 120}
USD_PER_KG = {"Starship target ($100/kg)": 100, "early Starship ($1,000/kg)": 1000, "Falcon 9 (~$3,250/kg)": 3246}


def main():
    sized = SAT["battery_kwh"]
    lit, saa, names = fleet.build_fleet(720)
    curve = {"local": [], "fleet": []}
    for kwh in SIZES:
        SAT["battery_kwh"] = float(kwh)
        for mode in curve:
            curve[mode].append(float(np.mean([fleet.simulate(mode, lit, saa, names, "LOW", 720, s)["work_kwh"]
                                               for s in SEEDS])))
    SAT["battery_kwh"] = sized
    target = float(np.interp(sized, SIZES, curve["local"]))           # no sharing, sized battery
    needed = float(np.interp(target, curve["fleet"], SIZES))          # sharing fleet matching it
    saved_kwh = sized - needed
    print(f"{sat_profile.LABEL}")
    print(f"No sharing with a {sized:.0f} kWh battery does {target:,.0f} kWh of work per 12 h (12 satellites).")
    print(f"A sharing fleet matches that with ~{needed:.0f} kWh -> {saved_kwh:.0f} kWh less battery per satellite.\n")
    out = {"profile": sat_profile.PROFILE, "sized_kwh": sized, "needed_with_sharing_kwh": round(needed, 1),
           "saved_kwh_per_sat": round(saved_kwh, 1), "curve": {"sizes": SIZES, **curve}, "cases": []}
    for wl, wh in WH_PER_KG.items():
        kg = saved_kwh * 1000 / wh
        print(f"Battery {wl}: {kg:,.0f} kg less per satellite ({kg:,.0f} t per 1,000 satellites)")
        for cl, usd in USD_PER_KG.items():
            per_sat = kg * usd
            out["cases"].append({"battery": wl, "launch": cl, "kg_per_sat": round(kg), "usd_per_sat": round(per_sat),
                                 "usd_per_1000": round(per_sat * 1000)})
            print(f"    {cl:28} ${per_sat:>12,.0f} per satellite   ${per_sat * 1000 / 1e6:>8,.1f}M per 1,000 satellites")
    json.dump(out, open("savings_results.json", "w"), indent=1)
    print("\nSaved savings_results.json")


if __name__ == "__main__":
    main()

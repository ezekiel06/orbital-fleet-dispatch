"""Step 9b: What a full day of real CANVAS telemetry says about batteries and heat."""
import json, statistics as S
rows = {}
for f in ("satnogs_68635.jsonl", "satnogs_68635_first2h.jsonl"):
    try:
        for l in open(f):
            r = json.loads(l); rows[r["time_stamp_seconds"]] = r
    except FileNotFoundError:
        pass
from datetime import datetime
ts = lambda r: datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00")).timestamp()
# keep only housekeeping "beacon" frames, ordered by when the ground station heard them
rows = sorted((r for r in rows.values() if "bcn_eps_battery_soc" in r), key=ts)
print(f"{len(rows)} unique frames, {rows[0]['timestamp']} -> {rows[-1]['timestamp']}")
# passes = gaps > 10 min between frames
passes, cur = [], [rows[0]]
for a, b in zip(rows, rows[1:]):
    if ts(b) - ts(a) > 600:
        passes.append(cur); cur = []
    cur.append(b)
passes.append(cur)
print(f"{len(passes)} separate ground-station passes heard")
soc = [int(r["bcn_eps_battery_soc"]) for r in rows]
print(f"battery charge over the day: min {min(soc)}%, max {max(soc)}%, median {S.median(soc)}%")
for e, name in ((1, "shadow"), (0, "sunlight")):
    g = [r for r in rows if int(r["bcn_adcs_eclipse"]) == e]
    if not g:
        continue
    f = lambda k: S.mean(float(r[k]) for r in g)
    print(f"{name:9} n={len(g):3} | battery current {f('bcn_battery_curr'):7.0f} mA | solar current {f('bcn_pwr_mntr1_all_pnl_curr'):6.0f} mA "
          f"| bus {f('bcn_pwr_mntr3_main_bus_voltage')/1000:.2f} V | charge {f('bcn_eps_battery_soc'):.1f}%")
heat = [int(r["bcn_battery_heater_status"]) for r in rows]
print(f"battery heater on in {sum(1 for h in heat if h)} of {len(heat)} frames")
bt = [float(r["bcn_eps_batt1_temp"]) for r in rows]; pt = [float(r["bcn_solar_panel1_temp"]) for r in rows]
print(f"raw temp ranges: battery {min(bt):.0f}-{max(bt):.0f} (spread {max(bt)-min(bt):.0f}), solar panel {min(pt):.0f}-{max(pt):.0f} (spread {max(pt)-min(pt):.0f})")

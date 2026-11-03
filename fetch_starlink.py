"""Download today's Starlink orbits from CelesTrak and pick 12 real satellites
for the fleet: the 53-degree shell, 3 orbital planes x 4 satellites, all near
that shell's typical altitude.

Usage:  python fetch_starlink.py
Saves:  starlink.tle (all Starlink), starlink_fleet12.tle (the 12 picked)
"""
import math, urllib.request
import numpy as np

URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle"


def parse(lines):
    sats = []
    for k in range(0, len(lines) - 2, 3):
        n, l1, l2 = lines[k].strip(), lines[k + 1], lines[k + 2]
        if not (l1.startswith("1 ") and l2.startswith("2 ")):
            continue
        inc, raan, argp, ma, mm = (float(l2[a:b]) for a, b in ((8, 16), (17, 25), (34, 42), (43, 51), (52, 63)))
        ecc = float("0." + l2[26:33])
        alt = (398600.4418 / (mm * 2 * math.pi / 86400) ** 2) ** (1 / 3) - 6378.137
        sats.append(dict(name=n, inc=inc, raan=raan, u=(argp + ma) % 360, alt=alt, ecc=ecc, l1=l1, l2=l2))
    return sats


def angdist(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


def main():
    with urllib.request.urlopen(URL, timeout=120) as r:
        text = r.read().decode()
    open("starlink.tle", "w").write(text)
    sats = parse([l.rstrip() for l in text.splitlines() if l.strip()])
    shell = [s for s in sats if 52.9 < s["inc"] < 53.3 and 400 < s["alt"] < 600 and s["ecc"] < 0.0015]
    med = float(np.median([s["alt"] for s in shell]))
    shell = [s for s in shell if abs(s["alt"] - med) < 8]
    print(f"{len(sats)} Starlink satellites; 53-degree shell median altitude {med:.0f} km ({len(shell)} near it)")
    pick = []
    for p, target in enumerate([0, 120, 240]):
        plane = [s for s in shell if angdist(s["raan"], target) < 3]
        for k in range(4):
            best = min(plane, key=lambda s: angdist(s["u"], (k * 90 + p * 30) % 360))
            plane.remove(best); pick.append(best)
    with open("starlink_fleet12.tle", "w") as f:
        for s in pick:
            f.write(f"{s['name']}\n{s['l1']}\n{s['l2']}\n")
    print("Picked:", ", ".join(s["name"] for s in pick), "-> starlink_fleet12.tle")


if __name__ == "__main__":
    main()

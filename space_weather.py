"""Step 3: Live space weather from NOAA, turned into a chip-risk level.

Two free NOAA feeds:
  - Kp index: a 0-9 score of how disturbed Earth's magnetic field is
    (5+ is a storm). Storms heat the upper air and add drag and charging.
  - Proton flux (GOES satellites, >=10 MeV): high-energy particles from the
    sun. These are what cause "bit flips" (random data errors) in chips.
    10+ "pfu" (particles per sq cm per second per steradian) is an official
    solar radiation storm.

Risk levels (the error multipliers are assumptions for the sim):
  LOW       normal conditions                         errors x1
  ELEVATED  Kp >= 5 or protons >= 1 pfu               errors x3
  HIGH      Kp >= 7 or protons >= 10 pfu              errors x10

Usage:  python space_weather.py            # live
        python space_weather.py --fake-storm   # test what a storm looks like
"""
import argparse, json, urllib.request

KP_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
PROTON_URL = "https://services.swpc.noaa.gov/json/goes/primary/integral-protons-1-day.json"
ERROR_MULT = {"LOW": 1, "ELEVATED": 3, "HIGH": 10}


def _get(url):
    with urllib.request.urlopen(url, timeout=15) as r:
        return json.loads(r.read().decode())


def latest_kp():
    data = _get(KP_URL)
    last = data[-1]
    # NOAA has used two formats: list of dicts (current) and list of lists.
    if isinstance(last, dict):
        return float(last["Kp"]), last["time_tag"]
    return float(last[1]), last[0]


def latest_protons():
    rows = [r for r in _get(PROTON_URL) if r.get("energy") == ">=10 MeV"]
    last = rows[-1]
    return float(last["flux"]), last["time_tag"]


def risk_level(kp, protons):
    if kp >= 7 or protons >= 10:
        return "HIGH"
    if kp >= 5 or protons >= 1:
        return "ELEVATED"
    return "LOW"


def current(fake_storm=False):
    """Returns a dict the scheduler can use. Falls back to LOW if NOAA is unreachable."""
    if fake_storm:
        kp, kp_t, pr, pr_t, src = 7.33, "fake", 25.0, "fake", "FAKE STORM"
    else:
        try:
            kp, kp_t = latest_kp()
            pr, pr_t = latest_protons()
            src = "NOAA live"
        except Exception as e:
            print(f"NOAA unreachable ({e}); assuming LOW risk.")
            kp, kp_t, pr, pr_t, src = 0.0, "n/a", 0.0, "n/a", "fallback"
    level = risk_level(kp, pr)
    return {"kp": kp, "kp_time": kp_t, "protons_pfu": pr, "protons_time": pr_t,
            "risk": level, "error_mult": ERROR_MULT[level], "source": src}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake-storm", action="store_true")
    w = current(ap.parse_args().fake_storm)
    print(f"Source:   {w['source']}")
    print(f"Kp index: {w['kp']:.2f}  (at {w['kp_time']})")
    print(f"Protons:  {w['protons_pfu']:.2f} pfu  (at {w['protons_time']})")
    print(f"Risk:     {w['risk']}  (chip errors x{w['error_mult']})")

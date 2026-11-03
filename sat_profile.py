"""Which satellite and orbit the whole pipeline simulates.

Pick with the SAT_PROFILE environment variable (default "v3"):
  v3       our original made-up satellite on the ISS orbit / 12 real Starlink sats
  ai1_53   SpaceX AI1-style satellite, 650 km, 53 degree tilt (Starlink-like, lots of shadow)
  ai1_sso  same satellite, 650 km dawn-dusk sun-synchronous orbit, run in June
           (its worst shadow season; Aug-Apr it has no shadow at all)

AI1 numbers (company claims / press reports, not SpaceX engineering data):
  ~120 kW sustained compute (150 kW peak, not modeled), ~110 m2 deployable liquid
  radiator, expected operating orbit ~600-800 km. Not public: solar array size,
  battery size, exact orbit tilt.
Our estimates to fill the gaps:
  - solar 185 kW: enough to run 125 kW on average with ~24% of each lap in shadow
    (the 53-degree, 650 km orbit), plus a small margin
  - radiator radiates from both faces (radiator_check.py shows one face can't keep up)
  - battery: picked per orbit by battery_sizing.py (240 kWh at 53 degrees, 180 kWh dawn-dusk)
  - otherwise the same hardware in both orbits

Usage (Windows):  set SAT_PROFILE=ai1_53   then   python run_all.py --train
Usage (Mac/Linux): SAT_PROFILE=ai1_53 python run_all.py --train
"""
import math, os

PROFILE = os.environ.get("SAT_PROFILE", "v3")

AI1_SAT = {
    "solar_kw": 185.0,
    "gpu_max_kw": 120.0,
    "bus_kw": 5.0,
    # battery_sizing.py picks the smallest battery within 2% of max output, per orbit:
    # 240 kWh on the 53-degree orbit, 180 kWh dawn-dusk. Override with SAT_BATTERY_KWH.
    "battery_kwh": float(os.environ.get("SAT_BATTERY_KWH", 240.0 if PROFILE == "ai1_53" else 180.0)),
    "radiator_m2": 220.0,          # 110 m2 panel, both faces
    "heat_capacity_j_per_k": 1.9e6,
}

ALT_KM = 650.0
SSO_INC = None  # computed below


def _sso_inclination(alt_km):
    R, mu, J2 = 6378.137, 398600.4418, 1.08263e-3
    a = R + alt_km
    n = math.sqrt(mu / a ** 3)
    prec = 2 * math.pi / (365.2422 * 86400)
    return math.degrees(math.acos(-prec / (1.5 * J2 * (R / a) ** 2 * n)))


SSO_INC = _sso_inclination(ALT_KM)

ORBITS = {
    "ai1_53": {"alt": ALT_KM, "inc": 53.0, "sso": False, "start": None,
               "label": "AI1 specs · 53° tilt · 650 km"},
    "ai1_sso": {"alt": ALT_KM, "inc": SSO_INC, "sso": True, "start": (2026, 6, 15),
                "label": "AI1 specs · dawn-dusk sun-sync · 650 km · June (worst shadow season)"},
}

IS_AI1 = PROFILE in ORBITS
ORBIT = ORBITS.get(PROFILE)
LABEL = ORBIT["label"] if IS_AI1 else "ISS orbit / real Starlink fleet"


# Tight-power stress tests (SAT_SCENARIO), on top of an AI1 profile:
#   small_batt  battery cut to 120 kWh (half the sized 240 kWh)
#   aging       4 of the 12 satellites worn: battery at 60% capacity, solar at 85%
#   burst150    GPUs run at the reported 150 kW peak, with demand to match
SCENARIO = os.environ.get("SAT_SCENARIO", "")
AGING_SATS = {0, 4, 8}  | {11}   # one or two per orbital plane (12-satellite fleet)

# Fleet size (SAT_FLEET = planes x satellites per plane). Default 3x4 = 12.
PLANES_N, PER_PLANE_N = (int(x) for x in os.environ.get("SAT_FLEET", "3x4").lower().split("x"))
N_SATS = PLANES_N * PER_PLANE_N
if N_SATS != 12:
    LABEL += f" · {N_SATS} satellites"
if SCENARIO:
    LABEL += f" · stress test: {SCENARIO}"


def apply(sat):
    """Overwrite the default satellite numbers in place when an AI1 profile is chosen."""
    if IS_AI1:
        sat.update(AI1_SAT)
        if SCENARIO == "small_batt":
            sat["battery_kwh"] = 120.0
        elif SCENARIO == "burst150":
            sat["gpu_max_kw"] = 150.0
    return sat


def sat_params(sat, idx):
    """Numbers for satellite number idx in the fleet (only differ in the aging test)."""
    worn_here = idx in AGING_SATS if N_SATS == 12 else idx % 3 == 0   # one in three worn
    if SCENARIO == "aging" and worn_here:
        worn = dict(sat)
        worn["battery_kwh"] = sat["battery_kwh"] * 0.60
        worn["solar_kw"] = sat["solar_kw"] * 0.85
        return worn
    return sat


def job_rate_scale(sat):
    """Keep the fleet about as busy as before when the GPUs get bigger or the fleet grows."""
    return sat["gpu_max_kw"] / 100.0 * N_SATS / 12

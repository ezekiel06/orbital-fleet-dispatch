"""Step 15a: Is a 110 m2 radiator enough for SpaceX's AI1 compute satellite?

Reported AI1 figures: ~120 kW sustained / 150 kW peak compute power and a
~110 m2 deployable liquid radiator (company claims / press reports, not
SpaceX engineering data).

Radiators shed heat as infrared light. How much depends on area, how well the
surface radiates (emissivity, ~0.9 for good coatings), and temperature to the
4th power. A flat panel can radiate from one face or both, which roughly
doubles capacity. This finds the temperature the radiator settles at (steady
state) for each case. Radiator temperature is a floor for chip temperature;
real chips run some degrees hotter.

Usage:  python radiator_check.py
"""
from power_thermal import EARTH_HEAT

SIGMA, EMIS, AREA = 5.670e-8, 0.90, 110.0
BUS_KW, SUN_HEAT_KW = 5.0, 10.0   # always-on systems; sunlight hitting the craft (assumed)
# Share of each radiator face's view taken up by Earth (see README for the formula and sources)
VIEWS = {"facing deep space (no Earth)": 0.0, "edge-on to Earth (default)": 0.24, "one face straight down": (0.823 + 0.0) / 2}


def earth_kw(area, view, lit=True):
    """Earth's infrared glow + reflected sunlight absorbed by radiators of this area (kW)."""
    e = EARTH_HEAT
    ir = EMIS * e["earth_ir_wm2"] * view * area
    alb = e["rad_absorptivity"] * e["albedo"] * e["solar_wm2"] * e["albedo_sunlit_avg"] * view * area if lit else 0
    return (ir + alb) / 1000


def settle_temp_c(heat_kw, area):
    return (heat_kw * 1000 / (EMIS * SIGMA * area)) ** 0.25 - 273.15


def capacity_kw(temp_c, area):
    return EMIS * SIGMA * area * (temp_c + 273.15) ** 4 / 1000


if __name__ == "__main__":
    print("Heat the radiator can shed at a given temperature:")
    for t in (60, 80, 100):
        print(f"  {t:>3} C: one face {capacity_kw(t, AREA):5.0f} kW | both faces {capacity_kw(t, 2 * AREA):5.0f} kW")
    print("\nWhere the radiator settles (steady state):")
    for gpu in (120, 150):
        heat = gpu + BUS_KW + SUN_HEAT_KW
        one, two = settle_temp_c(heat, AREA), settle_temp_c(heat, 2 * AREA)
        print(f"  GPUs {gpu} kW (total heat {heat:.0f} kW): one face {one:5.0f} C | both faces {two:5.0f} C")
    print("\nWith Earth's heat glow and reflected sunlight (both faces, 220 m2, sunlit side):")
    for name, v in VIEWS.items():
        for gpu in (120, 150):
            heat = gpu + BUS_KW + SUN_HEAT_KW + earth_kw(2 * AREA, v)
            print(f"  {name:30} GPUs {gpu} kW: Earth adds {earth_kw(2*AREA, v):4.1f} kW -> settles at {settle_temp_c(heat, 2*AREA):5.1f} C")
    print("\nGPU limits in this sim: throttle at 80 C, hard limit 85 C.")

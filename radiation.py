"""Bit-flip (radiation error) rates, calibrated from real chip testing.

Source: Google's Project Suncatcher paper (arXiv 2511.19468), which
radiation-tested its Trillium TPU for a sun-synchronous orbit at ~650 km with
10 mm aluminum shielding (~150 rad per year):
  - silent data corruption: about one event per ~17 rad   -> ~9 per chip per year
  - uncorrectable memory (HBM) errors: one per ~44 rad    -> ~3.4 per chip per year
  Total job-breaking errors: ~12.4 per chip per year (our derivation from their figures).

Converted to "per kW of GPUs", using ~1.28 kW of server power per accelerator
(DGX H100: ~10.2 kW for 8 GPUs):
  12.4 / 1.28 / 525,600 minutes  ~= 1.8e-5 errors per kW per minute, orbit average.
So a 60 kW job sees ~0.0011 errors a minute and a 15 kW job ~0.00028.
(v1/v2 used a flat 0.002 per job per minute: ~2-7x too high.)

That orbit average already includes passes through the South Atlantic Anomaly
(SAA). For the fleet, which models the SAA as 5x worse, the rate outside it is
lower so the average stays the same: with satellites in the SAA ~14% of the
time, outside rate = average / (1 - 0.14 + 5 * 0.14) ~= 1.2e-5.

Still assumptions (not in Google's data): the SAA x5 factor, the storm
multipliers (x3 / x10 in space_weather.py), and that every error is caught and
the job restarted rather than silently giving a wrong answer.
"""
ERRORS_PER_CHIP_YEAR = 12.4
KW_PER_CHIP = 1.275
MIN_PER_YEAR = 525_600
SAA_MULT = 5
SAA_TIME_FRACTION = 0.14

ORBIT_AVG_PER_KW_MIN = ERRORS_PER_CHIP_YEAR / KW_PER_CHIP / MIN_PER_YEAR          # ~1.85e-5
OUTSIDE_SAA_PER_KW_MIN = ORBIT_AVG_PER_KW_MIN / (1 - SAA_TIME_FRACTION + SAA_MULT * SAA_TIME_FRACTION)


def p_error(kw, weather_mult, in_saa=None):
    """Chance that a running job of `kw` hits a job-breaking error this minute.
    in_saa=None: orbit-average rate (single-satellite models with no SAA).
    in_saa=True/False: fleet model with the SAA handled explicitly."""
    if in_saa is None:
        base = ORBIT_AVG_PER_KW_MIN
    else:
        base = OUTSIDE_SAA_PER_KW_MIN * (SAA_MULT if in_saa else 1)
    return kw * base * weather_mult


if __name__ == "__main__":
    print(f"orbit average: {ORBIT_AVG_PER_KW_MIN:.2e} per kW-minute | outside SAA: {OUTSIDE_SAA_PER_KW_MIN:.2e}")
    for kw in (15, 30, 60):
        print(f"  {kw} kW job, calm: {p_error(kw, 1):.5f} per minute (old v2: 0.00200)")

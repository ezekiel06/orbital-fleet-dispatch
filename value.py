"""Step 13: What is the hybrid dispatcher's extra computing worth in dollars?

Method:
  1. Extra computing = hybrid's work minus the comparison's work (kWh of GPU work),
     from hybrid_results.json (12 real Starlink satellites, fresh test runs),
     blended over a weather mix, scaled to one satellite for one year.
  2. Dollar value = that kWh x what one kWh of GPU-server power earns when
     rented out at cloud GPU prices.

Rental value per kWh of server power (prices checked Nov 2026, see sources):
  H100: ~$2.00-$3.30 per GPU-hour; a DGX H100 server draws ~10.2 kW for 8 GPUs
        (~1.28 kW per GPU incl. CPUs etc.)       -> ~$1.57-$2.59 per kWh
  B200: ~$6.25-$6.69 per GPU-hour (typical on-demand); a DGX B200 draws
        ~14.3 kW for 8 GPUs (~1.79 kW per GPU)   -> ~$3.50-$3.74 per kWh

Assumptions (state these when you present it):
  - Every extra kWh of computing finds a paying customer at those prices.
  - The satellite design is ours (100 kW of GPUs), not a real company's.
  - Space weather mix: "realistic" = calm 90% / moderate 8% / big storm 2%
    (big storms are rare, a few days a year near solar maximum);
    "stormy" = 50/25/25, the harsh mix used in training.

Usage:  python value.py
Saves:  value_results.json
"""
import json

PRICES = {  # $ per kWh of server power: (low, high)
    "H100": (2.00 / 1.275, 3.30 / 1.275),
    "B200": (6.25 / 1.7875, 6.69 / 1.7875),
}
MIXES = {"realistic": {"LOW": 0.90, "ELEVATED": 0.08, "HIGH": 0.02},
         "stormy": {"LOW": 0.50, "ELEVATED": 0.25, "HIGH": 0.25}}
RUNS_PER_YEAR = 365 * 24 / 6   # each test run is 6 hours
import sat_profile
FLEET = sat_profile.N_SATS     # satellites in the test fleet (SAT_FLEET, default 12)


def main():
    h = json.load(open("hybrid_results.json"))["results"]
    out = {"prices_per_kwh": {k: [round(v[0], 2), round(v[1], 2)] for k, v in PRICES.items()}, "mixes": {}}
    for mname, mix in MIXES.items():
        work = {p: sum(mix[w] * h[w]["mean"][p]["work_kwh"] for w in mix) for p in ("local", "switch", "hybrid")}
        res = {}
        for base, label in (("local", "vs no sharing"), ("switch", "vs best simple rule")):
            extra_kwh = (work["hybrid"] - work[base]) / FLEET * RUNS_PER_YEAR  # per satellite per year
            pct = (work["hybrid"] / work[base] - 1) * 100
            val = {g: [round(extra_kwh * lo), round(extra_kwh * hi)] for g, (lo, hi) in PRICES.items()}
            res[label] = {"extra_pct": round(pct, 2), "extra_kwh_per_sat_year": round(extra_kwh), "usd_per_sat_year": val}
        out["mixes"][mname] = res
        print(f"\n=== Weather mix: {mname} ({', '.join(f'{k} {int(v*100)}%' for k, v in mix.items())}) ===")
        for label, r in res.items():
            lo = min(v[0] for v in r["usd_per_sat_year"].values()); hi = max(v[1] for v in r["usd_per_sat_year"].values())
            print(f"  {label:20} +{r['extra_pct']:.1f}% work = {r['extra_kwh_per_sat_year']:,} kWh/satellite/year "
                  f"= ${lo:,}-${hi:,} per satellite per year")
            for n in (100, 1000, 10000):
                print(f"      x {n:>6,} satellites: ${lo*n/1e6:,.1f}M-${hi*n/1e6:,.1f}M per year")
    with open("value_results.json", "w") as f:
        json.dump(out, f, indent=1)
    print("\nSaved value_results.json")


if __name__ == "__main__":
    main()

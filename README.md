# Orbital Fleet Dispatch

An open-source simulator for **AI data centers in orbit**: how much computing a fleet of
solar-powered GPU satellites can deliver, and how much better scheduling recovers, without
wearing out batteries or cooking chips.

It runs on real public data: live satellite orbits (CelesTrak), live space weather (NOAA),
real Starlink satellites, bit-flip rates calibrated from Google's TPU radiation tests, and
real satellite telemetry (ESA OPS-SAT, SatNOGS).

**Live pages:** [dashboard, orbit view and fleet sizer](https://ezekiel06.github.io/orbital-fleet-dispatch/), no install needed.

**Bring your own specs.** Every hardware number is a public estimate. Swap in real power,
battery, and radiator figures in `sat_profile.py` and rerun.

## Headline results

All results: 12-satellite fleet, 6-hour runs on held-out orbits, every scheduler compared on
identical runs, 95% ranges by bootstrap. Dollar figures use today's GPU rental prices
(H100 low end, B200 high end) and assume every extra kWh of computing is sold.

### 1. Sharing work across the fleet: ~5% more computing

Satellites pass jobs to each other over laser links, so work dodges Earth's shadow and the
South Atlantic Anomaly (a radiation hot spot). Tested on SpaceX AI1-style specs
(~120 kW compute, ~110 m² radiator, 650 km):

| Orbit | More computing vs. no sharing | Value per 1,000 satellites / year |
| --- | --- | --- |
| 53° tilt (Starlink-like) | +4.6% | $61.5M – $146.7M |
| Dawn-dusk sun-synchronous (June, worst case) | +5.6% | $75.5M – $180.2M |

### 2. A trained AI dispatcher: +1–1.6% more when power is tight

With comfortable hardware, plain sharing captures nearly everything and the AI adds ~0%.
When power gets tight, a reinforcement-learning dispatcher (PPO) beats the best simple rule:

| Stress test | AI vs. best simple rule | Runs won (of 40, per weather) | Value per 1,000 satellites / year |
| --- | --- | --- | --- |
| Small battery (120 kWh) | +1.5% | 29–37 | $18.9M – $45.1M |
| Aging fleet (4 of 12 satellites worn) | +1.6% | 28–36 | $21.1M – $50.4M |
| Bursting to 150 kW | +1.0% | 31–37 | $15.3M – $36.5M |

The battery never dropped below its 70% safety floor in any test.

### 3. Bigger fleets: both gains grow

Same AI1 53° setup at 12, 48, and 96 satellites (demand scaled to fleet size,
`scale_test.py`). Sharing has more places to put each job as the fleet grows:

| Fleet | Sharing vs. no sharing (calm / moderate / big storm) |
| --- | --- |
| 12 satellites | +3.3% / +3.0% / +1.8% |
| 48 satellites | +4.3% / +4.2% / +4.3% |
| 96 satellites | +4.8% / +4.9% / +4.4% |

The AI dispatcher's edge holds or grows with fleet size. On an aging fleet (one in three satellites
worn), against the best simple rule:

| Fleet | AI vs. best rule (realistic weather) | AI vs. best rule (stormy weather) | Fresh test runs won | Value per 1,000 satellites / year |
| --- | --- | --- | --- | --- |
| 12 satellites | +1.6% | +1.6% | 98 of 120 | $21M – $50M |
| 48 satellites | +2.2% | +2.6% | 96 of 96 | $30M – $72M |
| 96 satellites | +2.3% | +3.4% | 96 of 96 | $31M – $74M |

At 96 satellites the AI also won all 72 runs of the separate confirmation test, and no battery
went below its 70% floor. Realistic weather = calm 90% / moderate storm 8% / big storm 2%;
stormy = 50 / 25 / 25.

### 4. Laser-link limits: the gain survives real bandwidth

Moving a running job means sending its working state over a laser link. `link_test.py` caps each
satellite's lasers at real speeds and makes jobs carry more data. Each case runs two dispatchers:
**always** moves a job whenever another satellite has room; **smart** moves it only when the transfer
is quicker than waiting and short next to the work left. Sharing vs. no sharing, AI1 specs, 53°,
calm / moderate / big storm:

| Link setup | Data per job moved | 12 sats, always | 12 sats, smart | 48 sats, always | 48 sats, smart |
| --- | --- | --- | --- | --- | --- |
| Starlink today (3 × 100 Gbps) | 100 GB | +3.8 / +3.5 / +3.4% | +3.6 / +3.5 / +4.7% | +3.7 / +3.8 / +3.0% | +3.8 / +3.8 / +4.0% |
| Starlink today | 5 TB | +0.2 / −0.6 / +1.3% | +4.4 / +4.0 / +4.7% | +0.9 / +0.7 / +0.5% | +4.3 / +3.9 / +4.1% |
| Starlink V3 (6 × 400 Gbps) | 5 TB | +3.2 / +2.7 / +3.1% | +3.5 / +3.2 / +2.8% | +3.7 / +3.5 / +3.7% | +3.9 / +3.3 / +3.4% |
| Slow (1 × 10 Gbps) | 1 TB | −3.2 / −4.5 / −3.4% | +4.4 / +4.0 / +4.7% | −2.4 / −2.6 / −1.7% | +4.3 / +3.9 / +4.1% |
| Slow | 5 TB | −16.3 / −17.1 / −14.2% | +4.4 / +4.0 / +4.7% | −13.5 / −14.6 / −12.0% | +4.3 / +3.9 / +4.1% |

Most of the gain comes from *where new jobs start*, which needs only a tiny dispatch message, not
from moving running jobs. Moving heavy jobs blindly over slow links costs more than it saves; the
smart rule avoids that and keeps sharing at +3–5% in every case tested. It is on by default when
link limits are on (`SAT_MOVE=always` turns it off). Sources: Starlink sustains 100 Gbps per laser
link ([SPIE Photonics West via TechSpot](https://www.techspot.com/community/topics/starlink-satellites-exchange-petabytes-of-data-per-day-through-space-laser-links.284184));
Starlink V3 reports six 400 Gbps lasers and 1 Tbps downlink ([Converge Digest](https://convergedigest.com/starlink-v3-boosts-per-satellite-downlink-capacity-to-1-tbps/));
commercial laser links run 1–100 Gbps and Google's Suncatcher design needs ~10 Tbps with satellites
100–200 m apart ([paper](https://arxiv.org/abs/2511.19468)). Lasers per satellite today (3) is our assumption.

### 5. Real AI workloads: sharing gains more, not less

The tests above use three made-up job types. `helios_test.py` replays **real deep-learning jobs**
from SenseTime's [Helios GPU datacenter trace](https://github.com/S-Lab-System-Group/HeliosData)
(4 clusters, 2020, ~1.6 million GPU jobs): real arrival bursts, real GPU counts, real run times from
seconds to days (`workload.py`; 1 GPU = 1.275 kW). AI1 specs, 53°, 6 different 12-hour slices of
the trace per case, measuring GPU work actually delivered. Sharing vs. no sharing, calm / moderate /
big storm:

| Fleet | Demand vs. fleet GPU power | Sharing gain |
| --- | --- | --- |
| 12 satellites | 80% | +11.6 / +11.3 / +11.5% |
| 12 satellites | 100% | +6.3 / +7.9 / +7.8% |
| 12 satellites | 120% | +5.8 / +3.5 / +3.5% |
| 48 satellites | 80% | +9.7 / +11.0 / +7.8% |
| 48 satellites | 100% | +8.9 / +7.5 / +5.6% |
| 48 satellites | 120% | +4.1 / +4.6 / +2.3% |
| 48 satellites, Starlink-today links, 80 GB per GPU moved | 100% | +7.1 / +5.4 / +2.5% |

Sharing won 131 of 144 paired runs, and no battery went below its 70% floor. Real jobs arrive in
bursts, so a shared fleet has more idle capacity to soak them up than the steady made-up jobs
suggested. The losses (13 runs) were mostly on the small 12-satellite fleet, where one huge job can
block a satellite. Jobs larger than one satellite (over ~94 GPUs, about 0.1% of jobs) were left out;
running them would take several satellites working as one machine, which isn't modeled.

### 6. Money saved at launch: less battery to fly

A sharing fleet matches a non-sharing fleet's output with **~51 kWh less battery per satellite**
(240 → ~189 kWh). That's **300–424 kg** per satellite (at 170 / 120 Wh/kg space battery packs):

| Launch price | Saved per 1,000 satellites (one time) |
| --- | --- |
| $100/kg (Starship target) | $30M – $42M |
| $1,000/kg (illustrative early Starship) | $300M – $424M |

### 7. Design checks on SpaceX's reported AI1 numbers

- A 110 m² radiator works only if it sheds heat from **both faces**: it settles near 58 °C at
  120 kW, versus ~121 °C on one face (`radiator_check.py`).
- Keeping ~99% of full output through Earth's shadow on a 53° orbit takes **~240 kWh** of
  battery with a 70% floor (`battery_sizing.py`).

### 8. Problem detection on real ESA telemetry

On ESA's OPS-SAT anomaly benchmark, a supervised model catches 102 of 113 real problems
(F1 0.91) vs. 61 for fixed alarm limits, close to the paper's published Random Forest result.

## Try it in your browser

`web/index.html` is an interactive sizer: set GPU power, solar, battery, radiator, orbit,
space weather, and fleet wear, and it runs the same 12-satellite fleet model live in the page
(a JavaScript port of `fleet.py`, `web/sim.js`, which matches the Python within ~1–2%). It shows
the gain from sharing, its dollar value, design warnings (radiator too small, battery floor
broken), and the battery mass sharing saves. Open the file directly; no install needed.

`web/orbit.html` is an animated 3D view of the fleet: satellites circling Earth, passing through
its shadow and the South Atlantic Anomaly, with battery levels and laser-link job handoffs.

## Quick start

```bash
pip install -r requirements.txt
python run_all.py              # rerun every test with the included trained models (~10 min)
python run_all.py --train      # also retrain both AIs (~30-40 min)
python run_all.py --refresh    # download today's Starlink orbits first
```

Then open `dashboard.html`.

**Other setups.** Set `SAT_PROFILE` before running:

```bash
SAT_PROFILE=ai1_53  python run_all.py --train      # AI1 specs, 53° orbit
SAT_PROFILE=ai1_sso python run_all.py --train      # AI1 specs, dawn-dusk sun-synchronous
SAT_PROFILE=ai1_53 SAT_SCENARIO=aging python fleet_rl.py train   # stress tests:
                                                    # small_batt | aging | burst150
SAT_PROFILE=ai1_53 SAT_FLEET=6x8 python fleet_rl.py train         # 48 satellites (planes x per plane)
```

(Windows: `set SAT_PROFILE=ai1_53` on its own line first.) Run each setup in its own folder,
since results are written to the current folder.

**Real telemetry (optional):** `python fetch_opssat.py` then `python telemetry_check.py`.
For SatNOGS, put your own API token in `satnogs_token.txt` (git-ignored, never commit it) and
run `python satnogs_fetch.py`, then `python canvas_analysis.py`.

## How it works

| Step | File | What it does |
| --- | --- | --- |
| 1 | `orbit_sun.py` | Satellite position and sun/shadow, minute by minute |
| 2 | `power_thermal.py` | Solar, battery, GPU heat, radiator physics |
| 3 | `space_weather.py` | Live NOAA storm index and proton flux → chip risk |
| 4 | `radiation.py` | Bit-flip rates calibrated from Google's TPU radiation tests |
| 5 | `scheduler.py` | Hand-written rules baseline |
| 6 | `rl_env.py`, `train_rl.py`, `evaluate_rl.py` | Single-satellite AI with a hard safety limit |
| 7 | `fleet.py`, `fleet_rl.py` | 12-satellite fleet, laser-link sharing, fleet AI |
| 8 | `confirm_fleet.py`, `hybrid.py` | Paired 40-run tests; AI used only where it proved better |
| 9 | `value.py`, `savings.py` | Money made (extra computing) and saved (battery mass) |
| — | `scale_test.py` | Sharing gain at 12 / 48 / 96 satellites |
| — | `link_test.py` | Sharing gain with real laser-link speed limits |
| — | `workload.py`, `helios_test.py` | Real AI jobs from the Helios GPU trace |
| 10 | `radiator_check.py`, `battery_sizing.py` | Design checks |
| 11 | `telemetry_check.py`, `satnogs_fetch.py`, `canvas_analysis.py` | Real telemetry |
| — | `sat_profile.py` | Satellite and orbit setups; put your own specs here |
| — | `run_all.py`, `make_dashboard.py` | One-command pipeline and dashboard |

Saved results for every setup are in `results/`.

## Assumptions and limits

Being upfront about these is the point:

- **Hardware is estimated.** AI1 figures are company claims and press reports. Solar size
  (185 kW), battery size, and thermal mass are our estimates. Real specs aren't public.
- **Bit-flip rates** are derived from Google's published TPU test results. The South
  Atlantic Anomaly (×5) and storm multipliers (×3, ×10) are assumptions.
- **Links:** results above use unlimited links unless noted; `SAT_LINK=today|v3|slow` turns on
  speed limits. Multi-hop routing and laser range are not modeled.
- **Simple heat model:** the whole satellite has one temperature; no hot spots.
- **Every extra kWh is assumed sold** at today's rental prices. GPU prices move fast.
- **Launch prices:** Starship has no published price; the $/kg values are scenarios.
- **Fleet size:** tested at 12, 48, and 96 satellites, AI included. Thousands are untested.
- **Training is random:** 3 tries per AI, the best picked on separate validation orbits.
  Some tries lose to the rules, which is why that selection step matters.

## Related work

Other open projects model orbital compute from different angles, and they're worth a look:

- [orbital-compute](https://github.com/ShipItAndPray/orbital-compute): broad constellation
  simulator (power, thermal, radiation, laser links, ground stations, economics) with rule-based
  and look-ahead schedulers.
- [OrbitCompute](https://github.com/lucasligenza/OrbitCompute): interactive visual simulation
  of orbital compute nodes with energy- and thermal-aware schedulers.
- [OrbitalDC Studio](https://github.com/pratik-ingle/Space-data-centers): single-satellite
  5-year mission model with degradation and Monte Carlo sensitivity.
- [orbital-scheduler](https://github.com/gokkayai/orbital-scheduler): paired comparison of
  energy-aware vs. least-loaded scheduling.
- [Celestial](https://arxiv.org/abs/2204.06282): virtual testbeds for software on LEO edge
  satellites.

What this project adds: a reinforcement-learning dispatcher tested head to head against simple
rules (including where it loses), checks against real satellite telemetry (ESA OPS-SAT,
SatNOGS), bit-flip rates calibrated from Google's TPU radiation tests, live NOAA space weather,
and results at 12, 48, and 96 satellites.

## Data sources and credits

- Orbits: [CelesTrak](https://celestrak.org) GP data
- Space weather: [NOAA Space Weather Prediction Center](https://www.swpc.noaa.gov)
- Radiation calibration: [Google Project Suncatcher paper](https://arxiv.org/abs/2511.19468)
- Telemetry: [ESA OPS-SAT anomaly dataset (OPSSAT-AD)](https://zenodo.org/records/12588359),
  CC BY 4.0, and [benchmark paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12041257/);
  [SatNOGS DB](https://db.satnogs.org) (Libre Space Foundation). Raw data is not
  redistributed here; use the fetch scripts.
- AI1 specs: [SpaceX orbital compute summary](https://spacenexus.us/orbital-data-centers/spacex-orbital-compute)
- GPU prices: [H100 comparison](https://intuitionlabs.ai/pdfs/h100-rental-prices-compared-1-49-6-98-hr-across-15-cloud-providers-2026.pdf),
  [B200 pricing](https://www.morphllm.com/nvidia-b200)
- Battery packs: [Airbus LEO battery](https://www.satnow.com/products/batteries/airbus/98-1213-stellar-lowpower-lp-),
  [GS Yuasa module](https://www.satnow.com/products/batteries/gs-yuasa-lithium-power/98-1270-ma190-206)
- Launch prices: [New Space Economy, June 2026](https://newspaceeconomy.ca/2026/06/21/how-accurate-were-elon-musks-falcon-9-and-starship-kg-claims/)
- Real AI jobs: [Helios trace](https://github.com/S-Lab-System-Group/HeliosData), SenseTime / S-Lab, SC '21 paper; raw data not redistributed here
- RL: [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3); orbits: [Skyfield](https://rhodesmill.org/skyfield/)

## License

MIT. See `LICENSE`. Data keeps its own licenses (see above).

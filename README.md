# CEAC-SW-LCB simulator

Companion code for the project report

> **Scalable Capability- and Energy-Aware Clustered SW-LCB for Heterogeneous
> Spectrum Allocation in UAV-Assisted Emergency Communication Networks**

Siddhant Jain (24BCI0190), Divyam Taparia (24BCE2373), Utsav Kesharwani (24BCE2434).
Computer Networks course project, Review 3.

The simulator implements the emergency-network scenario of Li et al.,
*QoS-aware heterogeneous spectrum allocation for UAV-assisted emergency
communication networks*, Vehicular Communications 60:101042 (2026), and the
CEAC-SW-LCB extension proposed in Review 2. It produces the raw data behind
Section 5 of the report: Table 5 (main comparison), Table 6 (clustering), Table 7
(ablation), Table 8 (breakpoints), Table 9 (mechanism attribution) and Figures 2-4.

## Contents

- [Quick start](#quick-start)
- [Development checks](#development-checks)
- [Reproducing the paper's results](#reproducing-the-papers-results)
- [Precomputed outputs](#precomputed-outputs)
- [Experiments](#experiments)
- [Repository layout](#repository-layout)
- [What is simulated](#what-is-simulated)
- [Algorithms](#algorithms)
- [Output columns](#output-columns)
- [Reproducibility notes](#reproducibility-notes)
- [Citation and licence](#citation-and-licence)

## Quick start

Requirements: Python >= 3.12 and [uv](https://docs.astral.sh/uv/). The published
results were produced with Python 3.15.0, NumPy 2.5.4, pandas 3.0.6 and
Matplotlib 3.11.2.

```bash
cd simulation
uv sync

# one experiment, about 8 minutes for e1
uv run python run_experiments.py --only e1

# short smoke run (3 seeds, 300-400 slots), useful after code changes
uv run python run_experiments.py --quick --only e1 e6
```

`PYTHONPATH=.` is only needed if you invoke the modules outside `uv run`.

## Development checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check
```

All three pass on the current tree. The linter and type-checker are in the `dev`
dependency group, so `uv sync` installs them.

## Reproducing the paper's results

The full pipeline is sequential and takes roughly 35 minutes:

```bash
uv sync

uv run python run_experiments.py      # e1-e7, e9, e10  -> results/*.csv
uv run python closure_study.py        # e8               -> results/e8_closure.csv
uv run python shock_study.py          # breakpoint study -> results/tables/shock.md
uv run python attrib_breaks.py        # attribution      -> results/attrib_breaks.csv
uv run python dump_series.py          # time series      -> results/series.csv

uv run python analyse.py              # summary tables   -> results/tables/*.md
uv run python plots.py                # figures          -> figures/*.png
```

Running these commands overwrites `results/` and `figures/`. To compare a re-run
against the published artefacts, copy `results/` and `figures/` elsewhere first,
or change the output prefixes in the scripts.

## Precomputed outputs

The repository ships the outputs used in the report, so the results can be
inspected without re-running the simulator:

| Path | Contents |
|---|---|
| `results/*.csv` | one row per (algorithm, seed, parameter setting) |
| `results/tables/*.md` | generated summary tables |
| `results/series.csv` | slot-by-slot trace used for Figure 4 |
| `figures/fig*.png` | 300 dpi figures |
| `docs/RESULTS_final.md` | Section 5 as used in the report |
| `docs/DESIGN_CHANGES.md` | gaps in Review 2 found while implementing it |

A read-only provenance check on 2026-10-22 recomputed the `n_users = 50`,
`seed = 0` row of `e1_users.csv` and reproduced all summary statistics of all
six algorithms to floating-point noise (differences below 1e-13). See
[Reproducibility notes](#reproducibility-notes).

## Experiments

| id | Script | What it measures | Seeds | Slots | Output | Used in report |
|---|---|---|---|---|---|---|
| e1 | `run_experiments.py` | main comparison, 50-110 users, 6 algorithms | 8 | 1600 | `e1_users.csv` | Table 5, Figures 2-3 |
| e2 | `run_experiments.py` | density scaling, 70/150/220 users | 5 | 900 | `e2_density.csv` | Section 5.6 (density) |
| e3 | `run_experiments.py` | UAV speed 2-20 m/s, uniform deployment | 5 | 900 | `e3_speed.csv` | supplementary |
| e4 | `run_experiments.py` | window exponent alpha 0.12-0.40 | 5 | 900 | `e4_window.csv` | Section 5.2 (window) |
| e5 | `run_experiments.py` | energy weight beta3, throughput vs lifetime | 5 | 1600 | `e5_energy.csv` | Section 5.3 (beta3) |
| e6 | `run_experiments.py` | ablation, one mechanism removed at a time | 6 | 1200 | `e6_ablation.csv` | Table 7 |
| e7 | `run_experiments.py` | cluster quality, pruning threshold, head churn | 8 | 300 | `e7_clusters.csv`, `e7_pruning.csv` | Table 6, Section 5.6 (pruning) |
| e8 | `closure_study.py` | closure quality of GPS / RSSI / two-stage clustering | 8 | 200 | `e8_closure.csv` | Section 4.6 (calibration) |
| e9 | `run_experiments.py` | structured hotspot deployment, 5-45 m/s | 5 | 900 | `e9_mobility.csv` | Section 5.7 (speed) |
| e10 | `run_experiments.py` | breakpoints 0-0.08 per slot | 5 | 900 | `e10_breakpoints.csv` | Table 8 |
| - | `shock_study.py` | regret before and after a breakpoint | 3 | 900 | `results/tables/shock.md` | Section 5.7 (recovery) |
| - | `attrib_breaks.py` | CEAC mechanism attribution at 0.08 per slot | 10 | 900 | `attrib_breaks.csv` | Table 9 |
| - | `dump_series.py` | one slot-by-slot run for the transient figure | 1 | 1600 | `series.csv` | Figure 4 |

Table and figure numbers refer to the Review 3 report. e3 is an early uniform
deployment sweep, kept for completeness; its question is answered by e9, which
uses the structured deployment the paper discusses in Section 5.7. The e9/e10
pair tests the base paper's non-stationarity assumption: speed alone is flat,
while abrupt density shocks are expensive for every learner and are where the
clustering mechanisms start to pay.

## Repository layout

```
simulation/
  sim/
    config.py       parameters, service classes, POC overlap, arm layout
    scenario.py     nodes, capabilities, deployment, mobility, breakpoints
    radio.py        path loss, fading, blockage, rates, interference, utility
    cluster.py      two-stage clustering, pruning, head election, borders
    learners.py     random, greedy, UCB, D-eps-TS, SW-LCB, CEAC
    runner.py       Monte-Carlo driver, cost function, metrics, ablation flags
  run_experiments.py  e1-e7, e9, e10
  closure_study.py    e8
  shock_study.py      event-aligned regret around breakpoints
  attrib_breaks.py    mechanism attribution under frequent breakpoints
  dump_series.py      one slot-by-slot trace
  analyse.py          CSV -> results/tables/*.md
  plots.py            tables/CSV -> figures/*.png
  docs/
    RESULTS_final.md  Section 5 as used in the report
    DESIGN_CHANGES.md design decisions and deviations from Review 2
  results/            raw CSVs and generated tables
  figures/            generated figures
```

## What is simulated

| Layer | Module | Source |
|---|---|---|
| Deployment, mobility, capabilities, battery | `sim/scenario.py` | base paper Table 4 |
| Path loss, Nakagami-m fading, shadowing, rubble blockage | `sim/radio.py` | base paper Eqs. (7)-(18) |
| Co-channel interference, rates, QoS utility | `sim/radio.py` | base paper Eqs. (15)-(19), Table 3 |
| Two-stage GPS-then-RSSI clustering, pruning, head election | `sim/cluster.py` | Review 2 Sections 4-8 |
| Bandit learners | `sim/learners.py` | base paper Eq. (32); Review 2 Eq. (8) |
| Costs, coordination, metrics, Monte-Carlo driver | `sim/runner.py` | Review 2 Eq. (7) |

### Modelling decisions the papers leave open

These are deliberate; `docs/DESIGN_CHANGES.md` records each one:

* **Bandwidth sharing.** A channel's bandwidth is split equally among the nodes
  using it (PRB scheduling for LTE, airtime for 802.11). The base paper's
  per-device bandwidth is therefore an upper bound.
* **POC overlap.** 22 MHz channels on a 5 MHz raster; interference is weighted
  by the normalised spectral overlap.
* **Rubble.** Ten circular blockage footprints attenuate crossing links by
  20 dB. GPS cannot see them, RSSI can; this is the case Stage 2 exists for.
* **Power modes.** Each channel can be used at 20 dBm or 14 dBm. Review 2 Eq. (7)
  needs a per-arm transmit energy, which only exists if the arm can change the
  transmit power. Baselines transmit at the fixed 20 dBm of the base paper.
* **Battery.** A 300 J (ground) / 450 J (UAV) radio-energy budget; propulsion
  energy is excluded, as Review 2 states. A depleted node stops transmitting.
* **Structured mobility.** The default deployment is uniform with Gauss-Markov
  motion. `n_hotspots > 0` switches to drifting hotspots, patrolling UAVs and
  optional breakpoints. A breakpoint merges one team into another or disperses
  it, so local node density changes abruptly. That is the realisation of the
  base paper's breakpoint process used in e9/e10.
* **Utility slope.** Eq. (19) leaves kappa free; the sigmoid is centred on each
  service's own rate target.

## Algorithms

| Name | Arms | History | Coordination |
|---|---|---|---|
| Random | 16 (5 LTE + 11 POC channels) | none | none |
| Greedy | 16 | full | none |
| UCB | 16 | full | none |
| D-eps-TS | 16 | full, decaying epsilon | none |
| SW-LCB | 16 | sliding window, alpha = 0.21 | none |
| CEAC-SW-LCB | capability-filtered, two power modes | per-cluster adaptive window | cluster prior + boundary heads |

All learners minimise the same cost (Review 2 Eq. 7). The confidence term is
subtracted from the cost because the problem is a minimisation.

## Output columns

`conflicts_early`, `throughput_early`, `utility_early`, `qos_early` are means
over slots 100-500, before batteries start to deplete. `energy_per_mbit`,
`active_end` (fraction of nodes still powered at the end) and `total_mbit` cover
the full 1600-slot horizon. The unprefixed columns cover the post-warm-up
horizon. `convergence` is computed over the first 1200 slots and is not used in
the report, because depletion contaminates late-run conflicts.

## Reproducibility notes

* Every run uses common random numbers across algorithms: for a given seed all
  algorithms see the same positions, shadowing, fading and blockage. The
  environment RNG is seeded from the run seed, and each algorithm gets its own
  independent RNG for arm selection.
* Results are deterministic for a fixed seed and package set. Re-running the
  same command reproduces the CSV bit-for-bit on the same platform.
* The reported numbers were produced with the parameter values in
  `sim/config.py`. Changing them changes the results; if you do, re-run the full
  pipeline rather than comparing against the shipped CSVs.
* The report's Section 5 text is hand-written around generated tables and
  figures. `docs/RESULTS_final.md` is the kept copy; the earlier working draft
  and its placeholder-filling scripts have been retired.

## Citation and licence

If you use this code, please cite the project report; see `CITATION.cff`. Update
the citation with the DOI or venue once the paper is published.

The code is released under the MIT licence; see `LICENSE`.

The base paper, the review documents, the case studies and the compiled report
are not redistributed here. They remain the work of their respective authors.

"""Experiment driver for the Case Study 3 (Results and Discussion) section.

Usage
-----
    uv run python run_experiments.py               # everything
    uv run python run_experiments.py --only e1 e6  # selected experiments
    uv run python run_experiments.py --quick       # short smoke run

Every experiment writes ``results/<name>.csv`` with one row per
(algorithm, seed, parameter setting).  ``analyse.py`` turns those rows into the
summary tables and ``plots.py`` into the figures.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from sim.cluster import ClusterManager
from sim.config import Config
from sim.radio import Radio
from sim.runner import AlgoSpec, run_group
from sim.scenario import Scenario

RESULTS = "results"
BASE_BETAS = (0.55, 0.45, 0.0)
ENERGY_BETAS = (0.44, 0.36, 0.20)


# ----------------------------------------------------------------------------
def _specs(energy: bool = True) -> list[AlgoSpec]:
    b = BASE_BETAS
    specs = [
        AlgoSpec("Random", "random", b),
        AlgoSpec("Greedy", "greedy", b),
        AlgoSpec("UCB", "ucb", b),
        AlgoSpec("D-eps-TS", "ts", b),
        AlgoSpec("SW-LCB", "swlcb", b),
    ]
    if energy:
        specs.append(
            AlgoSpec(
                "CEAC-SW-LCB",
                "ceac",
                ENERGY_BETAS,
                cluster=True,
                capability=True,
                low_power=True,
                aggregate=True,
                boundary=True,
                adaptive=True,
            )
        )
    return specs


def _rows(res, exp, **params):
    out = []
    for name, r in res.items():
        row = dict(exp=exp, algo=name, seed=r.seed, **params)
        row.update(r.summary)
        out.append(row)
    return out


def _save(rows, name):
    df = pd.DataFrame(rows)
    path = f"{RESULTS}/{name}.csv"
    df.to_csv(path, index=False)
    print(f"  wrote {path}  ({len(df)} rows)", flush=True)
    return df


def _sweep(name, spec_fn, param_grid, seeds, cfg_fn, exp_name):
    rows = []
    for params in param_grid:
        cfg = cfg_fn(params)
        for seed in range(seeds):
            t0 = time.time()
            res = run_group(cfg, spec_fn(params), seed)
            rows += _rows(res, exp_name, **params)
            print(
                f"  {exp_name} {params} seed={seed} ({time.time() - t0:.1f}s)",
                flush=True,
            )
    return _save(rows, name)


# ----------------------------------------------------------------------------
def e1_users(quick=False):
    """Main comparison versus the number of ground users."""
    seeds = 3 if quick else Config.seeds
    grid = [{"n_users": n} for n in (50, 70, 90, 110)]
    cfg0 = Config(slots=400 if quick else Config.slots)
    _sweep(
        "e1_users",
        lambda p: _specs(True),
        grid,
        seeds,
        lambda p: replace(cfg0, n_users=p["n_users"]),
        "users",
    )


def e2_density(quick=False):
    """Does the clustering gain grow with node density?"""
    seeds = 3 if quick else 5
    specs = [
        AlgoSpec("Random", "random", BASE_BETAS),
        AlgoSpec("SW-LCB", "swlcb", BASE_BETAS),
        AlgoSpec(
            "CEAC-noCluster", "ceac", ENERGY_BETAS, capability=True, low_power=True
        ),
        AlgoSpec(
            "CEAC-SW-LCB",
            "ceac",
            ENERGY_BETAS,
            cluster=True,
            capability=True,
            low_power=True,
            aggregate=True,
            boundary=True,
            adaptive=True,
        ),
    ]
    grid = [{"n_users": n} for n in (70, 150, 220)]
    cfg0 = Config(slots=500 if quick else 900)
    _sweep(
        "e2_density",
        lambda p: specs,
        grid,
        seeds,
        lambda p: replace(cfg0, n_users=p["n_users"]),
        "density",
    )


def e3_speed(quick=False):
    """UAV speed sweep: non-stationarity increases with speed."""
    seeds = 3 if quick else 5
    specs = [
        AlgoSpec("UCB", "ucb", BASE_BETAS),
        AlgoSpec("SW-LCB", "swlcb", BASE_BETAS),
        AlgoSpec(
            "CEAC-SW-LCB",
            "ceac",
            ENERGY_BETAS,
            cluster=True,
            capability=True,
            low_power=True,
            aggregate=True,
            boundary=True,
            adaptive=True,
        ),
    ]
    grid = [{"uav_speed_max": v} for v in (2.0, 5.0, 10.0, 15.0, 20.0)]
    cfg0 = Config(n_users=90, slots=500 if quick else 900)
    _sweep(
        "e3_speed",
        lambda p: specs,
        grid,
        seeds,
        lambda p: replace(cfg0, uav_speed_max=p["uav_speed_max"]),
        "speed",
    )


def e4_window(quick=False):
    """Sliding-window parameter sweep (base paper Fig. 9 analogue)."""
    seeds = 3 if quick else 5
    specs = [
        AlgoSpec("SW-LCB", "swlcb", BASE_BETAS),
        AlgoSpec(
            "CEAC-SW-LCB",
            "ceac",
            ENERGY_BETAS,
            cluster=True,
            capability=True,
            low_power=True,
            aggregate=True,
            boundary=True,
            adaptive=False,
        ),
    ]
    grid = [{"alpha": a} for a in (0.12, 0.21, 0.30, 0.40)]
    cfg0 = Config(n_users=90, slots=500 if quick else 900)
    _sweep(
        "e4_window",
        lambda p: specs,
        grid,
        seeds,
        lambda p: replace(cfg0, alpha_base=p["alpha"]),
        "window",
    )


def e5_energy(quick=False):
    """Energy weight beta3: throughput against operating lifetime."""
    seeds = 3 if quick else 5
    grid = [{"beta3": b} for b in (0.0, 0.05, 0.10, 0.20)]

    def spec(p):
        b3 = p["beta3"]
        rest = 1.0 - b3
        betas = (round(rest * 0.5556, 3), round(rest * 0.4444, 3), b3)
        return [
            AlgoSpec("SW-LCB", "swlcb", BASE_BETAS),
            AlgoSpec(
                "CEAC-SW-LCB",
                "ceac",
                betas,
                cluster=True,
                capability=True,
                low_power=b3 > 0,
                aggregate=True,
                boundary=True,
                adaptive=True,
            ),
        ]

    cfg0 = Config(n_users=90, slots=600 if quick else Config.slots)
    _sweep("e5_energy", spec, grid, seeds, lambda p: cfg0, "energy")


def e6_ablation(quick=False):
    """Remove one mechanism at a time."""
    seeds = 3 if quick else 6
    specs = [
        AlgoSpec("SW-LCB", "swlcb", BASE_BETAS),
        AlgoSpec(
            "CEAC-noCluster", "ceac", ENERGY_BETAS, capability=True, low_power=True
        ),
        AlgoSpec(
            "CEAC-noCapability",
            "ceac",
            ENERGY_BETAS,
            cluster=True,
            capability=False,
            low_power=True,
            aggregate=True,
            boundary=True,
            adaptive=True,
        ),
        AlgoSpec(
            "CEAC-noEnergy",
            "ceac",
            BASE_BETAS,
            cluster=True,
            capability=True,
            low_power=False,
            aggregate=True,
            boundary=True,
            adaptive=True,
        ),
        AlgoSpec(
            "CEAC-noBoundary",
            "ceac",
            ENERGY_BETAS,
            cluster=True,
            capability=True,
            low_power=True,
            aggregate=True,
            boundary=False,
            adaptive=True,
        ),
        AlgoSpec(
            "CEAC-SW-LCB",
            "ceac",
            ENERGY_BETAS,
            cluster=True,
            capability=True,
            low_power=True,
            aggregate=True,
            boundary=True,
            adaptive=True,
        ),
    ]
    cfg = Config(n_users=90, slots=400 if quick else 1200)
    rows = []
    for seed in range(seeds):
        t0 = time.time()
        res = run_group(cfg, specs, seed)
        rows += _rows(res, "ablation")
        print(f"  ablation seed={seed} ({time.time() - t0:.1f}s)", flush=True)
    _save(rows, "e6_ablation")


# ----------------------------------------------------------------------------
def cluster_study(quick=False):
    """Clustering quality: purity, state size, messages, head churn, pruning."""
    cfg = Config(n_users=90)
    slots = 120 if quick else 300
    seeds = 3 if quick else 8

    variants = [
        (
            "two-stage",
            {"affinity_lambda": 0.5, "omega_mode": "adaptive", "head_hysteresis": True},
        ),
        (
            "GPS-only",
            {"affinity_lambda": 1.0, "omega_mode": "adaptive", "head_hysteresis": True},
        ),
        (
            "RSSI-only",
            {"affinity_lambda": 0.0, "omega_mode": "adaptive", "head_hysteresis": True},
        ),
        (
            "fixed-omega",
            {"affinity_lambda": 0.5, "omega_mode": "fixed", "head_hysteresis": True},
        ),
        (
            "no-hysteresis",
            {
                "affinity_lambda": 0.5,
                "omega_mode": "adaptive",
                "head_hysteresis": False,
            },
        ),
    ]

    rows = []
    prune_rows = []
    for label, kw in variants:
        c = replace(cfg, **kw)
        for seed in range(seeds):
            rng = np.random.default_rng(1000 + seed)
            sc = Scenario(c, rng)
            rd = Radio(c, sc, rng)
            cm = ClusterManager(c, sc, np.random.default_rng(2000 + seed))
            head_slots = np.zeros(sc.n)
            acc = {
                "purity": [],
                "state": [],
                "members": [],
                "boundary": [],
                "msgs": [],
                "ncl": [],
                "reelect": [],
                "kept": [],
                "regions": [],
            }

            for t in range(slots):
                if t > 0:
                    sc.step(rng)
                    rd.update()
                cs = cm.step(rd, t)
                heads = cs.heads[cs.heads >= 0]
                head_slots[heads] += 1

                strong = rd.edge & (rd.pwr_poc >= -92.0)
                same = cs.assign[:, None] == cs.assign[None, :]
                n_strong = strong.sum()
                purity = (strong & same).sum() / n_strong if n_strong else np.nan

                # coordination set of a node: cluster members plus the
                # cross-cluster neighbours it keeps an affinity edge with
                cross = cs.kept & ~same
                state = (np.bincount(cs.assign)[cs.assign] - 1) + cross.sum(1)

                if t >= 20:
                    acc["purity"].append(purity)
                    acc["state"].append(state.mean())
                    acc["members"].append(np.bincount(cs.assign).mean())
                    acc["boundary"].append(cs.n_boundary)
                    acc["msgs"].append(cs.messages)
                    acc["ncl"].append(cs.n_clusters)
                    acc["reelect"].append(cs.n_relections)
                    acc["kept"].append(cs.kept.sum() / 2)

                if t == slots - 1:
                    # pruning sweep on the final affinity matrix
                    for tau in (0.15, 0.25, 0.35, 0.45, 0.55, 0.65):
                        keep = cs.affinity >= tau
                        np.fill_diagonal(keep, False)
                        retained = (
                            (strong & keep).sum() / n_strong if n_strong else np.nan
                        )
                        prune_rows.append(
                            {
                                "variant": label,
                                "seed": seed,
                                "tau": tau,
                                "edges": float(keep.sum() / 2),
                                "retained": float(retained),
                                "purity_after": float(
                                    (strong & keep & same).sum()
                                    / max((strong & keep).sum(), 1)
                                ),
                            }
                        )

            rows.append(
                {
                    "variant": label,
                    "seed": seed,
                    "slots": slots,
                    "purity": float(np.nanmean(acc["purity"])),
                    "state_per_node": float(np.mean(acc["state"])),
                    "state_ratio": float(np.mean(acc["state"]) / (sc.n - 1)),
                    "members": float(np.mean(acc["members"])),
                    "boundary": float(np.mean(acc["boundary"])),
                    "messages": float(np.mean(acc["msgs"])),
                    "n_clusters": float(np.mean(acc["ncl"])),
                    "reelect_per_slot": float(np.mean(acc["reelect"])),
                    "kept_edges": float(np.mean(acc["kept"])),
                    "head_cv": float(head_slots.std() / max(head_slots.mean(), 1e-9)),
                }
            )
            print(f"  cluster {label} seed={seed}", flush=True)

    _save(rows, "e7_clusters")
    _save(prune_rows, "e7_pruning")


# ----------------------------------------------------------------------------
def _mobility_specs():
    """Fewer algorithms: the question is window vs global history vs CEAC."""
    return [
        AlgoSpec("Greedy", "greedy", BASE_BETAS),
        AlgoSpec("UCB", "ucb", BASE_BETAS),
        AlgoSpec("SW-LCB", "swlcb", BASE_BETAS),
        AlgoSpec(
            "CEAC-SW-LCB",
            "ceac",
            ENERGY_BETAS,
            cluster=True,
            capability=True,
            low_power=True,
            aggregate=True,
            boundary=True,
            adaptive=True,
        ),
    ]


def e9_mobility(quick=False):
    """High mobility in a structured (hotspot) deployment, no breakpoints."""
    seeds = 3 if quick else 5
    grid = [{"uav_speed_max": v} for v in (5.0, 15.0, 30.0, 45.0)]
    cfg0 = Config(
        n_users=90,
        slots=300 if quick else 900,
        n_hotspots=4,
        uav_patrol=True,
        break_prob=0.0,
    )
    _sweep(
        "e9_mobility",
        lambda p: _mobility_specs(),
        grid,
        seeds,
        lambda p: replace(cfg0, uav_speed_max=p["uav_speed_max"]),
        "mobility",
    )


def e10_breakpoints(quick=False):
    """Abrupt redeployments: the base paper's breakpoint process."""
    seeds = 3 if quick else 5
    grid = [{"break_prob": p} for p in (0.0, 0.005, 0.02, 0.08)]
    cfg0 = Config(
        n_users=90,
        slots=300 if quick else 900,
        n_hotspots=4,
        uav_patrol=True,
        uav_speed_max=15.0,
        break_mode="shock",
    )
    _sweep(
        "e10_breakpoints",
        lambda p: _mobility_specs(),
        grid,
        seeds,
        lambda p: replace(cfg0, break_prob=p["break_prob"]),
        "breakpoints",
    )


# ----------------------------------------------------------------------------
EXPERIMENTS = {
    "e1": e1_users,
    "e2": e2_density,
    "e3": e3_speed,
    "e4": e4_window,
    "e5": e5_energy,
    "e6": e6_ablation,
    "e7": cluster_study,
    "e9": e9_mobility,
    "e10": e10_breakpoints,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    import os

    os.makedirs(RESULTS, exist_ok=True)

    names = args.only or list(EXPERIMENTS)
    for n in names:
        print(f"=== {n} ===", flush=True)
        t0 = time.time()
        EXPERIMENTS[n](quick=args.quick)
        print(f"=== {n} done in {time.time() - t0:.0f}s ===\n", flush=True)


if __name__ == "__main__":
    main()

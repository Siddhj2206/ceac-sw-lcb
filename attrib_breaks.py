"""Where does CEAC's advantage under frequent breakpoints come from?

e10 shows CEAC-SW-LCB pulling ahead of SW-LCB as breakpoints become frequent
(+2% throughput at 0 per slot, +6% at 0.08).  CEAC differs from SW-LCB in five
ways, so this run removes them one at a time at the highest break rate.

Writes ``results/attrib_breaks.csv`` and ``results/tables/attrib_breaks.md``.
"""

from __future__ import annotations

import pandas as pd

from run_experiments import BASE_BETAS, ENERGY_BETAS
from sim.config import Config
from sim.runner import AlgoSpec, run_group

SEEDS = 10


def specs():
    return [
        AlgoSpec("SW-LCB", "swlcb", BASE_BETAS),
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


def main() -> None:
    cfg = Config(
        n_users=90,
        slots=900,
        n_hotspots=4,
        uav_patrol=True,
        uav_speed_max=15.0,
        break_mode="shock",
        break_prob=0.08,
    )
    rows = []
    for seed in range(SEEDS):
        res = run_group(cfg, specs(), seed)
        for name, r in res.items():
            row = {"seed": seed, "algo": name}
            row.update(r.summary)
            rows.append(row)
        print(f"  seed {seed}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv("results/attrib_breaks.csv", index=False)

    m = df.groupby("algo")[
        ["conflicts_early", "throughput_early", "qos_early", "regret"]
    ].mean()
    ref = m.loc["SW-LCB"]
    lines = [
        "# Table: attribution under frequent breakpoints (0.08 per slot)",
        "",
        f"Mean over {SEEDS} runs of 900 slots.",
        "",
        "| Configuration | Conflicts | Throughput (Mbps) | vs SW-LCB | QoS | Cumulative regret | vs SW-LCB |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in m.index:
        r = m.loc[name]
        lines.append(
            f"| {name} | {r.conflicts_early:.2f} | {r.throughput_early:.1f} | "
            f"{100 * (r.throughput_early / ref.throughput_early - 1):+.1f}% | "
            f"{r.qos_early:.3f} | {r.regret:.0f} | "
            f"{100 * (r.regret / ref.regret - 1):+.1f}% |"
        )
    with open("results/tables/attrib_breaks.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(m.round(3).to_string())


if __name__ == "__main__":
    main()

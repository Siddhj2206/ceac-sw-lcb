"""What does a breakpoint cost a learner?

The aggregate metrics in e10 cannot separate two effects: the density change
itself (which no learner can undo) and the staleness of the learner's estimate
(which a window should reduce).  The myopic regret - the gap between the arm
played and the best arm against the *current* occupancy - isolates the second.

For every breakpoint we compare the mean regret in the 5 and 10 slots after the
event with the 20 slots before it.  The ratio is averaged over all events and
seeds, with a 95% confidence interval over events.

Writes ``results/tables/shock.md`` and prints the summary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from run_experiments import _mobility_specs
from sim.config import Config
from sim.runner import run_group

PRE = 20
SEEDS = 3


def main() -> None:
    cfg = Config(
        n_users=90,
        slots=900,
        n_hotspots=4,
        uav_patrol=True,
        uav_speed_max=15.0,
        break_mode="shock",
        break_prob=0.02,
    )
    rows = []
    for seed in range(SEEDS):
        res = run_group(cfg, _mobility_specs(), seed)
        breaks = res["UCB"].series["break"]
        breaks = breaks[(breaks >= PRE + 5) & (breaks <= cfg.slots - 10)]
        for name, r in res.items():
            reg = r.series["regret"]
            for b in breaks:
                pre = float(reg[b - PRE : b].mean())
                if pre <= 1e-9:
                    continue
                rows.append(
                    {
                        "seed": seed,
                        "algo": name,
                        "slot": int(b),
                        "pre": pre,
                        "post5": float(reg[b : b + 5].mean()),
                        "post10": float(reg[b : b + 10].mean()),
                    }
                )
        print(f"  seed {seed}: {len(breaks)} breakpoints", flush=True)

    df = pd.DataFrame(rows)
    df["r5"] = df.post5 / df.pre
    df["r10"] = df.post10 / df.pre

    out = []
    for name, g in df.groupby("algo"):
        r5, r10 = g.r5.values, g.r10.values
        out.append(
            {
                "algo": name,
                "pre": g.pre.mean(),
                "post5": g.post5.mean(),
                "r5": 100 * (r5.mean() - 1),
                "ci5": 100 * 1.96 * r5.std(ddof=1) / np.sqrt(len(r5)),
                "r10": 100 * (r10.mean() - 1),
                "ci10": 100 * 1.96 * r10.std(ddof=1) / np.sqrt(len(r10)),
                "post10": g.post10.mean(),
                "events": len(g),
            }
        )
    out = pd.DataFrame(out)

    lines = [
        "# Table: cost of a breakpoint (excess myopic regret)",
        "",
        "Mean over all breakpoints and 3 seeds; 95% confidence interval over events.",
        "",
        "| Algorithm | Regret before | First 5 slots | Change | First 10 slots | Change | Events |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in out.iterrows():
        lines.append(
            f"| {r.algo} | {r.pre:.2f} | {r.post5:.2f} | {r.r5:+.1f}% ± {r.ci5:.1f} | "
            f"{r.post10:.2f} | {r.r10:+.1f}% ± {r.ci10:.1f} | {r.events} |"
        )
    with open("results/tables/shock.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(out.round(2).to_string(index=False))


if __name__ == "__main__":
    main()

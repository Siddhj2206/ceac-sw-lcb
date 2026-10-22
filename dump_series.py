"""Dump one slot-by-slot run for the time-series figure."""

from __future__ import annotations

import pandas as pd

from run_experiments import _specs
from sim.config import Config
from sim.runner import run_group

cfg = Config(n_users=90, slots=1600)
res = run_group(cfg, _specs(True), seed=3)

rows = []
for name, r in res.items():
    for t in range(cfg.slots):
        rows.append(
            {
                "algo": name,
                "slot": t,
                "conflicts": float(r.series["conflicts"][t]),
                "alive": float(r.series["alive"][t]),
                "qos": float(r.series["qos"][t]),
                "rate": float(r.series["rate"][t]),
                "energy": float(r.series["energy"][t]),
            }
        )

pd.DataFrame(rows).to_csv("results/series.csv", index=False)
print("wrote results/series.csv", len(rows), "rows")

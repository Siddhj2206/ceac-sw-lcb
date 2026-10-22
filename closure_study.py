"""Closure quality of the clustering variants.

The metric that Stage 2 is designed to improve is not interference containment
(which GPS clustering maximises by construction) but the quality of the
member-head link: a member that is geographically close to its head can still be
separated from it by rubble, in which case the head cannot coordinate it.

This script measures, per variant:
  * member-head affinity (mean over members),
  * fraction of members whose link to the head is below the join threshold,
  * size-1 clusters (nodes that could not join any head),
  * cross-cluster affinity edges (the pairs heads must coordinate).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from sim.cluster import ClusterManager
from sim.config import Config
from sim.radio import Radio
from sim.scenario import Scenario

VARIANTS = [
    ("two-stage", {"affinity_lambda": 0.5}),
    ("GPS-only", {"affinity_lambda": 1.0}),
    ("RSSI-only", {"affinity_lambda": 0.0}),
]

cfg = Config(n_users=90)
rows = []
for label, kw in VARIANTS:
    c = replace(cfg, **kw)
    for seed in range(8):
        rng = np.random.default_rng(1000 + seed)
        sc = Scenario(c, rng)
        rd = Radio(c, sc, rng)
        cm = ClusterManager(c, sc, np.random.default_rng(2000 + seed))
        acc = []
        for t in range(200):
            if t > 0:
                sc.step(rng)
                rd.update()
            cs = cm.step(rd, t)
            if t < 20:
                continue
            heads = cs.heads
            vals, below, single = [], [], 0
            for node in range(sc.n):
                h = heads[cs.assign[node]]
                if h < 0:
                    continue
                if h == node:
                    single += np.sum(cs.assign == cs.assign[node]) == 1
                    continue
                a = cs.affinity[node, h]
                vals.append(a)
                below.append(a < c.affinity_tau)
            acc.append(
                {
                    "head_aff": float(np.mean(vals)) if vals else np.nan,
                    "below": float(np.mean(below)) if below else np.nan,
                    "single": single,
                    "cross": float(
                        (cs.kept & (cs.assign[:, None] != cs.assign[None, :])).sum() / 2
                    ),
                }
            )
        m = pd.DataFrame(acc).mean()
        rows.append(
            {
                "variant": label,
                "seed": seed,
                "head_affinity": m.head_aff,
                "closure": 1.0 - m.below,
                "singleton_clusters": m.single,
                "cross_cluster_edges": m.cross,
            }
        )

df = pd.DataFrame(rows)
df.to_csv("results/e8_closure.csv", index=False)
print(
    df.groupby("variant")[
        ["head_affinity", "closure", "singleton_clusters", "cross_cluster_edges"]
    ]
    .mean()
    .round(3)
    .to_string()
)

lines = [
    "# Table: closure quality of the clustering variants",
    "",
    "| Variant | Mean member-head affinity | Members holding a kept edge to their head | Singleton clusters | Cross-cluster affinity edges |",
    "|---|---|---|---|---|",
]
for v, r in (
    df.groupby("variant")[
        ["head_affinity", "closure", "singleton_clusters", "cross_cluster_edges"]
    ]
    .mean()
    .iterrows()
):
    lines.append(
        f"| {v} | {r.head_affinity:.3f} | {100 * r.closure:.1f}% | "
        f"{r.singleton_clusters:.2f} | {r.cross_cluster_edges:.1f} |"
    )
with open("results/tables/closure.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("wrote results/tables/closure.md")

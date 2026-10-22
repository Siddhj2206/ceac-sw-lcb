# Table: high mobility in a hotspot deployment (no breakpoints)

| speed (m/s) | Algorithm | Conflicts | Throughput (Mbps) | QoS | Cumulative regret |
|---|---|---|---|---|---|
| 5 | CEAC-SW-LCB | 5.67 ± 1.09 | 52.8 ± 3.8 | 0.441 ± 0.021 | 479 ± 16 |
| 5 | Greedy | 5.36 ± 0.99 | 50.5 ± 5.4 | 0.432 ± 0.015 | 539 ± 12 |
| 5 | SW-LCB | 5.35 ± 0.94 | 52.2 ± 4.6 | 0.448 ± 0.025 | 531 ± 14 |
| 5 | UCB | 5.32 ± 0.96 | 53.3 ± 5.3 | 0.455 ± 0.020 | 527 ± 13 |
| 15 | CEAC-SW-LCB | 5.53 ± 1.06 | 54.1 ± 2.6 | 0.452 ± 0.021 | 476 ± 18 |
| 15 | Greedy | 5.31 ± 0.93 | 52.5 ± 3.9 | 0.440 ± 0.020 | 537 ± 17 |
| 15 | SW-LCB | 5.31 ± 0.91 | 53.0 ± 3.0 | 0.453 ± 0.027 | 531 ± 18 |
| 15 | UCB | 5.31 ± 0.92 | 54.3 ± 2.6 | 0.453 ± 0.018 | 526 ± 15 |
| 30 | CEAC-SW-LCB | 5.57 ± 1.06 | 51.8 ± 3.6 | 0.443 ± 0.031 | 479 ± 20 |
| 30 | Greedy | 5.30 ± 0.98 | 50.4 ± 3.9 | 0.432 ± 0.024 | 539 ± 16 |
| 30 | SW-LCB | 5.35 ± 0.90 | 50.9 ± 4.6 | 0.443 ± 0.032 | 535 ± 21 |
| 30 | UCB | 5.34 ± 0.86 | 51.4 ± 4.2 | 0.451 ± 0.035 | 533 ± 18 |
| 45 | CEAC-SW-LCB | 5.55 ± 1.06 | 53.5 ± 3.5 | 0.446 ± 0.021 | 476 ± 17 |
| 45 | Greedy | 5.42 ± 0.99 | 51.5 ± 4.1 | 0.441 ± 0.019 | 536 ± 13 |
| 45 | SW-LCB | 5.36 ± 0.94 | 51.9 ± 3.9 | 0.446 ± 0.024 | 533 ± 16 |
| 45 | UCB | 5.30 ± 1.01 | 53.6 ± 3.1 | 0.449 ± 0.024 | 526 ± 16 |

Cumulative regret is the sum over slots of the myopic oracle gap (the best feasible arm against the current channel occupancy). It is measured against each learner's own cost, so it is comparable between Greedy, UCB and SW-LCB, which share the same weights, but not between CEAC-SW-LCB and them.

# Table: abrupt redeployments (breakpoints)

| break rate (per slot) | Algorithm | Conflicts | Throughput (Mbps) | QoS | Cumulative regret |
|---|---|---|---|---|---|
| 0 | CEAC-SW-LCB | 5.53 ± 1.06 | 54.1 ± 2.6 | 0.452 ± 0.021 | 476 ± 18 |
| 0 | Greedy | 5.31 ± 0.93 | 52.5 ± 3.9 | 0.440 ± 0.020 | 537 ± 17 |
| 0 | SW-LCB | 5.31 ± 0.91 | 53.0 ± 3.0 | 0.453 ± 0.027 | 531 ± 18 |
| 0 | UCB | 5.31 ± 0.92 | 54.3 ± 2.6 | 0.453 ± 0.018 | 526 ± 15 |
| 0.005 | CEAC-SW-LCB | 5.52 ± 0.77 | 52.7 ± 2.6 | 0.451 ± 0.023 | 478 ± 17 |
| 0.005 | Greedy | 5.33 ± 0.74 | 50.3 ± 3.2 | 0.439 ± 0.027 | 546 ± 17 |
| 0.005 | SW-LCB | 5.28 ± 0.74 | 52.0 ± 3.6 | 0.455 ± 0.026 | 535 ± 13 |
| 0.005 | UCB | 5.27 ± 0.72 | 52.9 ± 3.6 | 0.461 ± 0.019 | 530 ± 11 |
| 0.02 | CEAC-SW-LCB | 6.97 ± 0.93 | 44.8 ± 3.5 | 0.395 ± 0.030 | 490 ± 20 |
| 0.02 | Greedy | 6.55 ± 0.88 | 41.9 ± 3.9 | 0.381 ± 0.028 | 559 ± 15 |
| 0.02 | SW-LCB | 6.57 ± 0.85 | 42.4 ± 3.8 | 0.393 ± 0.031 | 551 ± 17 |
| 0.02 | UCB | 6.52 ± 0.88 | 42.7 ± 3.5 | 0.397 ± 0.035 | 549 ± 16 |
| 0.08 | CEAC-SW-LCB | 7.75 ± 0.91 | 38.6 ± 2.0 | 0.357 ± 0.014 | 502 ± 19 |
| 0.08 | Greedy | 7.39 ± 0.77 | 35.0 ± 1.8 | 0.340 ± 0.015 | 572 ± 16 |
| 0.08 | SW-LCB | 7.31 ± 0.80 | 36.4 ± 2.1 | 0.347 ± 0.016 | 569 ± 17 |
| 0.08 | UCB | 7.30 ± 0.78 | 36.3 ± 2.0 | 0.346 ± 0.016 | 570 ± 16 |

Cumulative regret is the sum over slots of the myopic oracle gap (the best feasible arm against the current channel occupancy). It is measured against each learner's own cost, so it is comparable between Greedy, UCB and SW-LCB, which share the same weights, but not between CEAC-SW-LCB and them.

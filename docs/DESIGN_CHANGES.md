# Gaps found while implementing Review 2

Every item below is a point where the Review 2 design document either does not
specify something the implementation needs, or specifies something that turns out
to be internally inconsistent. Each entry records what the simulator does instead,
so Review 2 can be updated to match.

## 1. Utility slope κ (base paper Eq. 19, used by Review 2 Eq. 7)

Review 2 uses `U` in the cost but never defines κ. A single κ cannot serve text
(100 kbps) and video (800 kbps) at once: with κ = 3 per Mbps a text service
saturates at 1 and a video service sits at 0.2.

**Implemented:** κ = 3 / TP_se, i.e. the sigmoid is centred on each service's own
target, so U = 0.5 exactly at the requirement and U ≈ 0.95 at twice the
requirement. Utilities are then comparable across service classes.

## 2. Per-arm transmit energy (Review 2 Eq. 7)

`cost = β1 I/Imax + β2 max(0,1−U/Ureq) + β3 E_tx/E_res` needs `E_tx` to change
with the arm. With one fixed power level the middle term is identical for every
channel, so β3 can never change a decision.

**Implemented:** every channel is available at 20 dBm and at 14 dBm, so the arm
is (channel, power). `E_tx` is 0.1 J and 0.025 J per slot respectively. This is
the smallest change that makes the energy term meaningful (Review 2 §9 talks
about "high-power options" being penalised, which presumes the same idea).

## 3. Energy scaling

`E_tx/E_res` is of order 1e-4 with realistic batteries, so it cannot compete with
the interference and QoS terms.

**Implemented:** the energy cost is `(E_tx/E_slot) · (E0/E_res)²`, clipped to
[0, 1]. It is small while the battery is healthy and grows quadratically as the
battery drains, which is the stated intent ("the same transmission costs more on
a nearly empty battery").

## 4. The join threshold τ is inconsistent with r_c

Review 2 sets a single threshold (Eq. 4) and then uses it for membership
(Algorithm 4). With r_c = 200 m the affinity of Eq. (3) is

    s ≈ 0.5·(1 − d/200) + 0.5·ρ(RSSI)

which reaches 0.35 only at d ≈ 120 m, while a k-means cluster with
k = ⌈A/(π r_c²)⌉ = 8 has an effective radius of ~200 m and a mean member-to-head
distance of ~130 m. At τ = 0.35, 65% of members therefore have no kept edge to
their own head, and Algorithm 4's fallback path is taken by most of the network.

**Implemented:** τ is calibrated from the sweep in Table 8 and set to **0.18**,
which gives 60% closure and keeps the graph at ~370 edges (vs ~4900 pairs). The
paper recommends that Review 2 state the calibration rule explicitly:
`τ` should be chosen so that the chosen `r_c` and `k` give ≥ 60% closure.

## 5. Border refinement (Algorithm 4, line 9)

The pseudocode requires `s_i,h(c′) ≥ τ` for a border move. The first
implementation only required the target to beat the incumbent by δ_b, which lets
nodes oscillate between two unreachable heads.

**Fixed:** the threshold is now part of the move condition, as written.

## 6. The fallback when no head is reachable (Algorithm 4, line 4)

The pseudocode says a node with no reachable head "becomes the head of its own
cluster". With τ = 0.18 this creates a large number of one-node clusters, whose
"coordination" is meaningless and whose heads churn.

**Implemented:** such a node stays with its region head and the link is flagged
as a **weak link**; the weak-link fraction is reported as a clustering quality
metric. Recommendation: change line 4 of Algorithm 4 to "join the region head
and flag the link as weak".

## 7. Window adaptation rule (Review 2 §9 / Eq. 8)

Review 2 says the window "shrinks when members move fast or rewards fluctuate,
and grows when conditions are stable" but gives no rule.

**Implemented:** per cluster, α_q = α_max − (α_max − α_min)·(mean cluster
speed / v_max), clipped to [0.12, 0.30], and W_t = min(⌈λ t^α_q⌉, t) with λ = 50.
Low mobility therefore gives a long, stable window and high mobility a short,
responsive one, as specified.

## 8. Boundary conflict resolution (Review 2 §9, last paragraph)

Review 2 explicitly leaves this open ("This document does not fix the rule heads
use to resolve reported boundary conflicts").

**Implemented:** a boundary node adds a penalty `+0.08 · (fraction of its
cross-cluster neighbours that used that channel in the previous slot)` to the LCB
index of every arm on that channel. The head only has to exchange the
(channel, power) pair of its boundary nodes, so the message count stays at
`O(N + |B|)` per slot.

## 9. How clustering feeds the learner

Review 2 says the learner runs "inside the cluster" but not whether members share
statistics.

**Implemented:** a cluster mean is used as a **prior for arms the node has not
observed in its window**:
`s_eff = γ·s_cluster + (1−γ)·s_global`, γ = 0.35. Arms the node has observed keep
its own estimate, because the interference a node sees is node-specific; blending
observed arms with the cluster mean measurably hurts (about −4% throughput in a
pilot).

## 10. Radio energy budget

Review 2 excludes propulsion and hovering energy but does not give a budget.

**Implemented:** 300 J for a ground terminal and 450 J for a UAV of radio energy.
A node that exhausts its budget stops transmitting, which is what makes the
lifetime results meaningful. The base paper's 5000 s horizon is shortened to
1600 s because the transient, not the steady state, is where the mechanisms
differ; this is stated in the results.

## 11. What "non-stationary" means (Review 2 Section 9, base paper Fig. 11)

Review 2 motivates the sliding window with "time-varying interference" and
inherits the base paper's Gauss-Markov mobility. That model is *well mixed*:
every node samples the same deployment area, so the number of interferers per
channel is stationary even at high speed, and a window has nothing to track. A
window only helps when the *reward distribution* changes, which in this game
means the local node density or the channel occupancy changes persistently.

**Implemented:** an optional structured deployment (`n_hotspots > 0`) with
drifting hotspot centres, UAVs patrolling between them, and a breakpoint process
(`break_prob`, `break_mode`). A breakpoint moves one hotspot either onto another
(local density roughly doubles) or to a new area (it drops), i.e. it is a density
shock rather than a teleport that preserves density. Recommendation: Review 2
should say explicitly that the breakpoint process is a density/occupancy shock,
because a position change that preserves the neighbourhood statistics cannot
create non-stationarity for a bandit.

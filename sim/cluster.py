"""Two-stage clustering and cluster-head election (Review 2, Sections 4-8).

Stage 1 (Section 4): dead-reckoned positions, k-means over GPS coordinates with
a warm start, k = ceil(A / (pi rc^2)).
Stage 2 (Section 5): EWMA-smoothed RSSI, affinity score of Eq. (3) and the
single-threshold pruning rule of Eq. (4).
Head election (Section 6): capability and mobility filters, the variance-adaptive
score of Eqs. (5)-(6) and the hysteresis rule of Algorithm 3.
Membership and borders (Section 7): Algorithm 4, including boundary nodes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config
from .scenario import Scenario


@dataclass
class ClusterState:
    assign: np.ndarray  # (n,) cluster id per node
    heads: np.ndarray  # (k,) head node index, -1 if unused
    boundary: np.ndarray  # (n,) bool
    alpha: np.ndarray  # (n,) adaptive window exponent
    rssi: np.ndarray  # (n, n) smoothed RSSI, dBm
    kept: np.ndarray  # (n, n) bool, pruned affinity edges
    affinity: np.ndarray  # (n, n) affinity scores
    n_clusters: int
    n_relections: int
    n_boundary: int
    messages: int
    head_scores: np.ndarray  # (k,) scores of the elected heads


class ClusterManager:
    def __init__(self, cfg: Config, scen: Scenario, rng: np.random.Generator):
        self.cfg = cfg
        self.scen = scen
        self.n = scen.n
        self.rng = rng

        n = self.n
        k_max = int(np.ceil(cfg.area**2 / (np.pi * cfg.head_radius**2))) + 2
        self.k_max = k_max
        self.assign = np.zeros(n, dtype=int)
        self.heads = np.full(k_max, -1, dtype=int)
        self.boundary = np.zeros(n, dtype=bool)
        self.rssi = None
        self.kept = np.zeros((n, n), dtype=bool)
        self.affinity = np.zeros((n, n))
        self.centers: np.ndarray | None = None
        self.k = self._k()
        self.n_relections = 0
        self.alpha = np.full(n, cfg.alpha_base)
        self._prev_head = {}

    # ------------------------------------------------------------------
    def _k(self) -> int:
        cfg = self.cfg
        k = int(np.ceil(cfg.area * cfg.area / (np.pi * cfg.head_radius**2)))
        return max(2, min(k, self.n // 3))

    def _kmeans(self, pts: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        n = pts.shape[0]
        if self.centers is not None and len(self.centers) == k:
            centers = self.centers.copy()  # warm start
        else:
            idx = self.rng.choice(n, size=k, replace=False)
            centers = pts[idx].copy()

        labels = np.zeros(n, dtype=int)
        for _ in range(cfg.kmeans_iters):
            d = ((pts[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
            new = d.argmin(1)
            if np.array_equal(new, labels):
                labels = new
                break
            labels = new
            for c in range(k):
                m = labels == c
                if m.any():
                    centers[c] = pts[m].mean(0)
                else:
                    far = np.linalg.norm(pts - centers[labels], axis=1).argmax()
                    centers[c] = pts[far]
        self.centers = centers
        return labels, centers

    # ------------------------------------------------------------------
    def step(self, radio, t: int) -> ClusterState:
        cfg, scen = self.cfg, self.scen

        # ---------------- Stage 1: GPS regions -------------------------
        if t % cfg.stage1_period == 0 or self.centers is None:
            k = self._k()
            labels, _ = self._kmeans(scen.pos[:, :2], k)
            self.assign = labels
            self.k = k
            # a node that changed region inherits that region's head mapping
        else:
            k = self.k

        # ---------------- Stage 2: affinity and pruning ----------------
        rssi = radio.rssi_matrix()
        if self.rssi is None:
            self.rssi = rssi.copy()
        else:
            a = cfg.rssi_smooth
            self.rssi = a * rssi + (1.0 - a) * self.rssi

        d_geo = np.sqrt(((scen.pos[:, None, :2] - scen.pos[None, :, :2]) ** 2).sum(-1))

        rho = np.clip(
            (self.rssi - cfg.rssi_min) / (cfg.rssi_max - cfg.rssi_min), 0.0, 1.0
        )
        g = np.clip(1.0 - d_geo / cfg.head_radius, 0.0, None)
        aff = cfg.affinity_lambda * g + (1.0 - cfg.affinity_lambda) * rho

        # pairs in the same or in adjacent regions only
        centers = self.centers
        assert centers is not None
        cd = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(-1))
        adjacent = cd <= 2.0 * cfg.head_radius
        reg_ok = adjacent[self.assign][:, self.assign]
        np.fill_diagonal(reg_ok, False)

        kept = (aff >= cfg.affinity_tau) & reg_ok
        np.fill_diagonal(kept, False)
        self.affinity = aff
        self.kept = kept

        # ---------------- Head election (Algorithm 3) ------------------
        elig_cap = scen.has_lte & scen.has_poc
        eligible_all = (
            elig_cap
            & (scen.speed <= scen.vmax)
            & (scen.energy_frac >= cfg.energy_floor)
        )
        ef = scen.energy_frac

        heads = np.full(self.heads.shape[0], -1, dtype=int)
        scores = np.zeros(self.heads.shape[0])
        reelect = 0
        for c in range(k):
            members = np.where(self.assign == c)[0]
            if members.size == 0:
                continue
            cand = members[eligible_all[members]]
            if cand.size == 0:
                cand = members
            if cand.size == 0:
                continue

            # connectivity = mean affinity over kept edges inside the region
            conn = np.zeros(cand.size)
            for i, m in enumerate(cand):
                nb = np.where(kept[m, members])[0]
                conn[i] = aff[m, nb].mean() if nb.size else 0.0

            e = ef[cand]
            sE = e.var()
            sK = conn.var()
            lo, hi = cfg.omega_bounds
            if cfg.omega_mode == "fixed" or (sE < 1e-9 and sK < 1e-9):
                omega = 0.5
            else:
                omega = sE / (sE + sK) if (sE + sK) > 1e-12 else 0.5
            omega = float(np.clip(omega, lo, hi))

            sc = omega * e + (1.0 - omega) * conn
            best_local = int(np.argmax(sc))
            best = int(cand[best_local])

            prev = self._prev_head.get(c, -1)
            if cfg.head_hysteresis and prev in cand and ef[prev] >= cfg.energy_floor:
                pb = int(np.where(cand == prev)[0][0])
                if sc[best_local] <= sc[pb] + cfg.head_margin:
                    best = prev
                    best_local = pb
            if best != prev:
                reelect += 1
                self._prev_head[c] = best

            heads[c] = best
            scores[c] = sc[best_local]

        # ---------------- Membership and borders (Algorithm 4) ---------
        head_of_region = {}
        for c in range(k):
            if heads[c] >= 0:
                head_of_region[c] = heads[c]

        boundary = np.zeros(self.n, dtype=bool)
        sizes = np.zeros(self.heads.shape[0], dtype=int)
        for c in range(k):
            if heads[c] >= 0:
                sizes[c] = int((self.assign == c).sum())

        for i in range(self.n):
            if i in heads:
                continue
            c = self.assign[i]
            gi = head_of_region.get(c, -1)
            ok = (
                gi >= 0
                and aff[i, gi] >= cfg.affinity_tau
                and sizes[c] < cfg.cluster_nmax
            )
            if ok:
                continue
            # unassigned or over the size cap: join the best reachable head
            cands = [
                (aff[i, h], cc)
                for cc, h in head_of_region.items()
                if h != i and aff[i, h] >= cfg.affinity_tau
            ]
            if cands:
                _, cc = max(cands)
                self.assign[i] = cc

        # border refinement
        for i in range(self.n):
            c = self.assign[i]
            hi = head_of_region.get(c, -1)
            if hi < 0 or i == hi:
                continue
            best_other, best_val = None, aff[i, hi] + cfg.border_margin
            for cc, h in head_of_region.items():
                if cc == c or h == i:
                    continue
                # Algorithm 4: the target head must also satisfy the join
                # threshold, not merely be better than the current one.
                if (
                    aff[i, h] > best_val
                    and aff[i, h] >= cfg.affinity_tau
                    and sizes[cc] < cfg.cluster_nmax
                ):
                    best_other, best_val = cc, aff[i, h]
            if best_other is not None:
                self.assign[i] = best_other
                sizes[c] -= 1
                sizes[best_other] += 1

        # boundary nodes: kept edge leaving the cluster
        for i in range(self.n):
            others = kept[i]
            if others.any():
                boundary[i] = bool((self.assign[others] != self.assign[i]).any())

        # ---------------- Adaptive window parameter --------------------
        # Review 2 Section 9: the window shrinks with mobility, so alpha falls
        # as the cluster speed rises.
        v_ref = max(self.scen.vmax.max(), 1e-6)
        for c in range(k):
            m = self.assign == c
            if not m.any():
                continue
            mob = float(np.clip(self.scen.speed[m].mean() / v_ref, 0.0, 1.0))
            a = cfg.alpha_max - (cfg.alpha_max - cfg.alpha_min) * mob
            self.alpha[m] = a

        self.heads = heads
        self.boundary = boundary
        n_boundary = int(boundary.sum())
        # member->head reports + head broadcast, plus boundary summaries
        messages = 2 * self.n + 2 * n_boundary

        return ClusterState(
            assign=self.assign.copy(),
            heads=heads,
            boundary=boundary,
            alpha=self.alpha.copy(),
            rssi=self.rssi,
            kept=kept,
            affinity=aff,
            n_clusters=int((heads >= 0).sum()),
            n_relections=reelect,
            n_boundary=n_boundary,
            messages=messages,
            head_scores=scores,
        )

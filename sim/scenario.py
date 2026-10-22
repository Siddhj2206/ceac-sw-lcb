"""Node population, capabilities, mobility and battery state."""

from __future__ import annotations

import numpy as np

from .config import (
    KIND_DUAL,
    KIND_LTE_ONLY,
    KIND_POC_ONLY,
    KIND_UAV,
    SERVICE_NAMES,
    SERVICES,
    Config,
)


class Scenario:
    """Holds the node population and advances it one slot at a time.

    Every node is a transmitter with its own receiver a short distance away
    (``link_dist``).  This follows the receiver-centric link model of the base
    paper, Eqs. (15)-(18): the desired power is the power received on the node's
    own local link, and interference is the sum of powers received at that same
    receiver from every other node sharing the channel.
    """

    def __init__(self, cfg: Config, rng: np.random.Generator):
        self.cfg = cfg
        n_uav, n_usr = cfg.n_uav, cfg.n_users
        self.n = n_uav + n_usr

        # --- kinds and radios ------------------------------------------------
        kind = np.empty(self.n, dtype=int)
        kind[:n_uav] = KIND_UAV
        u = rng.random(n_usr)
        kind[n_uav:] = np.where(
            u < cfg.p_dual,
            KIND_DUAL,
            np.where(u < cfg.p_dual + cfg.p_lte_only, KIND_LTE_ONLY, KIND_POC_ONLY),
        )
        self.kind = kind
        self.has_lte = (
            (kind == KIND_UAV) | (kind == KIND_DUAL) | (kind == KIND_LTE_ONLY)
        )
        self.has_poc = (
            (kind == KIND_UAV) | (kind == KIND_DUAL) | (kind == KIND_POC_ONLY)
        )
        self.is_uav = kind == KIND_UAV

        # --- service class and QoS target -----------------------------------
        idx = rng.choice(len(SERVICE_NAMES), size=self.n, p=cfg.service_weights)
        self.service = idx
        self.service_name = [SERVICE_NAMES[i] for i in idx]
        self.tp_mbps = np.array([SERVICES[SERVICE_NAMES[i]]["rate_mbps"] for i in idx])

        # --- initial energy --------------------------------------------------
        self.energy0 = np.where(self.is_uav, cfg.energy_uav_j, cfg.energy_user_j)
        self.energy = self.energy0.copy()

        # --- positions and mobility ------------------------------------------
        self.area = cfg.area
        pos = np.empty((self.n, 3))
        pos[:, :2] = rng.uniform(0.0, cfg.area, size=(self.n, 2))
        pos[:n_uav, 2] = rng.uniform(*cfg.uav_alt, size=n_uav)
        pos[n_uav:, 2] = 2.0
        self.pos = pos

        self.vmax = np.where(self.is_uav, cfg.uav_speed_max, cfg.user_speed_max)
        self.speed = rng.uniform(0.3, 1.0, size=self.n) * self.vmax
        self.heading = rng.uniform(0.0, 2.0 * np.pi, size=self.n)
        self.nominal = 0.6 * self.vmax

        # --- local receiver offset -------------------------------------------
        ang = rng.uniform(0.0, 2.0 * np.pi, size=self.n)
        dist = rng.uniform(*cfg.link_dist, size=self.n)
        self.rx_off = np.stack(
            [dist * np.cos(ang), dist * np.sin(ang), np.zeros(self.n)], axis=1
        )

        self.slot = 0

        # Node groups (used by the structured mobility model).
        self.uav_idx = np.where(self.is_uav)[0]
        self.usr_idx = np.where(~self.is_uav)[0]

        # --- optional structured (non-stationary) deployment -----------------
        # Everything below is skipped at the default n_hotspots = 0, so the
        # uniform Gauss-Markov deployment is bit-for-bit unchanged.
        self.hot_centers: np.ndarray | None = None
        self.hot_vel: np.ndarray | None = None
        self.user_hot: np.ndarray | None = None
        self.user_off: np.ndarray | None = None
        self.uav_target: np.ndarray | None = None
        self.breakpoints = 0
        self.break_slots: list[int] = []
        self.break_kinds: list[str] = []

        if cfg.n_hotspots > 0:
            nh = cfg.n_hotspots
            self.hot_centers = rng.uniform(0.2, 0.8, size=(nh, 2)) * cfg.area
            ang = rng.uniform(0.0, 2.0 * np.pi, size=nh)
            self.hot_vel = cfg.hotspot_drift * np.stack(
                [np.cos(ang), np.sin(ang)], axis=1
            )

            m = self.usr_idx.size
            self.user_hot = rng.integers(0, nh, size=m)
            r = cfg.hotspot_radius * np.sqrt(rng.random(m))
            th = rng.uniform(0.0, 2.0 * np.pi, size=m)
            self.user_off = np.stack([r * np.cos(th), r * np.sin(th)], axis=1)
            self.pos[self.usr_idx, 0] = (
                self.hot_centers[self.user_hot, 0] + self.user_off[:, 0]
            )
            self.pos[self.usr_idx, 1] = (
                self.hot_centers[self.user_hot, 1] + self.user_off[:, 1]
            )

            if cfg.uav_patrol:
                self.uav_target = rng.integers(0, nh, size=self.uav_idx.size)

    # ------------------------------------------------------------------
    @property
    def rx_pos(self) -> np.ndarray:
        """Receiver positions: each receiver moves with its transmitter."""
        return self.pos + self.rx_off

    # ------------------------------------------------------------------
    def step(self, rng: np.random.Generator) -> None:
        """Advance mobility by one slot."""
        if self.hot_centers is None:
            self._gauss_step(rng, np.arange(self.n))
        else:
            self._hotspot_step(rng)
        self.slot += 1

    # ------------------------------------------------------------------
    def _gauss_step(self, rng: np.random.Generator, idx: np.ndarray) -> None:
        """Gauss-Markov motion (base paper Table 4) for the nodes in ``idx``."""
        cfg = self.cfg
        xi = cfg.gm_xi
        m = idx.size

        nom = self.nominal[idx]
        sp = (
            xi * self.speed[idx]
            + (1.0 - xi) * nom
            + rng.normal(0.0, 0.25 * nom, size=m)
        )
        surge = rng.random(m) < cfg.surge_prob
        sp = np.where(surge, sp * 1.8, sp)
        sp = np.clip(sp, 0.2, 1.6 * self.vmax[idx])
        self.speed[idx] = sp

        hd = self.heading[idx] + rng.normal(0.0, 0.18, size=m)
        self.heading[idx] = hd

        self.pos[idx, 0] += sp * np.cos(hd)
        self.pos[idx, 1] += sp * np.sin(hd)

        # reflect at the deployment boundary
        for ax in range(2):
            lo = idx[self.pos[idx, ax] < 0.0]
            hi = idx[self.pos[idx, ax] > self.area]
            self.pos[lo, ax] = -self.pos[lo, ax]
            self.pos[hi, ax] = 2.0 * self.area - self.pos[hi, ax]
            self.heading[lo] = np.pi - self.heading[lo]
            self.heading[hi] = np.pi - self.heading[hi]

    # ------------------------------------------------------------------
    def _hotspot_step(self, rng: np.random.Generator) -> None:
        """Hotspot-centred deployment: drift, patrol and abrupt breakpoints."""
        assert self.hot_centers is not None
        assert self.hot_vel is not None
        assert self.user_off is not None
        assert self.user_hot is not None
        cfg = self.cfg
        nh = cfg.n_hotspots

        # 1. hotspot centres drift and reflect at the boundary
        self.hot_centers += self.hot_vel
        for ax in range(2):
            lo = self.hot_centers[:, ax] < 0.0
            hi = self.hot_centers[:, ax] > self.area
            self.hot_centers[lo, ax] = -self.hot_centers[lo, ax]
            self.hot_centers[hi, ax] = 2.0 * self.area - self.hot_centers[hi, ax]
            self.hot_vel[lo, ax] *= -1.0
            self.hot_vel[hi, ax] *= -1.0

        # 2. users random-walk inside their hotspot
        jitter = 0.35 * cfg.user_speed_max
        self.user_off += rng.normal(0.0, jitter, size=self.user_off.shape)
        r = np.linalg.norm(self.user_off, axis=1)
        over = r > cfg.hotspot_radius
        if np.any(over):
            self.user_off[over] *= (cfg.hotspot_radius / r[over])[:, None]
        self.pos[self.usr_idx, 0] = (
            self.hot_centers[self.user_hot, 0] + self.user_off[:, 0]
        )
        self.pos[self.usr_idx, 1] = (
            self.hot_centers[self.user_hot, 1] + self.user_off[:, 1]
        )

        # 3. UAVs patrol between hotspot centres (or fly freely without patrol)
        if self.uav_target is not None:
            uav = self.uav_idx
            delta = self.hot_centers[self.uav_target] - self.pos[uav, :2]
            dist = np.linalg.norm(delta, axis=1)
            arrived = dist < 40.0
            if np.any(arrived):
                self.uav_target[arrived] = rng.integers(0, nh, size=int(arrived.sum()))
            step = np.minimum(self.vmax[uav], dist)
            safe = np.maximum(dist, 1e-9)
            self.pos[uav, 0] += step * delta[:, 0] / safe
            self.pos[uav, 1] += step * delta[:, 1] / safe
        else:
            self._gauss_step(rng, self.uav_idx)

        # 4. breakpoint: one hotspot and its users redeploy abruptly
        if cfg.break_prob > 0.0 and rng.random() < cfg.break_prob:
            h = int(rng.integers(0, nh))
            mode = cfg.break_mode
            if mode == "shock":
                mode = "merge" if rng.random() < 0.5 else "disperse"
            if mode == "merge" and nh > 1:
                # converge on another hotspot: local density roughly doubles
                others = [j for j in range(nh) if j != h]
                j = int(rng.choice(others))
                ang = rng.uniform(0.0, 2.0 * np.pi)
                off = 0.6 * cfg.hotspot_radius
                new = self.hot_centers[j] + off * np.array([np.cos(ang), np.sin(ang)])
                self.hot_centers[h] = np.clip(new, 30.0, cfg.area - 30.0)
            else:
                # disperse / teleport: a new location, preserving local density
                self.hot_centers[h] = rng.uniform(0.2, 0.8, size=2) * cfg.area
            ang = rng.uniform(0.0, 2.0 * np.pi)
            self.hot_vel[h] = cfg.hotspot_drift * np.array([np.cos(ang), np.sin(ang)])
            members = np.where(self.user_hot == h)[0]
            k = members.size
            if k:
                r2 = cfg.hotspot_radius * np.sqrt(rng.random(k))
                th2 = rng.uniform(0.0, 2.0 * np.pi, size=k)
                self.user_off[members, 0] = r2 * np.cos(th2)
                self.user_off[members, 1] = r2 * np.sin(th2)
                nodes = self.usr_idx[members]
                self.pos[nodes, 0] = self.hot_centers[h, 0] + self.user_off[members, 0]
                self.pos[nodes, 1] = self.hot_centers[h, 1] + self.user_off[members, 1]
            self.breakpoints += 1
            self.break_slots.append(self.slot + 1)
            self.break_kinds.append(mode)

    # ------------------------------------------------------------------
    def consume_energy(self, n_band_slots: np.ndarray, dt: float = 1.0) -> None:
        """Deduct radio energy.  ``n_band_slots`` is the number of active bands."""
        cfg = self.cfg
        e_tx = (10.0 ** (cfg.p_tx_dbm / 10.0)) / 1000.0  # watts
        self.energy = np.maximum(
            0.0,
            self.energy - (e_tx * n_band_slots + cfg.idle_power_w) * dt,
        )

    # ------------------------------------------------------------------
    @property
    def energy_frac(self) -> np.ndarray:
        return self.energy / self.energy0

    def summary(self) -> dict:
        return {
            "n_uav": int(self.is_uav.sum()),
            "n_dual": int((self.kind == KIND_DUAL).sum()),
            "n_lte_only": int((self.kind == KIND_LTE_ONLY).sum()),
            "n_poc_only": int((self.kind == KIND_POC_ONLY).sum()),
        }

"""Channel model, interference graph and link-level evaluation.

Implements the propagation and rate model of the base paper (Eqs. 7-19):
distance path loss with Nakagami-m fading and log-normal shadowing, co-channel
interference, Shannon rates and the logistic QoS utility.

Three modelling choices that the base paper leaves open are made explicit:

* Bandwidth sharing.  Every channel's bandwidth is divided equally among the
  nodes transmitting on it in the current slot, which is what PRB scheduling
  (LTE) and airtime contention (802.11) do.  The base paper's per-device
  bandwidth therefore acts as an upper bound.
* POC overlap.  IEEE 802.11b/g channels are 22 MHz wide and 5 MHz apart, so
  adjacent channels leak into each other.  Interference is weighted by the
  normalised spectral overlap (``config.POC_OVERLAP``).
* Power modes.  Each channel can be used at 20 dBm or at 14 dBm.  The per-arm
  transmit energy that Review 2 Eq. (7) needs is what distinguishes the two.
"""

from __future__ import annotations

import numpy as np

from .config import KAPPA_PER_TP, POC_OVERLAP, Config
from .scenario import Scenario


def _dbm_to_w(dbm):
    return 10.0 ** ((np.asarray(dbm, dtype=float) - 30.0) / 10.0)


class Radio:
    """Per-slot physical layer shared by every algorithm in a Monte-Carlo run."""

    def __init__(self, cfg: Config, scen: Scenario, rng: np.random.Generator):
        self.cfg = cfg
        self.scen = scen
        self.n = scen.n
        self.rng = rng

        self.noise_w = float(_dbm_to_w(cfg.noise_dbm))
        self.ov = np.asarray(POC_OVERLAP, dtype=float)[: cfg.n_poc, : cfg.n_poc]
        self.fspl_lte = 20.0 * np.log10(4.0 * np.pi * cfg.f_lte_ghz * 1e9 / 3e8)
        self.fspl_poc = 20.0 * np.log10(4.0 * np.pi * cfg.f_poc_ghz * 1e9 / 3e8)

        # linear gain of each power mode relative to the 20 dBm reference
        self.power_gain = 10.0 ** (
            (np.asarray(cfg.power_dbm, dtype=float) - cfg.p_tx_dbm) / 10.0
        )
        # radio energy of one slot at each power mode, joules
        self.power_energy = (
            10.0 ** (np.asarray(cfg.power_dbm, dtype=float) / 10.0) / 1000.0
        )

        # rubble footprints, fixed for the run
        self.obs_xy = rng.uniform(0.0, cfg.area, size=(cfg.n_obstacles, 2))
        self.obs_r = rng.uniform(*cfg.obstacle_radius, size=cfg.n_obstacles)

        self.shadow_lte = rng.normal(0.0, cfg.shadow_sigma_db, size=(self.n, self.n))
        self.shadow_poc = rng.normal(0.0, cfg.shadow_sigma_db, size=(self.n, self.n))

        self.dist = np.ones((self.n, self.n))
        self.pwr_lte = np.zeros((self.n, self.n))
        self.pwr_poc = np.zeros((self.n, self.n))
        self.edge = np.zeros((self.n, self.n), dtype=bool)

        self.update()

    # ------------------------------------------------------------------
    def update(self) -> None:
        """Recompute the channel gains of the current slot."""
        cfg = self.cfg
        pos = self.scen.pos
        rx = self.scen.rx_pos

        d = np.sqrt(
            ((pos[:, None, :2] - rx[None, :, :2]) ** 2).sum(-1)
            + (pos[:, None, 2] - rx[None, :, 2]) ** 2
        )
        d = np.maximum(d, 1.0)
        self.dist = d
        pl = 10.0 * cfg.path_loss_exp * np.log10(d)

        m = cfg.nakagami_m
        gain = self.rng.gamma(m, 1.0 / m, size=(self.n, self.n))
        fade_db = 10.0 * np.log10(np.maximum(gain, 1e-12))

        mid = 0.5 * (pos[:, None, :2] + rx[None, :, :2])
        d2 = ((mid[:, :, None, :] - self.obs_xy[None, None, :, :]) ** 2).sum(-1)
        blocked = (d2 < (self.obs_r**2)[None, None, :]).any(-1)
        self.blocked = blocked
        atten = np.where(blocked, cfg.blockage_db, 0.0)

        self.pwr_lte = (
            cfg.p_tx_dbm - (self.fspl_lte + pl) + self.shadow_lte + fade_db - atten
        )
        self.pwr_poc = (
            cfg.p_tx_dbm - (self.fspl_poc + pl) + self.shadow_poc + fade_db - atten
        )

        thr = cfg.intf_threshold_dbm
        e = self.pwr_lte >= thr
        e |= thr <= self.pwr_lte.T
        e |= self.pwr_poc >= thr
        e |= thr <= self.pwr_poc.T
        np.fill_diagonal(e, False)
        self.edge = e

        self.p_lte_w = _dbm_to_w(self.pwr_lte)
        self.p_poc_w = _dbm_to_w(self.pwr_poc)
        self.signal_lte_w = np.diag(self.p_lte_w).copy()
        self.signal_poc_w = np.diag(self.p_poc_w).copy()

    # ------------------------------------------------------------------
    def _onehot(self, ch: np.ndarray, n_ch: int) -> np.ndarray:
        a = np.zeros((self.n, n_ch))
        on = ch >= 0
        a[np.arange(self.n)[on], ch[on]] = 1.0
        return a

    def evaluate(self, lte_ch, lte_pw, poc_ch, poc_pw) -> dict:
        """Evaluate one allocation.

        ``*_ch`` is a channel index per node (-1 when the band is unused) and
        ``*_pw`` is a power-mode index (0 = 20 dBm, 1 = 14 dBm).
        """
        cfg = self.cfg
        n, S, H = self.n, cfg.n_lte, cfg.n_poc
        ar = np.arange(n)

        v_lte = lte_ch >= 0
        v_poc = poc_ch >= 0
        g_lte = np.where(v_lte, self.power_gain[np.maximum(lte_pw, 0)], 0.0)
        g_poc = np.where(v_poc, self.power_gain[np.maximum(poc_pw, 0)], 0.0)

        a_lte = self._onehot(lte_ch, S) * g_lte[:, None]
        a_poc = self._onehot(poc_ch, H) * g_poc[:, None]

        # ---- bandwidth sharing ---------------------------------------
        cnt_lte = self._onehot(lte_ch, S).sum(0)
        cnt_poc = self._onehot(poc_ch, H).sum(0)
        b_lte = cfg.b_lte_mhz * 1e6 / np.maximum(cnt_lte, 1.0)
        b_poc = cfg.b_poc_mhz * 1e6 / np.maximum(cnt_poc, 1.0)

        # ---- interference at every receiver ---------------------------
        I_lte = a_lte.T @ self.p_lte_w  # (S, n)
        I_poc = a_poc.T @ self.p_poc_w  # (H, n)

        s_lte = self.signal_lte_w * g_lte
        s_poc = self.signal_poc_w * g_poc

        i_lte = np.maximum(
            np.where(v_lte, I_lte[np.maximum(lte_ch, 0), ar] - s_lte, 0.0), 0.0
        )
        ov_rows = np.where(v_poc[:, None], self.ov[np.maximum(poc_ch, 0)], 0.0)
        i_poc = np.maximum(
            (ov_rows * I_poc.T).sum(1) - np.where(v_poc, s_poc, 0.0), 0.0
        )

        sinr_lte = np.where(v_lte, s_lte / (i_lte + self.noise_w), 0.0)
        sinr_poc = np.where(v_poc, s_poc / (i_poc + self.noise_w), 0.0)

        rate_lte = np.where(
            v_lte, b_lte[np.maximum(lte_ch, 0)] * np.log2(1.0 + sinr_lte), 0.0
        )
        rate_poc = np.where(
            v_poc, b_poc[np.maximum(poc_ch, 0)] * np.log2(1.0 + sinr_poc), 0.0
        )
        rate = (rate_lte + rate_poc) / 1e6

        # ---- interference conflicts (base paper Eq. 26) ----------------
        g_l = self.edge @ self._onehot(lte_ch, S)
        g_p = self.edge @ self._onehot(poc_ch, H)
        conf_lte = np.where(v_lte, g_l[ar, np.maximum(lte_ch, 0)], 0.0)
        conf_poc = np.where(v_poc, g_p[ar, np.maximum(poc_ch, 0)], 0.0)

        # ---- per-arm signal-to-interference for the cost function ------
        base_l = I_lte.T - (self._onehot(lte_ch, S) * s_lte[:, None])  # (n, S)
        base_p = (self.ov @ I_poc).T - (self._onehot(poc_ch, H) * s_poc[:, None])
        s0_l = self.signal_lte_w[:, None]
        s0_p = self.signal_poc_w[:, None]
        # arm layout a = power * n_ch + channel
        arm_sinr_l = np.concatenate(
            [s0_l * gp / (base_l + self.noise_w) for gp in self.power_gain], axis=1
        )
        arm_sinr_p = np.concatenate(
            [s0_p * gp / (base_p + self.noise_w) for gp in self.power_gain], axis=1
        )

        tp = self.scen.tp_mbps
        kappa = KAPPA_PER_TP / tp
        util = 1.0 / (1.0 + np.exp(-kappa * (rate - tp)))
        energy = np.where(
            v_lte, self.power_energy[np.maximum(lte_pw, 0)], 0.0
        ) + np.where(v_poc, self.power_energy[np.maximum(poc_pw, 0)], 0.0)

        return {
            "rate": rate,
            "rate_lte": rate_lte / 1e6,
            "rate_poc": rate_poc / 1e6,
            "sinr_lte": sinr_lte,
            "sinr_poc": sinr_poc,
            "i_lte": i_lte,
            "i_poc": i_poc,
            "conflicts": conf_lte + conf_poc,
            "conflicts_lte": conf_lte,
            "conflicts_poc": conf_poc,
            "util": util,
            "qos_met": rate >= tp,
            "arm_sinr_l": arm_sinr_l,
            "arm_sinr_p": arm_sinr_p,
            "n_active": v_lte.astype(float) + v_poc.astype(float),
            "energy_j": energy,
            "bits": rate * 1e6,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def sinr_cost(arm_sinr: np.ndarray) -> np.ndarray:
        """Normalised interference cost: 0 at +20 dB SINR, 1 at -20 dB."""
        db = 10.0 * np.log10(np.maximum(arm_sinr, 1e-12))
        return np.clip((20.0 - db) / 40.0, 0.0, 1.0)

    def rssi_matrix(self) -> np.ndarray:
        """Symmetric RSSI estimate used by the clustering layer (dBm)."""
        p = self.pwr_poc
        return np.maximum(p, p.T)

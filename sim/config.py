"""Configuration for the CEAC-SW-LCB simulator.

Physical parameters follow Li et al. (2026), "QoS-aware heterogeneous spectrum
allocation for UAV-assisted emergency communication networks", Vehicular
Communications 60:101042 (Table 4), which is the base paper of the project.

The CEAC mechanisms and the tunable parameters follow the Review 2 design
document (two-stage GPS-then-RSSI clustering, cluster-head election, per-cluster
SW-LCB with the energy-aware cost of Eq. (7) and the adaptive window of
Eq. (8)).  Values that Review 2 explicitly leaves "to be set by simulation" are
marked REVIEW2.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

# ----------------------------------------------------------------------------
# Emergency service classes (base paper Table 3)
# ----------------------------------------------------------------------------
# rate_mbps is the QoS target TP_se; data_mbit is the payload range.
SERVICES = {
    "text": {"data_mbit": (0.1, 0.5), "rate_mbps": 0.100, "weight": 0.20},
    "image": {"data_mbit": (0.9, 1.5), "rate_mbps": 0.150, "weight": 0.20},
    "voice": {"data_mbit": (1.8, 2.5), "rate_mbps": 0.340, "weight": 0.30},
    "video": {"data_mbit": (8.0, 12.0), "rate_mbps": 0.800, "weight": 0.30},
}
SERVICE_NAMES = list(SERVICES)

# Utility slope.  Eq. (19) of the base paper leaves kappa unspecified.  We centre
# the sigmoid on each service's own requirement (kappa = KAPPA_PER_TP / TP_se) so
# that utilities are comparable across service classes.
KAPPA_PER_TP = 3.0

# ----------------------------------------------------------------------------
# IEEE 802.11b/g partially overlapping channels
# ----------------------------------------------------------------------------
POC_WIDTH_MHZ = 22.0
POC_SPACING_MHZ = 5.0


def poc_overlap(delta: int) -> float:
    """Normalised spectral overlap between POC channels ``delta`` apart."""
    delta = abs(delta)
    frac = (POC_WIDTH_MHZ - POC_SPACING_MHZ * delta) / POC_WIDTH_MHZ
    return max(0.0, min(1.0, frac))


POC_OVERLAP = [[poc_overlap(i - j) for j in range(11)] for i in range(11)]


@dataclass
class Config:
    """Scenario and algorithm parameters."""

    # ---- deployment (base paper Sec. 5.1) ----
    area: float = 1000.0
    n_uav: int = 10
    n_users: int = 70
    uav_alt: tuple[float, float] = (50.0, 200.0)
    uav_speed_max: float = 15.0
    user_speed_max: float = 6.0
    gm_xi: float = 0.75
    surge_prob: float = 0.05

    # Structured (non-stationary) mobility.  n_hotspots = 0 keeps the uniform,
    # well-mixed deployment used everywhere else.  With > 0 the ground users are
    # deployed around drifting hotspots and the UAVs patrol between them; a
    # breakpoint teleports one hotspot, which is the abrupt regime change the
    # base paper models with its breakpoint process (Upsilon_E in O(E^chi)).
    n_hotspots: int = 0
    hotspot_radius: float = 150.0
    hotspot_drift: float = 2.0
    uav_patrol: bool = False
    break_prob: float = 0.0
    # What a breakpoint does: "teleport" keeps the local density (a team moves
    # elsewhere), "merge" makes a team converge on another one (density doubles
    # at the destination) and "disperse" moves it to an empty area (density
    # drops).  Only merge/disperse change the reward distribution, which is what
    # a sliding window can track; "shock" draws one of the two at random, which
    # is what e10 uses.
    break_mode: str = "merge"

    # Local link of each transmitter: distance to its own receiver, metres.
    # The receiver moves with the transmitter, so the desired signal stays
    # usable while interference depends on the moving topology.
    link_dist: tuple[float, float] = (30.0, 120.0)

    # ---- spectrum (base paper Table 4) ----
    n_lte: int = 5
    n_poc: int = 11
    b_lte_mhz: float = 4.0  # 20 MHz carrier / 5 channels
    b_poc_mhz: float = 2.0  # 22 MHz carrier / 11 channels
    f_lte_ghz: float = 0.8
    f_poc_ghz: float = 2.4
    p_tx_dbm: float = 20.0
    # Two transmit-power modes per channel.  Review 2 Eq. (7) needs a per-arm
    # E_tx, which only exists if the arm can change the transmit power.
    power_dbm: tuple[float, ...] = (20.0, 14.0)
    noise_dbm: float = -114.0
    # A pair is an interference edge when the received power exceeds this level
    # (10 dB above the noise floor).
    intf_threshold_dbm: float = -88.0
    path_loss_exp: float = 3.0
    shadow_sigma_db: float = 4.0
    nakagami_m: float = 1.5
    # Post-disaster blockage: rubble/building footprints that attenuate any link
    # crossing them.  GPS cannot see these, RSSI can, which is the rationale for
    # Stage 2 of Review 2.
    n_obstacles: int = 10
    obstacle_radius: tuple[float, float] = (40.0, 80.0)
    blockage_db: float = 20.0

    # ---- emergency services ----
    service_weights: tuple[float, ...] = (0.20, 0.20, 0.30, 0.30)

    # ---- cost function, Review 2 Eq. (7) ----
    # cost = b1*I/Imax + b2*max(0, 1-U/Ureq) + b3*Etx/Eres
    beta_base: tuple[float, float, float] = (0.55, 0.45, 0.0)
    beta_energy: tuple[float, float, float] = (0.44, 0.36, 0.20)

    # ---- bandit ----
    window_lambda: float = 50.0  # REVIEW2: lambda in W_t = ceil(lambda t^alpha)
    alpha_base: float = 0.21  # base paper Fig. 9 best setting
    alpha_min: float = 0.12  # REVIEW2 adaptive-window bounds
    alpha_max: float = 0.30
    xi: float = 0.02  # exploration constant, Review 2 Eq. (8)
    ts_eps_scale: float = 5.0  # D-eps-TS: eps_t = min(1, scale * A / t)
    ucb_c: float = 0.15
    gamma_cluster: float = 0.35  # weight of the cluster-aggregated estimate
    boundary_penalty: float = 0.08

    # ---- clustering (Review 2 Sections 4-8) ----
    head_radius: float = 200.0  # r_c
    affinity_lambda: float = 0.5  # lambda in Eq. (3)
    affinity_tau: float = 0.18  # tau in Eq. (4), calibrated (Table 8)
    rssi_min: float = -105.0
    rssi_max: float = -40.0
    rssi_smooth: float = 0.30  # alpha, EWMA factor
    omega_bounds: tuple[float, float] = (0.25, 0.75)
    omega_mode: str = "adaptive"  # "adaptive" (Eq. 6) or "fixed" (0.5)
    head_hysteresis: bool = True
    head_margin: float = 0.05  # delta
    border_margin: float = 0.05  # delta_b
    energy_floor: float = 0.10  # E_th, fraction of initial energy
    cluster_nmax: int = 24
    stage1_period: int = 20  # T_g
    kmeans_iters: int = 12

    # ---- energy ----
    energy_uav_j: float = 450.0
    energy_user_j: float = 300.0
    e_slot_ref_j: float = 0.5  # normalises the per-slot energy cost
    idle_power_w: float = 0.01

    # ---- capability mix ----
    # UAVs are dual-radio; ground users are split as follows (REVIEW2).
    p_dual: float = 0.45
    p_lte_only: float = 0.25  # remaining ground users are POC-only

    # ---- run control ----
    slots: int = 1600
    warmup: int = 100
    seeds: int = 8
    measure_tail: float = 0.20  # steady-state window for convergence
    early_until: int = 500  # "all nodes alive" window for spectrum metrics
    convergence_tol: float = 0.05

    def n_lte_arms(self) -> int:
        return self.n_lte * len(self.power_dbm)

    def n_poc_arms(self) -> int:
        return self.n_poc * len(self.power_dbm)

    def split_arm(self, a, n_ch: int):
        return a % n_ch, a // n_ch

    def variant(self, **kw) -> Config:
        return replace(self, **kw)


# Node kinds ----------------------------------------------------------------
KIND_UAV = 0
KIND_LTE_ONLY = 1
KIND_POC_ONLY = 2
KIND_DUAL = 3

# Arm layout ----------------------------------------------------------------
# LTE band: arms 0..n_lte-1 are channels, arm n_lte is "off".
# POC band: arms 0..n_poc-1 are channels, arm n_poc is "off".

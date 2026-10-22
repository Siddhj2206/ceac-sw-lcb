"""Monte-Carlo runner.

A group of algorithms is evaluated in a single pass over the same channel
realisation (common random numbers), so every algorithm sees identical node
positions, shadowing and fading for a given seed.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .cluster import ClusterManager, ClusterState
from .config import Config
from .learners import Bandit
from .radio import Radio
from .scenario import Scenario


@dataclass
class AlgoSpec:
    name: str
    mode: str
    betas: tuple = (0.55, 0.45, 0.0)
    cluster: bool = False
    capability: bool = False
    low_power: bool = False
    aggregate: bool = False
    boundary: bool = False
    adaptive: bool = False


@dataclass
class RunResult:
    name: str
    seed: int
    series: dict
    summary: dict


class _State:
    """Per-algorithm mutable state."""

    def __init__(self, cfg: Config, spec: AlgoSpec, scen: Scenario, rng):
        self.spec = spec
        self.learner = Bandit(cfg, scen.n, spec.mode, rng)
        self.energy = scen.energy0.copy()
        self.prev_l = np.full(scen.n, -1)
        self.prev_p = np.full(scen.n, -1)
        self.messages = 0

        n = scen.n
        n_pw = len(cfg.power_dbm)
        al, ap = cfg.n_lte_arms(), cfg.n_poc_arms()

        self.mask_l = np.zeros((n, al), dtype=bool)
        self.mask_p = np.zeros((n, ap), dtype=bool)
        # power-mode columns: arm a = p*n_ch + channel
        for p in range(n_pw):
            if p > 0 and not spec.low_power:
                continue
            self.mask_l[:, p * cfg.n_lte : (p + 1) * cfg.n_lte] = True
            self.mask_p[:, p * cfg.n_poc : (p + 1) * cfg.n_poc] = True
        if spec.capability:
            self.mask_l[~scen.has_lte, :] = False
            self.mask_p[~scen.has_poc, :] = False

        self.phys_l = scen.has_lte
        self.phys_p = scen.has_poc


def _blend_factory(cs: ClusterState, gamma: float):
    groups = [np.where(cs.assign == c)[0] for c in range(cs.heads.size)]

    def blend(mean, cnt, glob, band):
        """Cluster mean as a prior for arms this node has not observed yet.

        Unseen arms get gamma * (cluster mean) + (1 - gamma) * (global mean);
        observed arms keep their own windowed estimate.
        """
        out = mean.copy()
        for mem in groups:
            if mem.size <= 1:
                continue
            sub = mean[mem]
            if not np.isfinite(sub).any():
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                agg = np.nanmean(sub, axis=0)
            use = gamma * agg[None, :] + (1.0 - gamma) * glob[mem]
            unknown = ~np.isfinite(sub) & np.isfinite(agg)[None, :]
            out[mem] = np.where(unknown, use, sub)
        return out

    return blend


def _boundary_penalty(
    cs: ClusterState, cfg: Config, prev_ch: np.ndarray, n_ch: int
) -> np.ndarray:
    """Penalise channels that cross-cluster neighbours used in the last slot."""
    n = cs.assign.size
    pen = np.zeros((n, n_ch * len(cfg.power_dbm)))
    idx = np.where(cs.boundary)[0]
    for m in idx:
        nb = np.where(cs.kept[m] & (cs.assign != cs.assign[m]))[0]
        if nb.size == 0:
            continue
        ch = prev_ch[nb]
        ch = ch[(ch >= 0) & (ch < n_ch)]
        if ch.size == 0:
            continue
        counts = np.bincount(ch, minlength=n_ch) / nb.size
        # same penalty for both power modes of a channel
        pen[m] = np.tile(counts, len(cfg.power_dbm))
    return cfg.boundary_penalty * pen


def _cost_matrices(
    cfg: Config, scen: Scenario, radio: Radio, out: dict, energy: np.ndarray, betas
):
    b1, b2, b3 = betas
    n_bands = scen.has_lte.astype(float) + scen.has_poc.astype(float)
    bt = scen.tp_mbps / np.maximum(n_bands, 1.0)

    ci_l = radio.sinr_cost(out["arm_sinr_l"])
    ci_p = radio.sinr_cost(out["arm_sinr_p"])

    # smooth QoS shortfall of each band, relative to that band's share of the target
    u_l = 1.0 / (1.0 + np.exp(-3.0 * (out["rate_lte"] / bt - 1.0)))
    u_p = 1.0 / (1.0 + np.exp(-3.0 * (out["rate_poc"] / bt - 1.0)))
    sh_l = 1.0 - u_l
    sh_p = 1.0 - u_p

    # per-arm transmit energy
    e_l = np.concatenate([np.full(cfg.n_lte, e) for e in radio.power_energy])
    e_p = np.concatenate([np.full(cfg.n_poc, e) for e in radio.power_energy])
    # energy is cheap while the battery is healthy and expensive when it is not
    scale = np.clip((scen.energy0 / np.maximum(energy, 1.0)) ** 2, 0.0, 10.0)

    ec_l = np.clip(e_l[None, :] / cfg.e_slot_ref_j * scale[:, None], 0.0, 1.0)
    ec_p = np.clip(e_p[None, :] / cfg.e_slot_ref_j * scale[:, None], 0.0, 1.0)

    cost_l = b1 * ci_l + b2 * sh_l[:, None] + b3 * ec_l
    cost_p = b1 * ci_p + b2 * sh_p[:, None] + b3 * ec_p
    return cost_l, cost_p


def run_group(
    cfg: Config, specs: list[AlgoSpec], seed: int, use_cluster: bool = True
) -> dict[str, RunResult]:
    env_rng = np.random.default_rng(seed * 7919 + 11)
    scen = Scenario(cfg, env_rng)
    radio = Radio(cfg, scen, env_rng)

    cm = (
        ClusterManager(cfg, scen, np.random.default_rng(seed * 7919 + 23))
        if use_cluster
        else None
    )

    ar = np.arange(scen.n)
    states = {}
    for i, sp in enumerate(specs):
        rng = np.random.default_rng(seed * 104729 + i + 1)
        states[sp.name] = _State(cfg, sp, scen, rng)

    keys = (
        "conflicts",
        "rate",
        "rate_lte",
        "rate_poc",
        "util",
        "qos",
        "jain",
        "active",
        "regret",
        "energy",
        "alive",
    )
    series = {sp.name: {k: [] for k in keys} for sp in specs}

    for t in range(cfg.slots):
        if t > 0:
            scen.step(env_rng)
            radio.update()

        cs = cm.step(radio, t) if cm is not None else None

        for sp in specs:
            st = states[sp.name]
            alpha = cs.alpha if (cs is not None and sp.adaptive) else None
            stats = st.learner.stats(t, alpha=alpha)

            blend = (
                _blend_factory(cs, cfg.gamma_cluster)
                if (cs is not None and sp.aggregate)
                else None
            )

            pen_l = pen_p = None
            if cs is not None and sp.boundary:
                pen_l = _boundary_penalty(cs, cfg, st.prev_l, cfg.n_lte)
                pen_p = _boundary_penalty(cs, cfg, st.prev_p, cfg.n_poc)

            a_l, a_p = st.learner.choose(
                t, stats, st.mask_l, st.mask_p, pen_l, pen_p, blend=blend
            )

            ch_l, pw_l = cfg.split_arm(a_l, cfg.n_lte)
            ch_p, pw_p = cfg.split_arm(a_p, cfg.n_poc)
            alive = st.energy > 0.0
            ch_l = np.where(scen.has_lte & alive, ch_l, -1)
            ch_p = np.where(scen.has_poc & alive, ch_p, -1)

            out = radio.evaluate(ch_l, pw_l, ch_p, pw_p)
            cost_l, cost_p = _cost_matrices(cfg, scen, radio, out, st.energy, sp.betas)

            cl = cost_l[ar, a_l]
            cp = cost_p[ar, a_p]
            b2 = sp.betas[1]
            cl = np.where(scen.has_lte, cl, b2)
            cp = np.where(scen.has_poc, cp, b2)
            st.learner.observe(t, a_l, cl, a_p, cp, scen.has_lte, scen.has_poc)

            # ---------- metrics ----------
            rate = out["rate"]
            s = series[sp.name]
            n_active = float(out["n_active"].sum())
            s["conflicts"].append(float(out["conflicts"].sum() / max(n_active, 1.0)))
            s["rate"].append(rate.sum())
            s["rate_lte"].append(out["rate_lte"].sum())
            s["rate_poc"].append(out["rate_poc"].sum())
            s["util"].append(out["util"].mean())
            s["qos"].append(out["qos_met"].mean())
            s["jain"].append(rate.sum() ** 2 / (scen.n * (rate**2).sum() + 1e-12))
            s["active"].append(out["n_active"].mean())
            s["energy"].append(float((st.energy / scen.energy0).mean()))
            s["alive"].append(float((st.energy > 1e-9).mean()))

            if sp.mode in ("swlcb", "ceac", "ucb", "greedy", "ts"):
                ok_l = st.mask_l & st.phys_l[:, None]
                ok_p = st.mask_p & st.phys_p[:, None]
                best_l = (
                    np.where(ok_l, cost_l, np.inf).min(1)
                    if ok_l.any()
                    else np.full(scen.n, np.inf)
                )
                best_p = (
                    np.where(ok_p, cost_p, np.inf).min(1)
                    if ok_p.any()
                    else np.full(scen.n, np.inf)
                )
                best = np.minimum(best_l, best_p)
                got = cl + cp
                best = np.where(np.isfinite(best), best, got)
                s["regret"].append(float(np.maximum(got - best, 0.0).mean()))
            else:
                s["regret"].append(0.0)

            st.energy = np.maximum(
                0.0, st.energy - (out["energy_j"] + cfg.idle_power_w)
            )
            st.prev_l = np.where(ch_l >= 0, ch_l, -1)
            st.prev_p = np.where(ch_p >= 0, ch_p, -1)

            if cs is not None:
                st.messages += cs.messages

    results = {}
    breaks = np.asarray(scen.break_slots, dtype=int)
    for sp in specs:
        arrs = {k: np.asarray(v) for k, v in series[sp.name].items()}
        arrs["break"] = breaks
        results[sp.name] = RunResult(
            name=sp.name,
            seed=seed,
            series=arrs,
            summary=_summarise(cfg, arrs, states[sp.name], scen),
        )
    return results


def _summarise(cfg: Config, s: dict, state: _State, scen: Scenario) -> dict:
    lo = cfg.warmup
    steady = int(cfg.slots * (1.0 - cfg.measure_tail))
    early = slice(lo, min(cfg.early_until, cfg.slots))
    out = {
        "conflicts_early": float(s["conflicts"][early].mean()),
        "throughput_early": float(s["rate"][early].mean()),
        "throughput_lte_early": float(s["rate_lte"][early].mean()),
        "throughput_poc_early": float(s["rate_poc"][early].mean()),
        "utility_early": float(s["util"][early].mean()),
        "qos_early": float(s["qos"][early].mean()),
        "jain_early": float(s["jain"][early].mean()),
        "active_end": float(s["alive"][-1]),
        "total_mbit": float(s["rate"].sum()),
    }
    out.update(
        {
            "conflicts": s["conflicts"][lo:].mean(),
            "throughput": s["rate"][lo:].mean(),
            "throughput_lte": s["rate_lte"][lo:].mean(),
            "throughput_poc": s["rate_poc"][lo:].mean(),
            "utility": s["util"][lo:].mean(),
            "qos": s["qos"][lo:].mean(),
            "jain": s["jain"][lo:].mean(),
            "active": s["active"][lo:].mean(),
            "regret": s["regret"].sum(),
            "residual_energy": s["energy"][-1],
            "energy_per_mbit": _energy_per_mbit(s, scen),
            "messages": state.messages / cfg.slots,
            "convergence": _convergence(cfg, s["conflicts"], steady),
        }
    )
    return out


def _energy_per_mbit(s: dict, scen: Scenario) -> float:
    e_used = float((scen.energy0 * (1.0 - s["energy"][-1])).sum())
    total_mbit = float(s["rate"].sum())
    return e_used / total_mbit if total_mbit > 0 else float("nan")


def _convergence(cfg: Config, series: np.ndarray, steady_from: int) -> float:
    """First slot after which the 50-slot mean stays within tol of steady state.

    The reference is taken from slots 300-600, before battery depletion starts
    to change the population of transmitters.
    """
    k = 50
    # Only the pre-depletion part of the run is meaningful for convergence;
    # once nodes start switching off the conflict level falls for a reason that
    # has nothing to do with the learner.
    horizon = min(cfg.slots, 1200)
    series = series[:horizon]
    ref_lo, ref_hi = min(300, max(horizon - 1, 1)), min(600, horizon)
    if ref_hi <= ref_lo:
        ref_lo, ref_hi = 0, horizon
    ref = float(np.nanmean(series[ref_lo:ref_hi]))
    if ref <= 0:
        ref = max(float(np.nanmean(series[cfg.warmup :])), 1e-9)
    tol = cfg.convergence_tol * max(ref, 1e-9)
    ma = np.convolve(series, np.ones(k) / k, mode="valid")
    ok = np.abs(ma - ref) <= tol
    bad = np.where(~ok)[0]
    if bad.size == 0:
        return float(min(cfg.warmup, cfg.slots))
    return float(min(bad[-1] + 1, cfg.slots))


def run_single(cfg: Config, spec: AlgoSpec, seed: int) -> RunResult:
    return run_group(cfg, [spec], seed)[spec.name]

"""Bandit learners.

Every learner minimises the same per-band cost of Review 2 Eq. (7).  They differ
only in how the index is formed and how much history it uses:

* ``random``  uniform choice, no learning.
* ``greedy``  lowest mean cost, no exploration bonus.
* ``ucb``     confidence bound over the full history (base-paper baseline).
* ``ts``      decaying-epsilon Thompson sampling (base-paper baseline).
* ``swlcb``   sliding-window LCB, W_t = min(ceil(lambda t^alpha), t) (base paper).
* ``ceac``    sliding-window LCB with the per-cluster adaptive window; the
              estimate is blended with the cluster aggregate by the runner.

Cold start follows Algorithm 1 of the base paper: every feasible arm is played
once before the index is evaluated.  After that the index alone decides, which
is what a sliding-window learner needs; forcing re-exploration of every forgotten
arm (the textbook SW-UCB rule) makes the policy cycle and performs worse than
random in a congestion game.

The confidence term is subtracted from the cost because the problem is a
minimisation; this is the cost-form equivalent of the reward index of the base
paper and matches Review 2 Eq. (8).
"""

from __future__ import annotations

import numpy as np

from .config import Config


class Bandit:
    def __init__(self, cfg: Config, n: int, mode: str, rng: np.random.Generator):
        self.cfg = cfg
        self.n = n
        self.mode = mode
        self.rng = rng
        self.a_lte = cfg.n_lte_arms()
        self.a_poc = cfg.n_poc_arms()
        self.temporal = mode in ("swlcb", "ceac")

        shape_l = (n, self.a_lte)
        shape_p = (n, self.a_poc)

        # Global history, always maintained (cold start + window fallback).
        self.sum_all_l = np.zeros(shape_l)
        self.cnt_all_l = np.zeros(shape_l)
        self.sum_all_p = np.zeros(shape_p)
        self.cnt_all_p = np.zeros(shape_p)

        if self.temporal:
            T = cfg.slots + 1
            self.cum_l = np.zeros((T, *shape_l), dtype=np.float32)
            self.cnt_l = np.zeros((T, *shape_l), dtype=np.int32)
            self.cum_p = np.zeros((T, *shape_p), dtype=np.float32)
            self.cnt_p = np.zeros((T, *shape_p), dtype=np.int32)

    # ------------------------------------------------------------------
    def window(self, t: int, alpha: float | np.ndarray | None = None):
        cfg = self.cfg
        tt = max(t, 1)
        if self.mode == "ceac":
            a = cfg.alpha_base if alpha is None else alpha
            w = np.ceil(cfg.window_lambda * tt ** np.asarray(a, dtype=float))
            return np.clip(w, 3, tt).astype(int)
        if self.mode == "swlcb":
            a = cfg.alpha_base if alpha is None else alpha
            return int(np.clip(np.ceil(cfg.window_lambda * tt**a), 3, tt))
        return tt

    # ------------------------------------------------------------------
    def stats(self, t: int, alpha=None):
        """Windowed and global statistics for both bands."""
        if not self.temporal:
            return (
                self.sum_all_l,
                self.cnt_all_l,
                self.sum_all_p,
                self.cnt_all_p,
                self.sum_all_l,
                self.cnt_all_l,
                self.sum_all_p,
                self.cnt_all_p,
            )

        w = self.window(t, alpha)
        rows = np.arange(self.n)
        if np.isscalar(w):
            lo = max(t - int(w), 0)
            vals = [
                self.cum_l[t, rows] - self.cum_l[lo, rows],
                self.cnt_l[t, rows] - self.cnt_l[lo, rows],
                self.cum_p[t, rows] - self.cum_p[lo, rows],
                self.cnt_p[t, rows] - self.cnt_p[lo, rows],
            ]
        else:
            lo = np.maximum(t - np.asarray(w), 0)
            vals = [
                self.cum_l[t, rows] - self.cum_l[lo, rows],
                self.cnt_l[t, rows] - self.cnt_l[lo, rows],
                self.cum_p[t, rows] - self.cum_p[lo, rows],
                self.cnt_p[t, rows] - self.cnt_p[lo, rows],
            ]
        return (*vals, self.sum_all_l, self.cnt_all_l, self.sum_all_p, self.cnt_all_p)

    # ------------------------------------------------------------------
    def choose(
        self,
        t: int,
        stats,
        mask_l: np.ndarray,
        mask_p: np.ndarray,
        penalty_l: np.ndarray | None = None,
        penalty_p: np.ndarray | None = None,
        blend=None,
    ):
        sl, cl, sp, cp, sal, cal, sap, cap = stats
        a_l = self._choose_band(t, sl, cl, sal, cal, mask_l, penalty_l, blend, "l")
        a_p = self._choose_band(t, sp, cp, sap, cap, mask_p, penalty_p, blend, "p")
        return a_l, a_p

    def _index(self, t, s, c, sa, ca, mask, penalty, blend, band):
        """LCB index on cost for the greedy/UCB/SW-LCB/CEAC learners."""
        glob = sa / np.maximum(ca, 1.0)
        mean = np.where(c > 0, s / np.maximum(c, 1.0), np.nan)
        if blend is not None:
            mean = blend(mean, c, glob, band)
        mean = np.where(np.isfinite(mean), mean, glob)
        score = mean.copy()

        if self.mode in ("ucb", "swlcb", "ceac"):
            if self.mode == "ucb":
                bonus = self.cfg.ucb_c * np.sqrt(np.log(max(t, 2)) / np.maximum(c, 1.0))
            else:
                w = self.window(t)
                ww = (
                    float(np.min(w))
                    if np.isscalar(w)
                    else float(np.min(np.maximum(w, 2)))
                )
                bonus = np.sqrt(self.cfg.xi * np.log(max(ww, 2.0)) / np.maximum(c, 1.0))
            score = score - bonus

        if penalty is not None:
            score = score + penalty
        score = np.where(mask, score, np.inf)
        return score, np.inf

    def _choose_band(self, t, s, c, sa, ca, mask, penalty, blend, band):
        mode = self.mode
        n, A = c.shape

        if mode == "ts":
            a_post = 1.0 + c - s
            b_post = 1.0 + s
            score = self.rng.beta(np.maximum(a_post, 1e-3), np.maximum(b_post, 1e-3))
            eps = min(1.0, self.cfg.ts_eps_scale * A / max(t, 1))
            if t > 0:
                # epsilon-exploration must be uniform, not a constant score
                flip = self.rng.random((n, A)) < eps
                score = np.where(flip, self.rng.random((n, A)), score)
            score = np.where(mask, score, -np.inf)
            return score.argmax(1)

        if mode == "random":
            score = self.rng.random((n, A))
            score = np.where(mask, score, np.inf)
            return score.argmin(1)

        # ---- cold start: play every feasible arm once ------------------
        cold = (ca == 0) & mask
        if cold.any():
            score = np.where(cold, ca, np.inf)
            score = score + self.rng.random((n, A)) * 1e-9
            choice = score.argmin(1)
            # rows without a cold arm keep normal behaviour
            warm_rows = ~cold.any(1)
            if warm_rows.any():
                idx = self._index(t, s, c, sa, ca, mask, penalty, blend, band)[
                    0
                ].argmin(1)
                choice = np.where(warm_rows, idx, choice)
            return choice

        score, _ = self._index(t, s, c, sa, ca, mask, penalty, blend, band)
        score = np.where(mask, score, np.inf) + self.rng.random((n, A)) * 1e-9
        return score.argmin(1)

    # ------------------------------------------------------------------
    def observe(self, t: int, arms_l, costs_l, arms_p, costs_p, upd_l, upd_p):
        rows = np.arange(self.n)
        il = rows[upd_l]
        ip = rows[upd_p]

        self.sum_all_l[il, arms_l[il]] += costs_l[il]
        self.cnt_all_l[il, arms_l[il]] += 1
        self.sum_all_p[ip, arms_p[ip]] += costs_p[ip]
        self.cnt_all_p[ip, arms_p[ip]] += 1

        if self.temporal:
            self.cum_l[t + 1] = self.cum_l[t]
            self.cnt_l[t + 1] = self.cnt_l[t]
            self.cum_p[t + 1] = self.cum_p[t]
            self.cnt_p[t + 1] = self.cnt_p[t]
            self.cum_l[t + 1, il, arms_l[il]] += costs_l[il].astype(np.float32)
            self.cnt_l[t + 1, il, arms_l[il]] += 1
            self.cum_p[t + 1, ip, arms_p[ip]] += costs_p[ip].astype(np.float32)
            self.cnt_p[t + 1, ip, arms_p[ip]] += 1

"""Figures for the Results and Discussion section."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RES = "results"
FIG = "figures"

ALGOS = ["Random", "Greedy", "UCB", "D-eps-TS", "SW-LCB", "CEAC-SW-LCB"]
COLORS = {
    "Random": "#8c8c8c",
    "Greedy": "#b07aa1",
    "UCB": "#4e79a7",
    "D-eps-TS": "#f28e2b",
    "SW-LCB": "#59a14f",
    "CEAC-SW-LCB": "#e15759",
}
MARKERS = {
    "Random": "o",
    "Greedy": "s",
    "UCB": "^",
    "D-eps-TS": "v",
    "SW-LCB": "D",
    "CEAC-SW-LCB": "*",
}

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def _ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def _lines(ax, df, xcol, ycol, yerr=True, label_map=None):
    for a in ALGOS:
        sub = df[df.algo == a]
        if sub.empty:
            continue
        xs = sorted(sub[xcol].unique())
        m = np.array([sub[sub[xcol] == x][ycol].mean() for x in xs])
        e = np.array([_ci(sub[sub[xcol] == x][ycol]) for x in xs])
        name = label_map.get(a, a) if label_map else a
        ax.plot(xs, m, label=name, color=COLORS[a], marker=MARKERS[a], ms=5, lw=1.6)
        if yerr:
            ax.fill_between(xs, m - e, m + e, color=COLORS[a], alpha=0.15, lw=0)


def _legend(ax, ncol=2):
    ax.legend(fontsize=7.5, ncol=ncol, frameon=False, loc="best")


# ----------------------------------------------------------------------------
def fig_users():
    df = pd.read_csv(f"{RES}/e1_users.csv")

    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.5))
    _lines(axs[0], df, "n_users", "conflicts_early")
    axs[0].set_xlabel("Number of ground users")
    axs[0].set_ylabel("Interference conflicts per active node")
    axs[0].set_title("(a) Absolute conflict level", fontsize=9)

    ref = df[df.algo == "SW-LCB"].groupby("n_users").conflicts_early.mean()
    for a in ALGOS:
        if a == "SW-LCB":
            continue
        sub = df[df.algo == a].groupby("n_users").conflicts_early.mean()
        rel = 100 * (sub / ref - 1)
        axs[1].plot(
            sub.index, rel, marker=MARKERS[a], color=COLORS[a], ms=5, lw=1.6, label=a
        )
    axs[1].axhline(0, color="k", lw=0.8, ls="--")
    axs[1].set_xlabel("Number of ground users")
    axs[1].set_ylabel("Conflicts relative to SW-LCB (%)")
    axs[1].set_title("(b) Difference from SW-LCB", fontsize=9)
    axs[0].legend(fontsize=7, frameon=False, ncol=2)
    axs[1].legend(fontsize=7, frameon=False)
    fig.savefig(f"{FIG}/fig1_conflicts_vs_users.png")
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.9))
    for ax, col, title in (
        (axs[0], "throughput_lte_early", "(a) LTE licensed spectrum"),
        (axs[1], "throughput_poc_early", "(b) IEEE 802.11b/g POC spectrum"),
    ):
        _lines(ax, df, "n_users", col)
        ax.set_xlabel("Number of ground users")
        ax.set_ylabel("Aggregate throughput (Mbps)")
        ax.set_title(title, fontsize=9)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=6,
        frameon=False,
        fontsize=7.5,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.subplots_adjust(bottom=0.28)
    fig.savefig(f"{FIG}/fig2_throughput_vs_users.png")
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.4))
    for ax, col, ylab, title in (
        (axs[0], "utility_early", "Mean service utility", "(a) Service utility"),
        (
            axs[1],
            "qos_early",
            "Fraction meeting QoS target",
            "(b) QoS satisfaction ratio",
        ),
    ):
        _lines(ax, df, "n_users", col)
        ax.set_xlabel("Number of ground users")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=9)
    axs[0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.savefig(f"{FIG}/fig3_utility_qos_vs_users.png")
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.4))
    _lines(axs[0], df, "n_users", "energy_per_mbit")
    axs[0].set_xlabel("Number of ground users")
    axs[0].set_ylabel("Radio energy per delivered Mbit (J/Mbit)")
    axs[0].set_title("(a) Energy efficiency", fontsize=9)
    axs[0].legend(fontsize=7, frameon=False)

    users = sorted(df.n_users.unique())
    x = np.arange(len(users))
    w = 0.36
    base = df[df.algo != "CEAC-SW-LCB"].groupby("n_users").active_end
    ceac = df[df.algo == "CEAC-SW-LCB"].groupby("n_users").active_end
    bm = [100 * base.get_group(u).mean() for u in users]
    be = [_ci(100 * base.get_group(u)) for u in users]
    cm = [100 * ceac.get_group(u).mean() for u in users]
    ce = [_ci(100 * ceac.get_group(u)) for u in users]
    axs[1].bar(
        x - w / 2,
        bm,
        w,
        yerr=be,
        capsize=3,
        color="#8c8c8c",
        label="Baselines (mean of five)",
    )
    axs[1].bar(
        x + w / 2,
        cm,
        w,
        yerr=ce,
        capsize=3,
        color=COLORS["CEAC-SW-LCB"],
        label="CEAC-SW-LCB",
    )
    axs[1].set_xticks(x)
    axs[1].set_xticklabels([str(u) for u in users])
    axs[1].set_ylim(0, 115)
    axs[1].set_xlabel("Number of ground users")
    axs[1].set_ylabel("Nodes with energy left at end (%)")
    axs[1].set_title("(b) Nodes still powered at the end of the run", fontsize=9)
    axs[1].legend(fontsize=7.5, frameon=False, loc="lower left")
    fig.savefig(f"{FIG}/fig4_energy_vs_users.png")
    plt.close(fig)


def fig_series():
    path = f"{RES}/series.csv"
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)

    def smooth(x, w):
        return pd.Series(x).rolling(w, min_periods=1).mean()

    fig, axs = plt.subplots(1, 3, figsize=(11.0, 3.2))
    for a in ALGOS:
        sub = df[df.algo == a]
        if sub.empty:
            continue
        axs[0].plot(
            sub.slot, smooth(sub.conflicts.values, 5), color=COLORS[a], lw=1.3, label=a
        )
        axs[1].plot(sub.slot, 100 * sub.alive, color=COLORS[a], lw=1.6, label=a)
        axs[2].plot(
            sub.slot, smooth(sub.qos.values, 25), color=COLORS[a], lw=1.5, label=a
        )
    axs[0].set_xlim(0, 300)
    axs[0].set_xlabel("Slot")
    axs[0].set_ylabel("Conflicts per active node")
    axs[0].set_title("(a) Convergence during the first 300 s", fontsize=9)

    axs[1].set_xlabel("Slot")
    axs[1].set_ylabel("Nodes with energy (%)")
    axs[1].set_title("(b) Battery depletion", fontsize=9)

    axs[2].set_xlabel("Slot")
    axs[2].set_ylabel("QoS satisfaction ratio")
    axs[2].set_title("(c) Service satisfaction over the horizon", fontsize=9)
    axs[1].legend(fontsize=7, frameon=False)
    fig.savefig(f"{FIG}/fig5_time_series.png")
    plt.close(fig)


def fig_ablation():
    df = pd.read_csv(f"{RES}/e6_ablation.csv")
    order = [
        "SW-LCB",
        "CEAC-noCluster",
        "CEAC-noCapability",
        "CEAC-noEnergy",
        "CEAC-noBoundary",
        "CEAC-SW-LCB",
    ]
    labels = [
        "SW-LCB\n(baseline)",
        "CEAC\n−clustering",
        "CEAC\n−capability",
        "CEAC\n−energy",
        "CEAC\n−boundary",
        "CEAC\nfull",
    ]
    metrics = [
        ("conflicts_early", "Conflicts per active node", False),
        ("throughput_early", "Throughput (Mbps)", False),
        ("energy_per_mbit", "Energy (J/Mbit)", True),
        ("qos_early", "QoS satisfaction ratio", False),
    ]
    fig, axs = plt.subplots(1, 4, figsize=(12.5, 3.3))
    for ax, (col, lab, lower_better) in zip(axs, metrics, strict=True):
        vals, errs = [], []
        for a in order:
            s = df[df.algo == a][col]
            vals.append(s.mean() if len(s) else np.nan)
            errs.append(_ci(s))
        x = np.arange(len(order))
        cols = [
            "#59a14f"
            if a == "SW-LCB"
            else "#e15759"
            if a == "CEAC-SW-LCB"
            else "#f2a2a0"
            for a in order
        ]
        ax.bar(x, vals, yerr=errs, capsize=3, color=cols, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(lab, fontsize=8)
        if lower_better:
            ax.set_ylim(0, max(vals) * 1.25)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig6_ablation.png")
    plt.close(fig)


def fig_window():
    df = pd.read_csv(f"{RES}/e4_window.csv")
    fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.2))
    for a in ("SW-LCB", "CEAC-SW-LCB"):
        sub = df[df.algo == a]
        axs[0].plot(
            sub.alpha, sub.throughput_early, marker="o", color=COLORS[a], label=a
        )
        axs[1].plot(
            sub.alpha, sub.conflicts_early, marker="o", color=COLORS[a], label=a
        )
    axs[0].set_ylabel("Throughput (Mbps)")
    axs[1].set_ylabel("Conflicts per active node")
    for ax in axs:
        ax.set_xlabel("Window exponent α")
    axs[0].legend(fontsize=8, frameon=False)
    fig.savefig(f"{FIG}/fig7_window_alpha.png")
    plt.close(fig)


def fig_density():
    df = pd.read_csv(f"{RES}/e2_density.csv")
    fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.3))
    fig.subplots_adjust(wspace=0.42)
    ref = df[df.algo == "SW-LCB"].groupby("n_users").throughput_early.mean()
    for a, c in (
        ("CEAC-SW-LCB", COLORS["CEAC-SW-LCB"]),
        ("CEAC-noCluster", "#f2a2a0"),
        ("Random", COLORS["Random"]),
    ):
        sub = df[df.algo == a].groupby("n_users").throughput_early.mean()
        axs[0].plot(sub.index, 100 * (sub / ref - 1), marker="o", color=c, label=a)
    axs[0].axhline(0, color="k", lw=0.8, ls="--")
    axs[0].set_xlabel("Number of ground users")
    axs[0].set_ylabel("Throughput relative to SW-LCB (%)")
    axs[0].legend(fontsize=7.5, frameon=False)
    axs[0].set_title("(a) Spectrum performance versus density", fontsize=9)

    cdf = pd.read_csv(f"{RES}/e7_clusters.csv")
    st = cdf[cdf.variant == "two-stage"].state_per_node.mean()
    n_nodes = 100
    axs[1].bar(
        ["Full-graph\ncoordination", "Cluster-based\n(CEAC)"],
        [n_nodes - 1, st],
        color=["#4e79a7", "#e15759"],
    )
    axs[1].set_ylabel("Coordination entries per node")
    axs[1].set_title("(b) Coordination state at N = 100", fontsize=9)
    for i, v in enumerate([n_nodes - 1, st]):
        axs[1].text(i, v + 1.5, f"{v:.0f}", ha="center", fontsize=9)
    fig.savefig(f"{FIG}/fig8_density_and_state.png")
    plt.close(fig)


def fig_pruning():
    df = pd.read_csv(f"{RES}/e7_pruning.csv")
    m = df.groupby("tau")[["edges", "retained", "purity_after"]].mean()
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    ax.plot(m.index, m.edges, marker="o", color="#4e79a7", label="Affinity edges kept")
    ax.set_xlabel("Pruning threshold τ")
    ax.set_ylabel("Edges kept per slot")
    ax2 = ax.twinx()
    ax2.plot(
        m.index,
        100 * m.retained,
        marker="s",
        color="#e15759",
        label="Strong interference pairs retained (%)",
    )
    ax2.set_ylabel("Strong interference pairs retained (%)")
    ax2.grid(False)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(
        lines,
        [str(line.get_label()) for line in lines],
        fontsize=7.5,
        frameon=False,
        loc="upper right",
    )
    fig.savefig(f"{FIG}/fig9_pruning.png")
    plt.close(fig)


def fig_energy_tradeoff():
    df = pd.read_csv(f"{RES}/e5_energy.csv")
    sw = df[df.algo == "SW-LCB"]
    ce = (
        df[df.algo == "CEAC-SW-LCB"]
        .groupby("beta3")[["throughput_early", "energy_per_mbit", "active_end"]]
        .mean()
    )

    fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.4))
    fig.subplots_adjust(wspace=0.32)
    x = np.arange(len(ce))

    axs[0].bar(x, 100 * ce.active_end, color="#e15759", width=0.6)
    axs[0].axhline(
        100 * sw.active_end.mean(), color="#59a14f", ls="--", lw=1.4, label="SW-LCB"
    )
    for i, v in enumerate(100 * ce.active_end):
        axs[0].text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=8)
    axs[0].set_xticks(x)
    axs[0].set_xticklabels([f"{b:.2f}" for b in ce.index])
    axs[0].set_ylim(0, 115)
    axs[0].set_xlabel("Energy weight β₃")
    axs[0].set_ylabel("Nodes with battery at end (%)")
    axs[0].set_title("(a) Operating lifetime", fontsize=9)
    axs[0].legend(fontsize=8, frameon=False)

    ref = sw.throughput_early.mean()
    axs[1].bar(x, ce.throughput_early, color="#e15759", width=0.6)
    axs[1].axhline(ref, color="#59a14f", ls="--", lw=1.4, label="SW-LCB")
    for i, v in enumerate(ce.throughput_early):
        axs[1].text(
            i, v + 0.15, f"{100 * (v / ref - 1):+.1f}%", ha="center", fontsize=8
        )
    axs[1].set_xticks(x)
    axs[1].set_xticklabels([f"{b:.2f}" for b in ce.index])
    axs[1].set_ylim(85, 99)
    axs[1].set_xlabel("Energy weight β₃")
    axs[1].set_ylabel("Throughput (Mbps)")
    axs[1].set_title("(b) Throughput cost relative to SW-LCB", fontsize=9)
    axs[1].legend(fontsize=8, frameon=False, loc="lower left")
    fig.savefig(f"{FIG}/fig10_energy_tradeoff.png")
    plt.close(fig)


def _cat_lines(ax, df, xcol, ycol, xticklabels, scale=1.0):
    xs = sorted(df[xcol].unique())
    pos = np.arange(len(xs))
    for a in ALGOS:
        sub = df[df.algo == a]
        if sub.empty:
            continue
        m = np.array([sub[sub[xcol] == x][ycol].mean() for x in xs]) * scale
        e = np.array([_ci(sub[sub[xcol] == x][ycol]) for x in xs]) * scale
        ax.plot(pos, m, label=a, color=COLORS[a], marker=MARKERS[a], ms=5, lw=1.6)
        ax.fill_between(pos, m - e, m + e, color=COLORS[a], alpha=0.15, lw=0)
    ax.set_xticks(pos)
    ax.set_xticklabels(xticklabels)
    return xs


def fig_dynamics():
    """High mobility and breakpoints: absolute cost and adaptation quality."""
    mob = pd.read_csv(f"{RES}/e9_mobility.csv")
    brk = pd.read_csv(f"{RES}/e10_breakpoints.csv")
    rates = sorted(brk.break_prob.unique())
    labels = [
        f"{r:g}\n(never)" if r == 0 else f"{r:g}\n(1/{max(round(1 / r), 1)})"
        for r in rates
    ]

    fig, axs = plt.subplots(1, 3, figsize=(11.2, 3.3))
    fig.subplots_adjust(wspace=0.34)

    _lines(axs[0], mob, "uav_speed_max", "throughput_early")
    axs[0].set_xlabel("UAV speed (m/s)")
    axs[0].set_ylabel("Throughput (Mbps)")
    axs[0].set_title("(a) Hotspot deployment, no breakpoints", fontsize=9)

    _cat_lines(axs[1], brk, "break_prob", "throughput_early", labels)
    axs[1].set_xlabel("Break rate per slot")
    axs[1].set_ylabel("Throughput (Mbps)")
    axs[1].set_title("(b) Abrupt redeployments", fontsize=9)

    # throughput relative to SW-LCB (regret is measured on each learner's own
    # cost, so it cannot be compared between CEAC and the baselines)
    ref = brk[brk.algo == "SW-LCB"].groupby("break_prob").throughput_early.mean()
    for a in ("Greedy", "UCB", "CEAC-SW-LCB"):
        sub = brk[brk.algo == a].groupby("break_prob").throughput_early.mean()
        axs[2].plot(
            np.arange(len(rates)),
            100 * (sub / ref - 1),
            marker=MARKERS[a],
            color=COLORS[a],
            ms=5,
            lw=1.6,
            label=a,
        )
    axs[2].axhline(0, color=COLORS["SW-LCB"], lw=1.4, ls="--", label="SW-LCB")
    axs[2].set_xticks(np.arange(len(rates)))
    axs[2].set_xticklabels(labels)
    axs[2].set_xlabel("Break rate per slot")
    axs[2].set_ylabel("Throughput vs SW-LCB (%)")
    axs[2].set_title("(c) Who gains as breakpoints get frequent", fontsize=9)

    axs[0].legend(fontsize=7, frameon=False)
    axs[2].legend(fontsize=7, frameon=False)
    fig.savefig(f"{FIG}/fig11_dynamics.png")
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    fig_users()
    fig_series()
    fig_ablation()
    fig_window()
    fig_density()
    fig_pruning()
    fig_energy_tradeoff()
    if os.path.exists(f"{RES}/e9_mobility.csv"):
        fig_dynamics()
    print("figures written to", FIG)


if __name__ == "__main__":
    main()

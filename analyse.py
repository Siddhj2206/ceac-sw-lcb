"""Turn the raw experiment CSVs into the tables used by the Results section."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

RES = "results"
TAB = "results/tables"

ALGO_ORDER = ["Random", "Greedy", "UCB", "D-eps-TS", "SW-LCB", "CEAC-SW-LCB"]


def ci(series):
    s = np.asarray(series, dtype=float)
    s = s[np.isfinite(s)]
    if len(s) <= 1:
        return 0.0
    return 1.96 * s.std(ddof=1) / np.sqrt(len(s))


def agg(df, by, cols):
    """Mean and 95% CI per group, without relying on groupby.apply."""
    m_rows, e_rows = {}, {}
    for key, g in df.groupby(by):
        m_rows[key] = {c: g[c].mean() for c in cols}
        e_rows[key] = {c: ci(g[c]) for c in cols}
    m = pd.DataFrame.from_dict(m_rows, orient="index")[cols]
    e = pd.DataFrame.from_dict(e_rows, orient="index")[cols]
    return m, e


def cell(m, e, idx, c, nd=2):
    return f"{m.loc[idx, c]:.{nd}f} ± {e.loc[idx, c]:.{nd}f}"


def write(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def pct_change(num, den, col: str) -> float:
    """Percent change of ``col`` from ``den`` to ``num``."""
    return 100.0 * (num[col].mean() / den[col].mean() - 1.0)


def table_users():
    df = pd.read_csv(f"{RES}/e1_users.csv")
    cols = [
        "conflicts_early",
        "throughput_early",
        "throughput_lte_early",
        "throughput_poc_early",
        "utility_early",
        "qos_early",
        "energy_per_mbit",
        "active_end",
    ]
    m, e = agg(df, ["n_users", "algo"], cols)
    lines = [
        "# Table: main comparison versus number of users",
        "",
        (
            "Mean ± 95% confidence interval over "
            f"{df.groupby(['n_users', 'algo']).size().max()} independent runs "
            "(1600 slots each; interference, throughput, utility and QoS are measured "
            "over slots 100-500, while every node still has battery)."
        ),
        "",
        "| Users | Algorithm | Interference conflicts | Total throughput (Mbps) | LTE (Mbps) | POC (Mbps) | Mean utility | QoS satisfied | Energy (J/Mbit) | Nodes alive at end |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for idx, _ in m.iterrows():
        n, a = idx
        lines.append(
            f"| {n} | {a} | {cell(m, e, idx, 'conflicts_early')} | {cell(m, e, idx, 'throughput_early', 1)} | "
            f"{cell(m, e, idx, 'throughput_lte_early', 1)} | {cell(m, e, idx, 'throughput_poc_early', 1)} | "
            f"{cell(m, e, idx, 'utility_early', 3)} | {cell(m, e, idx, 'qos_early', 3)} | "
            f"{cell(m, e, idx, 'energy_per_mbit', 3)} | {cell(m, e, idx, 'active_end', 2)} |"
        )
    write(f"{TAB}/users.md", "\n".join(lines) + "\n")

    # relative improvement against SW-LCB and Random
    rel = []
    rel.append("# Table: relative performance (percent change, positive = better)")
    rel.append("")
    rel.append(
        "| Users | Reference | Conflicts | Throughput | Utility | QoS | Energy per Mbit |"
    )
    rel.append("|---|---|---|---|---|---|---|")
    for n in sorted(df.n_users.unique()):
        sub = df[df.n_users == n]
        base = sub[sub.algo == "SW-LCB"]
        rand = sub[sub.algo == "Random"]
        for name, b in (("vs SW-LCB", base), ("vs Random", rand)):
            row = sub[sub.algo == "CEAC-SW-LCB"]
            if row.empty:
                continue
            # fewer conflicts is better, so flip the sign for that column
            rel.append(
                f"| {n} | {name} | {-pct_change(row, b, 'conflicts_early'):+.1f}% | "
                f"{pct_change(row, b, 'throughput_early'):+.1f}% | "
                f"{pct_change(row, b, 'utility_early'):+.1f}% | "
                f"{pct_change(row, b, 'qos_early'):+.1f}% | "
                f"{-pct_change(row, b, 'energy_per_mbit'):+.1f}% |"
            )
    write(f"{TAB}/relative.md", "\n".join(rel) + "\n")
    return df


def simple_table(name, path, by, cols, title, nd=3):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    m, e = agg(df, by, cols)
    head = "| " + " | ".join(["Setting"] + [c.replace("_", " ") for c in cols]) + " |"
    lines = [f"# Table: {title}", "", head, "|" + "---|" * (len(cols) + 1)]
    for idx, _ in m.iterrows():
        key = idx if isinstance(idx, tuple) else (idx,)
        lbl = " / ".join(str(k) for k in key)
        vals = " | ".join(cell(m, e, idx, c, nd) for c in cols)
        lines.append(f"| {lbl} | {vals} |")
    write(f"{TAB}/{name}.md", "\n".join(lines) + "\n")
    return df


def table_energy():
    df = pd.read_csv(f"{RES}/e5_energy.csv")
    rows = []
    for b3 in sorted(df.beta3.unique()):
        sub = df[df.beta3 == b3]
        sw = sub[sub.algo == "SW-LCB"]
        ce = sub[sub.algo == "CEAC-SW-LCB"]
        rows.append(
            {
                "beta3": b3,
                "throughput": ce.throughput_early.mean(),
                "d_throughput": 100
                * (ce.throughput_early.mean() / sw.throughput_early.mean() - 1),
                "energy": ce.energy_per_mbit.mean(),
                "d_energy": 100
                * (ce.energy_per_mbit.mean() / sw.energy_per_mbit.mean() - 1),
                "alive": ce.active_end.mean(),
                "qos": ce.qos.mean(),
            }
        )
    out = pd.DataFrame(rows)
    lines = [
        "# Table: energy weight beta3 (throughput / lifetime trade-off)",
        "",
        "| beta3 | Throughput (Mbps) | vs SW-LCB | Energy (J/Mbit) | vs SW-LCB | Nodes alive at end | Mean QoS |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in out.iterrows():
        lines.append(
            f"| {r.beta3:.2f} | {r.throughput:.1f} | {r.d_throughput:+.1f}% | "
            f"{r.energy:.3f} | {r.d_energy:+.1f}% | {r.alive:.2f} | {r.qos:.3f} |"
        )
    write(f"{TAB}/energy.md", "\n".join(lines) + "\n")
    return out


def table_ablation():
    df = pd.read_csv(f"{RES}/e6_ablation.csv")
    order = [
        "SW-LCB",
        "CEAC-noCluster",
        "CEAC-noCapability",
        "CEAC-noEnergy",
        "CEAC-noBoundary",
        "CEAC-SW-LCB",
    ]
    cols = [
        "conflicts_early",
        "throughput_early",
        "utility_early",
        "qos_early",
        "energy_per_mbit",
        "active_end",
    ]
    m, e = agg(df, "algo", cols)
    lines = [
        "# Table: ablation of the CEAC mechanisms",
        "",
        "| Configuration | Conflicts | Throughput (Mbps) | Utility | QoS | Energy (J/Mbit) |",
        "|---|---|---|---|---|---|",
    ]
    for a in order:
        if a not in m.index:
            continue
        lines.append(
            f"| {a} | {cell(m, e, a, 'conflicts_early')} | {cell(m, e, a, 'throughput_early', 1)} | "
            f"{cell(m, e, a, 'utility_early', 3)} | {cell(m, e, a, 'qos_early', 3)} | {cell(m, e, a, 'energy_per_mbit', 3)} |"
        )
    write(f"{TAB}/ablation.md", "\n".join(lines) + "\n")
    return m


def table_clusters():
    df = pd.read_csv(f"{RES}/e7_clusters.csv")
    m = df.groupby("variant")[
        [
            "purity",
            "state_per_node",
            "state_ratio",
            "boundary",
            "messages",
            "reelect_per_slot",
            "head_cv",
            "kept_edges",
            "n_clusters",
            "members",
        ]
    ].mean()
    lines = [
        "# Table: clustering behaviour (90 ground users + 10 UAVs, 300 slots)",
        "",
        "| Variant | Clusters | Members/cluster | Boundary nodes | Intra-cluster strong-interference share | Coordination set / node | Share of N-1 | Messages / slot | Head re-elections / slot |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for v, r in m.iterrows():
        lines.append(
            f"| {v} | {r.n_clusters:.1f} | {r.members:.1f} | {r.boundary:.1f} | "
            f"{r.purity:.3f} | {r.state_per_node:.1f} | {r.state_ratio:.3f} | "
            f"{r.messages:.0f} | {r.reelect_per_slot:.2f} |"
        )
    write(f"{TAB}/clusters.md", "\n".join(lines) + "\n")

    p = pd.read_csv(f"{RES}/e7_pruning.csv")
    pm = p.groupby("tau")[["edges", "retained", "purity_after"]].mean()
    lines = [
        "# Table: affinity pruning threshold tau",
        "",
        "| tau | Affinity edges | Strong interference pairs retained | Retained pairs that are intra-cluster |",
        "|---|---|---|---|",
    ]
    for tau, r in pm.iterrows():
        lines.append(
            f"| {tau:.2f} | {r.edges:.0f} | {r.retained:.3f} | {r.purity_after:.3f} |"
        )
    write(f"{TAB}/pruning.md", "\n".join(lines) + "\n")
    return m, pm


def table_armspace():
    """Feasible arm space per node class (structural, not measured)."""
    from sim.config import Config

    c = Config(n_users=90)
    full = c.n_lte_arms() + c.n_poc_arms()  # every channel at every power
    n = c.n_uav + c.n_users
    dual = c.n_uav + c.p_dual * c.n_users
    lte = c.p_lte_only * c.n_users
    poc = (1.0 - c.p_dual - c.p_lte_only) * c.n_users
    mean = (dual * full + lte * c.n_lte_arms() + poc * c.n_poc_arms()) / n

    rows = [
        ("Dual-radio UAV / terminal", full, full),
        ("LTE-only terminal", full, c.n_lte_arms()),
        ("POC-only terminal", full, c.n_poc_arms()),
        ("Mean over the deployment mix", full, mean),
    ]
    lines = [
        "# Table: feasible arm space per node (5 LTE + 11 POC channels, two power modes)",
        "",
        "| Node class | All arms exposed | Capability-aware set | Change | Cold-start length (slots) |",
        "|---|---|---|---|---|",
    ]
    for label, a, b in rows:
        lines.append(
            f"| {label} | {a:.0f} | {b:.1f} | {100 * (b / a - 1):+.0f}% | {b:.0f} |"
        )
    lines += [
        "",
        (
            f"Baselines transmit at the fixed 20 dBm of the base paper and therefore "
            f"expose {c.n_lte + c.n_poc} arms (one power mode)."
        ),
    ]
    write(f"{TAB}/arms.md", "\n".join(lines) + "\n")


def table_dynamics():
    """Mobility and breakpoint experiments: performance and cumulative regret."""
    specs = [
        (
            "mobility",
            f"{RES}/e9_mobility.csv",
            "uav_speed_max",
            "speed (m/s)",
            "high mobility in a hotspot deployment (no breakpoints)",
        ),
        (
            "breakpoints",
            f"{RES}/e10_breakpoints.csv",
            "break_prob",
            "break rate (per slot)",
            "abrupt redeployments (breakpoints)",
        ),
    ]
    for name, path, key, label, title in specs:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        cols = ["conflicts_early", "throughput_early", "qos_early", "regret"]
        m, e = agg(df, [key, "algo"], cols)
        lines = [
            f"# Table: {title}",
            "",
            f"| {label} | Algorithm | Conflicts | Throughput (Mbps) | QoS | Cumulative regret |",
            "|---|---|---|---|---|---|",
        ]
        for idx, _ in m.iterrows():
            k, a = idx
            lines.append(
                f"| {k:g} | {a} | {cell(m, e, idx, 'conflicts_early')} | "
                f"{cell(m, e, idx, 'throughput_early', 1)} | "
                f"{cell(m, e, idx, 'qos_early', 3)} | "
                f"{cell(m, e, idx, 'regret', 0)} |"
            )
        lines += [
            "",
            (
                "Cumulative regret is the sum over slots of the myopic oracle gap "
                "(the best feasible arm against the current channel occupancy). It is "
                "measured against each learner's own cost, so it is comparable between "
                "Greedy, UCB and SW-LCB, which share the same weights, but not between "
                "CEAC-SW-LCB and them."
            ),
        ]
        write(f"{TAB}/{name}.md", "\n".join(lines) + "\n")

        # compact relative summary for the prose
        print(f"\n# {title}")
        for k in sorted(df[key].unique()):
            sub = df[df[key] == k]
            sw = sub[sub.algo == "SW-LCB"]
            parts = []
            for a in ("Greedy", "UCB", "CEAC-SW-LCB"):
                al = sub[sub.algo == a]
                dt = 100 * (al.throughput_early.mean() / sw.throughput_early.mean() - 1)
                dr = 100 * (al.regret.mean() / sw.regret.mean() - 1)
                parts.append(f"{a} thr {dt:+.1f}% regret {dr:+.1f}%")
            print(
                f"  {key}={k:g}: SW-LCB thr {sw.throughput_early.mean():.1f} "
                f"regret {sw.regret.mean():.0f} | " + "; ".join(parts)
            )


def summary(df_users):
    lines = ["# Headline numbers (auto-generated)", ""]
    for n in sorted(df_users.n_users.unique()):
        sub = df_users[df_users.n_users == n]
        ce = sub[sub.algo == "CEAC-SW-LCB"]
        sw = sub[sub.algo == "SW-LCB"]
        rd = sub[sub.algo == "Random"]
        ts = sub[sub.algo == "D-eps-TS"]
        lines.append(
            f"- n={n}: CEAC throughput {ce.throughput_early.mean():.1f} Mbps "
            f"({100 * (ce.throughput_early.mean() / sw.throughput_early.mean() - 1):+.1f}% vs SW-LCB, "
            f"{100 * (ce.throughput_early.mean() / rd.throughput_early.mean() - 1):+.1f}% vs Random, "
            f"{100 * (ce.throughput_early.mean() / ts.throughput_early.mean() - 1):+.1f}% vs D-eps-TS); "
            f"conflicts {ce.conflicts_early.mean():.2f} vs {sw.conflicts_early.mean():.2f}; "
            f"energy {ce.energy_per_mbit.mean():.3f} vs {sw.energy_per_mbit.mean():.3f} J/Mbit."
        )
    write(f"{TAB}/headline.md", "\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    os.makedirs(TAB, exist_ok=True)
    df = table_users()
    table_energy()
    table_ablation()
    table_clusters()
    table_armspace()
    simple_table(
        "density",
        f"{RES}/e2_density.csv",
        ["n_users", "algo"],
        ["conflicts_early", "throughput_early", "utility_early", "qos_early"],
        "density scaling",
    )
    simple_table(
        "window",
        f"{RES}/e4_window.csv",
        ["alpha", "algo"],
        ["conflicts_early", "throughput_early", "utility_early"],
        "sliding-window parameter alpha",
    )
    simple_table(
        "speed",
        f"{RES}/e3_speed.csv",
        ["uav_speed_max", "algo"],
        ["conflicts_early", "throughput_early", "utility_early"],
        "UAV speed sensitivity",
    )
    table_dynamics()
    summary(df)
    print(f"\ntables written to {TAB}/")


if __name__ == "__main__":
    main()

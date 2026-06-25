"""
plot_vss.py — standalone plotting for the cumulative VSS results CSV.

Reads the ``vss_results.csv`` produced by ``vss.py`` and renders, in the output
directory:

  * ``vss_eev_rp_step{step}.png`` — one per step: EEV (deterministic / EV policy)
    vs RP (stochastic / SAA policy) grouped bars per waiting list; the gap
    between the bar pair *is* the VSS, annotated with VSS and VSS%.
  * ``vss_summary.png`` — VSS (= EEV - RP) grouped by waiting list, one bar per
    step, so the value of the stochastic solution is comparable across instances
    and model steps at a glance.

This is the canonical home for the VSS plotting logic; ``vss.py`` imports
``plot_vss_results`` from here so a solve run auto-plots with the same code.

Usage
-----
# Plot output/vss_results.csv into output/:
python src/plot_vss.py

# Custom CSV / output directory:
python src/plot_vss.py --csv output/vss_results.csv --output-dir output
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd

# ── Plot palette (deterministic EV vs stochastic SAA) ────────────────────────
_EEV_COLOR = "#f28e2b"   # EEV — deterministic / expected-value policy
_RP_COLOR  = "#4e79a7"   # RP  — stochastic SAA policy
_VSS_COLOR = "#59a14f"   # VSS — value of the stochastic solution
_STEP_LABELS = {1: "Timing only", 2: "Sequencing + Timing", 3: "Full MILP"}


def plot_vss_results(csv_path: str, output_dir: str) -> None:
    """Plot the cumulative VSS results CSV.

    Produces, in *output_dir*:
      * ``vss_eev_rp_step{step}.png`` — one per step: EEV (deterministic) vs RP
        (stochastic) grouped bars per waiting list; the bar-pair gap *is* the VSS.
      * ``vss_summary.png`` — VSS (= EEV - RP) grouped by waiting list, one bar
        per step, so the value of the stochastic solution is comparable across
        instances and model steps at a glance.
    """
    import matplotlib
    matplotlib.use("Agg")            # headless: write PNG without a display
    import matplotlib.pyplot as plt
    import numpy as np

    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"{csv_path} is empty — nothing to plot.")
        return

    steps = sorted(df["step"].unique())
    bar_w = 0.38

    # ── Per-step EEV vs RP grouped bars ───────────────────────────────────────
    for step in steps:
        df_s = df[df["step"] == step].sort_values("workload")
        wls = df_s["workload"].tolist()
        x = np.arange(len(wls))

        fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(wls) + 3), 5))
        b_eev = ax.bar(x - bar_w / 2, df_s["eev"], bar_w,
                       color=_EEV_COLOR, label="EEV — deterministic (EV)")
        b_rp = ax.bar(x + bar_w / 2, df_s["rp"], bar_w,
                      color=_RP_COLOR, label="RP — stochastic (SAA)")
        ax.bar_label(b_eev, fmt="%.0f", padding=2, fontsize=8)
        ax.bar_label(b_rp, fmt="%.0f", padding=2, fontsize=8)

        # Annotate VSS (and % of EEV) above each workload pair.
        top = max(df_s["eev"].max(), df_s["rp"].max())
        for xi, (_, row) in zip(x, df_s.iterrows()):
            ax.annotate(f"VSS={row['vss']:.0f}\n({row['vss_pct']:.0f}%)",
                        xy=(xi, top * 1.04), ha="center", va="bottom",
                        fontsize=8, color=_VSS_COLOR, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([f"wl{w}" for w in wls])
        ax.set_xlabel("Waiting list")
        ax.set_ylabel("Out-of-sample objective")
        ax.set_ylim(0, top * 1.18)
        ax.set_title(f"VSS — Step {step} ({_STEP_LABELS.get(step, step)})\n"
                     f"EEV vs RP out-of-sample; gap = VSS")
        # Legend outside the axes so it never overlaps bars or VSS annotations.
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
        ax.grid(axis="y", color="lightgray", linewidth=0.6)
        ax.set_axisbelow(True)
        fig.tight_layout()

        plot_path = os.path.join(output_dir, f"vss_eev_rp_step{step}.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"VSS plot saved to: {plot_path}")

    # ── Summary: VSS by workload, one bar per step ────────────────────────────
    wls = sorted(df["workload"].unique())
    x = np.arange(len(wls))
    n_steps = len(steps)
    width = 0.8 / n_steps

    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(wls) + 3), 5))
    for i, step in enumerate(steps):
        df_s = df[df["step"] == step].set_index("workload")
        vals = [float(df_s.loc[w, "vss"]) if w in df_s.index else 0.0 for w in wls]
        dx = (i - (n_steps - 1) / 2) * width
        bars = ax.bar(x + dx, vals, width,
                      label=f"Step {step} ({_STEP_LABELS.get(step, step)})")
        ax.bar_label(bars, fmt="%.0f", padding=2, fontsize=7)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"wl{w}" for w in wls])
    ax.set_xlabel("Waiting list")
    ax.set_ylabel("VSS = EEV − RP")
    ax.set_title("Value of the Stochastic Solution by waiting list and step\n"
                 "(negative bars: SAA not solved to optimality — see saa_gap)")
    ax.legend(loc="best", frameon=False)
    ax.grid(axis="y", color="lightgray", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()

    summary_path = os.path.join(output_dir, "vss_summary.png")
    fig.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"VSS summary plot saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot the cumulative VSS results CSV produced by vss.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", default=None,
                        help="Path to vss_results.csv "
                             "(default: <output-dir>/vss_results.csv)")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for the PNG plots (default: ../output)")
    args = parser.parse_args()

    root = os.path.join(os.path.dirname(__file__), "..")
    output_dir = args.output_dir or os.path.join(root, "output")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = args.csv or os.path.join(output_dir, "vss_results.csv")

    if not os.path.exists(csv_path):
        print(f"No results to plot: {csv_path} does not exist. "
              f"Run vss.py first.")
        return

    plot_vss_results(csv_path, output_dir)


if __name__ == "__main__":
    main()

"""
plot_saa_convergence.py — SAA convergence plots by sampling method.

Reads the per-method SAA replication CSVs produced for a single waiting list,
``saa_wl{wl}_step{step}{method}.csv`` (columns: ``method, N, replication,
obj_out_of_sample``), and renders, in the output directory, one plot per step:

  * ``saa_convergence_step{step}.png`` — out-of-sample evaluation value against
    the number of training scenarios N, with one line per sampling method
    (random, lhs, is). The line is the mean over replications; a light shaded
    band shows ±1 standard deviation.

Usage
-----
# Plot output/saa_wl4_step{1,2,3}{method}.csv into output/:
python src/plot_saa_convergence.py

# Custom waiting list / output directory:
python src/plot_saa_convergence.py --wl 4 --output-dir output
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

# ── Plot palette (one colour per sampling method) ────────────────────────────
_METHODS = ["random", "lhs", "is"]
_METHOD_COLORS = {
    "random": "#4e79a7",
    "lhs":    "#59a14f",
    "is":     "#e15759",
}
_METHOD_LABELS = {
    "random": "Random",
    "lhs":    "Latin Hypercube (LHS)",
    "is":     "Importance Sampling (IS)",
}
_STEPS = [1, 2, 3]
_STEP_LABELS = {1: "Timing only", 2: "Sequencing + Timing", 3: "Full MILP"}


def plot_saa_convergence(wl: int, output_dir: str) -> None:
    """Plot SAA out-of-sample convergence per step for waiting list *wl*.

    For each step a separate figure ``saa_convergence_step{step}.png`` is written
    to *output_dir*, comparing the sampling methods. Each line is the mean
    out-of-sample objective over replications as a function of N, surrounded by a
    light ±1 std band.
    """
    import matplotlib
    matplotlib.use("Agg")            # headless: write PNG without a display
    import matplotlib.pyplot as plt

    for step in _STEPS:
        fig, ax = plt.subplots(figsize=(7, 5))
        plotted = False

        for method in _METHODS:
            csv_path = os.path.join(
                output_dir, f"saa_wl{wl}_step{step}{method}.csv")
            if not os.path.exists(csv_path):
                print(f"  skipping (not found): {csv_path}")
                continue

            df = pd.read_csv(csv_path)
            if df.empty:
                print(f"  skipping (empty): {csv_path}")
                continue

            stats = (df.groupby("N")["obj_out_of_sample"]
                       .agg(["mean", "std"])
                       .sort_index())
            stats["std"] = stats["std"].fillna(0.0)  # std is NaN for 1 replication

            color = _METHOD_COLORS[method]
            ax.plot(stats.index, stats["mean"], marker="o", color=color,
                    label=_METHOD_LABELS[method])
            ax.fill_between(stats.index,
                            stats["mean"] - stats["std"],
                            stats["mean"] + stats["std"],
                            color=color, alpha=0.18)
            plotted = True

        if not plotted:
            plt.close(fig)
            print(f"No data for step {step} — skipping plot.")
            continue

        ax.set_xlabel("Number of training scenarios $N$")
        ax.set_ylabel("Out-of-sample evaluation value")
        ax.set_title(f"SAA convergence — Step {step} "
                     f"({_STEP_LABELS.get(step, step)})\n"
                     f"mean over replications; band = $\\pm 1$ std")
        ax.legend(loc="best", frameon=False)
        ax.grid(color="lightgray", linewidth=0.6)
        ax.set_axisbelow(True)
        fig.tight_layout()

        plot_path = os.path.join(output_dir, f"saa_convergence_step{step}.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"SAA convergence plot saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot SAA out-of-sample convergence by sampling method.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--wl", type=int, default=4,
                        help="Waiting list id used in the CSV file names.")
    parser.add_argument("--output-dir", default=None,
                        help="Directory holding the CSVs / for the PNGs "
                             "(default: ../output)")
    args = parser.parse_args()

    root = os.path.join(os.path.dirname(__file__), "..")
    output_dir = args.output_dir or os.path.join(root, "output")
    os.makedirs(output_dir, exist_ok=True)

    plot_saa_convergence(args.wl, output_dir)


if __name__ == "__main__":
    main()

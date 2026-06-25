"""
plot_blueprint_comparison.py — bar charts of the blueprint-vs-baseline
operational-cost comparison, one PNG per step.

Reads ``output/blueprint_comparison.csv`` (written by ``apply_blueprint.py``)
and writes one ``blueprint_comparison_step{1,2,3}.png`` per step present in the
file. No MILP solve happens here — this is a pure plotting step.

For each step: x-axis = comparison seeds; two side-by-side stacked bars per seed
— baseline (saturated colours) and blueprint (lighter tints) — each stacked by
the out-of-sample objective parts W/I/O/D. A seed/mode whose solve was
infeasible (blank cost columns) is drawn as an empty slot annotated
"infeasible".

Usage
-----
    python src/plot_blueprint_comparison.py
    python src/plot_blueprint_comparison.py --csv output/blueprint_comparison.csv \
                                            --output-dir report/figures
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Objective-part colours, kept consistent with the other report figures
# (mirrors ``_PART_COLOR`` / ``_PART_COLOR_OUT`` in ``main.py``).
_PART_COLOR = {            # baseline (saturated)
    "W": "#4e79a7",   # waiting
    "I": "#f28e2b",   # idle
    "O": "#59a14f",   # overtime
    "D": "#e15759",   # specialty mismatch
}
_PART_COLOR_BP = {         # blueprint (lighter tints)
    "W": "#a7c2de",
    "I": "#fbd2a3",
    "O": "#aed6ab",
    "D": "#f3aeaf",
}
_PART_LABEL = {
    "W": "Waiting (W)",
    "I": "Idle (I)",
    "O": "Overtime (O)",
    "D": "Specialty (D)",
}

_STEP_LABELS = {1: "Timing only", 2: "Sequencing + Timing", 3: "Full MILP"}
_PARTS = ["W", "I", "O", "D"]


def plot_blueprint_comparison(csv_path: str, output_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")            # headless: write PNG without a display
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    df = pd.read_csv(csv_path)
    bar_w = 0.38

    for step in (1, 2, 3):
        df_s = df[df["step"] == step]
        if df_s.empty:
            continue

        seeds = sorted(df_s["comparison_seed"].unique())
        x = np.arange(len(seeds))

        fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(seeds) + 3), 3.6))

        for mode, dx, palette in [
            ("baseline",  -bar_w / 2, _PART_COLOR),
            ("blueprint", +bar_w / 2, _PART_COLOR_BP),
        ]:
            df_m = df_s[df_s["mode"] == mode].set_index("comparison_seed")
            bottom = np.zeros(len(seeds))
            for part in _PARTS:
                col = f"oos_{part}"
                vals = np.array([
                    float(df_m.loc[sd, col])
                    if sd in df_m.index and pd.notna(df_m.loc[sd, col])
                    else 0.0
                    for sd in seeds
                ])
                ax.bar(x + dx, vals, bar_w, bottom=bottom,
                       color=palette[part], edgecolor="white", linewidth=0.5)
                bottom += vals

            # Annotate infeasible / missing solves (blank cost columns).
            for i, sd in enumerate(seeds):
                feasible = (
                    sd in df_m.index and pd.notna(df_m.loc[sd, "oos_total"])
                )
                if not feasible:
                    ax.text(x[i] + dx, 0.5, "infeasible", rotation=90,
                            ha="center", va="bottom", fontsize=7, color="0.3")

        handles = (
            [Patch(facecolor=_PART_COLOR[p],
                   label=f"{_PART_LABEL[p]} — baseline") for p in _PARTS]
            + [Patch(facecolor=_PART_COLOR_BP[p],
                     label=f"{_PART_LABEL[p]} — blueprint") for p in _PARTS]
        )
        ax.legend(handles=handles, title="Objective part",
                  loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)

        ax.set_xticks(x)
        ax.set_xticklabels([f"s{sd}" for sd in seeds])
        ax.set_xlabel("Comparison seed")
        ax.set_ylabel("Out-of-sample cost")
        ax.set_title(
            f"Blueprint vs baseline — Step {step} ({_STEP_LABELS[step]})\n"
            f"left bar = baseline (free), right bar = blueprint (restricted)"
        )

        fig.tight_layout()
        out_path = os.path.join(output_dir, f"blueprint_comparison_step{step}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out_path}")


def main() -> None:
    root = os.path.join(os.path.dirname(__file__), "..")
    default_out = os.path.join(root, "report", "figures")

    parser = argparse.ArgumentParser(
        description="Plot the blueprint-vs-baseline comparison PNGs from an "
                    "existing CSV (no MILP solve).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        default=os.path.join(root, "output", "blueprint_comparison.csv"),
        help="Path to the blueprint comparison CSV",
    )
    parser.add_argument(
        "--output-dir", default=default_out,
        help="Directory to write the PNGs into",
    )
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        sys.exit(f"ERROR: CSV not found: {args.csv}\n"
                 f"Run `python src/apply_blueprint.py` first to generate it.")

    os.makedirs(args.output_dir, exist_ok=True)
    plot_blueprint_comparison(args.csv, args.output_dir)


if __name__ == "__main__":
    main()

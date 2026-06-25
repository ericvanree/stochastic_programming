"""
plot_breakdown.py — regenerate the objective-breakdown PNGs from an existing CSV.

Runs only the plotting step (no MILP solve): reads
``output/objective_breakdown.csv`` (written by main.py) and writes one
``objective_breakdown_step{1,2,3}.png`` per step present in the file.

Usage
-----
    python src/plot_breakdown.py
    python src/plot_breakdown.py --csv output/objective_breakdown.csv \
                                 --output-dir output
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

# Importing main only defines its functions; the solve is guarded by
# `if __name__ == "__main__"`, so no model is built or optimised here.
from main import plot_objective_breakdown


def main() -> None:
    root = os.path.join(os.path.dirname(__file__), "..")
    default_out = os.path.join(root, "output")

    parser = argparse.ArgumentParser(
        description="Regenerate objective-breakdown PNGs from an existing CSV "
                    "(no MILP solve).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv", default=None,
        help="Path to the breakdown CSV (default: <output-dir>/objective_breakdown.csv)",
    )
    parser.add_argument(
        "--output-dir", default=default_out,
        help="Directory to write the PNGs into",
    )
    args = parser.parse_args()

    csv_path = args.csv or os.path.join(args.output_dir, "objective_breakdown.csv")
    if not os.path.exists(csv_path):
        sys.exit(f"ERROR: CSV not found: {csv_path}\n"
                 f"Run `python src/main.py` first to generate it.")

    os.makedirs(args.output_dir, exist_ok=True)
    plot_objective_breakdown(csv_path, args.output_dir)


if __name__ == "__main__":
    main()

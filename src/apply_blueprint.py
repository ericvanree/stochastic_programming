"""
apply_blueprint.py — use a fixed blueprint as a restriction and compare.

Idea
----
A blueprint (decided beforehand) caps how many patients of each surgery-type
subgroup (S / M / L) may be placed in each session: ``N[g, h]``.  This script
takes such a blueprint as a *given* (read from output/blueprint.csv) and measures
what it costs to obey it, by solving the regular step 1 → 2 → 3 scheduling chain
on a comparison instance **twice**:

  * baseline  — the schedule is free to assign patients to sessions however it
                likes (no blueprint);
  * blueprint — the same chain, but step 3's assignment is restricted by the
                blueprint quotas  ``sum_{p in g} Y[p, h] <= N[g, h]``.

Comparison instance
-------------------
The two-stage scheduling model uses three identical sessions, exactly like the
blueprint's stage-2 structure.  We therefore build a dedicated comparison set by
randomly sampling ``--n-sessions`` (default 3) documented ORT sessions from
ort_patient_data.csv and writing them to input/sample_step5_comparison.csv.  The
same file is reused for the baseline and blueprint solves so they differ only in
the blueprint restriction.

Usage
-----
# 1) produce a blueprint first (writes output/blueprint.csv):
python src/blueprint.py --n-realizations 3 --n-scenarios 5 --seed 42 --method lhs

# 2) apply it and compare on a 3-session ORT instance:
python src/apply_blueprint.py --seed 42 --n-train 20 --method lhs --time-limit 600

Outputs (in output/)
--------------------
  blueprint_comparison.csv — cumulative; one row per (comparison_seed, mode, step)
                             with in-sample objective, out-of-sample cost parts and
                             the realised per-session S/M/L composition.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import gurobipy as gp

from sampling import generate_scenarios
from blueprint import cluster_surgery_types, _G, _H
from saa import load_data, solve_policy_chain
from main import evaluate_out_of_sample_parts

_ROOT = os.path.join(os.path.dirname(__file__), "..")

# Steps solved/evaluated together (1 → 2 → 3, each warm-started from the last).
_STEPS = (1, 2, 3)

# Column order for the cumulative comparison CSV.
_COMP_COLS = (
    ["comparison_seed", "n_sessions", "n_train", "n_prime", "method", "is_k",
     "mode", "step", "status", "mip_gap_pct", "in_sample_obj",
     "oos_total", "oos_W", "oos_I", "oos_O", "oos_D", "n_patients"]
    + [f"comp_{g}_{h}" for g in _G for h in _H]      # realised composition
    + [f"N_{g}_{h}" for g in _G for h in _H]          # the blueprint quota used
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. COMPARISON SET + BLUEPRINT INPUT
# ─────────────────────────────────────────────────────────────────────────────

def build_comparison_set(seed: int, n_sessions: int, resample: bool) -> str:
    """Sample `n_sessions` ORT sessions from ort_patient_data.csv → comparison CSV.

    Collects all patients in the chosen sessions and writes them to
    input/sample_step5_comparison.csv (same column schema as sample_wl*.csv, so it
    loads through saa.load_data unchanged).  Reuses the existing file unless
    `resample` is set, so the baseline and blueprint solves share one instance and
    runs reproduce.
    """
    out_path = os.path.join(_ROOT, "input", "sample_step5_comparison.csv")
    if os.path.exists(out_path) and not resample:
        print(f"[1] Reusing existing comparison set: {out_path}  (use --resample to redraw)")
        return out_path

    src = os.path.join(_ROOT, "input", "ort_patient_data.csv")
    df = pd.read_csv(src)
    available = df["Session ID"].unique().tolist()
    if len(available) < n_sessions:
        raise ValueError(f"Only {len(available)} sessions available; need {n_sessions}.")

    rng = np.random.default_rng(seed)
    chosen = rng.choice(available, size=n_sessions, replace=False)
    comp = df[df["Session ID"].isin(chosen)].copy().reset_index(drop=True)
    comp.to_csv(out_path, index=False)
    print(f"[1] Sampled {n_sessions} ORT sessions {list(chosen)} -> "
          f"{len(comp)} patients written to {out_path}")
    return out_path


def read_blueprint_csv(path: str) -> dict:
    """Read output/blueprint.csv (group, session, quota) → {(g, h): quota}."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Blueprint file not found: {path}\n"
            f"Run blueprint.py first (it writes output/blueprint.csv)."
        )
    N: dict[tuple[str, int], int] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            N[(row["group"], int(row["session"]))] = int(row["quota"])
    missing = [(g, h) for g in _G for h in _H if (g, h) not in N]
    if missing:
        raise ValueError(f"Blueprint {path} is missing quotas for {missing}.")
    return N


# ─────────────────────────────────────────────────────────────────────────────
# 2. SOLVER CONFIG + EVALUATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _make_configure(time_limit: int, mip_gap: float):
    """Per-step solver-config callback for solve_policy_chain (mirrors vss.py)."""
    def configure(m, step: int, with_mip_focus: bool = False):
        m.setParam("OutputFlag", 0)
        m.setParam("TimeLimit", time_limit)
        if with_mip_focus:
            m.setParam("MIPFocus", 1)
        if step == 3:
            m.setParam("MIPGap", mip_gap)
            m.setParam("Presolve", 2)
            m.setParam("Cuts", 2)
            m.setParam("Heuristics", 0.3)
    return configure


def _composition(m, P, H, group_of) -> dict:
    """Realised per-session S/M/L counts of a solved model's assignment Y."""
    comp = {(g, h): 0 for g in _G for h in H}
    if m.SolCount == 0:
        return comp
    for p in P:
        for h in H:
            if m._Y[p, h].X > 0.5:
                comp[group_of[p], h] += 1
    return comp


# ─────────────────────────────────────────────────────────────────────────────
# 3. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def append_comparison_csv(path: str, rows: list[dict]) -> None:
    """Upsert comparison rows into one cumulative CSV, keyed by (seed, mode, step)."""
    df_new = pd.DataFrame(rows, columns=_COMP_COLS)
    if os.path.exists(path):
        try:
            df_all = pd.concat([pd.read_csv(path), df_new], ignore_index=True)
        except Exception as exc:
            print(f"Could not read existing {path} ({exc}); starting fresh.")
            df_all = df_new
    else:
        df_all = df_new
    df_all = (
        df_all
        .drop_duplicates(subset=["comparison_seed", "mode", "step"], keep="last")
        .sort_values(["comparison_seed", "mode", "step"])
        .reset_index(drop=True)
    )
    df_all.to_csv(path, index=False)
    print(f"\nComparison results written to: {path}  ({len(df_all)} rows total)")


def _rows_for_mode(mode, models, comp_by_step, N, args,
                   d_eval, S_eval, pi_eval, P, H, SPECS):
    """Build one result row per step for a given mode (baseline | blueprint)."""
    rows = []
    for step in _STEPS:
        m = models[step]
        solved = m.SolCount > 0
        parts = (evaluate_out_of_sample_parts(m, P, H, SPECS, d_eval, S_eval, pi_eval)
                 if solved else None)
        comp = comp_by_step[step]
        row = {
            "comparison_seed": args.seed, "n_sessions": args.n_sessions,
            "n_train": args.n_train, "n_prime": args.n_prime,
            "method": args.method, "is_k": args.is_k,
            "mode": mode, "step": step, "status": m.Status,
            "mip_gap_pct": (m.MIPGap * 100 if solved else float("nan")),
            "in_sample_obj": (m.ObjVal if solved else float("nan")),
            "oos_total": (parts["total"] if parts else float("nan")),
            "oos_W": (parts["W"] if parts else float("nan")),
            "oos_I": (parts["I"] if parts else float("nan")),
            "oos_O": (parts["O"] if parts else float("nan")),
            "oos_D": (parts["D"] if parts else float("nan")),
            "n_patients": len(P),
        }
        for g in _G:
            for h in _H:
                row[f"comp_{g}_{h}"] = comp.get((g, h), 0)
                row[f"N_{g}_{h}"] = N[(g, h)]
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Apply a fixed blueprint as a restriction and compare the "
                    "step 1→2→3 schedule with vs. without it.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for sampling the comparison set.")
    parser.add_argument("--n-sessions", type=int, default=3,
                        help="ORT sessions in the comparison set (must match the "
                             "blueprint's 3 sessions).")
    parser.add_argument("--cluster-seed", type=int, default=42,
                        help="Fixed k-means seed (must match the blueprint run).")
    parser.add_argument("--blueprint-csv", default=None,
                        help="Blueprint to apply (default: output/blueprint.csv).")
    parser.add_argument("--n-train", type=int, default=20,
                        help="Training scenarios for the scheduling solve.")
    parser.add_argument("--n-prime", type=int, default=2000,
                        help="Out-of-sample evaluation set size.")
    parser.add_argument("--method", default="lhs", choices=["random", "lhs", "is"])
    parser.add_argument("--is-k", type=float, default=0.5)
    parser.add_argument("--time-limit", type=int, default=300)
    parser.add_argument("--mip-gap", type=float, default=0.05)
    parser.add_argument("--resample", action="store_true",
                        help="Redraw the comparison set even if it already exists.")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(_ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)
    blueprint_csv = args.blueprint_csv or os.path.join(output_dir, "blueprint.csv")

    print(f"\n=== Apply blueprint as restriction  |  comparison seed={args.seed} ===")

    # ── 1. Comparison set + blueprint ─────────────────────────────────────────
    comp_path = build_comparison_set(args.seed, args.n_sessions, args.resample)
    (df, P, H, P0, SPECS, q_pq,
     session_ids, session_sequences, patient_to_h) = load_data(path=comp_path)
    print(f"    Patients: {len(P)}, Sessions: {len(H)}, Specialties: {SPECS}")

    if len(H) != len(_H):
        raise ValueError(
            f"Comparison set has {len(H)} sessions but the blueprint has {len(_H)}. "
            f"Use --n-sessions {len(_H)} (and a session sample that yields exactly "
            f"{len(_H)} distinct sessions)."
        )

    N = read_blueprint_csv(blueprint_csv)
    print(f"[2] Blueprint quotas from {blueprint_csv}: "
          + ", ".join(f"N[{g},{h}]={N[(g, h)]}" for g in _G for h in _H))

    # ── 2. Map comparison patients to S/M/L (fixed clustering) ────────────────
    type_to_group, _ = cluster_surgery_types(k=3, random_state=args.cluster_seed)
    stype_of = dict(zip(df["Patient ID"], df["Surgery type"].astype(int)))
    group_of = {p: type_to_group.get(stype_of[p], "M") for p in P}

    # ── 3. Scenarios: training set + shared out-of-sample eval set ────────────
    d_tr, S_tr, pi_tr = generate_scenarios(
        df, n_scenarios=args.n_train, method=args.method, seed=args.seed, is_k=args.is_k,
    )
    d_eval, S_eval, pi_eval = generate_scenarios(
        df, n_scenarios=args.n_prime, method="random",
        seed=args.seed + 99999, is_k=args.is_k,
    )

    configure = _make_configure(args.time_limit, args.mip_gap)

    chain_args = (P, P0, H, SPECS, q_pq, session_sequences, patient_to_h,
                  3, d_tr, S_tr, pi_tr, configure)

    # ── 4a. Baseline: free assignment ─────────────────────────────────────────
    print("\n[3] Solving baseline chain (no blueprint restriction) ...")
    base_models = solve_policy_chain(*chain_args, name_prefix="base", return_all=True)

    # ── 4b. Blueprint: step 3 restricted by the quotas ────────────────────────
    def add_quota(m3):
        for g in _G:
            for h in _H:
                pg = [p for p in P if group_of[p] == g]
                if pg:
                    m3.addConstr(
                        gp.quicksum(m3._Y[p, h] for p in pg) <= N[(g, h)],
                        name=f"blueprint_{g}_{h}",
                    )

    print("[4] Solving blueprint-restricted chain (quotas on step 3) ...")
    bp_models = solve_policy_chain(*chain_args, name_prefix="bp", return_all=True,
                                   step3_post_build=add_quota)

    # ── 5. Compositions + result rows ─────────────────────────────────────────
    base_comp = {s: _composition(base_models[s], P, H, group_of) for s in _STEPS}
    bp_comp   = {s: _composition(bp_models[s],   P, H, group_of) for s in _STEPS}

    rows = (
        _rows_for_mode("baseline", base_models, base_comp, N, args,
                       d_eval, S_eval, pi_eval, P, H, SPECS)
        + _rows_for_mode("blueprint", bp_models, bp_comp, N, args,
                         d_eval, S_eval, pi_eval, P, H, SPECS)
    )
    append_comparison_csv(os.path.join(output_dir, "blueprint_comparison.csv"), rows)

    # ── 6. Console summary ────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  Baseline vs blueprint  (out-of-sample total cost per step)")
    print(f"{'=' * 70}")
    print(f"  {'step':>4}  {'baseline':>12}  {'blueprint':>12}  {'cost of restriction':>22}")
    by = {(r["mode"], r["step"]): r for r in rows}
    for step in _STEPS:
        b = by[("baseline", step)]["oos_total"]
        q = by[("blueprint", step)]["oos_total"]
        delta = q - b if (b == b and q == q) else float("nan")
        print(f"  {step:>4}  {b:12.3f}  {q:12.3f}  {delta:22.3f}")

    print("\n  Step-3 realised composition (count per session)  vs  quota N[g,h]:")
    for g in _G:
        base_line = " ".join(f"{base_comp[3][(g, h)]:>2}" for h in _H)
        bp_line   = " ".join(f"{bp_comp[3][(g, h)]:>2}" for h in _H)
        quota     = " ".join(f"{N[(g, h)]:>2}" for h in _H)
        print(f"    {g}:  baseline [{base_line}]   blueprint [{bp_line}]   quota [{quota}]")


if __name__ == "__main__":
    main()

"""
saa.py — SAA convergence analysis CLI for stochastic OR scheduling.

Runs the SAA procedure for a given model step, waiting-list, and scenario
range.  Each replication is evaluated analytically (forward simulation of the
fixed schedule) on a fixed out-of-sample evaluation set.  Results are written
row-by-row to a CSV file; a convergence plot is produced at the end.

Usage
-----
python src/saa.py --step 2 --wl 4 --n-start 10 --n-step 5 --n-max 30 --n-prime 1000 --m-reps 10 --start-seed 10

Outputs (in --output-dir, default "output/")
--------------------------------------------
  saa_wl{wl}_step{step}.csv                -- per-replication results
  saa_convergence_wl{wl}_step{step}.html   -- interactive convergence plot
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from sampling import generate_scenarios
from main import build_model

# ── Constants (must match main.py) ───────────────────────────────────────────
_T_START = 480      # 08:00 in minutes from midnight
_T_CLOSE = 960      # 16:00 in minutes from midnight
_C       = 10       # changeover / cleaning time (minutes)
_M_BIG   = 5_000    # big-M constant
_BETA    = {"W": 0.6, "I": 0.2, "O": 0.2, "D": 100.0}

_ALL_METHODS = ["random", "lhs", "is"]

# ── Method display names ──────────────────────────────────────────────────────
_METHOD_LABEL = {
    "random": "Random Sampling (RS)",
    "lhs":    "Latin Hypercube Sampling (LHS)",
    "is":     "Importance Sampling (IS)",
}
_METHOD_COLOR = {
    "random": "#4e79a7",
    "lhs":    "#f28e2b",
    "is":     "#59a14f",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def _parse_pos(label: str) -> int:
    """'S189-P3' → 3"""
    return int(re.search(r'P(\d+)', label).group(1))


def load_data(wl: int):
    """
    Load patient data for waiting-list *wl* (4 | 7 | 10 | 13).

    Returns
    -------
    df, P, H, P0, SPECS, q_pq, session_ids, session_sequences, patient_to_h
    """
    # Resolve path relative to this script so it works regardless of cwd
    root = os.path.join(os.path.dirname(__file__), "..")
    data_file = os.path.join(root, "input", f"sample_wl{wl}.csv")
    df = pd.read_csv(data_file)

    P = df["Patient ID"].tolist()
    session_ids = sorted(df["Session ID"].unique().tolist())
    H = list(range(len(session_ids)))
    h_of = {sid: h for h, sid in enumerate(session_ids)}
    P0 = [0] + P
    SPECS = sorted(df["Specialty"].unique().tolist())

    specialty_of = {row["Patient ID"]: row["Specialty"] for _, row in df.iterrows()}
    q_pq = {(p, q): int(specialty_of[p] == q) for p in P for q in SPECS}

    patient_to_h = {
        row["Patient ID"]: h_of[row["Session ID"]] for _, row in df.iterrows()
    }

    session_sequences: dict[int, list] = {h: [] for h in H}
    for _, row in df.iterrows():
        h   = h_of[row["Session ID"]]
        pos = _parse_pos(row["Session-sequence position"])
        session_sequences[h].append((pos, row["Patient ID"]))
    for h in H:
        session_sequences[h].sort()
        session_sequences[h] = [p for _, p in session_sequences[h]]

    return df, P, H, P0, SPECS, q_pq, session_ids, session_sequences, patient_to_h


# ─────────────────────────────────────────────────────────────────────────────
# 2. FIXED-VARIABLE BUILDERS  (for steps 1 and 2)
# ─────────────────────────────────────────────────────────────────────────────

def _make_fixed_X(
    P0: list, H: list, session_sequences: dict[int, list]
) -> dict:
    """Build fixed_X from CSV sequences (used in Step 1)."""
    P = [p for p in P0 if p != 0]
    fixed_X = {(i, j, h): 0 for i in P0 for j in P0 for h in H if i != j}
    for h in H:
        seq = session_sequences[h]
        if not seq:
            continue
        fixed_X[0, seq[0], h] = 1
        for k in range(len(seq) - 1):
            fixed_X[seq[k], seq[k + 1], h] = 1
        fixed_X[seq[-1], 0, h] = 1
    return fixed_X


def _make_fixed_Y(P: list, H: list, patient_to_h: dict) -> dict:
    """Build fixed_Y from CSV assignment (used in Steps 1 and 2)."""
    fixed_Y = {(p, h): 0 for p in P for h in H}
    for p in P:
        fixed_Y[p, patient_to_h[p]] = 1
    return fixed_Y


# ─────────────────────────────────────────────────────────────────────────────
# 3. ANALYTIC FORWARD SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def simulate_schedule(
    X_vals: dict,
    Y_vals: dict,
    A_vals: dict,
    D_vals: dict,
    d_eval: dict,
    S_eval: list,
    pi_eval: dict,
    P: list,
    H: list,
    SPECS: list,
    beta: dict = _BETA,
    t_start: int = _T_START,
    t_close: int = _T_CLOSE,
    c: int = _C,
) -> float:
    """
    Analytically evaluate a fixed first-stage solution on out-of-sample scenarios.

    For each session the patient sequence is reconstructed by following the X
    arcs from the depot node (0).  Start times are propagated as:

        S[p, s] = max(A[p],  prev_finish)

    where prev_finish = S[prev, s] + d[prev, s] + c.  Waiting time, idle time,
    and overtime follow directly.  The specialty-mismatch penalty (D) is
    scenario-independent and added once.

    Parameters
    ----------
    X_vals  : {(i, j, h): float}  — sequencing arcs (0/1 values)
    Y_vals  : {(p, h): float}     — assignment (0/1 values)
    A_vals  : {p: float}          — appointment times (minutes)
    D_vals  : {(h, q): float}     — specialty violation counts
    d_eval  : {(p, s): float}     — eval scenario durations
    S_eval  : list                — eval scenario indices
    pi_eval : {s: float}          — eval scenario probabilities
    P, H, SPECS                   — patient/session/specialty sets

    Returns
    -------
    float — weighted out-of-sample objective value
    """
    # Reconstruct per-session sequence from X arcs (follow depot → p1 → … → pk)
    sequences: dict[int, list] = {}
    for h in H:
        seq: list = []
        cur = 0
        for _ in range(len(P)):
            nxt = next(
                (j for j in P if X_vals.get((cur, j, h), 0.0) > 0.5),
                None,
            )
            if nxt is None:
                break
            seq.append(nxt)
            cur = nxt
        sequences[h] = seq

    # Weighted sum over eval scenarios
    total_scenario_cost = 0.0
    for s in S_eval:
        w_s = i_s = o_s = 0.0
        for h in H:
            seq = sequences[h]
            if not seq:
                continue

            # Forward-propagate start times
            starts: dict = {}
            prev_finish = float(t_start)   # session opens at t_start
            for p in seq:
                dur = d_eval[p, s]
                s_p = max(A_vals[p], prev_finish)
                starts[p] = s_p
                prev_finish = s_p + dur + c

            # Waiting time (per patient)
            for p in seq:
                w_s += max(0.0, starts[p] - A_vals[p])

            # Idle time (between consecutive patients)
            for k in range(len(seq) - 1):
                p_cur, p_nxt = seq[k], seq[k + 1]
                gap = starts[p_nxt] - (starts[p_cur] + d_eval[p_cur, s] + c)
                i_s += max(0.0, gap)

            # Overtime (last patient's finish vs session close)
            last_p = seq[-1]
            finish_last = starts[last_p] + d_eval[last_p, s]
            o_s += max(0.0, finish_last - t_close)

        total_scenario_cost += pi_eval[s] * (
            beta["W"] * w_s + beta["I"] * i_s + beta["O"] * o_s
        )

    # Specialty-mismatch penalty (scenario-independent)
    d_penalty = beta["D"] * sum(D_vals.get((h, q), 0.0) for h in H for q in SPECS)

    return total_scenario_cost + d_penalty


# ─────────────────────────────────────────────────────────────────────────────
# 4. SAA LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_saa(args) -> tuple[list, str]:
    """Execute the full SAA convergence experiment and stream results to CSV."""
    print(f"\n{'=' * 65}")
    print(f"  SAA Convergence Experiment")
    print(f"  Step={args.step}  WL={args.wl}  "
          f"N={args.n_start}..{args.n_max} (Δ={args.n_step})")
    print(f"  M={args.m_reps} reps  N'={args.n_prime}  seed={args.start_seed}")
    print(f"  Methods: {args.methods}")
    print(f"{'=' * 65}\n")

    # ── Load patient data ─────────────────────────────────────────────────────
    df, P, H, P0, SPECS, q_pq, session_ids, session_sequences, patient_to_h = (
        load_data(args.wl)
    )
    print(f"Loaded WL {args.wl}: {len(P)} patients, {len(H)} sessions, "
          f"{len(SPECS)} specialties.\n")

    # ── Generate evaluation scenarios ONCE ───────────────────────────────────
    print(f"Generating N'={args.n_prime} evaluation scenarios "
          f"(random, seed={args.start_seed})...")
    d_eval, S_eval, pi_eval = generate_scenarios(
        df, args.n_prime, "random", args.start_seed
    )
    print(f"  Done — {len(S_eval)} eval scenarios.\n")

    # ── Prepare step-specific fixed variables ────────────────────────────────
    if args.step == 1:
        fixed_X_base = _make_fixed_X(P0, H, session_sequences)
        fixed_Y_base = _make_fixed_Y(P, H, patient_to_h)
    elif args.step == 2:
        fixed_X_base = None
        fixed_Y_base = _make_fixed_Y(P, H, patient_to_h)
    else:
        fixed_X_base = None
        fixed_Y_base = None

    # ── Gurobi time limit ────────────────────────────────────────────────────
    time_limit = args.time_limit if args.time_limit > 0 else (
        120 if args.step <= 2 else 300
    )

    # ── Output CSV ───────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(
        args.output_dir, f"saa_wl{args.wl}_step{args.step}.csv"
    )
    with open(csv_path, "w", newline="") as fh:
        csv.writer(fh).writerow(["method", "N", "replication", "obj_out_of_sample"])
    print(f"Streaming results to: {csv_path}\n")

    all_rows: list[list] = []
    N_values = list(range(args.n_start, args.n_max + 1, args.n_step))

    for method_idx, method in enumerate(args.methods):
        print(f"\n── Method: {_METHOD_LABEL.get(method, method)} ──")
        for N in N_values:
            n_ok = n_fail = 0
            rep_objs: list[float] = []

            for rep in range(args.m_reps):
                # Unique, reproducible seed for every (method, N, rep) triple
                seed_rep = (
                    args.start_seed
                    + method_idx * 100_000
                    + N * args.m_reps
                    + rep
                )

                # Generate N training scenarios
                d_train, S_train, pi_train = generate_scenarios(
                    df, N, method, seed_rep
                )

                # Build and solve the MILP
                try:
                    m = build_model(
                        f"SAA_{method}_N{N}_r{rep}",
                        P, P0, H,
                        S_train, d_train, pi_train,
                        SPECS, q_pq,
                        _BETA, _T_START, _T_CLOSE, _C, _M_BIG,
                        fixed_X=fixed_X_base,
                        fixed_Y=fixed_Y_base,
                    )
                    m.setParam("OutputFlag", 0)
                    m.setParam("TimeLimit", time_limit)
                    if args.step == 3:
                        m.setParam("MIPGap", 0.01)
                    m.optimize()

                    if m.SolCount > 0:
                        X_vals  = {k: m._X[k].X for k in m._X}
                        Y_vals  = {k: m._Y[k].X for k in m._Y}
                        A_vals  = {p: m._A[p].X for p in P}
                        D_vals  = {k: m._D[k].X for k in m._D}
                        obj_out = simulate_schedule(
                            X_vals, Y_vals, A_vals, D_vals,
                            d_eval, S_eval, pi_eval,
                            P, H, SPECS,
                        )
                        n_ok += 1
                    else:
                        obj_out = float("nan")
                        n_fail += 1

                except Exception as exc:
                    obj_out = float("nan")
                    n_fail += 1
                    print(f"\n  [WARN] N={N} rep={rep}: {exc}")

                row = [method, N, rep, obj_out]
                all_rows.append(row)
                rep_objs.append(obj_out)

                # Write immediately so partial results survive interruptions
                with open(csv_path, "a", newline="") as fh:
                    csv.writer(fh).writerow(row)

            # Summary for this N
            valid = [v for v in rep_objs if not np.isnan(v)]
            if valid:
                mean_v = np.mean(valid)
                std_v  = np.std(valid, ddof=1) if len(valid) > 1 else 0.0
                ci_hw  = 1.96 * std_v / np.sqrt(len(valid))
                print(
                    f"  N={N:3d}: mean={mean_v:8.2f}  std={std_v:7.2f}"
                    f"  CI±{ci_hw:6.2f}  "
                    f"({n_ok} ok / {n_fail} failed)"
                )
            else:
                print(f"  N={N:3d}: all {n_fail} replications failed — "
                      f"check Gurobi licence or time limit")

    print(f"\nAll results saved to: {csv_path}")
    return all_rows, csv_path


# ─────────────────────────────────────────────────────────────────────────────
# 5. CONVERGENCE PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_convergence(
    csv_path: str,
    wl: int,
    step: int,
    output_dir: str,
) -> None:
    """
    Read the SAA results CSV and produce an interactive convergence plot.

    One line per sampling method; shaded band = 95% confidence interval
    on the out-of-sample mean (based on M replications).
    """
    df_res = pd.read_csv(csv_path)

    step_labels = {1: "Timing only", 2: "Sequencing + Timing", 3: "Full MILP"}
    fig = go.Figure()

    for method in _ALL_METHODS:
        df_m = df_res[df_res["method"] == method].dropna(
            subset=["obj_out_of_sample"]
        )
        if df_m.empty:
            continue

        stats = (
            df_m.groupby("N")["obj_out_of_sample"]
            .agg(mean="mean", std="std", count="count")
            .reset_index()
        )
        # ddof=1 standard deviation; at least 1 obs to avoid /0
        stats["ci_hw"] = (
            1.96 * stats["std"] / np.sqrt(stats["count"].clip(lower=1))
        )

        color = _METHOD_COLOR.get(method, "#bab0ac")
        label = _METHOD_LABEL.get(method, method)

        # Mean line + markers
        fig.add_trace(go.Scatter(
            x=stats["N"],
            y=stats["mean"],
            mode="lines+markers",
            name=label,
            line=dict(color=color, width=2),
            marker=dict(size=6),
        ))

        # 95 % CI shaded band
        x_band = pd.concat([stats["N"], stats["N"][::-1]], ignore_index=True)
        y_band = pd.concat(
            [stats["mean"] + stats["ci_hw"],
             (stats["mean"] - stats["ci_hw"])[::-1]],
            ignore_index=True,
        )
        fig.add_trace(go.Scatter(
            x=x_band,
            y=y_band,
            fill="toself",
            fillcolor=color,
            opacity=0.15,
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
            name=f"{label} 95% CI",
        ))

    step_str = step_labels.get(step, f"Step {step}")
    fig.update_layout(
        title=(
            f"SAA Convergence — WL {wl}, {step_str}<br>"
            f"<sub>Out-of-sample objective (mean ± 95 % CI across M replications)</sub>"
        ),
        xaxis=dict(
            title="Number of scenarios N",
            showgrid=True, gridcolor="lightgray",
            dtick=5,
        ),
        yaxis=dict(
            title="Out-of-sample objective value",
            showgrid=True, gridcolor="lightgray",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            orientation="v",
            yanchor="top", y=1.0,
            xanchor="left", x=1.02,
        ),
        margin=dict(l=80, r=220, t=110, b=60),
        height=500,
    )

    plot_path = os.path.join(
        output_dir, f"saa_convergence_wl{wl}_step{step}.html"
    )
    fig.write_html(plot_path)
    print(f"Convergence plot saved to: {plot_path}")
    fig.show()


# ─────────────────────────────────────────────────────────────────────────────
# 6. CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SAA convergence analysis for stochastic OR scheduling.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--step", type=int, required=True, choices=[1, 2, 3],
        help="Model step: 1=timing only, 2=sequencing+timing, 3=full MILP",
    )
    p.add_argument(
        "--wl", type=int, required=True, choices=[4, 7, 10, 13],
        help="Waiting list / workload (4 | 7 | 10 | 13)",
    )
    p.add_argument(
        "--n-start", type=int, default=10,
        help="Initial number of training scenarios",
    )
    p.add_argument(
        "--n-step", type=int, default=5,
        help="Scenario count increment",
    )
    p.add_argument(
        "--n-max", type=int, required=True,
        help="Maximum number of training scenarios (inclusive)",
    )
    p.add_argument(
        "--n-prime", type=int, default=1000,
        help="Evaluation sample size N'",
    )
    p.add_argument(
        "--m-reps", type=int, default=10,
        help="Number of SAA replications per N value",
    )
    p.add_argument(
        "--start-seed", type=int, default=42,
        help="Master random seed (controls eval sample + all training draws)",
    )
    p.add_argument(
        "--methods", nargs="+", default=_ALL_METHODS,
        choices=_ALL_METHODS,
        help="Sampling methods to include",
    )
    p.add_argument(
        "--time-limit", type=int, default=0,
        help=(
            "Gurobi time limit per solve (seconds). "
            "0 = auto: 120 s for steps 1-2, 300 s for step 3."
        ),
    )
    p.add_argument(
        "--output-dir", type=str, default="output",
        help="Directory for CSV results and convergence plot",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    _, csv_path = run_saa(args)
    plot_convergence(csv_path, args.wl, args.step, args.output_dir)

"""
mvpi.py — Mean Value of Perfect Information (MVPI) estimation.

Procedure
---------
1. Solve the stochastic MILP with S training scenarios → here-and-now policy.
2. Generate I independent evaluation scenarios.
3. For each eval scenario i:
   a. z_policy(i) — cost of the fixed policy on scenario i (forward simulation).
   b. z_perfectinfo(i) — optimal cost when scenario i is known in advance (wait-and-see solve).
4. VPI(i) = z_policy(i) − z_perfectinfo(i)   (≥ 0 for minimisation problems).
5. Report per-scenario results and summary statistics (mean, std, 95 % CI).

Usage
-----
python src/mvpi.py --step 3 --wl 4 --n-train 11 --n-eval 50 --method is --seed 42

Outputs (in --output-dir, default "output/")
--------------------------------------------
  mvpi_wl{wl}_step{step}_S{n_train}_I{n_eval}.csv
      columns: scenario, z_policy, z_perfectinfo, vpi
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from sampling import generate_scenarios
from main import build_model
from saa import (
    load_data,
    simulate_schedule,
    solve_policy_chain,
    _make_fixed_X,
    _make_fixed_Y,
    _T_START,
    _T_CLOSE,
    _C,
    _BETA,
)

_ALL_METHODS = ["random", "lhs", "is"]


def run_mvpi(args) -> pd.DataFrame:
    """
    Execute the MVPI estimation and return a DataFrame with per-scenario results.
    """
    print(f"\n{'=' * 65}")
    print(f"  MVPI Estimation")
    print(f"  Step={args.step}  WL={args.wl}")
    print(f"  S={args.n_train} training scenarios  (method='{args.method}')")
    print(f"  I={args.n_eval}  evaluation scenarios  seed={args.seed}")
    print(f"{'=' * 65}\n")

    # ── Load patient data ─────────────────────────────────────────────────────
    df, P, H, P0, SPECS, q_pq, session_ids, session_sequences, patient_to_h = (
        load_data(args.wl)
    )
    print(f"Loaded WL {args.wl}: {len(P)} patients, {len(H)} sessions, "
          f"{len(SPECS)} specialties.\n")

    # ── Fixed-variable dicts for the chosen step ──────────────────────────────
    if args.step == 1:
        fixed_X = _make_fixed_X(P0, H, session_sequences)
        fixed_Y = _make_fixed_Y(P, H, patient_to_h)
    elif args.step == 2:
        fixed_X = None
        fixed_Y = _make_fixed_Y(P, H, patient_to_h)
    else:  # step 3 — full MILP
        fixed_X = None
        fixed_Y = None

    time_limit = args.time_limit if args.time_limit > 0 else (
        120 if args.step <= 2 else 300
    )

    # ── Step 1: Solve stochastic model with S training scenarios → policy ─────
    print(f"Generating {args.n_train} training scenarios "
          f"(method='{args.method}', seed={args.seed})...")
    d_train, S_train, pi_train = generate_scenarios(
        df, args.n_train, args.method, args.seed, args.is_k
    )
    print(f"  Done — {len(S_train)} scenarios.\n")

    print(f"Solving stochastic model (step {args.step}) for here-and-now policy...")

    # Build the policy via the same step 1→2→3 warm-start chain as main.py:
    # each step is seeded with the previous step's solution.
    def _configure(m, step, with_mip_focus=False):
        m.setParam("OutputFlag", 0)
        m.setParam("TimeLimit", time_limit)
        if with_mip_focus:
            m.setParam("MIPFocus", 1)
        if step == 3:
            m.setParam("MIPGap", 0.01)
            m.setParam("Presolve", 2)
            m.setParam("Cuts", 2)
            m.setParam("Heuristics", 0.3)

    m_policy = solve_policy_chain(
        P, P0, H, SPECS, q_pq, session_sequences, patient_to_h,
        args.step, d_train, S_train, pi_train,
        _configure, name_prefix="Policy",
    )

    if m_policy.SolCount == 0:
        raise RuntimeError(
            f"Policy solve failed (status={m_policy.Status}). "
            "Check Gurobi licence or increase --time-limit."
        )

    X_vals = {k: m_policy._X[k].X for k in m_policy._X}
    Y_vals = {k: m_policy._Y[k].X for k in m_policy._Y}
    A_vals = {p: m_policy._A[p].X for p in P}
    D_vals = {k: m_policy._D[k].X for k in m_policy._D}
    print(f"  Policy solved. Training objective = {m_policy.ObjVal:.4f}\n")

    # ── Step 2: Generate I independent evaluation scenarios ───────────────────
    # Use a seed that is disjoint from the training seed space used in saa.py.
    eval_seed = args.seed + 99_999
    print(f"Generating {args.n_eval} evaluation scenarios "
          f"(random, seed={eval_seed})...")
    d_eval, S_eval, pi_eval = generate_scenarios(
        df, args.n_eval, "random", eval_seed
    )
    print(f"  Done — {len(S_eval)} scenarios.\n")

    # ── Step 3: Compute z_policy(i) and z_perfectinfo(i) for each eval scenario ───────
    print(f"Computing VPI for each of {args.n_eval} evaluation scenarios...\n")
    print(f"  {'i':>4}  {'z_policy':>10}  {'z_perfectinfo':>10}  {'VPI':>10}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*10}  {'-'*10}")

    rows: list[dict] = []
    for idx, s in enumerate(S_eval):

        # (a) Evaluate the fixed policy on scenario s alone
        z_policy_s = simulate_schedule(
            X_vals, Y_vals, A_vals, D_vals,
            d_eval=d_eval,
            S_eval=[s],
            pi_eval={s: 1.0},
            P=P, H=H, SPECS=SPECS,
        )

        # (b) Perfect-information solve: optimise everything for scenario s in isolation.
        #     Re-map to a fresh single-scenario index (0) to keep the model small.
        d_pi   = {(p, 0): d_eval[p, s] for p in P}
        prob_pi = {0: 1.0}
        m_pi = build_model(
            f"PI_{s}",
            P, P0, H,
            [0], d_pi, prob_pi,
            SPECS, q_pq,
            _BETA, _T_START, _T_CLOSE, _C,
            fixed_X=fixed_X,
            fixed_Y=fixed_Y,
        )
        m_pi.setParam("OutputFlag", 0)
        m_pi.setParam("TimeLimit", time_limit)
        if args.step == 3:
            m_pi.setParam("MIPGap", 0.01)
            m_pi.setParam("Presolve", 2)
            m_pi.setParam("Cuts", 2)
            m_pi.setParam("Heuristics", 0.3)
        m_pi.optimize()

        if m_pi.SolCount == 0:
            z_pi_s = float("nan")
            vpi_s  = float("nan")
        else:
            z_pi_s = m_pi.ObjVal
            vpi_s  = z_policy_s - z_pi_s

        rows.append({
            "scenario": s,
            "z_policy": z_policy_s,
            "z_perfectinfo": z_pi_s,
            "vpi":      vpi_s,
        })

        pi_str  = f"{z_pi_s:10.3f}" if not np.isnan(z_pi_s) else f"{'FAILED':>10}"
        vpi_str = f"{vpi_s:10.3f}"  if not np.isnan(vpi_s)  else f"{'FAILED':>10}"
        print(f"  {idx + 1:4d}  {z_policy_s:10.3f}  {pi_str}  {vpi_str}")

    df_result = pd.DataFrame(rows)
    valid_vpi = df_result["vpi"].dropna()
    n_failed  = df_result["vpi"].isna().sum()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─' * 55}")
    print(f"  MVPI Summary  (I={args.n_eval}, step={args.step}, WL={args.wl})")
    print(f"{'─' * 55}")
    if len(valid_vpi) > 0:
        mean_vpi = valid_vpi.mean()
        std_vpi  = valid_vpi.std(ddof=1) if len(valid_vpi) > 1 else 0.0
        ci_hw    = 1.96 * std_vpi / np.sqrt(len(valid_vpi))
        print(f"  MVPI (mean VPI) : {mean_vpi:.4f}")
        print(f"  Std VPI         : {std_vpi:.4f}")
        print(f"  95 % CI         : ± {ci_hw:.4f}  "
              f"[{mean_vpi - ci_hw:.4f}, {mean_vpi + ci_hw:.4f}]")
        print(f"  Min VPI         : {valid_vpi.min():.4f}")
        print(f"  Max VPI         : {valid_vpi.max():.4f}")
    else:
        print("  No valid VPI values — all PI solves failed.")
    print(f"  Failed solves   : {n_failed}/{args.n_eval}")
    print(f"{'─' * 55}\n")

    # ── Save results to CSV ───────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(
        args.output_dir,
        f"mvpi_wl{args.wl}_step{args.step}_S{args.n_train}_I{args.n_eval}.csv",
    )
    df_result.to_csv(out_path, index=False, float_format="%.6f")
    print(f"Per-scenario results saved to: {out_path}")

    return df_result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MVPI estimation for stochastic OR scheduling.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--step", type=int, default=3, choices=[1, 2, 3],
        help="Model step: 1=timing only, 2=sequencing+timing, 3=full MILP",
    )
    p.add_argument(
        "--wl", type=int, required=True, choices=[4, 7, 10, 13],
        help="Waiting list workload (4 | 7 | 10 | 13)",
    )
    p.add_argument(
        "--n-train", type=int, default=11,
        help="Number of training scenarios S for the policy solve",
    )
    p.add_argument(
        "--n-eval", type=int, default=50,
        help="Number of evaluation scenarios I",
    )
    p.add_argument(
        "--method", type=str, default="is", choices=_ALL_METHODS,
        help="Sampling method for the training scenarios",
    )
    p.add_argument(
        "--is-k", type=float, default=1.0,
        help="IS shift magnitude k (delta ∈ {-k, 0, k}); ignored for random/lhs",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help=(
            "Master random seed for training scenarios. "
            "Evaluation scenarios use seed + 99999."
        ),
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
        help="Directory for the output CSV",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_mvpi(args)

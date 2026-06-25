"""
saa.py — SAA convergence analysis CLI for stochastic OR scheduling.

Runs the SAA procedure for all three model steps (timing-only, sequencing+timing,
full MILP), warm-starting each step from the previous.  Each replication is
evaluated analytically on a fixed out-of-sample evaluation set.  Results are
written row-by-row to per-step CSV files; convergence plots are produced at the
end.

Usage
-----
python src/saa.py --wl 4 --n-start 1 --n-step 3 --n-max 16 --n-prime 10000 \\
                  --m-reps 10 --time-limit 600 --is-k 0.5

Outputs (in --output-dir, default "output/")
--------------------------------------------
  saa_wl{wl}_step{1,2,3}.csv                -- per-replication results
  saa_convergence_wl{wl}_step{1,2,3}.html   -- interactive convergence plots
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

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
_BETA    = {"W": 0.6, "I": 0.2, "O": 0.2, "D": 100.0}

_ALL_METHODS = ["random", "lhs", "is"]


def _flush_results_csv(results: dict, path: str) -> None:
    """Write the full merged results dict to CSV, sorted by (method, N, rep)."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "N", "replication", "obj_out_of_sample"])
        for (meth, n, rep), obj in sorted(results.items()):
            w.writerow([meth, n, rep, obj])

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


def load_data(wl: int | None = None, *, path: str | None = None):
    """
    Load patient data for waiting-list *wl* (4 | 7 | 10 | 13), or from an explicit
    CSV *path* (used e.g. by apply_blueprint.py's comparison set).  Exactly one of
    *wl* / *path* must be given.

    Returns
    -------
    df, P, H, P0, SPECS, q_pq, session_ids, session_sequences, patient_to_h
    """
    if (wl is None) == (path is None):
        raise ValueError("Provide exactly one of `wl` or `path`.")
    if path is None:
        root = os.path.join(os.path.dirname(__file__), "..")
        path = os.path.join(root, "input", f"sample_wl{wl}.csv")
    df = pd.read_csv(path)

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
    return_parts: bool = False,
):
    """Analytically evaluate a fixed first-stage solution on out-of-sample scenarios.

    By default returns the total weighted objective (float).  When
    ``return_parts=True`` returns a dict with the **weighted** contribution of
    each objective part — ``{"W", "I", "O", "D", "total"}`` — so callers can
    decompose the out-of-sample objective the same way it is built in-sample.
    """
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

    # Expected (probability-weighted) raw quantities, accumulated across scenarios.
    exp_w = exp_i = exp_o = 0.0
    for s in S_eval:
        w_s = i_s = o_s = 0.0
        for h in H:
            seq = sequences[h]
            if not seq:
                continue
            starts: dict = {}
            prev_finish = float(t_start)
            for p in seq:
                dur = d_eval[p, s]
                s_p = max(A_vals[p], prev_finish)
                starts[p] = s_p
                prev_finish = s_p + dur + c
            for p in seq:
                w_s += max(0.0, starts[p] - A_vals[p])
            for k in range(len(seq) - 1):
                p_cur, p_nxt = seq[k], seq[k + 1]
                gap = starts[p_nxt] - (starts[p_cur] + d_eval[p_cur, s] + c)
                i_s += max(0.0, gap)
            last_p = seq[-1]
            finish_last = starts[last_p] + d_eval[last_p, s]
            o_s += max(0.0, finish_last - t_close)

        exp_w += pi_eval[s] * w_s
        exp_i += pi_eval[s] * i_s
        exp_o += pi_eval[s] * o_s

    w_contrib = beta["W"] * exp_w
    i_contrib = beta["I"] * exp_i
    o_contrib = beta["O"] * exp_o
    d_contrib = beta["D"] * sum(D_vals.get((h, q), 0.0) for h in H for q in SPECS)
    total = w_contrib + i_contrib + o_contrib + d_contrib

    if return_parts:
        return {
            "W": w_contrib, "I": i_contrib, "O": o_contrib,
            "D": d_contrib, "total": total,
        }
    return total


# ─────────────────────────────────────────────────────────────────────────────
# 4a. WARM-START HELPER (step 2 → step 3)
# ─────────────────────────────────────────────────────────────────────────────

def _warm_start_s2_to_s3(m2, m3, P, P0, H, S_train, SPECS) -> None:
    """
    Inject m2's solution into m3 as a MIP start, applying the within-specialty
    session permutation required to satisfy m3's symmetry-breaking constraints
    (same logic as main.py Step 3 warm start).
    """
    p_rank = {p: k for k, p in enumerate(sorted(P))}
    first_rank: dict[int, float] = {}
    spec_of_h: dict[int, object] = {}
    for h in H:
        if m2._Z[h].X < 0.5:
            first_rank[h] = float("inf")
            spec_of_h[h] = None
            continue
        p_first = next((p for p in P if m2._X[0, p, h].X > 0.5), None)
        first_rank[h] = p_rank[p_first] if p_first is not None else float("inf")
        spec_of_h[h] = next((q for q in SPECS if m2._V[h, q].X > 0.5), None)

    sorted_src: list[int] = []
    for q in SPECS:
        sessions_of_q = [h for h in H if spec_of_h.get(h) == q]
        sorted_src.extend(sorted(sessions_of_q, key=lambda h: first_rank[h]))
    for h in H:
        if h not in sorted_src:
            sorted_src.append(h)
    perm = {h_src: h_dst for h_src, h_dst in zip(sorted_src, sorted(H))}

    for p in P:
        m3._A[p].Start = m2._A[p].X
        for h in H:
            m3._Y[p, perm[h]].Start = m2._Y[p, h].X
    for i in P0:
        for j in P0:
            if i == j:
                continue
            for h in H:
                if (i, j, h) in m2._X and (i, j, perm[h]) in m3._X:
                    m3._X[i, j, perm[h]].Start = m2._X[i, j, h].X
    for h in H:
        m3._Z[perm[h]].Start = m2._Z[h].X
        for q in SPECS:
            m3._V[perm[h], q].Start = m2._V[h, q].X
            m3._D[perm[h], q].Start = m2._D[h, q].X
    for p in P:
        for s in S_train:
            m3._S[p, s].Start = m2._S[p, s].X
            m3._W[p, s].Start = m2._W[p, s].X
    for p in P:
        for pp in P:
            if p == pp:
                continue
            for s in S_train:
                m3._I[p, pp, s].Start = m2._I[p, pp, s].X
    for h in H:
        for s in S_train:
            m3._O[perm[h], s].Start = m2._O[h, s].X
    # Inject MTZ position values derived from X arcs so the MIP start is complete.
    for h in H:
        dest_h = perm[h]
        cur, pos = 0, 1
        for _ in range(len(P)):
            nxt = next((p for p in P if p != cur and m2._X[cur, p, h].X > 0.5), None)
            if nxt is None:
                break
            m3._U[nxt, dest_h].Start = float(pos)
            cur, pos = nxt, pos + 1
    m3.update()
    m3.setParam("MIPFocus", 1)


# ─────────────────────────────────────────────────────────────────────────────
# 4a'. POLICY-CHAIN HELPER  (shared by saa.py, vss.py, mvpi.py)
# ─────────────────────────────────────────────────────────────────────────────

def solve_policy_chain(
    P, P0, H, SPECS, q_pq,
    session_sequences, patient_to_h,
    target_step, d_train, S_train, pi_train,
    configure, name_prefix="policy", return_all=False,
    step3_post_build=None,
):
    """
    Build and solve the model up to *target_step*, warm-starting each step from
    the previous one — exactly the sequence used in main.py and the SAA worker.

    Step 1 fixes X and Y from the CSV; Step 2 frees X (warm-started by a
    name-by-name copy of the Step 1 solution); Step 3 frees everything
    (warm-started from Step 2 via `_warm_start_s2_to_s3`, which applies the
    symmetry-breaking session permutation).  Returns the solved model at
    *target_step* so that the resulting first-stage policy is identical in
    construction to the one produced by main.py.

    `configure(m, step, with_mip_focus=False)` is a caller-supplied callback that
    applies solver parameters (output flag, time limit, MIP gap, …) per step.

    When ``return_all=True`` the function instead returns a ``{step: model}``
    dict holding every solved step from 1 up to *target_step*, so the whole
    warm-started chain can be inspected in one solve (mirrors main.py producing
    m1/m2/m3 together).

    ``step3_post_build(m3)`` is an optional callback invoked on the step-3 model
    after it is built and warm-started but before ``optimize()`` — used by
    apply_blueprint.py to inject the blueprint subgroup-quota constraints on the
    (now free) assignment variables ``m3._Y``.
    """
    models: dict[int, "gp.Model"] = {}
    fixed_X_s1 = _make_fixed_X(P0, H, session_sequences)
    fixed_Y    = _make_fixed_Y(P, H, patient_to_h)

    # ── Step 1: X and Y fixed from CSV ───────────────────────────────────────
    m1 = build_model(
        f"{name_prefix}_s1", P, P0, H, S_train, d_train, pi_train,
        SPECS, q_pq, _BETA, _T_START, _T_CLOSE, _C,
        fixed_X=fixed_X_s1, fixed_Y=fixed_Y,
    )
    configure(m1, 1)
    m1.optimize()
    models[1] = m1
    if target_step == 1:
        return models if return_all else m1

    # ── Step 2: Y fixed, warm start from Step 1 ──────────────────────────────
    m2 = build_model(
        f"{name_prefix}_s2", P, P0, H, S_train, d_train, pi_train,
        SPECS, q_pq, _BETA, _T_START, _T_CLOSE, _C,
        fixed_Y=fixed_Y,
    )
    warm_s2 = m1.SolCount > 0
    if warm_s2:
        m1_by_name = {v.VarName: v for v in m1.getVars()}
        for v2 in m2.getVars():
            v1 = m1_by_name.get(v2.VarName)
            if v1 is not None:
                v2.Start = v1.X
        m2.update()
    configure(m2, 2, with_mip_focus=warm_s2)
    m2.optimize()
    models[2] = m2
    if target_step == 2:
        return models if return_all else m2

    # ── Step 3: all free, warm start from Step 2 (with permutation) ───────────
    m3 = build_model(
        f"{name_prefix}_s3", P, P0, H, S_train, d_train, pi_train,
        SPECS, q_pq, _BETA, _T_START, _T_CLOSE, _C,
    )
    # _warm_start_s2_to_s3 calls m3.update() and sets MIPFocus=1 internally.
    if m2.SolCount > 0:
        _warm_start_s2_to_s3(m2, m3, P, P0, H, S_train, SPECS)
    if step3_post_build is not None:
        step3_post_build(m3)
    configure(m3, 3, with_mip_focus=False)  # MIPFocus already set by warm start
    m3.optimize()
    models[3] = m3
    return models if return_all else m3


# ─────────────────────────────────────────────────────────────────────────────
# 4b. PER-REPLICATION WORKER  (module-level so ProcessPoolExecutor can pickle it)
# ─────────────────────────────────────────────────────────────────────────────

def _rep_worker(job: dict) -> tuple[float, float, float]:
    """
    Run one SAA replication across all 3 steps, warm-starting each from the
    previous.  Returns (obj_step1, obj_step2, obj_step3).  Must be at module
    level so that ProcessPoolExecutor (spawn context on Windows) can pickle it.
    """
    d_train, S_train, pi_train = generate_scenarios(
        job["df"], job["N"], job["method"], job["seed_rep"], job["is_k"]
    )
    P, P0, H, SPECS = job["P"], job["P0"], job["H"], job["SPECS"]
    q_pq = job["q_pq"]
    d_eval, S_eval, pi_eval = job["d_eval"], job["S_eval"], job["pi_eval"]
    time_limit   = job["time_limit"]
    gurobi_threads = job.get("gurobi_threads")
    rid = f"{job['method']}_N{job['N']}_r{job['rep']}"

    def _configure(m, step, with_mip_focus=False):
        m.setParam("OutputFlag", 0)
        m.setParam("TimeLimit", time_limit)
        if gurobi_threads is not None:
            m.setParam("Threads", gurobi_threads)
        if with_mip_focus:
            m.setParam("MIPFocus", 1)
        if step == 3:
            m.setParam("MIPGap", 0.05)   # 5 % gap; convergence tracks out-of-sample obj
            m.setParam("Presolve", 2)
            m.setParam("Cuts", 2)
            m.setParam("Heuristics", 0.3)

    def _eval(m):
        if m.SolCount == 0:
            return float("nan")
        return simulate_schedule(
            {k: m._X[k].X for k in m._X},
            {k: m._Y[k].X for k in m._Y},
            {p: m._A[p].X for p in P},
            {k: m._D[k].X for k in m._D},
            d_eval, S_eval, pi_eval, P, H, SPECS,
        )

    try:
        # ── Step 1: X and Y fixed from CSV ───────────────────────────────────
        m1 = build_model(
            f"SAA_s1_{rid}", P, P0, H, S_train, d_train, pi_train,
            SPECS, q_pq, _BETA, _T_START, _T_CLOSE, _C,
            fixed_X=job["fixed_X_s1"], fixed_Y=job["fixed_Y"],
        )
        _configure(m1, 1)
        m1.optimize()
        obj1 = _eval(m1)

        # ── Step 2: Y fixed, warm start from step 1 ───────────────────────────
        m2 = build_model(
            f"SAA_s2_{rid}", P, P0, H, S_train, d_train, pi_train,
            SPECS, q_pq, _BETA, _T_START, _T_CLOSE, _C,
            fixed_Y=job["fixed_Y"],
        )
        warm_s2 = m1.SolCount > 0
        if warm_s2:
            m1_by_name = {v.VarName: v for v in m1.getVars()}
            for v2 in m2.getVars():
                v1 = m1_by_name.get(v2.VarName)
                if v1 is not None:
                    v2.Start = v1.X
            m2.update()
        _configure(m2, 2, with_mip_focus=warm_s2)
        m2.optimize()
        obj2 = _eval(m2)

        # ── Step 3: all free, warm start from step 2 (with permutation) ──────
        m3 = build_model(
            f"SAA_s3_{rid}", P, P0, H, S_train, d_train, pi_train,
            SPECS, q_pq, _BETA, _T_START, _T_CLOSE, _C,
        )
        # _warm_start_s2_to_s3 calls m3.update() and sets MIPFocus=1 internally
        if m2.SolCount > 0:
            _warm_start_s2_to_s3(m2, m3, P, P0, H, S_train, SPECS)
        _configure(m3, 3, with_mip_focus=False)  # MIPFocus already set by warm start
        m3.optimize()
        obj3 = _eval(m3)

        return obj1, obj2, obj3

    except Exception as exc:
        print(f"\n  [WARN] N={job['N']} rep={job['rep']}: {exc}", flush=True)
        return float("nan"), float("nan"), float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# 4c. SAA LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_saa(args) -> dict[int, str]:
    """Execute the full SAA convergence experiment and stream results to CSV.
    Returns {step: csv_path} for steps 1, 2, 3.
    """
    print(f"\n{'=' * 65}")
    print(f"  SAA Convergence Experiment (steps 1, 2, 3)")
    print(f"  WL={args.wl}  N={args.n_start}..{args.n_max} (step={args.n_step})")
    print(f"  M={args.m_reps} reps  N'={args.n_prime}  seed={args.start_seed}")
    print(f"  Workers: {args.n_workers}  Methods: {args.methods}")
    print(f"{'=' * 65}\n")

    df, P, H, P0, SPECS, q_pq, session_ids, session_sequences, patient_to_h = (
        load_data(args.wl)
    )
    print(f"Loaded WL {args.wl}: {len(P)} patients, {len(H)} sessions, "
          f"{len(SPECS)} specialties.\n")

    print(f"Generating N'={args.n_prime} evaluation scenarios "
          f"(random, seed={args.start_seed})...")
    d_eval, S_eval, pi_eval = generate_scenarios(
        df, args.n_prime, "random", args.start_seed
    )
    print(f"  Done - {len(S_eval)} eval scenarios.\n")

    fixed_X_s1 = _make_fixed_X(P0, H, session_sequences)
    fixed_Y    = _make_fixed_Y(P, H, patient_to_h)

    time_limit = args.time_limit if args.time_limit > 0 else 300

    os.makedirs(args.output_dir, exist_ok=True)
    csv_paths = {
        step: os.path.join(args.output_dir, f"saa_wl{args.wl}_step{step}.csv")
        for step in (1, 2, 3)
    }

    # Load existing results per step
    existing: dict[int, dict[tuple, float]] = {step: {} for step in (1, 2, 3)}
    for step, path in csv_paths.items():
        if os.path.exists(path):
            try:
                df_ex = pd.read_csv(path)
                for _, row in df_ex.iterrows():
                    key = (row["method"], int(row["N"]), int(row["replication"]))
                    existing[step][key] = float(row["obj_out_of_sample"])
                print(f"Loaded {len(existing[step])} existing step-{step} results from: {path}")
            except Exception as e:
                print(f"Could not read {path} ({e}); starting fresh.")
        else:
            _flush_results_csv(existing[step], path)

    print(f"\nStreaming results to: {args.output_dir}/\n")

    N_values = list(range(args.n_start, args.n_max + 1, args.n_step))

    for method_idx, method in enumerate(args.methods):
        print(f"\n-- Method: {_METHOD_LABEL.get(method, method)} --")
        for N in N_values:
            step_objs: dict[int, list[float]] = {1: [], 2: [], 3: []}

            if args.n_workers > 1:
                n_cpu = os.cpu_count() or 1
                gurobi_threads = max(1, n_cpu // args.n_workers)
                jobs = [
                    {
                        "method": method, "N": N, "rep": rep,
                        "seed_rep": (
                            args.start_seed
                            + method_idx * 100_000
                            + N * args.m_reps
                            + rep
                        ),
                        "df": df, "P": P, "P0": P0, "H": H,
                        "SPECS": SPECS, "q_pq": q_pq,
                        "d_eval": d_eval, "S_eval": S_eval, "pi_eval": pi_eval,
                        "time_limit": time_limit,
                        "gurobi_threads": gurobi_threads,
                        "fixed_X_s1": fixed_X_s1,
                        "fixed_Y": fixed_Y,
                        "is_k": args.is_k,
                    }
                    for rep in range(args.m_reps)
                ]
                with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
                    future_to_rep = {
                        ex.submit(_rep_worker, job): job["rep"] for job in jobs
                    }
                    for fut in as_completed(future_to_rep):
                        rep = future_to_rep[fut]
                        try:
                            o1, o2, o3 = fut.result()
                        except Exception as exc:
                            print(f"\n  [WARN] N={N} rep={rep}: worker crashed: {exc}",
                                  flush=True)
                            o1 = o2 = o3 = float("nan")
                        for step, obj in [(1, o1), (2, o2), (3, o3)]:
                            step_objs[step].append(obj)
                            existing[step][(method, N, rep)] = obj
                        # Persist each replication the moment it finishes
                        for step in (1, 2, 3):
                            _flush_results_csv(existing[step], csv_paths[step])

            else:
                for rep in range(args.m_reps):
                    seed_rep = (
                        args.start_seed
                        + method_idx * 100_000
                        + N * args.m_reps
                        + rep
                    )
                    job = {
                        "method": method, "N": N, "rep": rep,
                        "seed_rep": seed_rep,
                        "df": df, "P": P, "P0": P0, "H": H,
                        "SPECS": SPECS, "q_pq": q_pq,
                        "d_eval": d_eval, "S_eval": S_eval, "pi_eval": pi_eval,
                        "time_limit": time_limit,
                        "gurobi_threads": None,  # use all available threads
                        "fixed_X_s1": fixed_X_s1,
                        "fixed_Y": fixed_Y,
                        "is_k": args.is_k,
                    }
                    o1, o2, o3 = _rep_worker(job)
                    for step, obj in [(1, o1), (2, o2), (3, o3)]:
                        step_objs[step].append(obj)
                        existing[step][(method, N, rep)] = obj
                    for step in (1, 2, 3):
                        _flush_results_csv(existing[step], csv_paths[step])

            # Summary for this N
            for step in (1, 2, 3):
                rep_objs = step_objs[step]
                n_ok   = sum(1 for v in rep_objs if not np.isnan(v))
                n_fail = args.m_reps - n_ok
                valid  = [v for v in rep_objs if not np.isnan(v)]
                if valid:
                    mean_v = np.mean(valid)
                    std_v  = np.std(valid, ddof=1) if len(valid) > 1 else 0.0
                    ci_hw  = 1.96 * std_v / np.sqrt(len(valid))
                    print(
                        f"  N={N:3d} step={step}: mean={mean_v:8.2f}  std={std_v:7.2f}"
                        f"  CI+/-{ci_hw:6.2f}  ({n_ok} ok / {n_fail} failed)"
                    )
                else:
                    print(f"  N={N:3d} step={step}: all {n_fail} replications failed - "
                          f"check Gurobi licence or time limit")

    print(f"\nAll results saved to: {args.output_dir}/")
    return csv_paths


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
        stats["ci_hw"] = (
            1.96 * stats["std"] / np.sqrt(stats["count"].clip(lower=1))
        )

        color = _METHOD_COLOR.get(method, "#bab0ac")
        label = _METHOD_LABEL.get(method, method)

        fig.add_trace(go.Scatter(
            x=stats["N"],
            y=stats["mean"],
            mode="lines+markers",
            name=label,
            line=dict(color=color, width=2),
            marker=dict(size=6),
        ))

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
            "Gurobi time limit per solve per step (seconds). "
            "0 = auto: 300 s."
        ),
    )
    p.add_argument(
        "--n-workers", type=int, default=1,
        help=(
            "Number of parallel worker processes for SAA replications. "
            "1 = sequential (default). When >1, Gurobi threads per solve "
            "are set to max(1, cpu_count // n_workers) to avoid oversubscription."
        ),
    )
    p.add_argument(
        "--output-dir", type=str, default="output",
        help="Directory for CSV results and convergence plots",
    )
    p.add_argument(
        "--is-k", type=float, default=1.0,
        help="Shift magnitude k for IS (delta ∈ {-k, 0, k}); ignored for other methods",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    csv_paths = run_saa(args)
    for step, path in csv_paths.items():
        plot_convergence(path, args.wl, step, args.output_dir)

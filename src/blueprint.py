"""
blueprint.py — Step 5: Three-Stage Stochastic Blueprint for OR9 ORT Scheduling.

Problem
-------
Decide, before the waiting list is known, how many patients of each surgery-type
subgroup (S / M / L) to allocate to each of the three OR9 ORT sessions in a week.

Three-stage structure (node-oriented):
  Stage 1 (blueprint, root node):
      N[g,h] — maximum number of patients of subgroup g allowed in session h.
      Decided before any waiting-list information is known.
  Stage 2 (SSP, node r):
      X_r, Y_r, A_r, Z_r, U_r — assignment + sequencing + timing given
      the realised patient set P_r.  Z_h = 1 forced (all three sessions open).
  Stage 3 (leaf node rs):
      Sv_rs, W_rs, I_rs, O_rs — costs given realised surgery durations.

Patient Subgroups
-----------------
Surgery types in ort_patient_data.csv are clustered in the
(mean_expected_duration, mean_sigma_error) space using k-means (k=3).
Clusters are labelled S (short), M (medium), L (long) by ascending mean duration.

Realization Sampling
--------------------
Waiting-list realizations are constructed by randomly sampling 3 OR9 session IDs
from step_5_sample_sessions.csv (all ORT, OR=OR9).  All patients from those
three sessions form one realization r.

Usage
-----
# One or several seeds (consistency study — quotas comparable because the S/M/L
# clustering is fixed by --cluster-seed and only the sampling varies per seed):
python src/blueprint.py --n-realizations 5 --n-scenarios 20 --seed 42 43 44
                        --beta5 1.0 --time-limit 600
                        --method lhs

Outputs (in output/)
--------------------
  blueprint.csv            — N[g,h] quota table (last seed solved)
  blueprint_clusters.csv   — surgery-type cluster centroids (S/M/L)
  blueprint_solutions.csv  — cumulative: one row per run (input settings +
                             solution + objective), keyed by settings incl. seed
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
from gurobipy import GRB
from scipy.cluster.vq import kmeans2

from sampling import generate_scenarios

# ── Constants ─────────────────────────────────────────────────────────────────
_T_START = 480    # 08:00 in minutes
_T_CLOSE = 960    # 16:00 in minutes
_C       = 10     # changeover time (minutes)
_BETA    = {"W": 0.6, "I": 0.2, "O": 0.2}
_G       = ["S", "M", "L"]
_H       = [0, 1, 2]   # three OR9 sessions

_ROOT = os.path.join(os.path.dirname(__file__), "..")


# ─────────────────────────────────────────────────────────────────────────────
# 1. PATIENT SUBGROUP CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

def cluster_surgery_types(k: int = 3, random_state: int = 42):
    """
    K-means clustering of ORT surgery types in (mean_expected_duration,
    mean_sigma_error) space.  Returns:
      type_to_group : dict  surgery_type (int) -> "S"/"M"/"L"
      centroids     : DataFrame with one row per cluster
    """
    path = os.path.join(_ROOT, "input", "ort_patient_data.csv")
    df = pd.read_csv(path)

    # Average expected_duration and sigma_error per surgery type
    agg = (
        df.groupby("Surgery type")[["expected_duration", "sigma_error"]]
          .mean()
          .dropna()
          .reset_index()
    )

    # Standardise both features (zero mean, unit std) so duration and sigma
    # contribute comparably to the Euclidean distance used by k-means.
    raw = agg[["expected_duration", "sigma_error"]].values.astype(float)
    X = (raw - raw.mean(axis=0)) / raw.std(axis=0)

    # k-means via scipy (kmeans++ init, fixed seed for reproducibility)
    _centroids, labels = kmeans2(
        X, k, seed=random_state, minit="++", missing="raise"
    )
    agg["raw_cluster"] = labels

    # Map raw cluster ids to S/M/L by ascending mean expected_duration
    order = (
        agg.groupby("raw_cluster")["expected_duration"]
           .mean()
           .sort_values()
           .index.tolist()
    )
    label_map = {raw: lbl for raw, lbl in zip(order, _G)}
    agg["group"] = agg["raw_cluster"].map(label_map)

    type_to_group: dict[int, str] = dict(zip(
        agg["Surgery type"].astype(int), agg["group"]
    ))

    centroids = (
        agg.groupby("group")[["expected_duration", "sigma_error"]]
           .mean()
           .rename(columns={"expected_duration": "mean_exp_dur",
                            "sigma_error": "mean_sigma"})
           .assign(n_types=agg.groupby("group")["Surgery type"].count())
           .reset_index()
    )

    return type_to_group, centroids


# ─────────────────────────────────────────────────────────────────────────────
# 2. REALIZATION SAMPLING
# ─────────────────────────────────────────────────────────────────────────────

def sample_realizations(
    n_realizations: int,
    sessions_per_realization: int = 3,
    seed: int = 42,
    or_filter: str | None = None,
) -> list[pd.DataFrame]:
    """
    Sample n_realizations patient sets from step_5_sample_sessions.csv.
    Each realization: randomly pick `sessions_per_realization` documented
    session IDs and collect all patients in those sessions.

    By default all documented ORT sessions are eligible (or_filter=None); an
    optional OR restriction can be supplied but is not used in the main run.
    """
    path = os.path.join(_ROOT, "input", "step_5_sample_sessions.csv")
    df = pd.read_csv(path)
    if or_filter:
        df = df[df["OR"] == or_filter].copy()

    available = df["Session ID"].unique().tolist()
    if len(available) < sessions_per_realization:
        raise ValueError(
            f"Only {len(available)} sessions available; "
            f"need {sessions_per_realization}."
        )

    rng = np.random.default_rng(seed)
    result = []
    for _ in range(n_realizations):
        chosen = rng.choice(available, size=sessions_per_realization, replace=False)
        patients = df[df["Session ID"].isin(chosen)].copy().reset_index(drop=True)
        result.append(patients)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3. THREE-STAGE MILP
# ─────────────────────────────────────────────────────────────────────────────

def build_blueprint_model(
    realizations: list[pd.DataFrame],
    scenarios_per_real: list[tuple[dict, list, dict]],
    type_to_group: dict[int, str],
    beta5: float = 1.0,
    t_start: int = _T_START,
    t_close: int = _T_CLOSE,
    c: int = _C,
    beta: dict = _BETA,
) -> gp.Model:
    """
    Build the three-stage deterministic equivalent in Gurobi.

    Stage 1 : N[g, h]  (blueprint quotas — shared root variable)
    Stage 2 : X_r, Y_r, A_r, Z_r, U_r  (per realization)
    Stage 3 : Sv_rs, W_rs, I_rs, O_rs   (per realization × scenario leaf)
    """
    n_real = len(realizations)
    pi_r   = 1.0 / n_real
    H = _H
    G = _G

    m = gp.Model("Blueprint_3stage")

    # ── Stage 1: blueprint quotas (shared across all realizations) ─────────────
    N = m.addVars(
        [(g, h) for g in G for h in H],
        vtype=GRB.INTEGER, lb=0, name="N",
    )

    # Session symmetry-breaking.  All three sessions are identical (same opening
    # and closing times, no session-specific data), so any permutation of the
    # session indices yields an equivalent solution.  We remove this |H|!
    # symmetry — which otherwise bloats the branch-and-bound tree — by ordering
    # the sessions on a lexicographic capacity key that weights L above M above
    # S.  Because the sessions are genuinely exchangeable, this cut preserves
    # optimality while fixing a canonical labelling (session 0 = heaviest).
    for idx in range(len(H) - 1):
        h, h1 = H[idx], H[idx + 1]
        m.addConstr(
            100 * N["L", h] + 10 * N["M", h] + N["S", h]
            >= 100 * N["L", h1] + 10 * N["M", h1] + N["S", h1],
            name=f"sym_session_order_{h}",
        )

    # Accumulate objective terms
    obj_terms: list = [beta5 * gp.quicksum(N[g, h] for g in G for h in H)]

    # ── Per-realization Stage 2 + 3 ───────────────────────────────────────────
    for r, (df_r, (d_r, S_r, pi_rs)) in enumerate(
        zip(realizations, scenarios_per_real)
    ):
        P_r  = df_r["Patient ID"].tolist()
        P0_r = [0] + P_r
        n_p  = len(P_r)
        sfx  = f"_r{r}"

        # Patient subgroup membership
        def _grp(p):
            row = df_r.loc[df_r["Patient ID"] == p, "Surgery type"]
            stype = int(row.iloc[0]) if not row.empty else -1
            return type_to_group.get(stype, "M")   # default M for unseen types

        P_gr = {g: [p for p in P_r if _grp(p) == g] for g in G}

        # Tight big-M
        max_d  = max(d_r.values()) if d_r else float(t_close - t_start)
        M_time = float(n_p * (max_d + c))

        # ── Stage 2 variables ─────────────────────────────────────────────────
        A = m.addVars(P_r, lb=t_start, ub=t_close, name=f"A{sfx}")
        X = m.addVars(
            [(i, j, h) for i in P0_r for j in P0_r for h in H if i != j],
            vtype=GRB.BINARY, name=f"X{sfx}",
        )
        Y = m.addVars(
            [(p, h) for p in P_r for h in H],
            vtype=GRB.BINARY, name=f"Y{sfx}",
        )
        Z = m.addVars(H, vtype=GRB.BINARY, name=f"Z{sfx}")
        U = m.addVars(
            [(p, h) for p in P_r for h in H],
            lb=1, ub=n_p, vtype=GRB.INTEGER, name=f"U{sfx}",
        )

        # ── Stage 2 constraints ───────────────────────────────────────────────

        # All three OR9 sessions are open (scope restriction)
        m.addConstrs((Z[h] == 1 for h in H), name=f"allopen{sfx}")

        # Each patient assigned to exactly one session
        m.addConstrs(
            (gp.quicksum(Y[p, h] for h in H) == 1 for p in P_r),
            name=f"assign{sfx}",
        )

        # Patient only in open sessions (trivially satisfied since Z[h]=1,
        # but kept explicit for completeness)
        m.addConstrs(
            (Y[p, h] <= Z[h] for p in P_r for h in H),
            name=f"open{sfx}",
        )

        # Routing: one departure from depot per open session
        m.addConstrs(
            (gp.quicksum(X[0, p, h] for p in P_r) == Z[h] for h in H),
            name=f"depot{sfx}",
        )

        # Flow conservation (out)
        m.addConstrs(
            (
                gp.quicksum(X[p, j, h] for j in P0_r if j != p) == Y[p, h]
                for p in P_r for h in H
            ),
            name=f"flow_out{sfx}",
        )

        # Flow conservation (in)
        m.addConstrs(
            (
                gp.quicksum(X[i, p, h] for i in P0_r if i != p) == Y[p, h]
                for p in P_r for h in H
            ),
            name=f"flow_in{sfx}",
        )

        # MTZ sub-tour elimination
        m.addConstrs(
            (
                U[p, h] - U[pp, h] + n_p * X[p, pp, h] <= n_p - 1
                for p in P_r for pp in P_r for h in H if p != pp
            ),
            name=f"mtz{sfx}",
        )

        # First patient in session starts at opening time
        for p in P_r:
            for h in H:
                m.addGenConstrIndicator(
                    X[0, p, h], True, A[p], GRB.LESS_EQUAL, float(t_start),
                    name=f"first{sfx}_{p}_{h}",
                )

        # Valid arc inequalities (tighten LP relaxation)
        for p in P_r:
            for pp in P_r:
                if p == pp:
                    continue
                for h in H:
                    m.addConstr(X[p, pp, h] <= Y[p, h],  name=f"vi_out{sfx}_{p}_{pp}_{h}")
                    m.addConstr(X[p, pp, h] <= Y[pp, h], name=f"vi_in{sfx}_{p}_{pp}_{h}")
        for p in P_r:
            for h in H:
                m.addConstr(X[0, p, h] <= Z[h], name=f"vi_dep{sfx}_{p}_{h}")

        # Blueprint linking constraint (Stage 1 ↔ Stage 2)
        for g in G:
            for h in H:
                pg = P_gr[g]
                if pg:
                    m.addConstr(
                        gp.quicksum(Y[p, h] for p in pg) <= N[g, h],
                        name=f"blueprint{sfx}_{g}_{h}",
                    )

        # ── Stage 3: one copy of cost variables + constraints per scenario ─────
        for s in S_r:
            sfx2 = f"_r{r}_s{s}"

            Sv  = m.addVars(P_r, lb=t_start, name=f"Sv{sfx2}")
            W   = m.addVars(P_r, lb=0, name=f"W{sfx2}")
            Iv  = m.addVars(
                [(p, pp) for p in P_r for pp in P_r if p != pp],
                lb=0, name=f"I{sfx2}",
            )
            O_h = m.addVars(H, lb=0, name=f"O{sfx2}")

            # Start-time propagation (Big-M)
            m.addConstrs(
                (
                    Sv[pp] >= Sv[p] + d_r[p, s] + c
                               - M_time * (1 - X[p, pp, h])
                    for p in P_r for pp in P_r for h in H if p != pp
                ),
                name=f"seq{sfx2}",
            )

            # Start time ≥ appointment time
            m.addConstrs((Sv[p] >= A[p] for p in P_r), name=f"lb_appt{sfx2}")

            # Waiting time
            m.addConstrs(
                (W[p] >= Sv[p] - A[p] for p in P_r),
                name=f"wait{sfx2}",
            )

            # Overtime per session
            m.addConstrs(
                (
                    O_h[h] >= Sv[p] + d_r[p, s] - t_close
                               - M_time * (1 - Y[p, h])
                    for p in P_r for h in H
                ),
                name=f"ot{sfx2}",
            )

            # Idle time between consecutive patients
            m.addConstrs(
                (
                    Iv[p, pp] >= (
                        Sv[pp] - Sv[p] - d_r[p, s] - c
                        - M_time * (
                            1 - gp.quicksum(X[p, pp, h] for h in H)
                        )
                    )
                    for p in P_r for pp in P_r if p != pp
                ),
                name=f"idle{sfx2}",
            )

            # Accumulate stage-3 cost weighted by π_r × π_{s|r}
            pi_s = pi_rs[s]
            obj_terms.append(
                pi_r * pi_s * (
                    beta["W"] * gp.quicksum(W[p] for p in P_r)
                    + beta["I"] * gp.quicksum(
                        Iv[p, pp] for p in P_r for pp in P_r if p != pp
                    )
                    + beta["O"] * gp.quicksum(O_h[h] for h in H)
                )
            )

    m.setObjective(gp.quicksum(obj_terms), GRB.MINIMIZE)
    m._N = N
    return m


# ─────────────────────────────────────────────────────────────────────────────
# 4. SOLUTIONS CSV (cumulative, one row per run)
# ─────────────────────────────────────────────────────────────────────────────

# The settings that identify a run; re-running the same settings overwrites the
# old row (latest wins) instead of duplicating it.
_SETTINGS_KEYS = ["seed", "n_realizations", "n_scenarios", "beta5", "method", "is_k"]

# Full wide column order for blueprint_solutions.csv: settings, outcome, then the
# flattened quota table N_<g>_<h>.
_SOLUTION_COLS = (
    _SETTINGS_KEYS
    + ["time_limit", "cluster_seed", "status", "mip_gap_pct",
       "objective", "slot_cost", "op_cost", "total_slots"]
    + [f"N_{g}_{h}" for g in _G for h in _H]
)


def append_solutions_csv(path: str, rows: list[dict]) -> None:
    """Append run rows to the cumulative solutions CSV, de-duplicating on settings.

    Mirrors the upsert-into-one-file pattern used by main.py:write_objective_breakdown:
    read the existing CSV (if any), concatenate, keep the latest row per settings
    tuple, sort and rewrite.
    """
    df_new = pd.DataFrame(rows, columns=_SOLUTION_COLS)

    if os.path.exists(path):
        try:
            df_old = pd.read_csv(path)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        except Exception as exc:
            print(f"Could not read existing {path} ({exc}); starting fresh.")
            df_all = df_new
    else:
        df_all = df_new

    df_all = (
        df_all
        .drop_duplicates(subset=_SETTINGS_KEYS, keep="last")
        .sort_values(_SETTINGS_KEYS)
        .reset_index(drop=True)
    )
    df_all.to_csv(path, index=False)
    print(f"Solutions written to: {path}  ({len(df_all)} rows total)")


# ─────────────────────────────────────────────────────────────────────────────
# 5. SINGLE-SEED SOLVE
# ─────────────────────────────────────────────────────────────────────────────

def run_one_seed(seed: int, type_to_group: dict, args, output_dir: str) -> dict:
    """Sample, build and solve the three-stage blueprint for one seed.

    Clustering is done once by the caller (fixed across seeds) and passed in via
    `type_to_group`, so only the realization/scenario sampling varies per seed and
    the resulting quotas remain comparable.  Writes the per-run blueprint.csv quota
    table and returns one flat result row for the cumulative solutions CSV.
    """
    R, S = args.n_realizations, args.n_scenarios

    print(f"\n{'=' * 70}")
    print(f"  Seed {seed}  |  R={R} realizations  |  S={S} scenarios each  |  "
          f"beta5={args.beta5}  method={args.method}  is_k={args.is_k}")
    print(f"{'=' * 70}")

    # ── Sample waiting-list realizations ──────────────────────────────────────
    realizations = sample_realizations(
        n_realizations=R, sessions_per_realization=3, seed=seed, or_filter=None,
    )
    for r, df_r in enumerate(realizations):
        n_by_g = {
            g: sum(1 for p in df_r["Patient ID"]
                   if type_to_group.get(
                       int(df_r.loc[df_r["Patient ID"] == p, "Surgery type"].iloc[0]), -1
                   ) == g)
            for g in _G
        }
        print(f"  r={r}: {len(df_r)} patients  {n_by_g}")

    # ── Generate duration scenarios per realization ───────────────────────────
    scen_per_real = []
    for r, df_r in enumerate(realizations):
        d_r, S_r, pi_r = generate_scenarios(
            df_r, n_scenarios=S, method=args.method,
            seed=seed + r * 1000, is_k=args.is_k,
        )
        scen_per_real.append((d_r, S_r, pi_r))

    # ── Build and solve the three-stage MILP ──────────────────────────────────
    print(f"\n  Building three-stage MILP ({R}x{S} = {R * S} leaves) ...")
    m = build_blueprint_model(
        realizations, scen_per_real, type_to_group, beta5=args.beta5,
    )
    m.setParam("OutputFlag", 1)
    m.setParam("TimeLimit", args.time_limit)
    m.setParam("MIPGap", args.mip_gap)
    m.setParam("Presolve", 2)
    m.setParam("Cuts", 2)
    m.setParam("Heuristics", 0.3)
    m.optimize()

    print(f"\n  Status: {m.Status}  |  Solutions found: {m.SolCount}")

    base = {k: getattr(args, k) for k in _SETTINGS_KEYS if k != "seed"}
    base.update(seed=seed, time_limit=args.time_limit, cluster_seed=args.cluster_seed,
                status=m.Status)

    if m.SolCount == 0:
        print("  No feasible solution found. Try reducing beta5 or the time limit.")
        base.update(mip_gap_pct=float("nan"), objective=float("nan"),
                    slot_cost=float("nan"), op_cost=float("nan"), total_slots=0)
        for g in _G:
            for h in _H:
                base[f"N_{g}_{h}"] = ""
        return base

    blueprint = {(g, h): round(m._N[g, h].X) for g in _G for h in _H}
    total_slots = sum(blueprint.values())
    slot_cost = args.beta5 * total_slots
    op_cost = m.ObjVal - slot_cost
    gap = m.MIPGap * 100

    print(f"  Objective: {m.ObjVal:.4f}  |  MIP gap: {gap:.2f}%")
    _print_blueprint_table(blueprint)
    print(f"  beta5*slots = {slot_cost:.2f}  |  Expected op. cost = {op_cost:.4f}")

    # Per-run quota table (last seed solved is what apply_blueprint.py reads).
    out_path = os.path.join(output_dir, "blueprint.csv")
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "session", "quota"])
        for g in _G:
            for h in _H:
                w.writerow([g, h, blueprint[g, h]])
    print(f"  Blueprint quota table written to: {out_path}")

    base.update(mip_gap_pct=gap, objective=m.ObjVal, slot_cost=slot_cost,
                op_cost=op_cost, total_slots=total_slots)
    for g in _G:
        for h in _H:
            base[f"N_{g}_{h}"] = blueprint[g, h]
    return base


def _print_blueprint_table(blueprint: dict) -> None:
    """Pretty-print the N[g, h] quota table with row/column totals."""
    print("\n  === Blueprint  N[g, h] ===")
    header = ["Subgroup"] + [f"Session {h}" for h in _H] + ["Row total"]
    col_w = 12
    print("  " + " ".join(f"{c:<{col_w}}" for c in header))
    print("  " + "-" * (col_w * len(header)))
    for g in _G:
        vals = [blueprint[g, h] for h in _H]
        print("  " + " ".join(f"{str(v):<{col_w}}" for v in [g] + vals + [sum(vals)]))
    totals = ["Total"] + [sum(blueprint[g, h] for g in _G) for h in _H] + [sum(blueprint.values())]
    print("  " + " ".join(f"{str(v):<{col_w}}" for v in totals))


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step 5: Three-stage stochastic blueprint for OR9 ORT scheduling."
    )
    parser.add_argument("--n-realizations", type=int, default=3,
                        help="Waiting-list realizations R (default 3)")
    parser.add_argument("--n-scenarios",    type=int, default=5,
                        help="Duration scenarios S per realization (default 5)")
    parser.add_argument("--seed",           type=int, nargs="+", default=[42],
                        help="One or more seeds; each solved and stored as a row "
                             "(consistency study). Only sampling varies per seed.")
    parser.add_argument("--cluster-seed",   type=int, default=42,
                        help="Fixed seed for k-means S/M/L clustering, so quotas "
                             "stay comparable across --seed values (default 42).")
    parser.add_argument("--beta5",          type=float, default=1.0,
                        help="Blueprint slot penalty beta5 (default 1.0)")
    parser.add_argument("--time-limit",     type=int,   default=300)
    parser.add_argument("--mip-gap",        type=float, default=0.05)
    parser.add_argument("--method",         default="is",
                        choices=["random", "lhs", "is"])
    parser.add_argument("--is-k",           type=float, default=1.0)
    parser.add_argument("--output-dir",     default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(_ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Cluster surgery types ONCE (fixed across seeds) ────────────────────
    print("\n[1] Clustering ORT surgery types from ort_patient_data.csv "
          f"(cluster_seed={args.cluster_seed}) ...")
    type_to_group, centroids = cluster_surgery_types(k=3, random_state=args.cluster_seed)
    print(f"  {len(type_to_group)} surgery types assigned to S/M/L clusters.")
    print(centroids.to_string(index=False))

    clust_path = os.path.join(output_dir, "blueprint_clusters.csv")
    centroids.to_csv(clust_path, index=False)
    print(f"  Written to: {clust_path}")

    # ── 2. Solve for each seed ────────────────────────────────────────────────
    rows = [run_one_seed(seed, type_to_group, args, output_dir) for seed in args.seed]

    append_solutions_csv(
        os.path.join(output_dir, "blueprint_solutions.csv"), rows,
    )

    # ── 3. Consistency summary across seeds ───────────────────────────────────
    if len(args.seed) > 1:
        print(f"\n{'=' * 70}")
        print("  Consistency summary across seeds")
        print(f"{'=' * 70}")
        hdr = f"  {'seed':>6}  {'status':>6}  {'obj':>10}  {'slots':>6}  " \
              + "  ".join(f"N_{g}_{h}" for g in _G for h in _H)
        print(hdr)
        for r in rows:
            quotas = "  ".join(f"{str(r[f'N_{g}_{h}']):>5}" for g in _G for h in _H)
            obj = r["objective"]
            obj_s = f"{obj:10.3f}" if isinstance(obj, float) and obj == obj else f"{'n/a':>10}"
            print(f"  {r['seed']:>6}  {r['status']:>6}  {obj_s}  "
                  f"{r['total_slots']:>6}  {quotas}")


if __name__ == "__main__":
    main()

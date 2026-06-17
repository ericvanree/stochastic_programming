# %% [markdown]
# # Two-Stage Stochastic OR Scheduling — Gurobi
#
# Implements the **deterministic equivalent (DE)** MILP for stochastic
# operating-room scheduling.
#
# ### Two-stage structure
# | Stage | Variables | Decided |
# |-------|-----------|---------|
# | First | $A_p, X_{pp'h}, Y_{ph}, Z_h, V_{hq}, D_{hq}, U_{ph}$ | Before scenario realisation |
# | Second | $S_{ps}, W_{ps}, I_{hs}, O_{hs}$ | Per scenario $s$ |
#
# ### Objective
# $$\min \sum_{s} \pi_s \bigl[\beta_1 \sum_p W_{ps} + \beta_2 \sum_h I_{hs} + \beta_3 \sum_h O_{hs}\bigr] + \beta_4 \sum_h \sum_q D_{hq}$$
#
# ### Three progressive steps (from `formula.tex`)
# 1. **Timing only** — sequencing $X_{pp'h}$ pre-fixed
# 2. **Sequencing + timing** — assignment $Y_{ph}$ pre-fixed
# 3. **Full MILP** — all first-stage variables free

# %% [markdown]
# ## 1. Imports

# %%
import re

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd

# %% [markdown]
# ## 2. Problem instance — loaded from `input/Question1Data.csv`
#
# One deterministic scenario using the **planned (expected) surgery durations**.

# %%
# ── Load CSV ───────────────────────────────────────────────────────────────────
DATA_FILE = r"input\Question1Data.csv"
df = pd.read_csv(DATA_FILE)
print(f"Loaded {len(df)} patients across {df['Session ID'].nunique()} sessions")

# ── Sets ──────────────────────────────────────────────────────────────────────
P           = df["Patient ID"].tolist()                      # patients
session_ids = sorted(df["Session ID"].unique().tolist())     # original IDs (6)
H           = list(range(len(session_ids)))                  # sessions 0..5
h_of        = {sid: h for h, sid in enumerate(session_ids)}  # session_id → index
P0          = [0] + P                                        # depot (0) ∪ patients
SPECS       = sorted(df["Specialty"].unique().tolist())      # e.g. ['CHI', 'ORT']

# ── Single deterministic scenario: planned (expected) duration ─────────────────
S  = [0]
pi = {0: 1.0}
d  = {(row["Patient ID"], 0): float(row["Planned surgery duration"])
      for _, row in df.iterrows()}

# ── Scalar parameters ─────────────────────────────────────────────────────────
t_start = 480      # session opening  (08:00 in min from midnight)
t_close = 960      # session closing  (16:00 in min from midnight)
c       = 15       # changeover / cleaning time between patients (min)
M_BIG   = 5_000    # big-M  (safely above any feasible time horizon)

# Objective weights β₁ (wait) β₂ (idle) β₃ (overtime) β₄ (specialty mismatch)
beta = {"W": 0.6, "I": 0.2, "O": 0.2, "D": 100.0}

# ── Specialty membership ───────────────────────────────────────────────────────
specialty_of = {row["Patient ID"]: row["Specialty"] for _, row in df.iterrows()}
q_pq         = {(p, q): int(specialty_of[p] == q) for p in P for q in SPECS}

# ── Patient → session assignment (from CSV) ────────────────────────────────────
patient_to_h = {
    row["Patient ID"]: h_of[row["Session ID"]] for _, row in df.iterrows()
}

# ── Within-session sequences from "Session-sequence position" column ───────────
def parse_pos(label: str) -> int:
    """'S189-P3' → 3"""
    return int(re.search(r'P(\d+)', label).group(1))

session_sequences: dict[int, list] = {h: [] for h in H}
for _, row in df.iterrows():
    h   = h_of[row["Session ID"]]
    pos = parse_pos(row["Session-sequence position"])
    session_sequences[h].append((pos, row["Patient ID"]))
for h in H:
    session_sequences[h].sort()
    session_sequences[h] = [p for _, p in session_sequences[h]]

# ── Display summary ────────────────────────────────────────────────────────────
print(f"\nSpecialties : {SPECS}")
print(f"Sessions    : {[f'{h}→ID{session_ids[h]}' for h in H]}")
print("\nSession sequences (from CSV):")
for h in H:
    seq = session_sequences[h]
    print(f"  Session {h} (ID={session_ids[h]:4d}): {seq}  ({len(seq)} patients)")

df_show = df[["Patient ID", "Specialty", "Session ID",
              "Session-sequence position", "Planned surgery duration"]].copy()
df_show["_pos"] = df_show["Session-sequence position"].apply(parse_pos)
df_show = df_show.sort_values(["Session ID", "_pos"]).drop(columns=["_pos"])
print("\nPatient data:")
print(df_show.to_string(index=False))

# %% [markdown]
# ## 3. Model builder

# %%
def build_model(name, fixed_X=None, fixed_Y=None):
    """
    Build the two-stage stochastic OR scheduling deterministic equivalent.

    Parameters
    ----------
    name    : str   — Gurobi model name / step label
    fixed_X : dict  — optional {(i, j, h): 0/1} — fixes the entire X matrix
                      (Step 1: timing only)
    fixed_Y : dict  — optional {(p, h): 0/1}    — fixes the entire Y matrix
                      (Step 2: sequencing + timing)

    Returns
    -------
    gp.Model with variable references stored as attributes _A, _X, _Y, _Z,
    _V, _D, _S, _W, _I, _O.
    """
    m = gp.Model(name)

    # ── First-stage variables ──────────────────────────────────────────────────
    # Appointment times
    A = m.addVars(P, lb=t_start, ub=t_close, name="A")

    # Sequencing: X[i,j,h] = 1 if patient j directly follows i in session h
    #   i, j ∈ P₀ = {0} ∪ P  (0 = depot);  i ≠ j
    X = m.addVars(
        [(i, j, h) for i in P0 for j in P0 for h in H if i != j],
        vtype=GRB.BINARY, name="X",
    )

    # Assignment: Y[p,h] = 1 if patient p is assigned to session h
    Y = m.addVars(
        [(p, h) for p in P for h in H],
        vtype=GRB.BINARY, name="Y",
    )

    # Session opening: Z[h] = 1 if session h is opened
    Z = m.addVars(H, vtype=GRB.BINARY, name="Z")

    # Specialty assignment: V[h,q] = 1 if session h is assigned specialty q
    V = m.addVars(
        [(h, q) for h in H for q in SPECS],
        vtype=GRB.BINARY, name="V",
    )

    # Specialty violation count (integer)
    D = m.addVars(
        [(h, q) for h in H for q in SPECS],
        lb=0, vtype=GRB.INTEGER, name="D",
    )

    # MTZ position variable: position of patient p in session h
    U = m.addVars(
        [(p, h) for p in P for h in H],
        lb=1, ub=len(P), vtype=GRB.INTEGER, name="U",
    )

    # ── Second-stage variables (one copy per scenario) ─────────────────────────
    # Start times
    Sv = m.addVars([(p, s) for p in P for s in S], lb=0, name="S")
    # Waiting time
    W  = m.addVars([(p, s) for p in P for s in S], lb=0, name="W")
    # Idle time between consecutive patient pairs
    Iv = m.addVars([(p, pp, s) for p in P for pp in P for s in S if p != pp], lb=0, name="I")
    # Overtime per session
    Ov = m.addVars([(h, s) for h in H for s in S], lb=0, name="O")

    # ── Objective ──────────────────────────────────────────────────────────────
    m.setObjective(
        gp.quicksum(
            pi[s] * (
                beta["W"] * gp.quicksum(W[p, s] for p in P)
                + beta["I"] * gp.quicksum(Iv[p, pp, s] for p in P for pp in P if p != pp)
                + beta["O"] * gp.quicksum(Ov[h, s] for h in H)
            )
            for s in S
        )
        + beta["D"] * gp.quicksum(D[h, q] for h in H for q in SPECS),
        GRB.MINIMIZE,
    )

    # ── Fix variables when requested ───────────────────────────────────────────
    if fixed_X is not None:
        for key, val in fixed_X.items():
            X[key].lb = X[key].ub = float(val)

    if fixed_Y is not None:
        for key, val in fixed_Y.items():
            Y[key].lb = Y[key].ub = float(val)

    # ── First-stage constraints ────────────────────────────────────────────────

    # Each patient assigned to exactly one session
    m.addConstrs(
        (gp.quicksum(Y[p, h] for h in H) == 1 for p in P),
        name="assign",
    )

    # Patient only assigned to an open session  (Y[p,h] ≤ Z[h])
    m.addConstrs(
        (Y[p, h] <= Z[h] for p in P for h in H),
        name="open",
    )

    # Exactly one specialty per open session; none for closed sessions
    m.addConstrs(
        (gp.quicksum(V[h, q] for q in SPECS) == Z[h] for h in H),
        name="spec",
    )

    # Specialty violation counting:
    # Σ_p q_pq * Y[p,h]  ≤  M * V[h,q] + D[h,q]
    m.addConstrs(
        (
            gp.quicksum(q_pq[p, q] * Y[p, h] for p in P)
            <= M_BIG * V[h, q] + D[h, q]
            for h in H for q in SPECS
        ),
        name="viol",
    )

    # Routing — exactly one departure from depot per open session
    m.addConstrs(
        (gp.quicksum(X[0, p, h] for p in P) == Z[h] for h in H),
        name="depot",
    )

    # Routing — flow conservation (out) for each patient
    # Σ_{j ∈ P₀, j≠p} X[p,j,h]  =  Y[p,h]
    m.addConstrs(
        (
            gp.quicksum(X[p, j, h] for j in P0 if j != p) == Y[p, h]
            for p in P for h in H
        ),
        name="flow_out",
    )

    # Routing — flow conservation (in) for each patient
    # Σ_{i ∈ P₀, i≠p} X[i,p,h]  =  Y[p,h]
    m.addConstrs(
        (
            gp.quicksum(X[i, p, h] for i in P0 if i != p) == Y[p, h]
            for p in P for h in H
        ),
        name="flow_in",
    )

    # MTZ sub-tour elimination (Miller-Tucker-Zemlin)
    # U[p,h] - U[p',h] + |P| * X[p,p',h]  ≤  |P| - 1
    m.addConstrs(
        (
            U[p, h] - U[pp, h] + len(P) * X[p, pp, h] <= len(P) - 1
            for p in P for pp in P for h in H if p != pp
        ),
        name="mtz",
    )

    # First patient in a session is scheduled at opening time:
    # A[p]  ≤  t_start + M·(1 − Σ_h X[0,p,h])
    m.addConstrs(
        (
            A[p] <= t_start + M_BIG * (1 - gp.quicksum(X[0, p, h] for h in H))
            for p in P
        ),
        name="start_first",
    )

    # ── Second-stage constraints (replicated over all scenarios) ───────────────

    # Start time propagation: if p' directly follows p, respect duration + changeover
    # S[p',s]  ≥  S[p,s] + d[p,s] + c  −  M·(1 − X[p,p',h])
    m.addConstrs(
        (
            Sv[pp, s] >= Sv[p, s] + d[p, s] + c - M_BIG * (1 - X[p, pp, h])
            for p in P for pp in P for h in H for s in S if p != pp
        ),
        name="seq",
    )

    # Appointment lower bound: patient cannot start before their appointment
    # S[p,s]  ≥  A[p]
    m.addConstrs(
        (Sv[p, s] >= A[p] for p in P for s in S),
        name="lb_appt",
    )

    # Waiting time: excess of actual start over appointment time
    # W[p,s]  ≥  S[p,s] − A[p]
    m.addConstrs(
        (W[p, s] >= Sv[p, s] - A[p] for p in P for s in S),
        name="wait",
    )

    # Overtime: finish beyond session closing time
    # O[h,s]  ≥  S[p,s] + d[p,s] − t_close  −  M·(1 − Y[p,h])
    m.addConstrs(
        (
            Ov[h, s] >= Sv[p, s] + d[p, s] - t_close - M_BIG * (1 - Y[p, h])
            for p in P for h in H for s in S
        ),
        name="overtime",
    )

    # Idle time between consecutive patients p → p':
    # I[p,p',s]  ≥  S[p',s] − S[p,s] − d[p,s] − c − M·(1 − Σ_h X[p,p',h])
    m.addConstrs(
        (
            Iv[p, pp, s] >= (
                Sv[pp, s] - Sv[p, s] - d[p, s] - c
                - M_BIG * (1 - gp.quicksum(X[p, pp, h] for h in H))
            )
            for p in P for pp in P for s in S if p != pp
        ),
        name="idle",
    )

    # Attach variable references for post-solve access
    m._A, m._X, m._Y, m._Z = A, X, Y, Z
    m._V, m._D = V, D
    m._S, m._W, m._I, m._O = Sv, W, Iv, Ov

    return m


# %% [markdown]
# ## 4. Solution printer

# %%
def print_solution(m, step_name):
    """Print a human-readable summary of a solved model."""
    if m.SolCount == 0:
        print(f"\n[{step_name}] No feasible solution found (status={m.Status})")
        return

    gap = m.MIPGap * 100

    print(f"\n{'=' * 60}")
    print(f"  {step_name}")
    print(f"  Objective = {m.ObjVal:.2f}   MIP gap = {gap:.2f}%")
    print(f"{'=' * 60}")

    # Sessions
    print("\nSessions:")
    for h in H:
        if m._Z[h].X > 0.5:
            assigned = [p for p in P if m._Y[p, h].X > 0.5]
            spec = next((q for q in SPECS if m._V[h, q].X > 0.5), "—")
            # Reconstruct ordered sequence from X arcs
            seq, cur = [], 0  # start traversal from depot
            for _ in range(len(assigned)):
                nxt = next((j for j in P if j != cur and m._X[cur, j, h].X > 0.5), None)
                if nxt is None:
                    break
                seq.append(nxt)
                cur = nxt
            print(f"  Session {h}: specialty={spec}  sequence={seq}")
        else:
            print(f"  Session {h}: CLOSED")

    # Appointment times
    print("\nAppointment times (min from midnight):")
    for p in P:
        print(f"  Patient {p}: {m._A[p].X:6.1f}  ({m._A[p].X / 60:4.2f}h)")

    # Per-scenario second-stage costs
    print(
        f"\nPer-scenario second-stage costs"
        f"  (β_W={beta['W']}, β_I={beta['I']}, β_O={beta['O']}):"
    )
    for s in S:
        wt = sum(m._W[p, s].X for p in P)
        it = sum(m._I[p, pp, s].X for p in P for pp in P if p != pp)
        ot = sum(m._O[h, s].X for h in H)
        weighted = beta["W"] * wt + beta["I"] * it + beta["O"] * ot
        print(
            f"  s={s}: wait={wt:6.1f}  idle={it:6.1f}  OT={ot:5.1f}"
            f"  → weighted={weighted:.1f}"
        )

    total_D = sum(m._D[h, q].X for h in H for q in SPECS)
    print(f"\nSpecialty violations (D): {total_D:.0f}")


# %% [markdown]
# ## 5. Step 1 — Timing only
#
# Both the **session assignment** and the **within-session sequence** are fixed
# to the values read from the CSV.  Only appointment times $A_p$ and the
# second-stage variables $(S, W, I, O)$ are optimised.

# %%
# Build fixed X from CSV sequences:
#   depot(0) → p1 → p2 → ... → pn → depot(0)  for each session h
# All other arcs are set to 0.
fixed_X_s1 = {(i, j, h): 0 for i in P0 for j in P0 for h in H if i != j}
for h in H:
    seq = session_sequences[h]
    fixed_X_s1[0, seq[0], h] = 1                      # depot → first patient
    for k in range(len(seq) - 1):
        fixed_X_s1[seq[k], seq[k + 1], h] = 1         # sequential arcs
    fixed_X_s1[seq[-1], 0, h] = 1                     # last patient → depot (flow balance)

m1 = build_model("Step1_TimingOnly", fixed_X=fixed_X_s1)
m1.setParam("TimeLimit", 120)
m1.optimize()
print_solution(m1, "Step 1 — Timing only (session + sequence fixed from CSV)")

# %% [markdown]
# ## 6. Step 2 — Sequencing + timing
#
# Only the **session assignment** ($Y_{ph}$) is fixed to the CSV values.
# The within-session sequence and appointment times are optimised freely.

# %%
# Fix Y from CSV assignment; sequence (X) is free.
fixed_Y_s2 = {(p, h): 0 for p in P for h in H}
for p in P:
    fixed_Y_s2[p, patient_to_h[p]] = 1

m2 = build_model("Step2_SeqTiming", fixed_Y=fixed_Y_s2)
m2.setParam("TimeLimit", 120)
m2.optimize()
print_solution(m2, "Step 2 — Sequencing + timing (session fixed from CSV, sequence free)")

# %% [markdown]
# ## 7. Step 3 — Full MILP
#
# All first-stage decisions are free.  Gurobi optimises session assignment,
# sequence, and appointment times simultaneously.

# %%
m3 = build_model("Step3_FullMILP")
m3.setParam("TimeLimit", 300)
m3.setParam("MIPGap", 0.01)
m3.optimize()
print_solution(m3, "Step 3 — Full MILP (all decisions free)")

# %% [markdown]
# ## 8. Objective comparison
#
# Step 3 ≤ Step 2 ≤ Step 1 because each step relaxes constraints:
# more freedom ⟹ lower or equal optimal objective.

# %%
print("\nObjective value comparison:")
for label, model in [
    ("Step 1  (X fixed, timing only)   ", m1),
    ("Step 2  (Y fixed, seq + timing)  ", m2),
    ("Step 3  (full MILP)              ", m3),
]:
    if model.SolCount > 0:
        print(f"  {label}: {model.ObjVal:.4f}")
    else:
        print(f"  {label}: no feasible solution")

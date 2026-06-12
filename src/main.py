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
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd

np.random.seed(42)

# %% [markdown]
# ## 2. Problem instance

# %%
# ── Sets ──────────────────────────────────────────────────────────────────────
P      = [1, 2, 3, 4]            # patients
H      = [0, 1]                   # OR sessions
SPECS  = ["CHI", "ORT", "KNO"]   # surgical specialties  (Q)
n_scen = 5                        # number of SAA scenarios
S      = list(range(n_scen))      # scenario indices
P0     = [0] + P                  # P₀ = depot (0) ∪ patients

# ── Scalar parameters ─────────────────────────────────────────────────────────
t_start = 480      # session opening   (08:00 in min from midnight)
t_close = 960      # session closing   (16:00 in min from midnight)
c       = 15       # changeover / cleaning time between patients (min)
M_BIG   = 3_000    # big-M constant  (must exceed max feasible time horizon)

# Objective weights β₁ (wait) β₂ (idle) β₃ (overtime) β₄ (specialty mismatch)
beta = {"W": 1.0, "I": 0.5, "O": 3.0, "D": 5.0}

# Uniform scenario probabilities  Σ πₛ = 1
pi = {s: 1.0 / n_scen for s in S}

# Patient specialties:  q_pq[(p, q)] = 1  if patient p belongs to specialty q
specialty_of = {1: "CHI", 2: "CHI", 3: "ORT", 4: "KNO"}
q_pq = {(p, q): int(specialty_of[p] == q) for p in P for q in SPECS}

# Surgery durations  d[(p, s)]  drawn from LogNormal around mean_dur[p] minutes
mean_dur = {1: 90, 2: 70, 3: 110, 4: 60}
d = {}
for p in P:
    draws = np.random.lognormal(mean=np.log(mean_dur[p]), sigma=0.20, size=n_scen)
    for idx, s in enumerate(S):
        d[p, s] = float(draws[idx])

# Display duration table
df_dur = pd.DataFrame(
    {f"s={s}": [round(d[p, s], 1) for p in P] for s in S},
    index=[f"p={p}" for p in P],
)
print("Surgery durations (minutes):")
print(df_dur.to_string())

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
    # Idle time per session
    Iv = m.addVars([(h, s) for h in H for s in S], lb=0, name="I")
    # Overtime per session
    Ov = m.addVars([(h, s) for h in H for s in S], lb=0, name="O")

    # ── Objective ──────────────────────────────────────────────────────────────
    m.setObjective(
        gp.quicksum(
            pi[s] * (
                beta["W"] * gp.quicksum(W[p, s] for p in P)
                + beta["I"] * gp.quicksum(Iv[h, s] for h in H)
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

    # First patient in session starts no earlier than session opening
    # S[p,s]  ≥  t_start  −  M·(1 − X[0,p,h])
    m.addConstrs(
        (
            Sv[p, s] >= t_start - M_BIG * (1 - X[0, p, h])
            for p in P for h in H for s in S
        ),
        name="first_pat",
    )

    # Appointment lower bound: patient cannot start before their appointment
    # S[p,s]  ≥  A[p]  −  M·(1 − Y[p,h])
    m.addConstrs(
        (
            Sv[p, s] >= A[p] - M_BIG * (1 - Y[p, h])
            for p in P for h in H for s in S
        ),
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

    # Idle time: session window minus utilisation, corrected for overtime
    # I[h,s]  ≥  (t_close − t_start)·Z[h]
    #            − Σ_p d[p,s]·Y[p,h]
    #            − c·(Σ_p Y[p,h] − Z[h])
    #            + O[h,s]
    m.addConstrs(
        (
            Iv[h, s] >= (
                (t_close - t_start) * Z[h]
                - gp.quicksum(d[p, s] * Y[p, h] for p in P)
                - c * (gp.quicksum(Y[p, h] for p in P) - Z[h])
                + Ov[h, s]
            )
            for h in H for s in S
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
                nxt = next((j for j in P if m._X[cur, j, h].X > 0.5), None)
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
        it = sum(m._I[h, s].X for h in H)
        ot = sum(m._O[h, s].X for h in H)
        weighted = beta["W"] * wt + beta["I"] * it + beta["O"] * ot
        print(
            f"  s={s}: wait={wt:6.1f}  idle={it:6.1f}  OT={ot:5.1f}"
            f"  → weighted={weighted:.1f}"
        )

    total_D = sum(m._D[h, q].X for h in H for q in SPECS)
    print(f"\nSpecialty violations (D): {total_D:.0f}")


# %% [markdown]
# ## 5. Step 1 — Timing only (sequencing $X_{pp'h}$ pre-fixed)
#
# The within-session order is given: session 0 handles patients [1, 2],
# session 1 handles patients [3, 4].  Only appointment times $A_p$ and
# second-stage variables are optimised.

# %%
# Build predefined sequencing:
#   session 0:  depot(0) → 1 → 2 → depot(0)
#   session 1:  depot(0) → 3 → 4 → depot(0)
fixed_X_s1 = {(i, j, h): 0 for i in P0 for j in P0 for h in H if i != j}
# Session 0
fixed_X_s1[0, 1, 0] = 1
fixed_X_s1[1, 2, 0] = 1
fixed_X_s1[2, 0, 0] = 1  # return to depot
# Session 1
fixed_X_s1[0, 3, 1] = 1
fixed_X_s1[3, 4, 1] = 1
fixed_X_s1[4, 0, 1] = 1  # return to depot

m1 = build_model("Step1_TimingOnly", fixed_X=fixed_X_s1)
m1.setParam("TimeLimit", 60)
m1.optimize()
print_solution(m1, "Step 1 — Timing only (X fixed)")

# %% [markdown]
# ## 6. Step 2 — Sequencing + timing ($Y_{ph}$ pre-fixed)
#
# Patient-to-session assignment is fixed: patients {1, 2} → session 0,
# patients {3, 4} → session 1.  Within-session sequence is optimised.

# %%
fixed_Y_s2 = {(p, h): 0 for p in P for h in H}
fixed_Y_s2[1, 0] = 1
fixed_Y_s2[2, 0] = 1
fixed_Y_s2[3, 1] = 1
fixed_Y_s2[4, 1] = 1

m2 = build_model("Step2_SeqTiming", fixed_Y=fixed_Y_s2)
m2.setParam("TimeLimit", 60)
m2.optimize()
print_solution(m2, "Step 2 — Sequencing + timing (Y fixed)")

# %% [markdown]
# ## 7. Step 3 — Full MILP
#
# All first-stage decisions are free.  Gurobi solves the complete
# deterministic equivalent simultaneously.

# %%
m3 = build_model("Step3_FullMILP")
m3.setParam("TimeLimit", 120)
m3.setParam("MIPGap", 0.01)
m3.optimize()
print_solution(m3, "Step 3 — Full MILP")

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

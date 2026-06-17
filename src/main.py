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
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))  # ensure src/ is on path

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from sampling import generate_scenarios

# %% [markdown]
# ## 2. Problem instance — loaded from `input/Question1Data.csv`
#
# **Scenario parameters** — change these to control sampling:
# - `N_SCENARIOS = 0` → deterministic (single scenario at expected duration)
# - `N_SCENARIOS = k` → draw k samples using the chosen method
# - `SAMPLING_METHOD` → `'expected'` | `'random'` | `'lhs'` | `'ortho'` | `'is'`

# %%
# ══ Scenario parameters ═════════════════════════════════════════════════════════════════
N_SCENARIOS     = 4           # 0 = deterministic expected duration
SAMPLING_METHOD = "random"    # 'expected' | 'random' | 'lhs' | 'ortho' | 'is'
RANDOM_SEED     = 42

# ══ Load CSV ═════════════════════════════════════════════════════════════════
DATA_FILE = r"input\Question1Data.csv"
df = pd.read_csv(DATA_FILE)
print(f"Loaded {len(df)} patients across {df['Session ID'].nunique()} sessions")

# ══ Sets ══════════════════════════════════════════════════════════════════════
P           = df["Patient ID"].tolist()
session_ids = sorted(df["Session ID"].unique().tolist())
H           = list(range(len(session_ids)))
h_of        = {sid: h for h, sid in enumerate(session_ids)}
P0          = [0] + P
SPECS       = sorted(df["Specialty"].unique().tolist())

# ══ Scenario generation ══════════════════════════════════════════════════════════════
d, S, pi = generate_scenarios(df, N_SCENARIOS, SAMPLING_METHOD, RANDOM_SEED)
print(f"Scenarios   : {len(S)}  (method='{SAMPLING_METHOD}')")
if len(S) <= 5:
    dur_rows = pd.DataFrame(
        {f"s={s}": [round(d[p, s], 1) for p in P] for s in S},
        index=P,
    )
    print("\nDuration matrix (minutes):")
    print(dur_rows.to_string())

# ══ Scalar parameters ═════════════════════════════════════════════════════════
t_start = 480      # session opening  (08:00 in min from midnight)
t_close = 960      # session closing  (16:00 in min from midnight)
c       = 10       # changeover / cleaning time between patients (min)
M_BIG   = 5_000    # big-M  (safely above any feasible time horizon)

# Objective weights β₁ (wait) β₂ (idle) β₃ (overtime) β₄ (specialty mismatch)
beta = {"W": 0.6, "I": 0.2, "O": 0.2, "D": 100.0}

# ══ Specialty membership ═══════════════════════════════════════════════════════
specialty_of = {row["Patient ID"]: row["Specialty"] for _, row in df.iterrows()}
q_pq         = {(p, q): int(specialty_of[p] == q) for p in P for q in SPECS}

# ══ Patient → session assignment (from CSV) ════════════════════════════════════
patient_to_h = {
    row["Patient ID"]: h_of[row["Session ID"]] for _, row in df.iterrows()
}

# ══ Within-session sequences from "Session-sequence position" column ═══════════
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

# ══ Display summary ════════════════════════════════════════════════════════════
print(f"\nSpecialties : {SPECS}")
print(f"Sessions    : {[f'{h}→ID{session_ids[h]}' for h in H]}")
print("\nSession sequences (from CSV):")
for h in H:
    seq = session_sequences[h]
    print(f"  Session {h} (ID={session_ids[h]:4d}): {seq}  ({len(seq)} patients)")

df_show = df[["Patient ID", "Specialty", "Session ID",
              "Session-sequence position", "expected_duration", "sigma_error"]].copy()
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
    d_penalty = beta["D"] * total_D
    print(f"\nSpecialty violations (D): {total_D:.0f}  → penalty = {d_penalty:.2f}"
          f"  (β_D={beta['D']})")
    print(f"Total objective = {m.ObjVal:.4f}"
          f"  (scenario costs + D penalty)")


# %% [markdown]
# ## 4b. Solution extraction & dashboard

# %%
# Specialty colour palette (consistent across all plots)
_SPEC_COLOR = {
    "CHI": "#4e79a7",
    "ORT": "#f28e2b",
    "KNO": "#59a14f",
    "—":   "#bab0ac",
}


def extract_solution(m):
    """Extract the optimised solution into a plain dict for visualisation."""
    if m.SolCount == 0:
        return None

    sessions = {}
    for h in H:
        is_open = m._Z[h].X > 0.5
        spec = next((q for q in SPECS if m._V[h, q].X > 0.5), "—") if is_open else "—"
        assigned = [p for p in P if m._Y[p, h].X > 0.5]
        seq, cur = [], 0
        for _ in range(len(assigned)):
            nxt = next((j for j in P if j != cur and m._X[cur, j, h].X > 0.5), None)
            if nxt is None:
                break
            seq.append(nxt)
            cur = nxt
        sessions[h] = {
            "open":       is_open,
            "specialty":  spec,
            "session_id": session_ids[h],
            "sequence":   seq,
        }

    patients = {}
    for p in P:
        h_assigned = next((h for h in H if m._Y[p, h].X > 0.5), None)
        patients[p] = {
            "session":     h_assigned,
            "specialty":   specialty_of[p],
            "appointment": m._A[p].X,
            "duration":    {s: d[p, s] for s in S},
            "start":       {s: m._S[p, s].X for s in S},
            "wait":        {s: m._W[p, s].X for s in S},
        }

    return {"sessions": sessions, "patients": patients, "scenarios": list(S)}


def plot_dashboard(sol, title):
    """Interactive Gantt-style session timeline. Scenario selector via dropdown."""
    if sol is None:
        print("No solution to plot.")
        return

    scenarios = sol["scenarios"]
    open_h    = [h for h in H if sol["sessions"][h]["open"]]

    def row_label(h):
        sess = sol["sessions"][h]
        return f"Session {h}  |  {sess['specialty']}  |  ID {sess['session_id']}"

    fig = go.Figure()

    # ── 1. Background: session window (always visible) ─────────────────────────
    for h in open_h:
        fig.add_trace(go.Bar(
            x=[t_close - t_start], y=[row_label(h)], base=[t_start],
            orientation="h",
            marker=dict(color="rgba(200,200,200,0.35)", line=dict(width=0)),
            hoverinfo="skip", showlegend=False, name="_bg",
        ))
    n_bg = len(open_h)

    # ── 2. Per-scenario traces ──────────────────────────────────────────────────
    scenario_trace_ranges = {}
    for s in scenarios:
        t0 = len(fig.data)
        for h in open_h:
            seq = sol["sessions"][h]["sequence"]
            for p in seq:
                pat   = sol["patients"][p]
                start = pat["start"][s]
                dur   = pat["duration"][s]
                appt  = pat["appointment"]
                wait  = pat["wait"][s]
                pspec = pat["specialty"]
                color = _SPEC_COLOR.get(pspec, "#bab0ac")

                # Surgery bar
                fig.add_trace(go.Bar(
                    x=[dur], y=[row_label(h)], base=[start],
                    orientation="h",
                    marker=dict(color=color, opacity=0.85,
                                line=dict(color="white", width=1)),
                    text=(f"<b>Patient {p}</b><br>"
                          f"Specialty : {pspec}<br>"
                          f"Appt      : {int(appt)} min  ({appt/60:.2f} h)<br>"
                          f"Start     : {int(start)} min  ({start/60:.2f} h)<br>"
                          f"Duration  : {int(dur)} min<br>"
                          f"Wait      : {wait:.1f} min"),
                    hovertemplate="%{text}<extra></extra>",
                    visible=(s == scenarios[0]),
                    showlegend=False, name=f"s{s}_p{p}",
                ))

                # Appointment marker: 2-min wide black bar
                fig.add_trace(go.Bar(
                    x=[2], y=[row_label(h)], base=[appt - 1],
                    orientation="h",
                    marker=dict(color="black", opacity=1.0,
                                line=dict(width=0)),
                    hovertemplate=f"Appointment {p}: {int(appt)} min<extra></extra>",
                    visible=(s == scenarios[0]),
                    showlegend=False, name=f"appt_s{s}_p{p}",
                ))

        scenario_trace_ranges[s] = (t0, len(fig.data))

    # ── 3. Legend dummy traces (always visible) ─────────────────────────────────
    leg_start = len(fig.data)
    shown_specs = sorted({sol["patients"][p]["specialty"] for p in P})
    for spec in shown_specs:
        fig.add_trace(go.Bar(
            x=[0], y=[""], orientation="h",
            marker_color=_SPEC_COLOR.get(spec, "#bab0ac"),
            name=spec, showlegend=True, visible=True, hoverinfo="skip",
        ))
    fig.add_trace(go.Bar(
        x=[0], y=[""], orientation="h",
        marker_color="black",
        name="Appointment", showlegend=True, visible=True, hoverinfo="skip",
    ))
    leg_end = len(fig.data)

    # ── 4. Dropdown buttons ────────────────────────────────────────────────────────
    n_total  = len(fig.data)
    always_on = list(range(n_bg)) + list(range(leg_start, leg_end))
    buttons = []
    for s in scenarios:
        s0, s1 = scenario_trace_ranges[s]
        vis = [False] * n_total
        for i in always_on + list(range(s0, s1)):
            vis[i] = True
        buttons.append(dict(
            label=f"Scenario {s}",
            method="update",
            args=[{"visible": vis},
                  {"title": f"{title}  —  Scenario {s}"}],
        ))

    # ── 5. Layout ───────────────────────────────────────────────────────────────
    tick_vals = list(range(t_start, t_close + 1, 60))
    tick_text = [f"{v // 60:02d}:00" for v in tick_vals]

    fig.update_layout(
        title=f"{title}  —  Scenario {scenarios[0]}",
        barmode="overlay",
        bargap=0.3,
        xaxis=dict(
            title="Time of day",
            tickmode="array",
            tickvals=tick_vals,
            ticktext=tick_text,
            range=[t_start - 15, t_close + 15],
            gridcolor="lightgray",
            showgrid=True,
        ),
        yaxis=dict(
            title="Session",
            autorange="reversed",
            tickfont=dict(size=11),
        ),
        updatemenus=[dict(
            buttons=buttons,
            direction="down",
            showactive=True,
            x=0.0, xanchor="left",
            y=1.13, yanchor="top",
        )] if len(scenarios) > 1 else [],
        height=180 + 80 * len(open_h),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            title="Legend",
            orientation="v",
            yanchor="top", y=1.0,
            xanchor="left", x=1.02,
        ),
        margin=dict(l=230, r=120, t=100, b=60),
    )

    fig.show()


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
sol1 = extract_solution(m1)
plot_dashboard(sol1, "Step 1 — Timing only")

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
sol2 = extract_solution(m2)
plot_dashboard(sol2, "Step 2 — Sequencing + timing")

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
sol3 = extract_solution(m3)
plot_dashboard(sol3, "Step 3 — Full MILP")

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

# Stochastic Operating-Room Scheduling

A two-stage stochastic program for operating-room (OR) scheduling under uncertain
surgery durations, solved as a **deterministic-equivalent MILP** with Gurobi.
The model assigns patients to OR sessions, sequences them within each session, and
sets appointment times so as to minimise the expected cost of patient waiting,
OR idle time, overtime, and specialty mismatch.

The code also implements the standard stochastic-programming evaluation suite —
Sample Average Approximation (SAA) convergence, the Value of the Stochastic
Solution (VSS), and the (Mean) Value of Perfect Information (EVPI/MVPI) — plus a
three-stage **blueprint** extension.


## Model at a glance

| Stage | Variables | Decided |
|-------|-----------|---------|
| First (here-and-now) | assignment $Y_{ph}$, sequencing $X_{ijh}$, appointment times $A_p$, session opening $Z_h$, specialty $V_{hq}$/$D_{hq}$, MTZ positions $U_{ph}$ | Before durations are known |
| Second (recourse) | start times $S_{ps}$, waiting $W_{ps}$, idle $I_{pp's}$, overtime $O_{hs}$ | Per scenario $s$ |

**Objective** (weights $\beta$):

$$\min \sum_{s}\pi_s\Big[\beta_W\textstyle\sum_p W_{ps} + \beta_I\sum_{p\neq p'} I_{pp's} + \beta_O\sum_h O_{hs}\Big] + \beta_D\sum_{h,q} D_{hq}$$

Default weights: `W = 0.6`, `I = 0.2`, `O = 0.2`, `D = 100.0` (specialty mismatch
is heavily penalised). Sessions run `08:00–16:00` with a `10`-minute changeover.

**Three progressive solve steps** (each warm-started from the previous):

1. **Timing only** — sequencing `X` fixed from the data; only appointment/recourse variables free.
2. **Sequencing + timing** — assignment `Y` fixed; sequence and timing free.
3. **Full MILP** — all first-stage decisions free.

Since each step relaxes constraints, `obj(Step 3) ≤ obj(Step 2) ≤ obj(Step 1)`.

## Scenario sampling

Surgery durations are drawn from a **truncated log-normal** with real-space mean
`expected_duration` and std `sigma_error`, truncated to `[μ − 2σ, μ + 2σ]`.
`generate_scenarios(df, n, method, seed, is_k)` returns `(d, S, pi)` and supports:

| Method | Description |
|--------|-------------|
| `expected` | Single deterministic scenario at the expected duration |
| `random` | Plain SAA — independent truncated log-normal draws |
| `lhs` | Latin Hypercube Sampling — CDF-stratified, one draw per equiprobable interval |
| `is` | Mixture Importance Sampling — three boundary-shifted proposals with IS weights (shift magnitude `is_k`) |

## Repository layout

```
src/
  main.py                       # Data loading, model builder (build_model), single-run solve + dashboards
  sampling.py                   # Scenario generation (generate_scenarios) + the 4 methods
  saa.py                        # SAA convergence study + analytic out-of-sample simulator
  vss.py                        # Value of the Stochastic Solution (EEV vs RP)
  mvpi.py                       # (Mean) Value of Perfect Information (EVPI/MVPI)
  blueprint.py                  # Three-stage blueprint: per-subgroup session quotas N[g,h]
  apply_blueprint.py            # Solve with/without a blueprint and compare the cost
  plot_saa_convergence.py       # Convergence plots (mean ± std band) per step
  plot_vss.py                   # EEV-vs-RP bar charts + VSS summary
  plot_breakdown.py             # Objective breakdown (W/I/O/D) stacked bars
  plot_blueprint_comparison.py  # Baseline-vs-blueprint comparison plots
input/                          # Patient data (sample_wl{4,7,10,13}.csv) + source spreadsheet
output/                         # Generated CSVs, PNGs and interactive HTML dashboards
requirements.txt                # gurobipy, numpy, pandas, plotly, matplotlib, scipy
```

Input patient files carry the columns `Patient ID, Specialty, Session ID,
Session-sequence position, expected_duration, sigma_error` (plus raw fields).
Available workloads are `wl4`, `wl7`, `wl10`, `wl13`.

## Installation

Requires Python 3.10+ and a valid **Gurobi licence** (`gurobipy`). Academic and
free licences cap model size, so increase the scenario count with care.

```powershell
# from the repository root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

(On macOS/Linux use `source .venv/bin/activate`.)

## Usage

### Single run + interactive schedule (`main.py`)

Solves Steps 1→3 for one workload and writes Gantt-style HTML dashboards and the
objective breakdown. It is a percent-cell (`# %%`) script that also runs
cell-by-cell in VS Code. Edit the parameter block near the top to configure a run:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `WORKLOAD` | `13` | Input file `input/sample_wl{4,7,10,13}.csv` |
| `N_SCENARIOS` | `20` | `0` = deterministic; `k > 0` = draw `k` training scenarios |
| `SAMPLING_METHOD` | `lhs` | `expected` \| `random` \| `lhs` \| `is` |
| `IS_K` | `0.5` | IS shift magnitude (ignored for other methods) |
| `RANDOM_SEED` | `42` | Controls all stochastic draws |
| `N_PRIME` | `10000` | Out-of-sample evaluation size |

```powershell
python src/main.py
```

### SAA convergence (`saa.py`)

Varies the training-scenario count `N`, repeats `M` replications per `N`, and
evaluates each solution out-of-sample on a fixed `N'`-scenario set.

```powershell
python src/saa.py --step 3 --wl 4 --n-start 5 --n-step 5 --n-max 30 `
                  --n-prime 10000 --m-reps 10 --start-seed 42 `
                  --methods random lhs is --is-k 0.5
```

Outputs `output/saa_wl{wl}_step{step}{method}.csv`
(columns `method, N, replication, obj_out_of_sample`) and interactive HTML plots.
Then render the report-style mean ± std band plots:

```powershell
python src/plot_saa_convergence.py --wl 4
```

### Value of the Stochastic Solution (`vss.py`)

Compares the EV (deterministic) policy against the SAA policy on a shared
out-of-sample set: `VSS = EEV − RP`.

```powershell
python src/vss.py --n-train 20 --method lhs --n-prime 10000 --seed 42
python src/plot_vss.py
```

Writes `output/vss_results.csv` and `output/vss_eev_rp_step{1,2,3}.png`.

### Value of Perfect Information (`mvpi.py`)

Compares the policy cost against the wait-and-see (perfect-information) cost per
evaluation scenario: `VPI(i) = z_policy(i) − z_perf(i)`.

```powershell
python src/mvpi.py --step 3 --wl 4 --n-train 20 --n-eval 10 --method lhs --seed 42
```

### Blueprint extension (`blueprint.py`, `apply_blueprint.py`)

A three-stage model that decides per-subgroup (S/M/L) session quotas `N[g,h]`
before the waiting list is known, then a comparison of scheduling with vs.
without those quotas:

```powershell
# 1) build a blueprint (writes output/blueprint.csv)
python src/blueprint.py --n-realizations 3 --n-scenarios 20 --seed 42 --method lhs

# 2) apply it and compare against the unconstrained baseline
python src/apply_blueprint.py --seed 42 --n-train 20 --method lhs --time-limit 600
python src/plot_blueprint_comparison.py
```

## Outputs

All artefacts are written to [output/](output/):

- **CSVs** — per-replication SAA results, cumulative objective breakdowns, VSS,
  MVPI, and blueprint results (cumulative files upsert by key, so re-running a
  workload replaces its own rows).
- **PNGs** — convergence, VSS, objective-breakdown, and blueprint-comparison
  figures used in the report.
- **HTML** — interactive Gantt-style schedule dashboards (`schedule_wl{wl}_step{n}.html`).

## Notes

- Gurobi variable references are attached to each model object (`m._A`, `m._X`, …)
  for post-solve access.
- Out-of-sample evaluation (`saa.simulate_schedule`) propagates start times
  analytically for a fixed first-stage policy, without re-solving the MILP, so
  evaluation is fast and identical across the SAA/VSS/MVPI tools.
- Patient ID `0` is a dummy depot used by the routing/sequencing formulation;
  sub-tours are eliminated with Miller–Tucker–Zemlin constraints.


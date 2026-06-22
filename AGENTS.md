# Stochastic OR Scheduling — Agent Instructions

## Project Overview

Two-stage stochastic operating-room (OR) scheduling solved as a **deterministic equivalent (DE) MILP** using Gurobi.  
The model assigns patients to OR sessions, sequences them within sessions, and optimises appointment times under uncertain surgery durations.

- **Stage 1** (before scenarios): assignment ($Y_{ph}$), sequencing ($X_{ijh}$), appointment times ($A_p$), session opening ($Z_h$), specialty assignment ($V_{hq}$, $D_{hq}$), MTZ positions ($U_{ph}$)  
- **Stage 2** (per scenario): start times ($S_{ps}$), waiting ($W_{ps}$), idle ($I_{hs}$), overtime ($O_{hs}$)

See [report/sections/formula.tex](report/sections/formula.tex) for the full mathematical model.

## Project Structure

```
src/
  main.py        # Problem data loading, model construction (build_model), solve & visualise
  sampling.py    # Scenario generation: generate_scenarios() dispatcher + 4 methods
input/
  sample_wl4.csv      # Patient data — workload 4  (ID, Specialty, Session ID, expected_duration, sigma_error)
  sample_wl7.csv      # Patient data — workload 7
  sample_wl10.csv     # Patient data — workload 10
  sample_wl13.csv     # Patient data — workload 13
output/          # Generated plots and results land here (gitkeep only)
report/          # LaTeX report (main.tex, formula.tex, etc.)
requirements.txt # gurobipy, numpy, pandas, plotly, scipy
```

## Running the Model

```powershell
# Activate the virtual environment first
.\.venv\Scripts\Activate.ps1

# Run the full pipeline
python src/main.py
```

The script is structured as a Jupyter-style percent-cell file (`# %%`) and can also be run cell-by-cell in VS Code with the Python extension.

## Key Parameters (top of `src/main.py`)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `WORKLOAD` | `4` | Selects input file: `4` \| `7` \| `10` \| `13` → `input/sample_wlX.csv` |
| `N_SCENARIOS` | `11` | `0` = deterministic (single expected-value scenario); `k > 0` = draw k samples |
| `SAMPLING_METHOD` | `"is"` | `"expected"` \| `"random"` \| `"lhs"` \| `"is"` |
| `RANDOM_SEED` | `42` | Controls all stochastic draws |

## Scenario Sampling (`src/sampling.py`)

Entry point: `generate_scenarios(df, n_scenarios, method, seed, is_k)` returns `(d, S, pi)`:
- `d` — duration dict `{(patient_id, scenario_idx): float}`
- `S` — list of scenario indices
- `pi` — probability dict `{scenario_idx: float}` (uniform for SAA/LHS; IS weights for `"is"`)

Durations are **truncated log-normal** with real-space mean = `expected_duration`, std = `sigma_error`, truncated to `[mu − 2σ, mu + 2σ]`.

| Method | Description |
|--------|-------------|
| `"expected"` | Single deterministic scenario at `expected_duration` |
| `"random"` | SAA — independent truncated log-normal draws |
| `"lhs"` | Latin Hypercube Sampling — CDF-stratified, one draw per equiprobable interval; independently permuted across patients |
| `"is"` | Mixture Importance Sampling — three-group boundary-shifted proposals with IS weights |

### Importance Sampling (`"is"`) and `is_k`

`generate_is(df, n_scenarios, rng, k=1.0)` partitions the `n_scenarios` into **three equal-sized groups** (sizes differ by at most 1 when `n % 3 ≠ 0`). Each group draws from a proposal whose real-space mean is shifted by `delta * sigma_p`, where `delta ∈ {-k, 0, +k}`:

| Group | Shift | Purpose |
|-------|-------|---------|
| 0 | `−k · σ_p` | Under-run tail (short surgery scenarios) |
| 1 | `0` | Nominal distribution |
| 2 | `+k · σ_p` | Over-run tail (long surgery scenarios) |

All three proposals share the **same truncation bounds** as the nominal distribution (`[mu_p − 2σ_p, mu_p + 2σ_p]`), ensuring `f(x) > 0` everywhere a sample can fall.

**IS weights** are computed as likelihood ratios in log-space per patient, summed across patients, then normalised with log-sum-exp:

```
log w_s = Σ_p [ log f_p(x_{p,s} | mu_p) − log g_p(x_{p,s} | mu_p + delta_s·sigma_p) ]
```

After normalisation the effective sample size `1 / Σ w²` is printed — use it to judge whether `is_k` is well-tuned.

**Tuning `is_k`:**
- `k = 1.0` (default) — shifts proposals by ±1 standard deviation; reasonable starting point.
- Larger `k` (e.g. 1.5–2.0) — places more weight on the tails; effective only if over/under-runs beyond ±σ are consequential.
- Too large `k` — weight degeneracy (one scenario dominates); watch effective n printed at runtime.
- `k = 0` — all three groups collapse to the nominal; equivalent to `"random"` but slower.

**`is_k` in the SAA CLI** — passed via `--is-k` (default `1.0`); stored in `args.is_k` and forwarded to every call of `generate_scenarios(..., is_k=args.is_k)`.

## SAA Convergence Analysis (`src/saa.py`)

Runs the full SAA study: varies `N` (training scenarios), repeats `M` replications per `N`, evaluates each solution out-of-sample with `N'` scenarios, streams results to CSV, and plots convergence.

```powershell
python src/saa.py --step 2 --wl 4 --n-start 10 --n-step 5 --n-max 30 \
                  --n-prime 1000 --m-reps 10 --start-seed 42 \
                  --methods random lhs is --is-k 1.0
```

### Key CLI arguments

| Argument | Default | Effect |
|----------|---------|--------|
| `--step` | required | `1` = timing only, `2` = seq+timing, `3` = full MILP |
| `--wl` | required | Waiting-list size: `4` \| `7` \| `10` \| `13` |
| `--n-start` | `10` | First training-scenario count |
| `--n-step` | `5` | Increment between N values |
| `--n-max` | required | Last training-scenario count (inclusive) |
| `--n-prime` | `1000` | Out-of-sample evaluation size N′ |
| `--m-reps` | `10` | SAA replications per N value |
| `--start-seed` | `42` | Master seed (controls eval sample + training draws) |
| `--methods` | all three | Space-separated subset of `random lhs is` |
| `--is-k` | `1.0` | IS shift magnitude k (ignored for `random`/`lhs`) |
| `--n-workers` | `1` | Parallel worker processes; Gurobi threads auto-scaled |
| `--time-limit` | `0` | Gurobi seconds per solve (0 = 120 s steps 1–2, 300 s step 3) |
| `--output-dir` | `output` | Directory for CSV + HTML plot |

### SAA outputs

- `output/saa_wl{wl}_step{step}.csv` — columns: `method, N, replication, obj_out_of_sample`
- `output/saa_convergence_wl{wl}_step{step}.html` — Plotly convergence chart (mean ± 95 % CI)

### SAA seed scheme

Each `(method, N, rep)` triple gets a deterministic unique seed:
```python
seed_rep = start_seed + method_idx * 100_000 + N * m_reps + rep
```
The evaluation sample always uses `start_seed` with `method="random"` — it is **fixed across all training methods**.

### Out-of-sample evaluation (`simulate_schedule`)

`simulate_schedule()` in `saa.py` analytically propagates start times for a fixed first-stage solution without re-solving the MILP:
- Reconstructs patient sequences from `X` arcs (depot → p1 → … → pk).
- Computes `S[p,s] = max(A[p], prev_finish)` per scenario.
- Returns the weighted objective under `pi_eval`.

## Model Builder

`build_model(name, fixed_X=None, fixed_Y=None)` in `main.py` supports three progressive solve steps:

1. **Timing only** — pass `fixed_X` (sequencing locked, only appointment times optimised)
2. **Sequencing + timing** — pass `fixed_Y` (assignment locked)
3. **Full MILP** — no fixed variables

## Objective Weights

```python
beta = {"W": 0.6, "I": 0.2, "O": 0.2, "D": 100.0}
# W = patient waiting, I = idle time, O = overtime, D = specialty mismatch penalty
```

## Conventions

- **Big-M**: `M_BIG = 5_000` (minutes). Do not tighten arbitrarily — it must exceed any feasible time horizon.
- **Depot node**: patient ID `0` is a dummy depot representing session start in the routing formulation.
- **MTZ**: sub-tour elimination uses Miller-Tucker-Zemlin variables `U[p,h]`.
- **Session time**: `t_start = 480` (08:00), `t_close = 960` (16:00), `c = 10` min changeover.
- Gurobi model variables are stored as attributes on the returned model object (`m._A`, `m._X`, etc.) for post-solve access.

## Dependencies

Requires a valid **Gurobi licence** (`gurobipy`). Academic/free licences are limited to ~2000 variables — increase `N_SCENARIOS` carefully.

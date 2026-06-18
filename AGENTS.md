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
  Question1Data.csv   # Patient data (ID, Specialty, Session ID, expected_duration, sigma_error)
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
| `N_SCENARIOS` | `11` | `0` = deterministic (single expected-value scenario); `k > 0` = draw k samples |
| `SAMPLING_METHOD` | `"is"` | `"expected"` \| `"random"` \| `"lhs"` \| `"is"` |
| `RANDOM_SEED` | `42` | Controls all stochastic draws |

## Scenario Sampling (`src/sampling.py`)

All methods return `(d, S, pi)`: duration dict `{(patient_id, scenario_idx): float}`, scenario index list, probability vector.

| Method | Description |
|--------|-------------|
| `"expected"` | Single deterministic scenario at `expected_duration` |
| `"random"` | SAA — independent truncated log-normal draws |
| `"lhs"` | Latin Hypercube Sampling — stratified coverage of the CDF |
| `"is"` | Mixture Importance Sampling — boundary-shifted proposals with IS weights |

Durations are modelled as **truncated log-normal** with real-space mean = `expected_duration` and std = `sigma_error`, truncated to `[mu ± 2σ]`.

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

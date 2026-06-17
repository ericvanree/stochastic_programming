"""
sampling.py — scenario generation methods for stochastic OR scheduling.

Each public function has the signature:

    generate(df, n_scenarios, rng) -> dict[(patient_id, scenario_idx): float]

where:
  df          : the patient DataFrame (must contain 'Patient ID',
                'expected_duration', 'sigma_error')
  n_scenarios : number of scenarios to draw  (int > 0)
  rng         : np.random.Generator

Available methods (pass as SAMPLING_METHOD string):
  "expected"  — deterministic; single scenario = expected_duration
  "random"    — truncated log-normal random sampling (SAA)
  # "lhs"     — Latin Hypercube Sampling        (TODO)
  # "ortho"   — Orthogonal Sampling              (TODO)
  # "is"      — Importance Sampling              (TODO)
"""

import numpy as np
from scipy.stats import truncnorm


# ── Helpers ────────────────────────────────────────────────────────────────────

def _lognorm_params(mu_d: float, sigma_d: float):
    """
    Convert (mean, std) of the *real-space* duration distribution to the
    (mu, sigma) of the underlying normal:

        ln X ~ N(mu_ln, sigma_ln^2)
        E[X]   = exp(mu_ln + 0.5 * sigma_ln^2)  = mu_d
        Var[X] = (exp(sigma_ln^2) - 1) * exp(2*mu_ln + sigma_ln^2) = sigma_d^2
    """
    cv2      = (sigma_d / mu_d) ** 2          # coefficient of variation squared
    sigma_ln = np.sqrt(np.log(1.0 + cv2))
    mu_ln    = np.log(mu_d) - 0.5 * sigma_ln ** 2
    return mu_ln, sigma_ln


def _sample_truncated_lognormal(
    mu_d: float,
    sigma_d: float,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw n samples from a log-normal with real-space mean mu_d and std sigma_d,
    truncated to [mu_d - 2*sigma_d, mu_d + 2*sigma_d].

    Strategy: sample from the underlying truncated normal in log-space and
    exponentiate (equivalent to truncating the log-normal in real space).
    """
    mu_ln, sigma_ln = _lognorm_params(mu_d, sigma_d)

    # Truncation bounds in log-space
    lo_real = max(mu_d - 2.0 * sigma_d, 1e-3)   # never go below ~0
    hi_real = mu_d + 2.0 * sigma_d

    lo_ln = (np.log(lo_real) - mu_ln) / sigma_ln
    hi_ln = (np.log(hi_real) - mu_ln) / sigma_ln

    z = truncnorm.rvs(lo_ln, hi_ln, loc=0.0, scale=1.0, size=n, random_state=rng)
    return np.exp(mu_ln + sigma_ln * z)


# ── Public generation functions ────────────────────────────────────────────────

def generate_expected(df, n_scenarios: int, rng: np.random.Generator) -> dict:
    """Deterministic: single scenario at expected_duration (ignores n_scenarios)."""
    return {
        (row["Patient ID"], 0): float(row["expected_duration"])
        for _, row in df.iterrows()
    }


def generate_random(df, n_scenarios: int, rng: np.random.Generator) -> dict:
    """
    SAA: draw n_scenarios independent samples from truncated log-normal.
    Each patient's duration is independent across scenarios.
    """
    d = {}
    for _, row in df.iterrows():
        p      = row["Patient ID"]
        mu_d   = float(row["expected_duration"])
        sig_d  = float(row["sigma_error"])
        draws  = _sample_truncated_lognormal(mu_d, sig_d, n_scenarios, rng)
        for s, val in enumerate(draws):
            d[p, s] = val
    return d


# ── Latin Hypercube (placeholder) ─────────────────────────────────────────────
def generate_lhs(df, n_scenarios: int, rng: np.random.Generator) -> dict:
    """Latin Hypercube Sampling — TODO."""
    raise NotImplementedError("LHS sampling not yet implemented.")


# ── Orthogonal (placeholder) ───────────────────────────────────────────────────
def generate_ortho(df, n_scenarios: int, rng: np.random.Generator) -> dict:
    """Orthogonal Sampling — TODO."""
    raise NotImplementedError("Orthogonal sampling not yet implemented.")


# ── Importance Sampling (placeholder) ─────────────────────────────────────────
def generate_is(df, n_scenarios: int, rng: np.random.Generator) -> dict:
    """Importance Sampling — TODO."""
    raise NotImplementedError("Importance sampling not yet implemented.")


# ── Registry ──────────────────────────────────────────────────────────────────
METHODS: dict[str, callable] = {
    "expected": generate_expected,
    "random":   generate_random,
    "lhs":      generate_lhs,
    "ortho":    generate_ortho,
    "is":       generate_is,
}


def generate_scenarios(
    df,
    n_scenarios: int,
    method: str = "expected",
    seed: int = 42,
) -> tuple[dict, list, dict]:
    """
    Top-level entry point.

    Parameters
    ----------
    df          : patient DataFrame
    n_scenarios : number of scenarios (0 → deterministic expected duration)
    method      : one of 'expected', 'random', 'lhs', 'ortho', 'is'
    seed        : random seed for reproducibility

    Returns
    -------
    d  : {(patient_id, scenario_idx): duration}
    S  : list of scenario indices
    pi : {scenario_idx: probability}
    """
    rng = np.random.default_rng(seed)

    if n_scenarios == 0 or method == "expected":
        d = generate_expected(df, 1, rng)
        S  = [0]
        pi = {0: 1.0}
        return d, S, pi

    if method not in METHODS:
        raise ValueError(f"Unknown sampling method '{method}'. "
                         f"Choose from: {list(METHODS)}")

    d  = METHODS[method](df, n_scenarios, rng)
    S  = list(range(n_scenarios))
    pi = {s: 1.0 / n_scenarios for s in S}   # uniform (equal weight SAA)
    return d, S, pi

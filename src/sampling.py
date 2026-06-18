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
  "lhs"       — Latin Hypercube Sampling
  "is"        — Mixture Importance Sampling (boundary-shifted proposals)
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
    """
    Latin Hypercube Sampling over a truncated log-normal distribution.

    Algorithm
    ---------
    For each patient p independently:
      1. Divide the CDF of the (patient-specific) truncated log-normal into
         n_scenarios equiprobable intervals: [(k-1)/n, k/n] for k=1..n.
      2. Draw one uniform sample from *each* interval:
             u_k ~ Uniform((k-1)/n, k/n)
      3. Map each u_k through the inverse CDF (ppf) to obtain a duration.
      4. Randomly permute the n durations and assign them to scenarios 0..n-1.

    This guarantees that every patient's duration samples cover all probability
    intervals exactly once, while the scenario-to-interval assignment is
    independently shuffled across patients.
    """
    n = n_scenarios
    d = {}
    for _, row in df.iterrows():
        p      = row["Patient ID"]
        mu_d   = float(row["expected_duration"])
        sig_d  = float(row["sigma_error"])

        mu_ln, sigma_ln = _lognorm_params(mu_d, sig_d)

        # Truncation bounds in standardised normal space (log-space)
        lo_real = max(mu_d - 2.0 * sig_d, 1e-3)
        hi_real = mu_d + 2.0 * sig_d
        lo_ln   = (np.log(lo_real) - mu_ln) / sigma_ln
        hi_ln   = (np.log(hi_real) - mu_ln) / sigma_ln

        # CDF of the truncated normal at the two bounds (used to scale intervals)
        dist    = truncnorm(lo_ln, hi_ln)
        cdf_lo  = dist.cdf(lo_ln)   # = 0 by definition of truncnorm, kept for clarity
        cdf_hi  = dist.cdf(hi_ln)   # = 1 by definition of truncnorm

        # Equiprobable interval edges in [0, 1] (probability space of truncated dist)
        edges = np.linspace(0.0, 1.0, n + 1)

        # Sample one uniform point per interval and invert CDF
        u_lo  = edges[:-1]                          # lower edge of each interval
        u_hi  = edges[1:]                           # upper edge of each interval
        u     = rng.uniform(u_lo, u_hi)             # one draw per interval, shape (n,)
        # Map u through the ppf of the truncated normal in log-space, then exponentiate
        z       = dist.ppf(u)                       # standard truncated-normal quantiles
        samples = np.exp(mu_ln + sigma_ln * z)      # back to real-space durations

        # Randomly permute to break correlation across patients
        perm = rng.permutation(n)
        for s in range(n):
            d[p, s] = float(samples[perm[s]])

    return d


# ── Mixture Importance Sampling ───────────────────────────────────────────────
def generate_is(df, n_scenarios: int, rng: np.random.Generator) -> tuple[dict, dict]:
    """
    Mixture Importance Sampling (MIS).

    Each of the |S| scenarios s uses its own proposal distribution g_s whose
    mean for every patient p is shifted to boundary point s of the interval
    [mu_p - sigma_p, mu_p + sigma_p]:

        k_s      = -1 + 2*s / (|S| - 1)          (linearly from -1 to +1)
        mu_{p,s} = mu_p + k_s * sigma_p

    The shift fraction k_s is the SAME for all patients in scenario s, so
    all patients' distributions move together across scenarios.

    One sample x_{p,s} is drawn per patient per scenario from g_{s,p}.
    All proposals are truncated to f's bounds [mu_p - 2*sigma_p, mu_p + 2*sigma_p]
    so that f(x_{p,s}) > 0 is guaranteed and no NaN weights arise.

    Weights
    -------
        log w_s = sum_p [ log f_p(x_{p,s}) - log g_{s,p}(x_{p,s}) ]

    Accumulated in log-space per patient, then normalised with log-sum-exp.

    Edge case: |S| = 1  =>  k_0 = 0 (original mean, no shift).
    """
    n = n_scenarios

    # Shift fractions: n equally-spaced values from -1 to +1
    if n == 1:
        shift_factors = np.array([0.0])
    else:
        shift_factors = np.linspace(-1.0, 1.0, n)   # shape (n,)

    # Precompute per-patient parameters for f (target) and g_s (proposals)
    patient_params = []
    for _, row in df.iterrows():
        p     = row["Patient ID"]
        mu_d  = float(row["expected_duration"])
        sig_d = float(row["sigma_error"])

        # Target f: truncated log-normal with real-space mean = mu_d
        mu_ln_f, sigma_ln_f = _lognorm_params(mu_d, sig_d)
        lo_f = max(mu_d - 2.0 * sig_d, 1e-3)
        hi_f = mu_d + 2.0 * sig_d
        a_f  = (np.log(lo_f) - mu_ln_f) / sigma_ln_f
        b_f  = (np.log(hi_f) - mu_ln_f) / sigma_ln_f
        dist_f = truncnorm(a_f, b_f)

        # Proposals g_s: one per scenario, shifted mean, same truncation bounds as f
        proposals = []
        for k in shift_factors:
            mu_d_s = np.clip(mu_d + k * sig_d, lo_f + 1e-6, hi_f - 1e-6)
            mu_ln_s, sigma_ln_s = _lognorm_params(mu_d_s, sig_d)
            a_s = (np.log(lo_f) - mu_ln_s) / sigma_ln_s
            b_s = (np.log(hi_f) - mu_ln_s) / sigma_ln_s
            proposals.append((mu_ln_s, sigma_ln_s, truncnorm(a_s, b_s)))

        patient_params.append(dict(
            p=p,
            mu_ln_f=mu_ln_f, sigma_ln_f=sigma_ln_f, dist_f=dist_f,
            proposals=proposals,
        ))

    d     = {}
    log_w = np.zeros(n)

    for pat in patient_params:
        p          = pat["p"]
        mu_ln_f    = pat["mu_ln_f"]
        sigma_ln_f = pat["sigma_ln_f"]
        dist_f     = pat["dist_f"]
        proposals  = pat["proposals"]

        for s in range(n):
            mu_ln_s, sigma_ln_s, dist_s = proposals[s]

            # Draw one sample from proposal g_s
            z_s = dist_s.rvs(random_state=rng)          # scalar
            x   = float(np.exp(mu_ln_s + sigma_ln_s * z_s))
            d[p, s] = x

            # log f_p(x) - log g_{s,p}(x)  (Jacobian -ln x cancels in ratio)
            z_f      = (np.log(x) - mu_ln_f) / sigma_ln_f
            logpdf_f = float(dist_f.logpdf(z_f)) - np.log(sigma_ln_f)
            logpdf_s = float(dist_s.logpdf(z_s)) - np.log(sigma_ln_s)
            log_w[s] += logpdf_f - logpdf_s

    # Normalise via log-sum-exp
    log_w -= np.max(log_w)
    w      = np.exp(log_w)
    w_norm = w / w.sum()

    pi    = {s: float(w_norm[s]) for s in range(n)}
    eff_n = 1.0 / float(np.sum(w_norm ** 2))
    print(f"  MIS shift factors: {np.round(shift_factors, 3).tolist()}")
    print(f"  MIS weight stats : min={w_norm.min():.4f}  max={w_norm.max():.4f}"
          f"  effective n = {eff_n:.1f} / {n}")
    return d, pi

# ── Registry ──────────────────────────────────────────────────────────────────
METHODS: dict[str, callable] = {
    "expected": generate_expected,
    "random":   generate_random,
    "lhs":      generate_lhs,
    "is":       generate_is,
}

# Methods that return (d, pi) instead of just d
_CUSTOM_WEIGHT_METHODS = {"is"}


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
    method      : one of 'expected', 'random', 'lhs', 'is'
    seed        : random seed for reproducibility

    Returns
    -------
    d  : {(patient_id, scenario_idx): duration}
    S  : list of scenario indices
    pi : {scenario_idx: probability}  (uniform for SAA/LHS, importance weights for IS)
    """
    rng = np.random.default_rng(seed)

    if n_scenarios == 0 or method == "expected":
        d = generate_expected(df, 1, rng)
        return d, [0], {0: 1.0}

    if method not in METHODS:
        raise ValueError(f"Unknown sampling method '{method}'. "
                         f"Choose from: {list(METHODS)}")

    if method in _CUSTOM_WEIGHT_METHODS:
        d, pi = METHODS[method](df, n_scenarios, rng)
    else:
        d  = METHODS[method](df, n_scenarios, rng)
        pi = {s: 1.0 / n_scenarios for s in range(n_scenarios)}

    S = list(range(n_scenarios))
    return d, S, pi

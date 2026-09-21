"""
BN-S stochastic-volatility Strang splitting Monte Carlo.

Model (Eq. 33):
    dX_t = [r - lambda*kappa(rho) - V_t/2] dt + sqrt(V_t) dW_t + rho dZ_{lambda*t}
    dV_t = -lambda V_t dt + dZ_{lambda*t}

Strang splitting one-step update over [t, t+h]:
    mu    = r - lambda * kappa(rho)
    DeltaZ ~ BDLP increment over lambda*h time units
    V_{t+h} = exp(-lambda h) V_t + exp(-lambda h/2) DeltaZ
    I_t^h   = (1-exp(-lambda h))/lambda * V_t + (1-exp(-lambda h/2))/lambda * DeltaZ
    X_{t+h} = X_t + mu h - I/2 + N(0, I) + rho DeltaZ

Euler comparator one-step update:
    DeltaZ ~ BDLP increment over lambda*h
    V_{t+h} = exp(-lambda h) V_t + DeltaZ           (exact OU update without half-exp)
    I_t^h   = h * max(V_t, 0)                        (left-point Euler)
    X_{t+h} = X_t + mu h - I/2 + N(0, I) + rho DeltaZ
"""

import time
from typing import Dict, Any, List, Tuple

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.bns.levy_increments import sample_bdlp_increment, compute_kappa

_BATCH = 50_000


# ---------------------------------------------------------------------------
# One-step updates
# ---------------------------------------------------------------------------

def bns_strang_step(
    X: np.ndarray,
    V: np.ndarray,
    h: float,
    lam: float,
    mu: float,
    rho: float,
    DeltaZ: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    BN-S Strang splitting one step (Eq. 33).

    Parameters
    ----------
    X, V    : (n,) log-price and variance
    h       : float  step size
    lam     : float  lambda (OU speed)
    mu      : float  r - lambda*kappa(rho)
    rho     : float  jump-correlation parameter
    DeltaZ  : (n,)  pre-sampled BDLP increment

    Returns
    -------
    X_new, V_new : (n,) each
    """
    e_h  = np.exp(-lam * h)
    e_h2 = np.exp(-lam * h / 2.0)

    V_new = e_h  * V   + e_h2 * DeltaZ
    I     = ((1.0 - e_h)  / lam) * V + ((1.0 - e_h2) / lam) * DeltaZ
    I     = np.maximum(I, 0.0)          # guard against rare negatives

    G     = rng.standard_normal(len(X))
    X_new = X + mu * h - 0.5 * I + np.sqrt(I) * G + rho * DeltaZ
    return X_new, V_new


def bns_euler_step(
    X: np.ndarray,
    V: np.ndarray,
    h: float,
    lam: float,
    mu: float,
    rho: float,
    DeltaZ: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    BN-S Euler comparator one step.

    Uses exact OU update for V but left-point (Euler) approximation for I.
    """
    e_h  = np.exp(-lam * h)

    V_new = e_h * V + DeltaZ
    I     = h * np.maximum(V, 0.0)

    G     = rng.standard_normal(len(X))
    X_new = X + mu * h - 0.5 * I + np.sqrt(I) * G + rho * DeltaZ
    return X_new, V_new


# ---------------------------------------------------------------------------
# Full path simulators
# ---------------------------------------------------------------------------

def simulate_bns_paths(
    method: str,
    n_paths: int,
    h: float,
    params: dict,
    bdlp_params: Dict[str, Any],
    kappa_rho: float,
    monitoring_dates: np.ndarray,          # monitoring times in [0, T]
    rng: np.random.Generator,
    batch_size: int = _BATCH,
) -> np.ndarray:
    """
    Simulate BN-S paths and return asset prices at monitoring dates.

    Parameters
    ----------
    method          : 'strang' or 'euler'
    n_paths         : int
    h               : float  step size
    params          : dict   with keys r, lam, rho, V0, S0, T
    bdlp_params     : dict   BDLP specification
    kappa_rho       : float  precomputed kappa(rho)
    monitoring_dates: (M,)   times at which to record S (including T)
    rng             : Generator
    batch_size      : int

    Returns
    -------
    S_mon : (n_paths, len(monitoring_dates)) asset prices at monitoring dates
    """
    lam  = params['lambda']
    rho  = params['rho']
    r    = params['r']
    V0   = params['V0']
    S0   = params['S0']
    T    = params['T']
    mu   = r - lam * kappa_rho

    n_steps = max(1, round(T / h))
    h_eff = T / n_steps                   # actual step to exactly cover T

    # Grid of step-end times
    t_grid = np.linspace(h_eff, T, n_steps)

    # For each monitoring date, find closest grid index (from left)
    mon_indices = np.searchsorted(t_grid, monitoring_dates, side='left')
    mon_indices = np.clip(mon_indices, 0, n_steps - 1)

    step_fn = bns_strang_step if method == 'strang' else bns_euler_step

    all_chunks: List[np.ndarray] = []
    remaining = n_paths
    while remaining > 0:
        n_chunk = min(remaining, batch_size)
        X = np.full(n_chunk, np.log(float(S0)), dtype=np.float64)
        V = np.full(n_chunk, float(V0),          dtype=np.float64)

        S_mon_chunk = np.empty((n_chunk, len(monitoring_dates)), dtype=np.float64)
        mon_recorded = np.zeros(len(monitoring_dates), dtype=bool)

        for step in range(n_steps):
            DeltaZ = sample_bdlp_increment(n_chunk, lam, h_eff, bdlp_params, rng)
            X, V = step_fn(X, V, h_eff, lam, mu, rho, DeltaZ, rng)

            for mi, t_mon in enumerate(monitoring_dates):
                if not mon_recorded[mi] and step == mon_indices[mi]:
                    S_mon_chunk[:, mi] = np.exp(X)
                    mon_recorded[mi] = True

        all_chunks.append(S_mon_chunk)
        remaining -= n_chunk

    return np.concatenate(all_chunks, axis=0)


# ---------------------------------------------------------------------------
# Payoffs
# ---------------------------------------------------------------------------

def asian_call_price(
    S_mon: np.ndarray,
    S0: float,
    K: float,
    r: float,
    T: float,
    include_spot: bool = True,
) -> float:
    """
    Arithmetic Asian call price.

    Parameters
    ----------
    S_mon        : (n, M) asset prices at monitoring dates
    S0           : float  spot price at t=0
    K            : float  strike
    r, T         : float  rate and maturity
    include_spot : bool   if True, average includes S0 (M+1 terms)

    Returns
    -------
    float  discounted price
    """
    if include_spot:
        S_all = np.concatenate([np.full((S_mon.shape[0], 1), S0), S_mon], axis=1)
        avg = S_all.mean(axis=1)
    else:
        avg = S_mon.mean(axis=1)

    payoff = np.maximum(avg - K, 0.0)
    return float(np.exp(-r * T) * payoff.mean())


def vix_call_price(
    V_T: np.ndarray,
    A_coef: float,
    B_coef: float,
    K: float,
    r: float,
    T: float,
) -> float:
    """
    VIX call payoff: exp(-rT) * E[max(sqrt(A*V_T + B) - K, 0)].

    Parameters
    ----------
    V_T    : (n,)  terminal variance
    A_coef : float  affine coefficient for V_T in VIX^2
    B_coef : float  affine constant
    K      : float  VIX strike
    r, T   : float

    Returns
    -------
    float
    """
    vix2 = np.maximum(A_coef * V_T + B_coef, 0.0)
    vix  = np.sqrt(vix2)
    payoff = np.maximum(vix - K, 0.0)
    return float(np.exp(-r * T) * payoff.mean())


# ---------------------------------------------------------------------------
# Affine VIX coefficients
# ---------------------------------------------------------------------------

def vix_affine_coeffs(
    lam: float,
    tau_vix: float,
    bdlp_params: Dict[str, Any],
) -> Tuple[float, float]:
    """
    Compute A(tau_vix) and B(tau_vix) for VIX^2_T = A*V_T + B.

    A(tau) = (1 - exp(-lambda*tau)) / (lambda*tau)
    B(tau) = (1 - A(tau)) / lambda * integral(x * nu(dx)) - 2*kappa_rho_correction

    For the BN-S model:
        VIX_T^2 = (1/tau) integral_T^{T+tau} E_T[V_s] ds
                = A(tau) V_T + B(tau)
    where A(tau) = (1-exp(-lam*tau))/(lam*tau) and
    B(tau) = (1-A(tau))/lam * integral(x nu(dx))

    Note: the exact formula for B also includes jump-at-T contributions, but
    since V has no predictable jumps here, B is as above.

    Parameters
    ----------
    lam        : float
    tau_vix    : float  VIX horizon (e.g. 30/365)
    bdlp_params: dict

    Returns
    -------
    A_coef, B_coef : float, float
    """
    A_coef = (1.0 - np.exp(-lam * tau_vix)) / (lam * tau_vix)

    # integral(x * nu(dx)) = E[Z_1] for the BDLP per unit time
    t_type = bdlp_params['type']
    a      = bdlp_params.get('alpha', bdlp_params.get('alpha_g', None))
    b      = bdlp_params.get('beta',  bdlp_params.get('beta_g',  None))

    if t_type in ('ou_gamma', 'gamma_ou'):
        mean_z1 = a / b    # mean of Gamma(a,1/b) per unit time
    elif t_type in ('ou_ts', 'ts_ou'):
        alpha_ts = bdlp_params['alpha']
        beta_ts  = bdlp_params['beta']
        theta_ts = bdlp_params['theta']
        # E[Z_1] = theta_ts * Gamma(1-alpha_ts) / beta_ts^{1-alpha_ts}
        from scipy import special
        mean_z1 = theta_ts * special.gamma(1.0 - alpha_ts) / beta_ts ** (1.0 - alpha_ts)
    else:
        raise ValueError(f"Unknown BDLP type: {t_type}")

    B_coef = (1.0 - A_coef) / lam * mean_z1
    return float(A_coef), float(B_coef)


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment_grid(
    method: str,
    params: dict,
    bdlp_params: Dict[str, Any],
    kappa_rho: float,
    grid_pairs: List[Tuple[int, float]],
    m: int,
    benchmark: float,
    monitoring_dates: np.ndarray,
    payoff_fn,
    seed_base: int = 42,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Run m MC repetitions for each (n_paths, h) grid point.

    Parameters
    ----------
    payoff_fn : callable(S_mon, params) -> float
        Function taking S_mon (n, M) array and params dict, returning price.

    Returns
    -------
    all_prices : list[ndarray(m,)]
    all_cpus   : list[ndarray(m,)]
    """
    ss = np.random.SeedSequence(seed_base)
    child_seeds = ss.spawn(len(grid_pairs) * m)

    all_prices: List[np.ndarray] = []
    all_cpus:   List[np.ndarray] = []

    for gi, (n_paths, h) in enumerate(grid_pairs):
        prices = np.empty(m)
        cpus   = np.empty(m)
        for rep in range(m):
            rng = np.random.default_rng(child_seeds[gi * m + rep])
            t0 = time.perf_counter()
            S_mon = simulate_bns_paths(
                method, n_paths, h, params, bdlp_params, kappa_rho,
                monitoring_dates, rng,
            )
            price = payoff_fn(S_mon, params)
            elapsed = time.perf_counter() - t0
            prices[rep] = price
            cpus[rep]   = elapsed

        all_prices.append(prices)
        all_cpus.append(cpus)
        print(
            f"  [{method}] n={n_paths:>9,} h={h:.5f}  "
            f"mean={prices.mean():.6f}  bias={prices.mean()-benchmark:+.6f}  "
            f"cpu={cpus.mean():.2f}s"
        )

    return all_prices, all_cpus

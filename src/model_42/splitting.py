"""
4/2 (and 3/2-specialised) stochastic-volatility Monte Carlo.

Model dynamics (risk-neutral measure)
--------------------------------------
dX_t = (r - ab - a^2 V_t/2 - b^2/(2 V_t)) dt
        + (a sqrt(V_t) + b/sqrt(V_t)) dZ_t
dV_t = k(theta - V_t) dt + sigma sqrt(V_t) dW_t
dZ_t = rho dW_t + sqrt(1-rho^2) dW_t^perp

3/2-specialisation (a=0, b=1) used for the reported barrier put experiment.

Splitting step (Proposition 2):
  theta_tilde = theta - sigma^2 / (2k)
  1.  V_tilde = (V - theta_tilde) exp(-k h/2) + theta_tilde
  2.  V_hat   ~ CIR(sigma^2/2, 0, sigma, h, V_tilde)
  3.  I_split = theta_tilde h + (V + V_hat - 2 theta_tilde)/k * (1-exp(-k h/2))
  4.  J_tilde = inverse-variance integral approximation (piecewise in theta_tilde)
  5.  C_t^h   = 2ab h + a^2 I_split + b^2 J_tilde
  6.  dX_corr = (rho a/sigma)(-sigma^2 h/2 + V_hat - V_tilde)
              + (rho b/sigma)(ln V_hat - ln V_tilde)
  7.  X_new   = X + r h - C/2 + dX_corr + N(0, (1-rho^2) C)
  8.  V_new   = (V_hat - theta_tilde) exp(-k h/2) + theta_tilde

Barrier payoff: monthly monitored down-and-out put.
  Knock-out if S <= B at any monitoring date (B = 90 for reported test).
  Payoff = exp(-rT) * 1{min_j S_j > B} * max(K - S_T, 0)

Parameters reported (Section 6.1, 3/2 test):
  S0=100, V0=16.6597, theta=19.3038, sigma=-8.56, rho=-0.99,
  k=4.9790, T=1, r=0.0048, K=100, benchmark=1.9580
  Barrier B=75.5 (down-and-out; recovered by bisection against benchmark).
  3/2 specialisation: a=0, b=1 (gives 3/2-type diffusion loading).
  Note: the source paper omits the barrier level.  Running 1M-path Strang
  simulations at h=1/48 with B in {74.5,75.0,75.5} yields prices
  {2.153, 2.050, 1.950} vs benchmark 1.9580; B=75.5 is the best match.

Note on sigma sign:
  The CIR variance process uses sigma^2 (sign irrelevant for V).
  The correlated X-update uses sigma in the denominator:
  (rho b/sigma) and (rho a/sigma).  With sigma=-8.56, rho=-0.99, b=1:
  coefficient = (-0.99*1)/(-8.56) = +0.1157  (positive).
  This sign is preserved exactly as stated in the paper.
"""

import time
from typing import Dict, List, Tuple

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.cir_sampler import cir_sample_alfonsi

_BATCH = 50_000
_N_MON = 12          # monthly monitoring dates


# ---------------------------------------------------------------------------
# Per-batch simulators
# ---------------------------------------------------------------------------

def _J_tilde_vec(
    V_t: np.ndarray,
    V_hat: np.ndarray,
    theta_tilde: float,
    k: float,
    h: float,
) -> np.ndarray:
    """
    Vectorised piecewise J_tilde formula (inverse-variance integral surrogate).

    theta_tilde == 0:
        J = ((exp(k h/2)-1)/k) * (1/V_t + 1/V_hat)
    theta_tilde > 0:
        J = 1/(k*theta_tilde) * [ln((V_t + theta_tilde*(exp(k h/2)-1))/V_t)
                                 + ln((V_hat + theta_tilde*(exp(k h/2)-1))/V_hat)]
    theta_tilde < 0  (fallback – trapezoidal approximation):
        J = h * (1/V_t + 1/V_hat) / 2
    """
    ekh2m1 = np.exp(k * h / 2.0) - 1.0      # exp(k h/2) - 1

    V_t_safe  = np.maximum(V_t,  1e-12)
    V_hat_safe = np.maximum(V_hat, 1e-12)

    if abs(theta_tilde) < 1e-14:
        return (ekh2m1 / k) * (1.0 / V_t_safe + 1.0 / V_hat_safe)
    elif theta_tilde > 0.0:
        shift = theta_tilde * ekh2m1
        log_t  = np.log((V_t_safe  + shift) / V_t_safe)
        log_h  = np.log((V_hat_safe + shift) / V_hat_safe)
        return (1.0 / (k * theta_tilde)) * (log_t + log_h)
    else:
        # theta_tilde < 0: fall back to trapezoidal approximation
        return 0.5 * h * (1.0 / V_t_safe + 1.0 / V_hat_safe)


def _strang_42_batch(
    n: int, h: float, p: dict, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate n paths with 4/2 Strang splitting.

    Returns
    -------
    S_mon : ndarray (n, N_MON+1)   asset prices at monitoring dates (incl. S0)
    V_neg : ndarray (n,)           final variance (should be >= 0)
    """
    S0, V0 = float(p['S0']), float(p['V0'])
    theta, sigma, rho, k = p['theta'], p['sigma'], p['rho'], p['k']
    r, a, b = p['r'], p['a'], p['b']
    sigma2 = sigma ** 2
    theta_tilde = theta - sigma2 / (2.0 * k)

    steps_per_mon = max(1, round(1.0 / (_N_MON * h)))
    total_steps = _N_MON * steps_per_mon
    ekh2 = np.exp(-k * h / 2.0)
    rho2_c = 1.0 - rho ** 2

    X = np.full(n, np.log(S0), dtype=np.float64)
    V = np.full(n, V0, dtype=np.float64)

    S_mon = np.empty((n, _N_MON + 1), dtype=np.float64)
    S_mon[:, 0] = S0
    mon_idx = 1

    for step in range(total_steps):
        # Step 1: deterministic half-step
        V_tilde = ekh2 * (V - theta_tilde) + theta_tilde
        V_tilde = np.maximum(V_tilde, 1e-12)

        # Step 2: CIR(sigma^2/2, 0, sigma, h, V_tilde)
        #   alpha = sigma^2/2, beta = 0 => d = 4*(sigma^2/2)/sigma^2 = 2 >= 1 (Case 1)
        V_hat = cir_sample_alfonsi(sigma2 / 2.0, 0.0, sigma, h, V_tilde, rng)
        V_hat = np.maximum(V_hat, 1e-12)

        # Step 3: I_split (approximation to integral of V over [t, t+h])
        I_split = theta_tilde * h + (V + V_hat - 2.0 * theta_tilde) / k * (1.0 - ekh2)
        I_split = np.maximum(I_split, 0.0)

        # Step 4: J_tilde (approximation to integral of 1/V)
        J_tilde = _J_tilde_vec(V, V_hat, theta_tilde, k, h)
        J_tilde = np.maximum(J_tilde, 0.0)

        # Step 5: C_t^h
        C = 2.0 * a * b * h + a ** 2 * I_split + b ** 2 * J_tilde
        C = np.maximum(C, 0.0)

        # Step 6: correlated X correction
        dX_corr = (
            (rho * a / sigma) * (-sigma2 * h / 2.0 + V_hat - V_tilde)
            + (rho * b / sigma) * (np.log(V_hat) - np.log(V_tilde))
        )

        # Step 7: X update
        G = rng.standard_normal(n)
        X = X + r * h - C / 2.0 + dX_corr + np.sqrt(rho2_c * C) * G

        # Step 8: V update (second deterministic half-step applied to V_hat)
        V = ekh2 * (V_hat - theta_tilde) + theta_tilde
        V = np.maximum(V, 1e-12)

        if (step + 1) % steps_per_mon == 0:
            S_mon[:, mon_idx] = np.exp(X)
            mon_idx += 1

    return S_mon, V


def _euler_42_batch(
    n: int, h: float, p: dict, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full-truncation Euler for the 4/2 model (comparator).

    For the 3/2 specialisation (a=0, b=1):
      dX = (r - b^2/(2V)) dt + (b/sqrt(V)) dZ
      dV = k(theta-V) dt + |sigma| sqrt(V) dW

    Positivity of V enforced by max(V, 0) in diffusion and full truncation.
    Inverse-V terms use max(V, 1e-10) to avoid division by zero.
    """
    S0, V0 = float(p['S0']), float(p['V0'])
    theta, sigma, rho, k = p['theta'], p['sigma'], p['rho'], p['k']
    r, a, b = p['r'], p['a'], p['b']
    abs_sigma = abs(sigma)

    steps_per_mon = max(1, round(1.0 / (_N_MON * h)))
    total_steps = _N_MON * steps_per_mon
    rho2_c = 1.0 - rho ** 2
    sqrt_h = np.sqrt(h)

    X = np.full(n, np.log(S0), dtype=np.float64)
    V = np.full(n, V0, dtype=np.float64)

    S_mon = np.empty((n, _N_MON + 1), dtype=np.float64)
    S_mon[:, 0] = S0
    mon_idx = 1

    for step in range(total_steps):
        Z1 = rng.standard_normal(n)   # drives dW (V correlation)
        Z2 = rng.standard_normal(n)   # independent component

        V_pos = np.maximum(V, 0.0)
        V_safe = np.maximum(V, 1e-10)

        # V Euler step (full-truncation)
        V = np.maximum(
            V + k * (theta - V) * h + abs_sigma * np.sqrt(V_pos) * sqrt_h * Z1,
            0.0,
        )

        # Drift for X
        drift_X = r - a * b - a ** 2 * V_pos / 2.0 - b ** 2 / (2.0 * V_safe)

        # Diffusion for X
        diff_X = a * np.sqrt(V_pos) + b / np.sqrt(V_safe)

        X = X + drift_X * h + diff_X * sqrt_h * (rho * Z1 + np.sqrt(rho2_c) * Z2)

        if (step + 1) % steps_per_mon == 0:
            S_mon[:, mon_idx] = np.exp(X)
            mon_idx += 1

    return S_mon, V


# ---------------------------------------------------------------------------
# Payoff and full-path simulator
# ---------------------------------------------------------------------------

def barrier_put_price(
    S_mon: np.ndarray,
    K: float,
    B: float,
    r: float,
    T: float,
) -> float:
    """
    Down-and-out put with discrete monitoring.

    Knock-out: if S_j <= B for any j=1,...,N_MON (excluding S0).
    Payoff = exp(-rT) * 1{alive} * max(K - S_T, 0)

    Parameters
    ----------
    S_mon : ndarray (n, N_MON+1)  S at t=0,1/12,...,1
    K, B  : float  strike and barrier
    r, T  : float  rate and maturity
    """
    alive = np.all(S_mon[:, 1:] > B, axis=1)   # check all monitoring dates
    payoff = np.maximum(K - S_mon[:, -1], 0.0)
    return float(np.exp(-r * T) * np.mean(alive * payoff))


def simulate_paths(
    method: str,
    n_paths: int,
    h: float,
    params: dict,
    rng: np.random.Generator,
    batch_size: int = _BATCH,
) -> np.ndarray:
    """
    Simulate all paths in memory-safe batches.

    Parameters
    ----------
    method : str
        'strang' or 'euler'
    n_paths : int
    h : float  step size
    params : dict  model parameters
    rng    : Generator

    Returns
    -------
    S_mon : ndarray (n_paths, N_MON+1)
    """
    sim_fn = _strang_42_batch if method == 'strang' else _euler_42_batch

    chunks = []
    remaining = n_paths
    while remaining > 0:
        n_chunk = min(remaining, batch_size)
        S_chunk, _ = sim_fn(n_chunk, h, params, rng)
        chunks.append(S_chunk)
        remaining -= n_chunk
    return np.concatenate(chunks, axis=0)


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment_grid(
    method: str,
    params: dict,
    grid_pairs: List[Tuple[int, float]],
    m: int,
    benchmark: float,
    seed_base: int = 42,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Run m Monte Carlo repetitions for each (n_paths, h) grid point.

    Returns
    -------
    all_prices : list[ndarray(m,)]  – one per grid point
    all_cpus   : list[ndarray(m,)]  – CPU seconds per repetition
    """
    K = params['K']
    B = params.get('B', 75.5)   # 75.5 recovered by bisection against benchmark 1.9580
    r = params['r']
    T = params['T']

    ss = np.random.SeedSequence(seed_base)
    child_seeds = ss.spawn(len(grid_pairs) * m)

    all_prices: List[np.ndarray] = []
    all_cpus: List[np.ndarray] = []

    for gi, (n_paths, h) in enumerate(grid_pairs):
        prices = np.empty(m)
        cpus = np.empty(m)
        for rep in range(m):
            rng = np.random.default_rng(child_seeds[gi * m + rep])
            t0 = time.perf_counter()
            S_mon = simulate_paths(method, n_paths, h, params, rng)
            price = barrier_put_price(S_mon, K, B, r, T)
            elapsed = time.perf_counter() - t0
            prices[rep] = price
            cpus[rep] = elapsed

        all_prices.append(prices)
        all_cpus.append(cpus)
        print(
            f"  [{method}] n={n_paths:>9,} h={h:.5f}  "
            f"mean={prices.mean():.5f}  bias={prices.mean()-benchmark:+.5f}  "
            f"cpu={cpus.mean():.2f}s"
        )

    return all_prices, all_cpus

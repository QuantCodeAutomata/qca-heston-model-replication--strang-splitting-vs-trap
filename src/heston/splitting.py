"""
Heston stochastic-volatility Asian-call Monte Carlo.

Three discretisation schemes
-----------------------------
Strang splitting   – Proposition 1 of the paper; exact CIR sub-step.
Trapezoidal        – Euler V update + trapezoidal integral approximation.
Euler (FT)         – Full-truncation Euler for both V and X.

Model
-----
dS = r S dt + sqrt(V) S dZ          Z = rho W + sqrt(1-rho^2) W_perp
dV = k(theta - V) dt + sigma sqrt(V) dW

Reference parameters (Table 1 / Exp 1)
--------------------------------------
S0=100, V0=0.010201, theta=0.019, sigma=0.61, rho=-0.7, k=6.21,
T=1, r=0.0319, K=100.  Benchmark Asian call: 3.5665.

Monitoring dates: arithmetic average of S at t = 0, 1/12, ..., 12/12.
"""

import time
from typing import List, Tuple

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.cir_sampler import cir_sample_alfonsi

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BATCH = 50_000          # paths per memory-safe simulation chunk
_N_MON = 12              # number of monitoring intervals (monthly)


# ---------------------------------------------------------------------------
# Internal per-batch simulators  (return S grid, shape (n, N_MON+1))
# ---------------------------------------------------------------------------

def _strang_batch(n: int, h: float, p: dict, rng: np.random.Generator) -> np.ndarray:
    """
    Simulate n paths with Strang splitting; return S at monitoring dates.

    Strang step (Proposition 1):
      1. V_tilde = exp(-k h/2)(V-theta) + theta          half-step, deterministic
      2. V_hat   ~ CIR(0, 0, sigma, h, V_tilde)          full step,  stochastic
      3. I       = theta*h + (V + V_hat - 2*theta)/k * (1-exp(-k h/2))
      4. dX_corr = (rho/sigma)(V_hat - V_tilde)
      5. dX_gauss~ N(0, (1-rho^2) I)
      6. X_new   = X + r h - I/2 + dX_corr + dX_gauss
      7. V_new   = exp(-k h/2)(V_hat - theta) + theta    half-step, deterministic
    """
    S0, V0 = p['S0'], p['V0']
    theta, sigma, rho, k = p['theta'], p['sigma'], p['rho'], p['k']
    r = p['r']

    steps_per_mon = max(1, round(1.0 / (_N_MON * h)))
    total_steps   = _N_MON * steps_per_mon

    ekh2 = np.exp(-k * h / 2.0)          # exp(-k h/2)

    X = np.full(n, np.log(float(S0)), dtype=np.float64)
    V = np.full(n, float(V0),         dtype=np.float64)

    S_grid       = np.empty((n, _N_MON + 1), dtype=np.float64)
    S_grid[:, 0] = S0
    mon_idx      = 1

    rho2_c = 1.0 - rho * rho             # (1 - rho^2)

    for step in range(total_steps):
        # --- Step 1: deterministic half-step ---
        V_tilde = ekh2 * (V - theta) + theta       # always >= 0 when V >= 0

        # --- Step 2: CIR(0, 0, sigma, h, V_tilde) ---
        V_hat = cir_sample_alfonsi(0.0, 0.0, sigma, h, V_tilde, rng)

        # --- Step 3: integral approximation ---
        I = theta * h + ((V + V_hat - 2.0 * theta) / k) * (1.0 - ekh2)
        np.maximum(I, 0.0, out=I)         # guard against rare numerical negatives

        # --- Steps 4-6: log-price update ---
        dX_corr  = (rho / sigma) * (V_hat - V_tilde)
        gauss_sd = np.sqrt(rho2_c * I)
        X       += r * h - 0.5 * I + dX_corr + gauss_sd * rng.standard_normal(n)

        # --- Step 7: deterministic half-step ---
        V = ekh2 * (V_hat - theta) + theta

        # --- record monitoring date ---
        if (step + 1) % steps_per_mon == 0:
            S_grid[:, mon_idx] = np.exp(X)
            mon_idx += 1

    return S_grid


def _trapezoidal_batch(n: int, h: float, p: dict, rng: np.random.Generator) -> np.ndarray:
    """
    Simulate n paths with the Trapezoidal scheme.

    V is advanced by full-truncation Euler; the integrated variance is
    approximated by the trapezoidal rule I = h*(V_t + V_{t+h})/2.
    The correlation term is formed from the Euler V increment.

    Step:
      dW       = sqrt(h) * Z_V,   Z_V ~ N(0,1)
      V_unc    = V + k(theta-V)h + sigma*sqrt(V)*dW
      V_next   = max(V_unc, 0)
      I_trap   = h*(V + V_next)/2
      dX_corr  = (rho/sigma)*(V_next - V - k*(theta-V)*h)
      dX_gauss ~ N(0, (1-rho^2)*I_trap)
      X_next   = X + rh - I_trap/2 + dX_corr + dX_gauss
    """
    S0, V0 = p['S0'], p['V0']
    theta, sigma, rho, k = p['theta'], p['sigma'], p['rho'], p['k']
    r = p['r']

    steps_per_mon = max(1, round(1.0 / (_N_MON * h)))
    total_steps   = _N_MON * steps_per_mon

    sqrth  = np.sqrt(h)
    rho_s  = rho / sigma
    rho2_c = 1.0 - rho * rho

    X = np.full(n, np.log(float(S0)), dtype=np.float64)
    V = np.full(n, float(V0),         dtype=np.float64)

    S_grid       = np.empty((n, _N_MON + 1), dtype=np.float64)
    S_grid[:, 0] = S0
    mon_idx      = 1

    for step in range(total_steps):
        dW     = sqrth * rng.standard_normal(n)
        V_sqrt = np.sqrt(V)                          # V >= 0 invariant
        V_unc  = V + k * (theta - V) * h + sigma * V_sqrt * dW
        V_next = np.maximum(V_unc, 0.0)

        # I_trap uses truncated V_next so it stays non-negative.
        I_trap   = 0.5 * h * (V + V_next)
        # corr_term uses PRE-truncation V_unc so that
        #   rho/sigma*(V_unc - V - k*(theta-V)*h) = rho*sqrt(V)*dW exactly,
        # preserving the Brownian correlation regardless of truncation events.
        dX_corr  = rho_s * (V_unc - V - k * (theta - V) * h)
        gauss_sd = np.sqrt(rho2_c * I_trap)
        X       += r * h - 0.5 * I_trap + dX_corr + gauss_sd * rng.standard_normal(n)
        V        = V_next

        if (step + 1) % steps_per_mon == 0:
            S_grid[:, mon_idx] = np.exp(X)
            mon_idx += 1

    return S_grid


def _euler_batch(n: int, h: float, p: dict, rng: np.random.Generator) -> np.ndarray:
    """
    Simulate n paths with the full-truncation Euler scheme.

    Step:
      dW     = sqrt(h)*Z_V,    Z_V    ~ N(0,1)
      dW_perp= sqrt(h)*Z_perp, Z_perp ~ N(0,1),  independent
      V_next = max(V + k(theta-V)h + sigma*sqrt(V)*dW, 0)
      I_euler= h * V                               (left-point)
      X_next = X + rh - I_euler/2
                 + sqrt(V)*(rho*dW + sqrt(1-rho^2)*dW_perp)
    """
    S0, V0 = p['S0'], p['V0']
    theta, sigma, rho, k = p['theta'], p['sigma'], p['rho'], p['k']
    r = p['r']

    steps_per_mon = max(1, round(1.0 / (_N_MON * h)))
    total_steps   = _N_MON * steps_per_mon

    sqrth   = np.sqrt(h)
    rho2_sq = np.sqrt(1.0 - rho * rho)

    X = np.full(n, np.log(float(S0)), dtype=np.float64)
    V = np.full(n, float(V0),         dtype=np.float64)

    S_grid       = np.empty((n, _N_MON + 1), dtype=np.float64)
    S_grid[:, 0] = S0
    mon_idx      = 1

    for step in range(total_steps):
        dW      = sqrth * rng.standard_normal(n)
        dW_perp = sqrth * rng.standard_normal(n)
        V_sqrt  = np.sqrt(V)

        V_next = np.maximum(V + k * (theta - V) * h + sigma * V_sqrt * dW, 0.0)
        X     += r * h - 0.5 * h * V + V_sqrt * (rho * dW + rho2_sq * dW_perp)
        V      = V_next

        if (step + 1) % steps_per_mon == 0:
            S_grid[:, mon_idx] = np.exp(X)
            mon_idx += 1

    return S_grid


# ---------------------------------------------------------------------------
# Public path-returning wrappers  (batch internally for memory safety)
# ---------------------------------------------------------------------------

def _run_batched(batch_fn, n_paths: int, h: float, params: dict,
                 rng: np.random.Generator, batch_size: int) -> np.ndarray:
    """Stack per-batch S grids into shape (n_paths, N_MON+1)."""
    parts = []
    for start in range(0, n_paths, batch_size):
        n = min(batch_size, n_paths - start)
        parts.append(batch_fn(n, h, params, rng))
    return np.vstack(parts)


def heston_strang_paths(n_paths: int, h: float, params: dict,
                        rng: np.random.Generator,
                        batch_size: int = _BATCH) -> np.ndarray:
    """
    Heston Strang-splitting simulation.

    Returns
    -------
    ndarray, shape (n_paths, 13)
        S at dates t = 0, 1/12, …, 12/12.
    """
    return _run_batched(_strang_batch, n_paths, h, params, rng, batch_size)


def heston_trapezoidal_paths(n_paths: int, h: float, params: dict,
                             rng: np.random.Generator,
                             batch_size: int = _BATCH) -> np.ndarray:
    """Heston trapezoidal simulation; returns (n_paths, 13)."""
    return _run_batched(_trapezoidal_batch, n_paths, h, params, rng, batch_size)


def heston_euler_paths(n_paths: int, h: float, params: dict,
                       rng: np.random.Generator,
                       batch_size: int = _BATCH) -> np.ndarray:
    """Heston full-truncation Euler simulation; returns (n_paths, 13)."""
    return _run_batched(_euler_batch, n_paths, h, params, rng, batch_size)


# ---------------------------------------------------------------------------
# Pricing utility
# ---------------------------------------------------------------------------

def asian_call_price(S_grid: np.ndarray, K: float, r: float, T: float) -> float:
    """
    Discounted arithmetic-average Asian call price.

    Parameters
    ----------
    S_grid : ndarray, shape (n_paths, n_dates)
        Asset prices at all monitoring dates (including t=0).
    K : float
    r : float
    T : float

    Returns
    -------
    float
    """
    avg    = S_grid.mean(axis=1)
    payoff = np.maximum(avg - K, 0.0)
    return float(np.exp(-r * T) * payoff.mean())


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment_grid(
    method: str,
    params: dict,
    grid_pairs: List[Tuple],
    m: int = 50,
    seed_base: int = 42,
    batch_size: int = _BATCH,
    benchmark: float = None,          # accepted but unused; kept for API uniformity
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Run m Monte-Carlo repetitions for each (n_paths, h) grid point.

    Timing covers only the simulation + payoff section.
    Paths are processed in memory-safe chunks of ``batch_size``.

    Parameters
    ----------
    method : {'strang', 'trapezoidal', 'euler'}
    params : dict
        Heston model parameters including 'K', 'r', 'T'.
    grid_pairs : list of (n_paths, h)
    m : int
        Number of independent repetitions per grid point.
    seed_base : int
        Root seed for numpy SeedSequence.
    batch_size : int
        Paths per internal batch.

    Returns
    -------
    list of (prices, cpu_times) – each an ndarray of shape (m,).
    """
    _batch_fns = {
        'strang':       _strang_batch,
        'trapezoidal':  _trapezoidal_batch,
        'euler':        _euler_batch,
    }
    if method not in _batch_fns:
        raise ValueError(f"method must be one of {list(_batch_fns)}; got '{method}'")

    batch_fn = _batch_fns[method]
    K, r, T  = params['K'], params['r'], params['T']
    disc     = np.exp(-r * T)

    results = []
    for gi, (n_paths_raw, h) in enumerate(grid_pairs):
        n_paths = int(n_paths_raw)
        prices  = np.empty(m)
        times   = np.empty(m)

        for rep in range(m):
            ss  = np.random.SeedSequence(seed_base + gi * 10_000 + rep)
            rng = np.random.default_rng(ss)

            t0 = time.perf_counter()

            total_payoff = 0.0
            for start in range(0, n_paths, batch_size):
                n       = min(batch_size, n_paths - start)
                S       = batch_fn(n, h, params, rng)
                avg     = S.mean(axis=1)
                total_payoff += np.maximum(avg - K, 0.0).sum()

            price = disc * total_payoff / n_paths

            times[rep]  = time.perf_counter() - t0
            prices[rep] = price

        results.append((prices, times))

    all_prices = [r[0] for r in results]
    all_times  = [r[1] for r in results]
    return all_prices, all_times

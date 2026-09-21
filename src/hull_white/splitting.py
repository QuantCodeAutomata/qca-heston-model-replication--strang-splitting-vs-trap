"""
Hull-White stochastic-volatility European-call Monte Carlo.

Three discretisation schemes
-----------------------------
Strang splitting   – Proposition 3 of the paper.
Trapezoidal        – Exact GBM V step + trapezoidal integral approximation.
Euler              – Exact GBM V step + left-point (Euler) integral.

Model
-----
dX_t = (r - V_t/2) dt + sqrt(V_t) dZ_t
dV_t = mu V_t dt + xi V_t dW_t           (geometric Brownian motion)
dZ_t = rho dW_t + sqrt(1-rho^2) dW_perp_t

Auxiliary constant
------------------
c = mu - xi^2/4

Splitting of V's SDE
--------------------
A (deterministic): dV =  c V dt          =>  V(h) = exp(c h) V_0
B (stochastic)   : dV = -(xi^2/4) V dt + xi V dW
                   => V(h) = V_0 exp(-xi^2 h/4 + xi DeltaW),  DeltaW ~ N(0,h)

Reference parameters (Table 1 / Exp 3)
--------------------------------------
S0=100, V0=0.04, mu=0.2, xi=2, rho=-0.2, T=1, r=0.02, K=100.
Benchmark European call: 8.0361.
"""

import time
from typing import List, Tuple

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BATCH = 50_000


# ---------------------------------------------------------------------------
# Internal per-batch simulators  (return terminal S, shape (n,))
# ---------------------------------------------------------------------------

def _strang_batch(n: int, h: float, p: dict, rng: np.random.Generator) -> np.ndarray:
    """
    Hull-White Strang-splitting simulation for n paths.

    Step (Proposition 3):
      1.  V_tilde = exp(c h/2) * V_t          deterministic half-step
      2.  DeltaW ~ N(0, h)
      3.  V_hat  = V_tilde * exp(-xi^2 h/4 + xi * DeltaW)   stochastic full step
      4.  Ibar   = (V_t + V_hat) * (exp(c h/2) - 1) / c     (= h/2 if c == 0)
      5.  X_next = X + r h - Ibar/2
                     + (2 rho/xi)*(sqrt(V_hat) - sqrt(V_tilde))
                     + N(0, (1-rho^2) * Ibar)
      6.  V_next = exp(c h/2) * V_hat
    """
    S0, V0 = p['S0'], p['V0']
    mu, xi, rho, r, T = p['mu'], p['xi'], p['rho'], p['r'], p['T']
    c = mu - 0.25 * xi * xi

    steps  = max(1, round(T / h))
    ekh2   = np.exp(c * h / 2.0)        # exp(c h/2)
    rho2_c = 1.0 - rho * rho

    # Ibar weight = (exp(c h/2) - 1) / c, or h/2 when c ≈ 0
    if abs(c) < 1e-12:
        ibar_w = h / 2.0
    else:
        ibar_w = (ekh2 - 1.0) / c       # positive for both c>0 and c<0

    # stochastic B-step exponent constant
    b_const = -0.25 * xi * xi * h       # -xi^2 h/4

    X = np.full(n, np.log(float(S0)), dtype=np.float64)
    V = np.full(n, float(V0),         dtype=np.float64)

    for _ in range(steps):
        V_tilde  = ekh2 * V
        DeltaW   = np.sqrt(h) * rng.standard_normal(n)
        V_hat    = V_tilde * np.exp(b_const + xi * DeltaW)

        Ibar     = (V + V_hat) * ibar_w
        np.maximum(Ibar, 0.0, out=Ibar)

        corr     = (2.0 * rho / xi) * (np.sqrt(V_hat) - np.sqrt(V_tilde))
        gauss_sd = np.sqrt(rho2_c * Ibar)
        X       += r * h - 0.5 * Ibar + corr + gauss_sd * rng.standard_normal(n)

        V        = ekh2 * V_hat

    return np.exp(X)          # S_T, shape (n,)


def _trapezoidal_batch(n: int, h: float, p: dict, rng: np.random.Generator) -> np.ndarray:
    """
    Hull-White trapezoidal simulation for n paths.

    V is advanced by exact GBM (log-normal step).
    The integrated variance I = h*(V_t + V_{t+h})/2 (trapezoidal).
    The corr term uses the Itô approximation  rho * int sqrt(V) dW
                                           ≈ (2 rho/xi)*(sqrt(V_{t+h}) - sqrt(V_t)).

    Step:
      DeltaW   ~ N(0, h)
      V_next   = V_t * exp((mu - xi^2/2) h + xi * DeltaW)
      I_trap   = h*(V_t + V_next)/2
      corr     = (2 rho/xi)*(sqrt(V_next) - sqrt(V_t))
      X_next   = X + r h - I_trap/2 + corr + N(0, (1-rho^2) I_trap)
    """
    S0, V0 = p['S0'], p['V0']
    mu, xi, rho, r, T = p['mu'], p['xi'], p['rho'], p['r'], p['T']

    steps      = max(1, round(T / h))
    log_drift  = (mu - 0.5 * xi * xi) * h   # log drift for exact GBM step
    rho2_c     = 1.0 - rho * rho
    two_rho_xi = 2.0 * rho / xi

    # Itô correction: sqrt(V)·dW = d(2√V/ξ) − (μ/ξ − ξ/4)·√V·dt
    # => ρ·∫sqrt(V)dW ≈ (2ρ/ξ)·(√V_next − √V) − ρ·(μ/ξ − ξ/4)·J_trap
    ito_factor = -rho * (mu / xi - xi / 4.0)   # coefficient of J_trap correction

    X = np.full(n, np.log(float(S0)), dtype=np.float64)
    V = np.full(n, float(V0),         dtype=np.float64)

    for _ in range(steps):
        DeltaW  = np.sqrt(h) * rng.standard_normal(n)
        V_sqrt  = np.sqrt(V)
        V_next  = V * np.exp(log_drift + xi * DeltaW)
        Vn_sqrt = np.sqrt(V_next)

        I_trap   = 0.5 * h * (V + V_next)
        J_trap   = 0.5 * h * (V_sqrt + Vn_sqrt)
        corr     = two_rho_xi * (Vn_sqrt - V_sqrt) + ito_factor * J_trap
        gauss_sd = np.sqrt(rho2_c * I_trap)
        X       += r * h - 0.5 * I_trap + corr + gauss_sd * rng.standard_normal(n)

        V        = V_next

    return np.exp(X)


def _euler_batch(n: int, h: float, p: dict, rng: np.random.Generator) -> np.ndarray:
    """
    Hull-White Euler simulation for n paths.

    V is advanced by exact GBM (log-normal step); X uses the left-point
    Euler integral  I_euler = h * V_t.

    Step:
      DeltaW     ~ N(0, h),  DeltaW_perp ~ N(0, h)  (independent)
      V_next     = V_t * exp((mu - xi^2/2) h + xi * DeltaW)
      I_euler    = h * V_t
      X_next     = X + r h - I_euler/2
                     + sqrt(V_t)*(rho * DeltaW + sqrt(1-rho^2) * DeltaW_perp)
    """
    S0, V0 = p['S0'], p['V0']
    mu, xi, rho, r, T = p['mu'], p['xi'], p['rho'], p['r'], p['T']

    steps     = max(1, round(T / h))
    log_drift = (mu - 0.5 * xi * xi) * h
    rho2_sq   = np.sqrt(1.0 - rho * rho)

    X = np.full(n, np.log(float(S0)), dtype=np.float64)
    V = np.full(n, float(V0),         dtype=np.float64)

    for _ in range(steps):
        DeltaW      = np.sqrt(h) * rng.standard_normal(n)
        DeltaW_perp = np.sqrt(h) * rng.standard_normal(n)
        V_sqrt      = np.sqrt(V)

        V_next  = V * np.exp(log_drift + xi * DeltaW)
        X      += r * h - 0.5 * h * V + V_sqrt * (rho * DeltaW + rho2_sq * DeltaW_perp)
        V       = V_next

    return np.exp(X)


# ---------------------------------------------------------------------------
# Public path-returning wrappers
# ---------------------------------------------------------------------------

def _run_batched_hw(batch_fn, n_paths: int, h: float, params: dict,
                    rng: np.random.Generator, batch_size: int) -> np.ndarray:
    """Concatenate per-batch S_T arrays; result shape (n_paths,)."""
    parts = []
    for start in range(0, n_paths, batch_size):
        n = min(batch_size, n_paths - start)
        parts.append(batch_fn(n, h, params, rng))
    return np.concatenate(parts)


def hull_white_strang_paths(n_paths: int, h: float, params: dict,
                            rng: np.random.Generator,
                            batch_size: int = _BATCH) -> np.ndarray:
    """Hull-White Strang splitting; returns S_T, shape (n_paths,)."""
    return _run_batched_hw(_strang_batch, n_paths, h, params, rng, batch_size)


def hull_white_trapezoidal_paths(n_paths: int, h: float, params: dict,
                                 rng: np.random.Generator,
                                 batch_size: int = _BATCH) -> np.ndarray:
    """Hull-White trapezoidal; returns S_T, shape (n_paths,)."""
    return _run_batched_hw(_trapezoidal_batch, n_paths, h, params, rng, batch_size)


def hull_white_euler_paths(n_paths: int, h: float, params: dict,
                           rng: np.random.Generator,
                           batch_size: int = _BATCH) -> np.ndarray:
    """Hull-White Euler; returns S_T, shape (n_paths,)."""
    return _run_batched_hw(_euler_batch, n_paths, h, params, rng, batch_size)


# ---------------------------------------------------------------------------
# Pricing utility
# ---------------------------------------------------------------------------

def european_call_price(S_T: np.ndarray, K: float, r: float, T: float) -> float:
    """Discounted European call price from terminal stock prices."""
    payoff = np.maximum(S_T - K, 0.0)
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
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Run m Monte-Carlo repetitions for each (n_paths, h) grid point.

    Parameters
    ----------
    method : {'strang', 'trapezoidal', 'euler'}
    params : dict
        HW parameters including 'K', 'r', 'T'.
    grid_pairs : list of (n_paths, h)
    m : int
    seed_base : int
    batch_size : int

    Returns
    -------
    list of (prices, cpu_times) – each ndarray, shape (m,).
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
                S_T     = batch_fn(n, h, params, rng)
                total_payoff += np.maximum(S_T - K, 0.0).sum()

            price = disc * total_payoff / n_paths

            times[rep]  = time.perf_counter() - t0
            prices[rep] = price

        results.append((prices, times))

    all_prices = [r[0] for r in results]
    all_times  = [r[1] for r in results]
    return all_prices, all_times

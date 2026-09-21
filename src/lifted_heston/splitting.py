"""
Lifted Heston Strang splitting Monte Carlo.

Model (Proposition 4):
    dX_t = -(V_t^N / 2) dt + sqrt(V_t^N) dZ_t
    dV_{t,i} = -x_i (V_{t,i} - v0_i) dt + (theta - lambda V_t^N) dt
               + nu sqrt(V_t^N) dW_t    for i = 1,...,Nf
    V_t^N = sum_i w_i V_{t,i}
    dZ_t  = rho dW_t + sqrt(1-rho^2) dW_t^perp

Matrix form:
    A     = lambda * ones @ w_row + diag(x)   [Nf x Nf]
    b_vec = theta * ones + diag(x) @ v0_vec   [Nf]

Deterministic sub-problem solution:
    V^(1)(t+s) = exp(-A s) V + A^{-1}(I - exp(-A s)) b

Strang splitting one step over h  (Eq. 24):
  1. Det half-step:  V_half = exp(-A h/2) V + A^{-1}(I-exp(-Ah/2)) b
  2. V_tilde_N = w @ V_half
  3. Sample Vhat_N ~ CIR(0, 0, nu*sum(w), h, V_tilde_N)
  4. Common shift delta = (Vhat_N - V_tilde_N) / sum(w); V_hat = V_half + delta
  5. Det half-step:  V_full = exp(-A h/2) V_hat + A^{-1}(I-exp(-Ah/2)) b
  6. Compute integrated variance I_total = I_det1 + I_stoch + I_det2
  7. X_{t+h} = X + (-I_total/2) + (rho/(nu*sum(w)))*(Vhat_N-V_tilde_N)
             + N(0, (1-rho^2) I_total)
  8. V_{t+h} = V_full

Comparator (drift-implicit Euler), simplified:
    V_{t+h} = (I + diag(x_eff) h)^{-1} (V_t + b h + nu sqrt(max(V_t^N,0)) h G_V)
    X_{t+h} = X_t - (V_t^N)^+ h/2 + sqrt((V_t^N)^+) (rho G_W + sqrt(1-rho^2) G_p) sqrt(h)

Parameters:
    lambda=0.3, nu=0.3, theta=0.02, V0_N=0.02, rho=-0.7, S0=1
    x=[0.63781, 9.6554, 681.37], w=[0.66909, 3.3694, 184.50]
    Benchmarks: K=0.6 -> 0.40291, K=1 -> 0.056555

Initial values: v0_i = V0_N / sum(w) for all i, so that V0_N = sum(w) * v0_common.
"""

import time
from typing import Dict, List, Tuple

import numpy as np
from scipy.linalg import expm, solve
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.cir_sampler import cir_sample_alfonsi

_BATCH = 20_000          # paths per memory chunk (Nf=3 factor state is heavier)


# ---------------------------------------------------------------------------
# Precomputed matrices for a given step size h
# ---------------------------------------------------------------------------

class LiftedHestonCache:
    """
    Precomputed matrix quantities for the lifted Heston Strang step at fixed h.

    Attributes
    ----------
    exp_Ah2 : (Nf,Nf) exp(-A h/2)
    Ainv    : (Nf,Nf) A^{-1}
    M1_h2   : (Nf,Nf) A^{-1}(I - exp(-Ah/2))       – used in det half-step
    M2_h2   : (Nf,Nf) A^{-1}(h/2*I - M1_h2)         – used in integrated V
    sum_w   : float  sum_i w_i
    sigma_N : float  nu * sum_w  (diffusion coeff of V^N in stoch sub-problem)
    """

    def __init__(self, params: dict, h: float) -> None:
        lam = params['lambda']
        nu  = params['nu']
        x   = np.asarray(params['x'], dtype=float)
        w   = np.asarray(params['w'], dtype=float)
        v0  = params['v0_vec']          # (Nf,)
        theta = params['theta']
        Nf = len(x)

        ones = np.ones(Nf)
        # A = lambda * outer(ones, w) + diag(x)
        self.A = lam * np.outer(ones, w) + np.diag(x)
        self.b = theta * ones + np.diag(x) @ v0

        self.Ainv    = np.linalg.inv(self.A)
        I_Nf         = np.eye(Nf)
        self.exp_Ah2 = expm(-self.A * h / 2.0)
        self.M1_h2   = self.Ainv @ (I_Nf - self.exp_Ah2)                   # A^{-1}(I-exp(-Ah/2))
        self.M2_h2   = self.Ainv @ (h / 2.0 * I_Nf - self.M1_h2)          # A^{-1}(h/2 I - M1)

        self.w       = w
        self.sum_w   = float(w.sum())
        self.sigma_N = nu * self.sum_w       # diffusion of aggregate V^N under stoch sub-problem


def _integrate_det_variance(
    V_init: np.ndarray,   # (n_paths, Nf)
    cache: LiftedHestonCache,
) -> np.ndarray:
    """
    Compute w-weighted integral of V^(1)(s) for s in [0, h/2].

    integral_0^{h/2} V^N(s) ds
        = w @ M1_h2 @ V_init  +  w @ M2_h2 @ b_vec   (per path)

    Parameters
    ----------
    V_init : (n_paths, Nf)

    Returns
    -------
    (n_paths,) integrated V^N over [0, h/2]
    """
    # (n_paths, Nf) @ (Nf,) -> (n_paths,)
    wM1 = cache.w @ cache.M1_h2          # (Nf,) row vector
    wM2 = cache.w @ cache.M2_h2          # (Nf,) row vector
    return V_init @ wM1 + (wM2 @ cache.b)


def _det_half_step(
    V: np.ndarray,    # (n_paths, Nf)
    cache: LiftedHestonCache,
) -> np.ndarray:
    """
    Deterministic half-step: V_half = exp(-Ah/2) V + A^{-1}(I-exp(-Ah/2)) b.

    Parameters
    ----------
    V : (n_paths, Nf)

    Returns
    -------
    (n_paths, Nf)
    """
    det_const = cache.M1_h2 @ cache.b    # (Nf,)  constant part
    return V @ cache.exp_Ah2.T + det_const   # broadcasting over paths


# ---------------------------------------------------------------------------
# Per-batch simulators
# ---------------------------------------------------------------------------

def _strang_lh_batch(
    n: int,
    h: float,
    params: dict,
    cache: LiftedHestonCache,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate n paths with Lifted Heston Strang splitting.

    Returns
    -------
    S_T : (n,)  terminal asset prices
    """
    S0   = float(params['S0'])
    rho  = float(params['rho'])
    nu   = float(params['nu'])
    v0   = np.asarray(params['v0_vec'], dtype=float)   # (Nf,)
    Nf   = len(v0)

    n_steps = max(1, round(params['T'] / h))
    rho2_c  = 1.0 - rho ** 2
    sum_w   = cache.sum_w
    sigma_N = cache.sigma_N

    # Initialise paths
    X = np.full(n, np.log(S0), dtype=np.float64)
    V = np.tile(v0, (n, 1)).astype(np.float64)    # (n, Nf)

    for _ in range(n_steps):
        # Step 1: first deterministic half-step
        V_half = _det_half_step(V, cache)
        V_half = np.maximum(V_half, 0.0)

        # Step 2: aggregate
        V_tilde_N = V_half @ cache.w        # (n,)

        # Step 3: stoch full step on aggregate V^N via exact CIR
        #   CIR(0, 0, sigma_N, h, V_tilde_N)
        Vhat_N = cir_sample_alfonsi(0.0, 0.0, sigma_N, h, V_tilde_N, rng)
        Vhat_N = np.maximum(Vhat_N, 0.0)

        # Step 4: common shift and update all factor components
        delta = (Vhat_N - V_tilde_N) / sum_w    # (n,) per-factor increment
        V_hat = V_half + delta[:, np.newaxis]    # (n, Nf)
        V_hat = np.maximum(V_hat, 0.0)

        # Step 5: second deterministic half-step
        V_full = _det_half_step(V_hat, cache)
        V_full = np.maximum(V_full, 0.0)

        # Step 6: integrated variance
        I_det1  = _integrate_det_variance(V,     cache)              # (n,)
        I_stoch = 0.5 * (V_tilde_N + Vhat_N) * h                    # trapezoidal
        I_det2  = _integrate_det_variance(V_hat, cache)              # (n,)
        I_total = np.maximum(I_det1 + I_stoch + I_det2, 0.0)

        # Step 7: log-price update
        dX_corr = (rho / sigma_N) * (Vhat_N - V_tilde_N)
        G = rng.standard_normal(n)
        X = X - 0.5 * I_total + dX_corr + np.sqrt(rho2_c * I_total) * G

        # Step 8: update state
        V = V_full

    return np.exp(X)


def _di_euler_lh_batch(
    n: int,
    h: float,
    params: dict,
    cache: LiftedHestonCache,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Drift-implicit Euler comparator for Lifted Heston (Eq. 49 style).

    V_{t+h} = (I + A h)^{-1} [V_t + b h + nu sqrt((V_t^N)^+) h G_W sum_w]
    X_{t+h} = X_t - (V_t^N)^+ h/2 + sqrt((V_t^N)^+) [rho G_W + sqrt(1-rho^2) G_p] sqrt(h)
    """
    S0    = float(params['S0'])
    rho   = float(params['rho'])
    nu    = float(params['nu'])
    v0    = np.asarray(params['v0_vec'], dtype=float)
    sum_w = cache.sum_w

    n_steps = max(1, round(params['T'] / h))
    rho2_c  = 1.0 - rho ** 2
    sqrt_h  = np.sqrt(h)

    # Precompute (I + A h)^{-1}
    I_plus_Ah_inv = np.linalg.inv(np.eye(len(v0)) + cache.A * h)

    X = np.full(n, np.log(S0), dtype=np.float64)
    V = np.tile(v0, (n, 1)).astype(np.float64)   # (n, Nf)

    for _ in range(n_steps):
        VN_pos = np.maximum(V @ cache.w, 0.0)       # (V_t^N)^+

        G_W = rng.standard_normal(n)    # correlated Brownian
        G_p = rng.standard_normal(n)    # independent component

        # Implicit V update (common Brownian shock shifts all factors equally)
        stoch_shift = nu * np.sqrt(VN_pos) * sqrt_h * G_W    # (n,)
        V_rhs = V + cache.b * h + stoch_shift[:, np.newaxis]  # broadcast over Nf
        V = (V_rhs @ I_plus_Ah_inv.T)                         # (n, Nf)
        V = np.maximum(V, 0.0)

        # X update
        X = (
            X
            - 0.5 * VN_pos * h
            + np.sqrt(VN_pos) * sqrt_h * (rho * G_W + np.sqrt(rho2_c) * G_p)
        )

    return np.exp(X)


# ---------------------------------------------------------------------------
# Full-path simulator with batching
# ---------------------------------------------------------------------------

def simulate_paths(
    method: str,
    n_paths: int,
    h: float,
    params: dict,
    rng: np.random.Generator,
    batch_size: int = _BATCH,
) -> np.ndarray:
    """
    Simulate terminal asset prices in memory-safe batches.

    Parameters
    ----------
    method  : 'strang' or 'di_euler'
    n_paths : int
    h       : float  step size
    params  : dict   model parameters
    rng     : Generator

    Returns
    -------
    S_T : (n_paths,) terminal asset prices
    """
    cache = LiftedHestonCache(params, h)
    if method == 'strang':
        sim_fn = lambda n, rng: _strang_lh_batch(n, h, params, cache, rng)
    else:
        sim_fn = lambda n, rng: _di_euler_lh_batch(n, h, params, cache, rng)

    chunks = []
    remaining = n_paths
    while remaining > 0:
        n_chunk = min(remaining, batch_size)
        chunks.append(sim_fn(n_chunk, rng))
        remaining -= n_chunk
    return np.concatenate(chunks)


# ---------------------------------------------------------------------------
# European call payoff
# ---------------------------------------------------------------------------

def european_call_price(S_T: np.ndarray, K: float, r: float, T: float) -> float:
    """Discounted European call price from terminal asset samples."""
    return float(np.exp(-r * T) * np.mean(np.maximum(S_T - K, 0.0)))


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment_grid(
    method: str,
    params: dict,
    grid_pairs: List[Tuple[int, float]],
    m: int,
    benchmark: float,
    K_strike: float,
    seed_base: int = 42,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Run m repetitions per (n_paths, h) pair.

    Returns
    -------
    all_prices : list[ndarray(m,)]
    all_cpus   : list[ndarray(m,)]
    """
    r = params.get('r', 0.0)
    T = params['T']

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
            S_T = simulate_paths(method, n_paths, h, params, rng)
            price = european_call_price(S_T, K_strike, r, T)
            elapsed = time.perf_counter() - t0
            prices[rep] = price
            cpus[rep]   = elapsed

        all_prices.append(prices)
        all_cpus.append(cpus)
        print(
            f"  [{method} K={K_strike}] n={n_paths:>9,} h={h:.6f}  "
            f"mean={prices.mean():.6f}  bias={prices.mean()-benchmark:+.6f}  "
            f"cpu={cpus.mean():.2f}s"
        )

    return all_prices, all_cpus

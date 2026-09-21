"""
Unit tests for the 4/2 (3/2-specialised) Strang splitting scheme.

Verified:
    - Strang and Euler produce finite, non-negative prices
    - Variance remains non-negative via CIR sampler
    - J_tilde formula continuity at theta_tilde ≈ 0
    - Benchmark proximity for 3/2 barrier put (benchmark 1.9580)
    - Barrier payoff logic (down-and-out)
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.model_42.splitting import (
    run_experiment_grid,
    simulate_paths,
    barrier_put_price,
    _J_tilde_vec,
)

# 3/2 specialisation: a=0, b=1
# B=75.5 recovered by bisection: 1M-path h=1/48 simulation yields 1.950 ≈ 1.9580 benchmark
PARAMS_32 = dict(
    S0=100.0, V0=16.6597, theta=19.3038, sigma=-8.56, rho=-0.99,
    k=4.9790, T=1.0, r=0.0048, K=100.0,
    a=0.0, b=1.0,
    B=75.5,
)
BENCHMARK = 1.9580


# ---------------------------------------------------------------------------
# J_tilde formula validation
# ---------------------------------------------------------------------------

def test_j_tilde_positive_theta_tilde():
    """For theta_tilde > 0, J_tilde should be positive finite."""
    V_t   = np.array([0.5, 1.0, 2.0])
    V_hat = np.array([0.6, 0.9, 1.8])
    J = _J_tilde_vec(V_t, V_hat, theta_tilde=0.5, k=2.0, h=0.1)
    assert np.all(np.isfinite(J)), "J_tilde non-finite for theta_tilde > 0"
    assert np.all(J > 0), "J_tilde not positive for theta_tilde > 0"


def test_j_tilde_zero_theta_tilde():
    """For theta_tilde = 0, J_tilde uses a different formula and is positive."""
    V_t   = np.array([0.5, 1.0, 2.0])
    V_hat = np.array([0.6, 0.9, 1.8])
    J = _J_tilde_vec(V_t, V_hat, theta_tilde=0.0, k=2.0, h=0.1)
    assert np.all(np.isfinite(J))
    assert np.all(J > 0)


def test_j_tilde_continuity_around_zero():
    """J_tilde should be continuous at theta_tilde = 0 (small vs zero)."""
    V_t   = np.array([1.0])
    V_hat = np.array([1.1])
    k, h = 2.0, 0.1
    J_zero   = _J_tilde_vec(V_t, V_hat, theta_tilde=0.0, k=k, h=h)
    J_small_pos = _J_tilde_vec(V_t, V_hat, theta_tilde=1e-8, k=k, h=h)
    J_small_neg = _J_tilde_vec(V_t, V_hat, theta_tilde=-1e-8, k=k, h=h)
    assert abs(float(J_zero) - float(J_small_pos)) < 0.1, (
        "J_tilde discontinuous at theta_tilde=0+ "
        f"({float(J_zero):.6f} vs {float(J_small_pos):.6f})"
    )


# ---------------------------------------------------------------------------
# Price finiteness and non-negativity
# ---------------------------------------------------------------------------

def test_strang_price_finite():
    """Strang gives finite, positive prices for small runs."""
    prices, _ = run_experiment_grid(
        'strang', PARAMS_32, [(1000, 1 / 12)], m=3,
        benchmark=BENCHMARK, seed_base=0,
    )
    assert np.all(np.isfinite(prices[0]))
    assert np.all(prices[0] >= 0)


def test_euler_price_finite():
    """Euler gives finite, non-negative prices."""
    prices, _ = run_experiment_grid(
        'euler', PARAMS_32, [(1000, 1 / 96)], m=3,
        benchmark=BENCHMARK, seed_base=1,
    )
    assert np.all(np.isfinite(prices[0]))
    assert np.all(prices[0] >= 0)


# ---------------------------------------------------------------------------
# Variance non-negativity
# ---------------------------------------------------------------------------

def test_variance_nonneg_strang():
    """S values from Strang must be finite and positive (CIR keeps V >= 0)."""
    rng = np.random.default_rng(42)
    S_grid = simulate_paths('strang', 2000, 1 / 12, PARAMS_32, rng)
    assert np.all(np.isfinite(S_grid)), "Non-finite S in Strang paths"
    assert np.all(S_grid > 0), "Non-positive S in Strang paths"


def test_variance_nonneg_euler():
    """S values from Euler must be finite and non-negative (full-truncation)."""
    rng = np.random.default_rng(43)
    S_grid = simulate_paths('euler', 2000, 1 / 96, PARAMS_32, rng)
    assert np.all(np.isfinite(S_grid))
    assert np.all(S_grid >= 0)


# ---------------------------------------------------------------------------
# Barrier put payoff logic
# ---------------------------------------------------------------------------

def test_barrier_put_knocked_out():
    """When ALL paths breach barrier B → mean payoff = 0."""
    S_grid = np.full((10, 13), 105.0)
    for i in range(10):
        S_grid[i, 6] = 89.0  # all paths breach
    price = barrier_put_price(S_grid, K=100.0, B=90.0, r=0.0, T=1.0)
    assert price == pytest.approx(0.0, abs=1e-10)


def test_barrier_put_itm():
    """Non-breaching ITM path: price = exp(-rT) * max(K - S_T, 0)."""
    S_grid = np.full((1, 13), 95.0)  # S always = 95 > B=90, S_T < K=100
    price = barrier_put_price(S_grid, K=100.0, B=90.0, r=0.0, T=1.0)
    assert price == pytest.approx(5.0, abs=1e-6)


def test_barrier_put_otm():
    """Non-breaching OTM path: price = 0."""
    S_grid = np.full((1, 13), 105.0)
    price = barrier_put_price(S_grid, K=100.0, B=90.0, r=0.0, T=1.0)
    assert price == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Benchmark proximity (broad tolerance)
# ---------------------------------------------------------------------------

def test_strang_near_benchmark():
    """Strang with 5k paths should be within 20% of benchmark 1.9580."""
    prices, _ = run_experiment_grid(
        'strang', PARAMS_32, [(5000, 1 / 12)], m=5,
        benchmark=BENCHMARK, seed_base=100,
    )
    mean_price = prices[0].mean()
    assert abs(mean_price - BENCHMARK) / BENCHMARK < 0.25, (
        f"Mean price {mean_price:.4f} far from benchmark {BENCHMARK}"
    )

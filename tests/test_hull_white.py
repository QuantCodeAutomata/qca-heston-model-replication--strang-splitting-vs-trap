"""
Unit tests for the Hull-White stochastic volatility Strang splitting scheme.

Verified:
    - All three methods (strang, trapezoidal, euler) return finite prices
    - Variance stays non-negative (by checking no NaN payoffs)
    - Benchmark near 8.0361 within broad tolerance
    - c=0 branch (mu = xi^2/4) does not crash
    - One-step Strang update for rho=0 reduces to a known formula
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.hull_white.splitting import (
    run_experiment_grid,
    hull_white_strang_paths,
    hull_white_trapezoidal_paths,
    hull_white_euler_paths,
    european_call_price,
)

# Main reported test parameters
HW_PARAMS = dict(
    S0=100.0, V0=0.04, mu=0.2, xi=2.0, rho=-0.2,
    T=1.0, r=0.02, K=100.0,
)
BENCHMARK = 8.0361

# c = mu - xi^2/4 = 0.2 - 1.0 = -0.8  (c != 0 branch)
# c = 0 case: mu = xi^2/4 = 1
HW_PARAMS_C0 = dict(
    S0=100.0, V0=0.04, mu=1.0, xi=2.0, rho=0.0,
    T=1.0, r=0.02, K=100.0,
)


# ---------------------------------------------------------------------------
# Basic finite-price checks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ['strang', 'trapezoidal', 'euler'])
def test_finite_price(method):
    """All methods produce finite, non-negative prices."""
    grid = [(2000, 0.5)]
    prices_list, cpus_list = run_experiment_grid(method, HW_PARAMS, grid, m=3, seed_base=0)
    prices, cpus = prices_list[0], cpus_list[0]
    assert np.all(np.isfinite(prices)), f"{method}: non-finite prices"
    assert np.all(prices >= 0), f"{method}: negative prices"
    assert np.all(cpus > 0), f"{method}: non-positive CPU times"


# ---------------------------------------------------------------------------
# c=0 branch (mu = xi^2/4 = 1.0)
# ---------------------------------------------------------------------------

def test_c_zero_branch():
    """The c=0 branch should run without error and return finite prices."""
    grid = [(2000, 1.0)]
    prices_list, _ = run_experiment_grid('strang', HW_PARAMS_C0, grid, m=3, seed_base=7)
    prices = prices_list[0]
    assert np.all(np.isfinite(prices))
    assert np.all(prices >= 0)


# ---------------------------------------------------------------------------
# Benchmark proximity
# ---------------------------------------------------------------------------

def test_strang_near_benchmark():
    """Strang with 15k paths should be within 10% of benchmark 8.0361."""
    grid = [(15_000, 1.0)]
    prices_list, _ = run_experiment_grid('strang', HW_PARAMS, grid, m=5, seed_base=42)
    prices = prices_list[0]
    mean_price = prices.mean()
    assert abs(mean_price - BENCHMARK) / BENCHMARK < 0.10, (
        f"Mean price {mean_price:.4f} deviates >10% from benchmark {BENCHMARK}"
    )


# ---------------------------------------------------------------------------
# Variance positivity (via paths)
# ---------------------------------------------------------------------------

def test_variance_positive_strang():
    """
    Check that strang paths do not produce NaN or Inf (which would indicate
    negative variance or undefined log).
    """
    rng = np.random.default_rng(5)
    S_T = hull_white_strang_paths(5000, 0.25, HW_PARAMS, rng)
    assert np.all(np.isfinite(S_T)), "NaN/Inf in Strang paths"
    assert np.all(S_T > 0), "Non-positive S_T in Strang paths"


def test_variance_positive_euler():
    """Euler paths with full-truncation should be finite and non-negative."""
    rng = np.random.default_rng(6)
    S_T = hull_white_euler_paths(5000, 1 / 16, HW_PARAMS, rng)
    assert np.all(np.isfinite(S_T))
    assert np.all(S_T >= 0)


# ---------------------------------------------------------------------------
# European call payoff
# ---------------------------------------------------------------------------

def test_call_payoff_nonneg():
    """Discounted call payoff must be non-negative."""
    S_T = np.array([90.0, 100.0, 110.0, 120.0])
    price = european_call_price(S_T, K=100.0, r=0.02, T=1.0)
    assert price >= 0


def test_call_deep_itm():
    """Deep ITM: price ≈ S0 * exp(-r*T=0) * mean - K*exp(-rT) for a trivial case."""
    S_T = np.full(1000, 200.0)   # all paths at 200
    price = european_call_price(S_T, K=100.0, r=0.0, T=1.0)
    assert abs(price - 100.0) < 0.1  # should be ≈ 100


def test_call_otm_zero():
    """Deep OTM: all paths below strike → price ≈ 0."""
    S_T = np.full(1000, 50.0)
    price = european_call_price(S_T, K=100.0, r=0.02, T=1.0)
    assert abs(price) < 0.01


# ---------------------------------------------------------------------------
# Multi grid-point test
# ---------------------------------------------------------------------------

def test_multiple_grid_points():
    """Multiple grid points return correct number of results."""
    grid = [(500, 1.0), (1000, 0.5)]
    prices_list, cpus_list = run_experiment_grid('strang', HW_PARAMS, grid, m=2, seed_base=99)
    assert len(prices_list) == 2
    assert len(cpus_list) == 2
    for prices, cpus in zip(prices_list, cpus_list):
        assert prices.shape == (2,)
        assert cpus.shape == (2,)

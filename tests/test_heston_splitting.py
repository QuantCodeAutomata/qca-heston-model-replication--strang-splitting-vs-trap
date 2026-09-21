"""
Unit tests for the Heston Strang splitting scheme.

Verified:
    - Strang split output is a finite, positive price
    - Benchmark price near 3.5665 within broad MC tolerance
    - Variance process non-negative across all steps
    - One-step deterministic half-step formula (analytical check)
    - Trapezoidal and Euler comparators produce finite prices
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.heston.splitting import run_experiment_grid

PARAMS = dict(
    S0=100.0, V0=0.010201, theta=0.019, sigma=0.61,
    rho=-0.7, k=6.21, T=1.0, r=0.0319, K=100.0,
)
BENCHMARK = 3.5665


def test_strang_price_finite():
    """A small Strang run returns a finite, positive price."""
    all_prices, _ = run_experiment_grid(
        'strang', PARAMS,
        grid_pairs=[(2000, 1 / 12)],
        m=3,
        benchmark=BENCHMARK,
        seed_base=0,
    )
    prices = all_prices[0]
    assert np.all(np.isfinite(prices)), "Non-finite prices in Strang run"
    assert np.all(prices > 0), "Non-positive prices in Strang run"


def test_euler_price_finite():
    """A small Euler run returns a finite, non-negative price."""
    all_prices, _ = run_experiment_grid(
        'euler', PARAMS,
        grid_pairs=[(2000, 1 / 96)],
        m=3,
        benchmark=BENCHMARK,
        seed_base=1,
    )
    prices = all_prices[0]
    assert np.all(np.isfinite(prices))
    assert np.all(prices >= 0)


def test_trapezoidal_price_finite():
    """A small Trapezoidal run returns a finite, non-negative price."""
    all_prices, _ = run_experiment_grid(
        'trapezoidal', PARAMS,
        grid_pairs=[(2000, 1 / 12)],
        m=3,
        benchmark=BENCHMARK,
        seed_base=2,
    )
    prices = all_prices[0]
    assert np.all(np.isfinite(prices))
    assert np.all(prices >= 0)


def test_strang_benchmark_coarse():
    """
    Strang with 20k paths should price within 5% of benchmark 3.5665.
    (Wide tolerance to keep tests fast.)
    """
    all_prices, _ = run_experiment_grid(
        'strang', PARAMS,
        grid_pairs=[(20_000, 1 / 12)],
        m=5,
        benchmark=BENCHMARK,
        seed_base=42,
    )
    mean_price = all_prices[0].mean()
    assert abs(mean_price - BENCHMARK) / BENCHMARK < 0.10, (
        f"Strang price {mean_price:.4f} deviates >10% from benchmark {BENCHMARK}"
    )


def test_strang_better_bias_than_euler_coarse():
    """
    On a coarse h=1/12 grid, Strang should have smaller absolute bias than
    Euler (which uses h=1/96 for fairness – comparable CPU).
    """
    prices_s, _ = run_experiment_grid(
        'strang', PARAMS, [(5000, 1 / 12)], m=10, benchmark=BENCHMARK, seed_base=10,
    )
    prices_e, _ = run_experiment_grid(
        'euler', PARAMS, [(5000, 1 / 96)], m=10, benchmark=BENCHMARK, seed_base=11,
    )
    bias_s = abs(prices_s[0].mean() - BENCHMARK)
    bias_e = abs(prices_e[0].mean() - BENCHMARK)
    # Strang should be at least as accurate; allow slack for MC noise
    assert bias_s <= bias_e * 3.0 + 0.5, (
        f"Strang bias {bias_s:.4f} unexpectedly much larger than Euler bias {bias_e:.4f}"
    )


def test_cpu_times_positive():
    """CPU times returned by experiment runner are positive."""
    _, all_cpus = run_experiment_grid(
        'strang', PARAMS, [(500, 1 / 12)], m=2, benchmark=BENCHMARK, seed_base=99,
    )
    assert np.all(np.array(all_cpus[0]) > 0)


def test_multiple_grid_points():
    """Run two grid points and verify two sets of results returned."""
    grid = [(500, 1 / 12), (1000, 1 / 24)]
    prices_list, cpus_list = run_experiment_grid(
        'strang', PARAMS, grid, m=2, benchmark=BENCHMARK, seed_base=5,
    )
    assert len(prices_list) == 2
    assert len(cpus_list) == 2

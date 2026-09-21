"""
Unit tests for the Alfonsi exact CIR sampler.

Verified properties:
    - All samples non-negative
    - First and second moment match theoretical CIR transition moments
    - Edge cases: alpha=0 (absorption boundary), beta=0, sigma→0
    - Vectorised (array v0) and scalar v0 both work
    - Single-path case (size=1)
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.utils.cir_sampler import cir_sample_alfonsi, zeta


# ---------------------------------------------------------------------------
# zeta helper
# ---------------------------------------------------------------------------

def test_zeta_nonzero_beta():
    """zeta(beta, t) = (1-exp(-beta*t))/beta for beta != 0."""
    beta, t = 2.0, 0.5
    expected = (1.0 - np.exp(-beta * t)) / beta
    assert abs(zeta(beta, t) - expected) < 1e-12


def test_zeta_zero_beta():
    """zeta(0, t) = t (L'Hôpital limit)."""
    assert abs(zeta(0.0, 1.0) - 1.0) < 1e-12


def test_zeta_large_beta():
    """For large beta, zeta → 1/beta."""
    assert abs(zeta(1e6, 1.0) - 1e-6) < 1e-9


# ---------------------------------------------------------------------------
# Nonnegativity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alpha,beta,sigma,v0", [
    (0.5, 1.0, 0.5, 0.5),    # d = 4*0.5/0.25 = 8   >= 1
    (0.01, 1.0, 0.5, 0.1),   # d < 1  (sub-Feller)
    (2.0, 0.0, 1.0, 1.0),    # beta = 0
    (1.0, -0.5, 0.8, 0.5),   # negative beta (growing process)
    (0.0, 1.0, 0.5, 1.0),    # alpha = 0
])
def test_nonnegativity(alpha, beta, sigma, v0):
    """All CIR samples must be non-negative."""
    rng = np.random.default_rng(0)
    samples = cir_sample_alfonsi(alpha, beta, sigma, 1.0, v0, rng, size=5000)
    assert np.all(samples >= 0.0), f"Negative samples found for params ({alpha},{beta},{sigma})"


# ---------------------------------------------------------------------------
# Moment validation (CIR transition moments)
# ---------------------------------------------------------------------------

def theoretical_cir_moments(alpha, beta, sigma, dt, v0):
    """
    Analytical first and second moments of V_T | V_0 = v0.

    E[V_T] = exp(-beta*dt)*v0 + (alpha/beta)*(1-exp(-beta*dt))  [beta != 0]
    Var[V_T] = v0*(sigma^2/beta)*(exp(-beta*dt)-exp(-2*beta*dt))
               + (alpha*sigma^2/(2*beta^2))*(1-exp(-beta*dt))^2  [beta != 0]
    """
    z = zeta(beta, dt)
    e = np.exp(-beta * dt) if abs(beta) > 1e-12 else 1.0
    if abs(beta) > 1e-12:
        mean = e * v0 + alpha * z
        var  = (v0 * sigma ** 2 / beta * (e - e ** 2)
                + alpha * sigma ** 2 / (2.0 * beta ** 2) * (1.0 - e) ** 2)
    else:
        mean = v0 + alpha * dt
        var  = v0 * sigma ** 2 * dt + 0.5 * alpha * sigma ** 2 * dt ** 2
    return mean, var


@pytest.mark.parametrize("alpha,beta,sigma,v0,dt", [
    (2.0, 3.0, 0.5, 1.0, 0.1),
    (0.5, 1.0, 0.5, 0.5, 0.5),
    (1.0, 0.0, 0.5, 1.0, 1.0),
])
def test_moments(alpha, beta, sigma, v0, dt):
    """Empirical moments should match theoretical CIR moments within 3-sigma."""
    rng = np.random.default_rng(42)
    n = 100_000
    samples = cir_sample_alfonsi(alpha, beta, sigma, dt, v0, rng, size=n)

    emp_mean = samples.mean()
    emp_var  = samples.var(ddof=1)

    th_mean, th_var = theoretical_cir_moments(alpha, beta, sigma, dt, v0)

    # Allow 3-sigma tolerance (using CLT for sample mean)
    tol_mean = 3.0 * np.sqrt(th_var / n)
    tol_var  = 3.0 * th_var * np.sqrt(2.0 / n)  # approximation for variance std

    assert abs(emp_mean - th_mean) < tol_mean, (
        f"Mean mismatch: empirical={emp_mean:.5f}, theoretical={th_mean:.5f}"
    )
    assert abs(emp_var - th_var) < max(tol_var, 0.05 * th_var + 1e-6), (
        f"Var mismatch: empirical={emp_var:.5f}, theoretical={th_var:.5f}"
    )


# ---------------------------------------------------------------------------
# Array v0 (vectorised)
# ---------------------------------------------------------------------------

def test_vector_v0():
    """Accept array v0 and return samples of matching length."""
    rng = np.random.default_rng(1)
    v0 = np.array([0.1, 0.5, 1.0, 2.0])
    samples = cir_sample_alfonsi(1.0, 2.0, 0.5, 0.5, v0, rng)
    assert samples.shape == (4,)
    assert np.all(samples >= 0.0)


# ---------------------------------------------------------------------------
# Deterministic special case: sigma → 0
# ---------------------------------------------------------------------------

def test_deterministic_limit():
    """For very small sigma, sample should be near the ODE solution."""
    alpha, beta, sigma, dt, v0 = 1.0, 2.0, 1e-6, 0.5, 0.8
    rng = np.random.default_rng(7)
    sample = cir_sample_alfonsi(alpha, beta, sigma, dt, v0, rng, size=1)[0]
    ode_sol = np.exp(-beta * dt) * v0 + (alpha / beta) * (1.0 - np.exp(-beta * dt))
    assert abs(sample - ode_sol) < 1e-3


# ---------------------------------------------------------------------------
# Sub-Feller regime (d < 1)
# ---------------------------------------------------------------------------

def test_sub_feller_regime():
    """Sub-Feller (d < 1) samples should include zero but remain non-negative."""
    rng = np.random.default_rng(99)
    # d = 4*alpha/sigma^2 = 4*0.01/1.0 = 0.04
    samples = cir_sample_alfonsi(0.01, 1.0, 1.0, 1.0, 0.5, rng, size=10_000)
    assert np.all(samples >= 0.0)


# ---------------------------------------------------------------------------
# Single path (size=1)
# ---------------------------------------------------------------------------

def test_single_path():
    """size=1 returns a length-1 array."""
    rng = np.random.default_rng(3)
    s = cir_sample_alfonsi(1.0, 1.0, 0.5, 0.5, 0.5, rng, size=1)
    assert s.shape == (1,)
    assert s[0] >= 0.0

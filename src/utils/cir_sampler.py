"""
Alfonsi exact sampler for the Cox-Ingersoll-Ross (CIR) process.

Reference: Alfonsi (2010), "High order discretization schemes for the CIR
process: Application to affine term structure and Heston models",
Mathematics of Computation 79(269), pp. 209-237.

Process SDE:   dV_t = (alpha - beta*V_t) dt + sigma*sqrt(V_t) dW_t

Key parameters
--------------
d   = 4*alpha / sigma^2   (Feller parameter; d >= 2 guarantees non-absorption)
c_T = 4 / (sigma^2 * zeta(T))
lam = c_T * exp(-beta*T) * V_0   (non-centrality per path)

Exact transition:  V_T | V_0 ~ (1/c_T) * chi^2_d(lam)

Sampling scheme
---------------
Case 1  d >= 1:
    chi^2_d(lam) = (G + sqrt(lam))^2 + chi^2_{d-1}   G ~ N(0,1), independent
    => V_T = (exp(-beta*T/2)*sqrt(V0) + (sigma/2)*sqrt(zeta(T))*G)^2
             + Gamma((d-1)/2, scale=2/c_T)      [Gamma term is 0 when d==1]

Case 2  d < 1  (sub-Feller regime):
    chi^2_d(lam) | N ~ chi^2_{d+2N}            N ~ Poisson(lam/2)
    => V_T | N ~ Gamma(d/2 + N, scale=2/c_T)
"""

import numpy as np
from numpy.random import Generator
from typing import Union


def zeta(beta: float, t: float) -> float:
    """
    Compute zeta_beta(t) = (1 - exp(-beta*t)) / beta.

    For beta == 0 (flat drift) the L'Hopital limit is t.

    Parameters
    ----------
    beta : float
        Mean-reversion speed.  May be zero or negative.
    t : float
        Horizon length (> 0).

    Returns
    -------
    float
    """
    if abs(beta) < 1e-12:
        return t
    return (1.0 - np.exp(-beta * t)) / beta


def cir_sample_alfonsi(
    alpha: float,
    beta: float,
    sigma: float,
    dt: float,
    v0: Union[float, np.ndarray],
    rng: Generator,
    size: int = 1,
) -> np.ndarray:
    """
    Draw exact samples from the CIR transition kernel via Alfonsi's scheme.

    Samples V_dt given V_0 = v0 for the process

        dV = (alpha - beta*V) dt + sigma*sqrt(V) dW

    Parameters
    ----------
    alpha : float
        Long-run drift parameter (alpha > 0).
    beta : float
        Mean-reversion speed.  May be zero or negative.
    sigma : float
        Diffusion coefficient (sigma^2 is used; sign is irrelevant for V).
    dt : float
        Time step (> 0).
    v0 : float or ndarray
        Initial value(s).  If scalar, `size` independent samples are drawn
        from the same starting point.  If an array, one sample is drawn per
        element and `size` is ignored.
    rng : numpy.random.Generator
        Seeded random-number generator (e.g. ``numpy.random.default_rng``).
    size : int, optional
        Number of samples when v0 is scalar.  Default 1.

    Returns
    -------
    np.ndarray, shape (n,)
        Terminal CIR values.  n == size when v0 is scalar, n == len(v0)
        when v0 is an array.

    Notes
    -----
    The Feller parameter d = 4*alpha/sigma^2 determines the sampling branch:

    * d >= 1  (Case 1): non-central chi-squared decomposed as a squared
      Gaussian shift plus an independent central Gamma variate.

    * d < 1   (Case 2): Poisson-mixture of Gamma variates, fully vectorised
      over paths via array-valued shape arguments to ``rng.gamma``.
    """
    sigma2 = sigma ** 2
    z = zeta(beta, dt)
    c_t = 4.0 / (sigma2 * z)           # c(t) = 4 / (sigma^2 * zeta(t))
    d = 4.0 * alpha / sigma2            # Feller parameter

    # ------------------------------------------------------------------
    # Normalise v0 to a 1-D contiguous array; infer size from it when it
    # is not scalar so that the caller never needs to pass size explicitly.
    # ------------------------------------------------------------------
    v0_arr = np.asarray(v0, dtype=float)
    if v0_arr.ndim == 0:
        v0_arr = np.full(size, float(v0))
    else:
        size = v0_arr.size
        v0_arr = v0_arr.ravel()

    # exp(-beta*dt) and exp(-beta*dt/2) used in both branches
    exp_beta = np.exp(-beta * dt) if abs(beta) > 1e-12 else 1.0
    exp_half = np.sqrt(exp_beta)        # exp(-beta*dt/2)

    if d >= 1.0:
        return _case1(d, c_t, z, sigma, exp_half, v0_arr, rng, size)
    else:
        return _case2(d, c_t, exp_beta, v0_arr, rng)


# ---------------------------------------------------------------------------
# Private helpers – one per sampling branch
# ---------------------------------------------------------------------------

def _case1(
    d: float,
    c_t: float,
    z: float,
    sigma: float,
    exp_half: float,
    v0_arr: np.ndarray,
    rng: Generator,
    size: int,
) -> np.ndarray:
    """
    Case 1 sampler  (d >= 1).

    chi^2_d(lam) = (G + sqrt(lam))^2 + chi^2_{d-1}(0)

    After rescaling by 1/c_t the squared term simplifies to:

        (exp_half * sqrt(v0) + (sigma/2) * sqrt(z) * G)^2

    and the chi^2_{d-1}(0) term becomes Gamma((d-1)/2, scale=2/c_t).
    """
    G = rng.standard_normal(size)
    squared_term = (exp_half * np.sqrt(v0_arr) + (abs(sigma) / 2.0) * np.sqrt(z) * G) ** 2

    # chi^2_{d-1}(0) / c_t ~ Gamma((d-1)/2, scale=2/c_t); zero when d == 1.
    if d > 1.0:
        Z = rng.gamma(shape=(d - 1.0) / 2.0, scale=2.0 / c_t, size=size)
    else:
        Z = np.zeros(size)

    return squared_term + Z


def _case2(
    d: float,
    c_t: float,
    exp_beta: float,
    v0_arr: np.ndarray,
    rng: Generator,
) -> np.ndarray:
    """
    Case 2 sampler  (d < 1)  – vectorised Poisson-mixture.

    chi^2_d(lam) | N ~ chi^2_{d+2N}(0) ~ 2*Gamma((d+2N)/2, 1),  N ~ Poisson(lam/2)

    Rescaled:  V_T | N ~ Gamma(d/2 + N, scale=2/c_t)

    When d == 0 (alpha = 0) and N == 0 the Gamma shape is 0, which means the
    process is absorbed at zero; those paths return 0 exactly.
    """
    lam = c_t * exp_beta * v0_arr          # per-path non-centrality, shape (n,)
    N = rng.poisson(lam / 2.0)             # per-path N ~ Poisson(lam/2)
    shapes = d / 2.0 + N                   # per-path Gamma shape = d/2 + N

    if d == 0.0:
        # alpha = 0: shape[i] = N[i]; if N[i]=0 the path is absorbed at zero.
        result = np.zeros(v0_arr.size, dtype=np.float64)
        pos = N > 0
        if pos.any():
            result[pos] = rng.gamma(shape=shapes[pos], scale=2.0 / c_t)
        return result

    return rng.gamma(shape=shapes, scale=2.0 / c_t)


# ---------------------------------------------------------------------------
# Convenience wrapper for vectorised use (one sample per path in v0)
# ---------------------------------------------------------------------------

def cir_sample_alfonsi_vec(
    alpha: float,
    beta: float,
    sigma: float,
    dt: float,
    v0: np.ndarray,
    rng: Generator,
) -> np.ndarray:
    """
    Vectorised Alfonsi exact CIR sampler: one terminal sample per path.

    A thin wrapper around :func:`cir_sample_alfonsi` that makes the
    one-sample-per-path intent explicit in the call site.

    Parameters
    ----------
    alpha, beta, sigma, dt : float
        CIR parameters (see :func:`cir_sample_alfonsi`).
    v0 : ndarray, shape (n_paths,)
        Initial CIR value for each path.
    rng : numpy.random.Generator

    Returns
    -------
    np.ndarray, shape (n_paths,)
        One exact draw from V_{dt} | V_0 = v0[i] per path i.
    """
    return cir_sample_alfonsi(alpha, beta, sigma, dt, v0, rng)

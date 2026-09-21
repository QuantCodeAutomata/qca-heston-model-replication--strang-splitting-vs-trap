"""
BDLP increment samplers for BN-S stochastic volatility models.

Four Background Driving Lévy Process (BDLP) specifications:
    OU-Gamma   : Gamma-process BDLP (increments ~ Gamma)
    Gamma-OU   : Compound-Poisson BDLP with Exp jump sizes
    OU-TS      : Tempered-stable BDLP, alpha=0.5 → Inverse-Gaussian increments
    TS-OU      : Tempered-stable + compound-Poisson BDLP, alpha=0.25

Lévy cumulant kappa(rho) = E[exp(rho*Z_1) - 1] for the base BDLP at unit time.
All kappa values below are pre-derived from the Lévy-Khintchine formula.

OU-Gamma (alpha_g, beta_g):
    Lévy measure : nu(dx) = alpha_g * exp(-beta_g*x) / x   dx  (x>0)
    DeltaZ(h)    ~ Gamma(alpha_g * lambda * h,  scale=1/beta_g)
    kappa(rho)   = alpha_g * log(beta_g / (beta_g - rho))   for rho < beta_g
                 = -alpha_g * log(1 - rho/beta_g)

Gamma-OU (alpha_g, beta_g):
    Lévy measure : nu(dx) = alpha_g * beta_g * exp(-beta_g*x)  dx   (x>0)
                           (compound Poisson, each jump ~ Exp(beta_g), rate alpha_g)
    DeltaZ(h)    : N ~ Pois(alpha_g * lambda * h), jumps J_i ~ Exp(beta_g)
    kappa(rho)   = alpha_g * rho / (beta_g - rho)              for rho < beta_g

OU-TS (alpha_ts, beta_ts, c_scale) with alpha_ts=0.5:
    Lévy measure : nu(dx) = c_scale * x^{-3/2} * exp(-beta_ts*x) / Γ(-1/2+1)  dx
                  (using c_scale = theta, the paper's notation for the intensity scale)
    DeltaZ(h)    ~ IG(mean = c*sqrt(π/beta_ts),  lambda = 2π c²)
                  where c = c_scale * lambda * h
    kappa(rho)   = c_scale * Γ(-0.5) * ((beta_ts-rho)^0.5 - beta_ts^0.5)
                 = 2*c_scale*sqrt(π) * (sqrt(beta_ts) - sqrt(beta_ts-rho))
                                                               for rho < beta_ts

TS-OU (alpha_ts=0.25, beta_ts, c_scale):
    Lévy measure : nu(dx) = c_scale * x^{-1.25} * exp(-beta_ts*x)  dx   (x>0)
    DeltaZ(h) sampled via compound-Poisson approximation with small-jump correction.
    kappa(rho)   = c_scale * Γ(-0.25) * ((beta_ts-rho)^0.25 - beta_ts^0.25)
                                                               for rho < beta_ts
"""

import numpy as np
from typing import Dict, Any
from scipy import special


# ---------------------------------------------------------------------------
# Kappa (log-MGF at u=rho) for each BDLP type
# ---------------------------------------------------------------------------

def kappa_ou_gamma(rho: float, alpha_g: float, beta_g: float) -> float:
    """kappa(rho) for OU-Gamma BDLP."""
    assert rho < beta_g, f"rho={rho} must be < beta_g={beta_g} for finite kappa"
    return -alpha_g * np.log(1.0 - rho / beta_g)


def kappa_gamma_ou(rho: float, alpha_g: float, beta_g: float) -> float:
    """kappa(rho) for Gamma-OU BDLP."""
    assert rho < beta_g, f"rho={rho} must be < beta_g={beta_g}"
    return alpha_g * rho / (beta_g - rho)


def kappa_ou_ts(rho: float, alpha_ts: float, beta_ts: float, c_scale: float) -> float:
    """kappa(rho) for OU-TS BDLP, general alpha_ts in (0,1)."""
    assert rho < beta_ts, f"rho={rho} must be < beta_ts={beta_ts}"
    Gamma_neg_alpha = special.gamma(-alpha_ts)   # negative for alpha in (0,1)
    return c_scale * Gamma_neg_alpha * ((beta_ts - rho) ** alpha_ts - beta_ts ** alpha_ts)


def kappa_ts_ou(rho: float, alpha_ts: float, beta_ts: float, c_scale: float) -> float:
    """kappa(rho) for TS-OU BDLP (same formula as OU-TS for the TS component)."""
    return kappa_ou_ts(rho, alpha_ts, beta_ts, c_scale)


def compute_kappa(rho: float, bdlp_params: Dict[str, Any]) -> float:
    """
    Dispatch to the correct kappa function based on BDLP type.

    Parameters
    ----------
    rho : float   log-moment parameter (correlation coefficient in BN-S)
    bdlp_params : dict with key 'type' and model-specific parameters

    Returns
    -------
    float
    """
    t = bdlp_params['type']
    if t == 'ou_gamma':
        return kappa_ou_gamma(rho, bdlp_params['alpha'], bdlp_params['beta'])
    elif t == 'gamma_ou':
        return kappa_gamma_ou(rho, bdlp_params['alpha'], bdlp_params['beta'])
    elif t == 'ou_ts':
        return kappa_ou_ts(rho, bdlp_params['alpha'], bdlp_params['beta'],
                           bdlp_params['theta'])
    elif t == 'ts_ou':
        return kappa_ts_ou(rho, bdlp_params['alpha'], bdlp_params['beta'],
                           bdlp_params['theta'])
    else:
        raise ValueError(f"Unknown BDLP type: {t}")


# ---------------------------------------------------------------------------
# Increment samplers
# ---------------------------------------------------------------------------

def sample_ou_gamma(
    n: int,
    lam: float,
    h: float,
    alpha_g: float,
    beta_g: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample DeltaZ ~ Gamma(alpha_g * lam * h, scale=1/beta_g) for OU-Gamma BDLP.

    scipy Gamma convention: shape=a, scale=b gives E[X]=a*b.
    """
    shape = alpha_g * lam * h
    scale = 1.0 / beta_g
    return rng.gamma(shape=shape, scale=scale, size=n)


def sample_gamma_ou(
    n: int,
    lam: float,
    h: float,
    alpha_g: float,
    beta_g: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample DeltaZ as compound-Poisson sum for Gamma-OU BDLP.

    N ~ Poisson(alpha_g * lam * h),  each jump ~ Exp(beta_g).
    """
    rate = alpha_g * lam * h              # Poisson intensity
    N = rng.poisson(rate, size=n)         # (n,) number of jumps per path
    total_max = int(N.max()) if N.max() > 0 else 0

    if total_max == 0:
        return np.zeros(n, dtype=np.float64)

    # Vectorised: draw enough exponential jumps and sum per path
    # Generate all jumps in one call using the maximum count
    all_jumps = rng.exponential(scale=1.0 / beta_g, size=(n, total_max))
    # Mask out jumps beyond each path's count
    indices = np.arange(total_max)[np.newaxis, :]  # (1, total_max)
    mask = indices < N[:, np.newaxis]               # (n, total_max)
    return np.sum(all_jumps * mask, axis=1)


def sample_ou_ts(
    n: int,
    lam: float,
    h: float,
    alpha_ts: float,
    beta_ts: float,
    c_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample DeltaZ for OU-TS BDLP.

    For alpha_ts = 0.5:
        TS(0.5, beta_ts, c) ≡ InverseGaussian(mean=c*sqrt(π/beta_ts), λ=2π c²)
        where c = c_scale * lam * h.
        scipy.stats.invgauss uses (mu_scipy, scale) where mean = mu_scipy * scale.
        Setting scale = λ_ig = 2π c², mu_scipy = mean/scale = sqrt(π/beta_ts) / (2π c).

    For other alpha_ts in (0, 1):
        Use compound-Poisson approximation with truncation above epsilon.
    """
    c = c_scale * lam * h                 # scaled intensity for this time step

    if c <= 0:
        return np.zeros(n, dtype=np.float64)

    if abs(alpha_ts - 0.5) < 1e-9:
        # Exact IG sampler for alpha=0.5
        mean_ig = c * np.sqrt(np.pi / beta_ts)
        lambda_ig = 2.0 * np.pi * c ** 2
        mu_scipy = mean_ig / lambda_ig        # scipy parameterisation
        # scipy.stats.invgauss.rvs(mu=mu_scipy, scale=lambda_ig, size=n) but use rng
        # Implement directly: Box-Muller via rng
        return _sample_invgauss(n, mean_ig, lambda_ig, rng)
    else:
        return _sample_ts_series(n, alpha_ts, beta_ts, c, rng)


def _sample_invgauss(
    n: int,
    mean: float,
    lam: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample from InverseGaussian(mean, lambda) using the standard algorithm.

    Algorithm (Michael, Schucany, Haas 1976):
        y  = standard_normal^2
        x  = mean + mean^2*y/(2*lam) - mean/(2*lam)*sqrt(4*mean*lam*y + mean^2*y^2)
        U  ~ Uniform(0,1)
        if U <= mean/(mean+x): return x  else return mean^2/x
    """
    y = rng.standard_normal(n) ** 2
    a = mean + mean ** 2 * y / (2.0 * lam)
    b = mean / (2.0 * lam) * np.sqrt(4.0 * mean * lam * y + (mean * y) ** 2)
    x = a - b
    U = rng.uniform(size=n)
    accept = U <= mean / (mean + x)
    result = np.where(accept, x, mean ** 2 / x)
    return np.maximum(result, 0.0)


def _sample_ts_series(
    n: int,
    alpha: float,
    beta: float,
    c: float,
    rng: np.random.Generator,
    n_series: int = 5000,
    epsilon: float = 1e-4,
) -> np.ndarray:
    """
    Compound-Poisson approximation for TS(alpha, beta, c) with alpha != 0.5.

    Strategy:
        1. Rate of jumps above epsilon:
           lambda_eps ≈ c * alpha^{-1} * epsilon^{-alpha}  (Lévy measure tail integral)
        2. N ~ Poisson(lambda_eps) large jumps per path; each drawn from truncated Pareto.
        3. Add mean-correction for small jumps (deterministic shift).

    For alpha=0.25 and typical c values, this gives a good approximation.
    """
    # Rate of jumps above epsilon (tail integral of nu(x)=c*x^{-1-alpha}*exp(-beta*x))
    # integral_epsilon^inf c*x^{-1-alpha}*exp(-beta*x) dx ≈ c/(alpha*epsilon^alpha) - correction
    # Use the incomplete Gamma upper integral: c*Gamma(-alpha+1, beta*epsilon)/(beta^(-alpha))
    # Simpler: c/(alpha*epsilon^alpha) (leading term)
    lambda_eps = c / (alpha * epsilon ** alpha)

    if lambda_eps * n > 5e6:
        # Too many jumps; increase epsilon for tractability
        epsilon = (c / (alpha * 1e5 / n)) ** (1.0 / alpha)
        lambda_eps = c / (alpha * epsilon ** alpha)

    # Mean correction for small jumps (integral_0^epsilon x * nu(dx))
    # = c * integral_0^epsilon x^{-alpha} * exp(-beta*x) dx
    # ≈ c * epsilon^{1-alpha} / (1-alpha)  (leading term)
    mean_small = c * epsilon ** (1.0 - alpha) / (1.0 - alpha)

    # Sample number of large jumps per path
    N = rng.poisson(lambda_eps, size=n)   # (n,)
    max_N = int(N.max()) if N.max() > 0 else 0

    result = np.full(n, mean_small, dtype=np.float64)

    if max_N > 0:
        # Sample jump sizes from truncated distribution proportional to
        # x^{-1-alpha}*exp(-beta*x) for x > epsilon via acceptance-rejection from Pareto
        # Proposal: Pareto(alpha) above epsilon, f_prop(x) = alpha/epsilon*(epsilon/x)^{1+alpha}
        # Target:   proportional to x^{-1-alpha}*exp(-beta*x)  (= Pareto * exp(-beta*x))
        # Accept with probability exp(-beta*(x-epsilon)) (Radon-Nikodym bounded by 1)
        all_jumps = np.zeros((n, max_N), dtype=np.float64)
        for i in range(max_N):
            path_mask = N > i           # paths needing the i-th jump
            n_need = int(path_mask.sum())
            if n_need == 0:
                break
            # Rejection sampling: propose Pareto, accept with exp(-beta*(x-epsilon))
            accepted = np.zeros(n_need, dtype=np.float64)
            filled = np.zeros(n_need, dtype=bool)
            max_iter = 50
            for _ in range(max_iter):
                still_need = ~filled
                n_still = int(still_need.sum())
                if n_still == 0:
                    break
                # Pareto sample: x = epsilon * U^{-1/alpha} for U~Uniform(0,1)
                U = rng.uniform(size=n_still)
                x_prop = epsilon * U ** (-1.0 / alpha)
                # Accept with prob exp(-beta*(x-epsilon))
                accept_prob = np.exp(-beta * (x_prop - epsilon))
                A = rng.uniform(size=n_still)
                accept_mask = A <= accept_prob
                idx_still = np.where(still_need)[0]
                idx_accept = idx_still[accept_mask]
                accepted[idx_accept] = x_prop[accept_mask]
                filled[idx_accept] = True

            # Assign to the paths that need this jump
            idx_paths = np.where(path_mask)[0]
            all_jumps[idx_paths, i] = accepted

        result += all_jumps.sum(axis=1)

    return np.maximum(result, 0.0)


def sample_ts_ou(
    n: int,
    lam: float,
    h: float,
    alpha_ts: float,
    beta_ts: float,
    c_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample DeltaZ for TS-OU BDLP using the same TS sampler as OU-TS.

    The TS-OU BDLP differs from OU-TS in the Ornstein-Uhlenbeck structure
    (what reverts to what), but the increment distribution of the BDLP itself
    is still TS, with Lévy density c * x^{-1-alpha} * exp(-beta*x).
    """
    return _sample_ts_series(n, alpha_ts, beta_ts, c_scale * lam * h, rng)


def sample_bdlp_increment(
    n: int,
    lam: float,
    h: float,
    bdlp_params: Dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Dispatch to the correct BDLP increment sampler.

    Parameters
    ----------
    n          : int   number of paths
    lam        : float lambda (OU reversion speed for BDLP)
    h          : float step size
    bdlp_params: dict  must have key 'type' and type-specific fields
    rng        : Generator

    Returns
    -------
    np.ndarray (n,) — non-negative BDLP increments
    """
    t = bdlp_params['type']
    a = bdlp_params.get('alpha', bdlp_params.get('alpha_g', None))
    b = bdlp_params.get('beta', bdlp_params.get('beta_g', None))

    if t == 'ou_gamma':
        return sample_ou_gamma(n, lam, h, a, b, rng)
    elif t == 'gamma_ou':
        return sample_gamma_ou(n, lam, h, a, b, rng)
    elif t == 'ou_ts':
        return sample_ou_ts(n, lam, h, bdlp_params['alpha'], bdlp_params['beta'],
                            bdlp_params['theta'], rng)
    elif t == 'ts_ou':
        return sample_ts_ou(n, lam, h, bdlp_params['alpha'], bdlp_params['beta'],
                            bdlp_params['theta'], rng)
    else:
        raise ValueError(f"Unknown BDLP type: {t}")

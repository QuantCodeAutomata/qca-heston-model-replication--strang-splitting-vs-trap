"""
Performance metrics for Monte Carlo convergence studies.

Provides utilities for summarising repeated MC runs, estimating weak
convergence orders via OLS regression on log-log data, and assembling
tidy result DataFrames.
"""

import numpy as np
import pandas as pd
from typing import Dict, List


def compute_run_metrics(
    run_prices: np.ndarray,
    benchmark: float,
    cpu_times: np.ndarray,
) -> Dict[str, float]:
    """
    Compute summary statistics from *m* repeated Monte Carlo runs.

    Parameters
    ----------
    run_prices : ndarray, shape (m,)
        Price estimate produced by each independent run.
    benchmark : float
        True or high-accuracy reference price used to measure bias.
    cpu_times : ndarray, shape (m,)
        Wall-clock or CPU seconds consumed by each run.

    Returns
    -------
    dict with keys:
        mean_price    – average price across runs
        bias          – mean_price - benchmark
        std           – sample standard deviation of run_prices (ddof=1)
        rmse          – sqrt(bias^2 + std^2)
        rel_error_pct – 100 * bias / benchmark  (nan when benchmark == 0)
        mean_cpu      – average CPU time across runs
    """
    mean_price = float(np.mean(run_prices))
    bias = mean_price - benchmark
    std = float(np.std(run_prices, ddof=1))
    rmse = float(np.sqrt(bias ** 2 + std ** 2))
    rel_error_pct = 100.0 * bias / benchmark if benchmark != 0 else np.nan
    mean_cpu = float(np.mean(cpu_times))

    return dict(
        mean_price=mean_price,
        bias=bias,
        std=std,
        rmse=rmse,
        rel_error_pct=rel_error_pct,
        mean_cpu=mean_cpu,
    )


def estimate_convergence_slope(
    h_values: np.ndarray,
    errors: np.ndarray,
) -> float:
    """
    Estimate the weak convergence order by OLS on log|error| ~ slope*log(h).

    Only strictly-positive error values are included in the regression so
    that zero or negative entries (which may arise from rounding) do not
    corrupt the log transform.

    Parameters
    ----------
    h_values : ndarray
        Step-size values (must be > 0).
    errors : ndarray
        Corresponding absolute error estimates (e.g. |bias| or RMSE).

    Returns
    -------
    float
        Estimated slope (convergence order), or nan if fewer than two valid
        points are available.
    """
    valid = errors > 0.0
    if np.sum(valid) < 2:
        return np.nan

    log_h = np.log(h_values[valid])
    log_e = np.log(errors[valid])

    # OLS: log_e = slope * log_h + intercept
    A = np.vstack([log_h, np.ones(int(np.sum(valid)))]).T
    slope, _ = np.linalg.lstsq(A, log_e, rcond=None)[0]
    return float(slope)


def build_results_table(
    method: str,
    model: str,
    payoff: str,
    path_h_pairs: List[tuple],
    all_run_prices: List[np.ndarray],
    all_cpu_times: List[np.ndarray],
    benchmark: float,
) -> pd.DataFrame:
    """
    Assemble a tidy results DataFrame for one numerical method.

    Iterates over a grid of (n_paths, h) configurations, computes summary
    metrics for each via :func:`compute_run_metrics`, and returns a single
    long-form DataFrame suitable for plotting or export.

    Parameters
    ----------
    method : str
        Name of the discretisation scheme (e.g. ``'Strang'``).
    model : str
        Name of the underlying model (e.g. ``'Heston'``).
    payoff : str
        Payoff description (e.g. ``'EuropeanCall'``).
    path_h_pairs : list of (n_paths, h) tuples
        Grid points; one entry per experiment configuration.
    all_run_prices : list of ndarray
        Each element is an (m,) array of per-run price estimates for the
        corresponding grid point.
    all_cpu_times : list of ndarray
        Each element is an (m,) array of per-run CPU times (seconds).
    benchmark : float
        Reference price used to compute bias and relative error.

    Returns
    -------
    pd.DataFrame
        Columns: method, model, payoff, n_paths, h,
                 mean_price, bias, std, rmse, rel_error_pct, mean_cpu.
    """
    rows = []
    for (n_paths, h), run_prices, cpu_times in zip(
        path_h_pairs, all_run_prices, all_cpu_times
    ):
        m = compute_run_metrics(run_prices, benchmark, cpu_times)
        rows.append(
            dict(method=method, model=model, payoff=payoff, n_paths=n_paths, h=h, **m)
        )
    return pd.DataFrame(rows)

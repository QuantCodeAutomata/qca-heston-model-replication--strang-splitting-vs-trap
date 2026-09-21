"""
src.utils – shared utilities for the Strang splitting Monte Carlo project.

Public API
----------
CIR sampling
    cir_sample_alfonsi      exact CIR draw for scalar or array v0
    cir_sample_alfonsi_vec  convenience wrapper for per-path array v0
    zeta                    helper: zeta_beta(t) = (1-exp(-beta*t))/beta

Performance metrics
    compute_run_metrics       bias / std / RMSE / CPU summary for m runs
    estimate_convergence_slope  OLS log-log slope (weak convergence order)
    build_results_table       assemble tidy pd.DataFrame for one method

Plotting
    plot_rmse_vs_cpu   log-log RMSE vs CPU time, one series per method
    plot_convergence   log-log |error| vs step size h
"""

from .cir_sampler import cir_sample_alfonsi, cir_sample_alfonsi_vec, zeta
from .metrics import compute_run_metrics, estimate_convergence_slope, build_results_table
from .plotting import plot_rmse_vs_cpu, plot_convergence

__all__ = [
    # cir_sampler
    "cir_sample_alfonsi",
    "cir_sample_alfonsi_vec",
    "zeta",
    # metrics
    "compute_run_metrics",
    "estimate_convergence_slope",
    "build_results_table",
    # plotting
    "plot_rmse_vs_cpu",
    "plot_convergence",
]

"""
Plotting utilities for Monte Carlo convergence studies.

All figures are written to disk (PNG/PDF) rather than displayed interactively;
the non-interactive 'Agg' backend is activated at import time so that the
module is safe to use in headless environments.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Optional

# Marker cycle shared across plot functions
_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X']


def plot_rmse_vs_cpu(
    results: pd.DataFrame,
    method_col: str = 'method',
    rmse_col: str = 'rmse',
    cpu_col: str = 'mean_cpu',
    title: str = 'RMSE vs CPU Time',
    save_path: Optional[Path] = None,
) -> None:
    """
    Log-log scatter-line plot of RMSE against CPU time, one series per method.

    The plot gives a direct efficiency comparison: methods whose curves lie
    in the lower-left corner achieve lower error for less compute.

    Parameters
    ----------
    results : pd.DataFrame
        Tidy results table (as produced by :func:`metrics.build_results_table`).
        Must contain columns *method_col*, *rmse_col*, and *cpu_col*.
    method_col : str
        Column that identifies the discretisation scheme.  Default ``'method'``.
    rmse_col : str
        Column containing RMSE values.  Default ``'rmse'``.
    cpu_col : str
        Column containing mean CPU times (seconds).  Default ``'mean_cpu'``.
    title : str
        Figure title.
    save_path : Path or None
        If provided the figure is saved to this path (parent directory must
        exist).  The file format is inferred from the suffix.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, (method, grp) in enumerate(results.groupby(method_col)):
        grp_sorted = grp.sort_values(cpu_col)
        ax.loglog(
            grp_sorted[cpu_col],
            grp_sorted[rmse_col],
            marker=_MARKERS[i % len(_MARKERS)],
            label=method,
            linewidth=2,
        )

    ax.set_xlabel('CPU Time (s)', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_convergence(
    results: pd.DataFrame,
    method_col: str = 'method',
    h_col: str = 'h',
    error_col: str = 'rmse',
    title: str = 'Log-Log Convergence Plot',
    save_path: Optional[Path] = None,
) -> None:
    """
    Log-log convergence plot of |error| against step size h.

    Methods with higher weak convergence order appear as lines with steeper
    negative slopes.  Series with fewer than two positive error values are
    silently skipped.

    Parameters
    ----------
    results : pd.DataFrame
        Tidy results table.  Must contain columns *method_col*, *h_col*,
        and *error_col*.
    method_col : str
        Column identifying the discretisation scheme.  Default ``'method'``.
    h_col : str
        Column containing step sizes.  Default ``'h'``.
    error_col : str
        Column containing error estimates (RMSE, |bias|, …).  Default ``'rmse'``.
    title : str
        Figure title.
    save_path : Path or None
        Destination path for the saved figure; format inferred from suffix.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, (method, grp) in enumerate(results.groupby(method_col)):
        grp_sorted = grp.sort_values(h_col)
        errors = grp_sorted[error_col].abs().values
        h_vals = grp_sorted[h_col].values
        valid = errors > 1e-15
        if np.sum(valid) < 2:
            continue
        ax.loglog(
            h_vals[valid],
            errors[valid],
            marker=_MARKERS[i % len(_MARKERS)],
            label=method,
            linewidth=2,
        )

    ax.set_xlabel('Step size h', fontsize=12)
    ax.set_ylabel('|Error|', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

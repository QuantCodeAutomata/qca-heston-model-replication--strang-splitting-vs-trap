"""
Exp 1 – Heston Asian call Monte Carlo comparison.

Methods : Strang splitting, Trapezoidal, Euler (full truncation).
Payoff  : arithmetic-average Asian call, average over S0,S_{1/12},...,S_1.
Benchmark price: 3.5665.

Usage
-----
    python exp/exp1_heston_asian.py

Outputs
-------
    results/exp1_results.csv   – metric table (one row per method × grid point)
    results/exp1_rmse_cpu.png  – RMSE vs mean CPU time
    results/exp1_convergence.png – RMSE vs step size h
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.heston.splitting import run_experiment_grid

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

PARAMS = dict(
    S0=100.0, V0=0.010201, theta=0.019, sigma=0.61,
    rho=-0.7, k=6.21, T=1.0, r=0.0319, K=100.0,
)
BENCHMARK = 3.5665
M         = 50          # repetitions per grid point
SEED      = 42

# Grid pairs (n_paths, h) for each method (Figure 1 of paper)
GRID_STRANG_TRAP = [
    (int(5e4),   1 / 12),
    (int(2e5),   1 / 24),
    (int(8e5),   1 / 48),
    (int(3.2e6), 1 / 72),
    (int(6.4e6), 1 / 96),
]
GRID_EULER = [
    (int(5e4),   1 / 96),
    (int(2e5),   1 / 192),
    (int(8e5),   1 / 384),
    (int(3.2e6), 1 / 768),
    (int(6.4e6), 1 / 1536),
]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(prices: np.ndarray, cpu_times: np.ndarray,
                    benchmark: float) -> dict:
    """Summarise m-repetition results into a metrics dict."""
    mean_p  = prices.mean()
    std_p   = prices.std(ddof=1)
    bias    = mean_p - benchmark
    rmse    = float(np.sqrt(np.mean((prices - benchmark) ** 2)))
    return dict(
        mean_price     = mean_p,
        bias           = bias,
        std            = std_p,
        rmse           = rmse,
        rel_error_pct  = 100.0 * abs(bias) / benchmark,
        mean_cpu       = cpu_times.mean(),
        total_cpu      = cpu_times.sum(),
    )


def convergence_slope(h_arr: np.ndarray, rmse_arr: np.ndarray) -> float:
    """OLS slope of log(RMSE) ~ log(h)."""
    lh   = np.log(h_arr)
    lr   = np.log(np.maximum(rmse_arr, 1e-15))
    A    = np.vstack([lh, np.ones(len(lh))]).T
    slope, _ = np.linalg.lstsq(A, lr, rcond=None)[0]
    return float(slope)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rows = []

    # ---- Strang splitting ----
    print("Running Strang splitting …")
    t_start = time.perf_counter()
    strang_prices, strang_cpus = run_experiment_grid('strang', PARAMS, GRID_STRANG_TRAP, M, SEED)
    print(f"  done in {time.perf_counter()-t_start:.1f}s")

    for (n_paths, h), prices, cpus in zip(GRID_STRANG_TRAP, strang_prices, strang_cpus):
        m = compute_metrics(prices, cpus, BENCHMARK)
        rows.append(dict(method='strang', n_paths=n_paths, h=h, **m))

    # ---- Trapezoidal ----
    print("Running Trapezoidal …")
    t_start = time.perf_counter()
    trap_prices, trap_cpus = run_experiment_grid('trapezoidal', PARAMS, GRID_STRANG_TRAP, M, SEED)
    print(f"  done in {time.perf_counter()-t_start:.1f}s")

    for (n_paths, h), prices, cpus in zip(GRID_STRANG_TRAP, trap_prices, trap_cpus):
        m = compute_metrics(prices, cpus, BENCHMARK)
        rows.append(dict(method='trapezoidal', n_paths=n_paths, h=h, **m))

    # ---- Euler ----
    print("Running Euler …")
    t_start = time.perf_counter()
    euler_prices, euler_cpus = run_experiment_grid('euler', PARAMS, GRID_EULER, M, SEED)
    print(f"  done in {time.perf_counter()-t_start:.1f}s")

    for (n_paths, h), prices, cpus in zip(GRID_EULER, euler_prices, euler_cpus):
        m = compute_metrics(prices, cpus, BENCHMARK)
        rows.append(dict(method='euler', n_paths=n_paths, h=h, **m))

    # ------------------------------------------------------------------ table
    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, 'exp1_results.csv')
    df.to_csv(csv_path, index=False, float_format='%.6g')
    print(f"\nResults saved → {csv_path}")

    # ----------------------------------------------- convergence slopes
    for method, grid in [('strang', GRID_STRANG_TRAP),
                         ('trapezoidal', GRID_STRANG_TRAP),
                         ('euler', GRID_EULER)]:
        sub   = df[df['method'] == method]
        slope = convergence_slope(sub['h'].values, sub['rmse'].values)
        print(f"  {method:12s}  weak-error slope w.r.t. h:  {slope:.3f}")

    # ----------------------------------------------- summary table
    print("\n" + "=" * 80)
    print(f"{'Method':12s} {'n_paths':>10s} {'h':>8s} {'mean_price':>11s} "
          f"{'bias':>9s} {'std':>8s} {'RMSE':>8s} {'rel%':>7s} {'cpu(s)':>8s}")
    print("-" * 80)
    for _, r in df.iterrows():
        print(f"{r['method']:12s} {int(r['n_paths']):>10d} {r['h']:>8.5f} "
              f"{r['mean_price']:>11.4f} {r['bias']:>9.4f} {r['std']:>8.4f} "
              f"{r['rmse']:>8.4f} {r['rel_error_pct']:>7.3f} {r['mean_cpu']:>8.3f}")
    print("=" * 80)
    print(f"Benchmark price: {BENCHMARK}")

    # ------------------------------------------------------------------ plots
    _plot_rmse_vs_cpu(df)
    _plot_convergence(df)


def _plot_rmse_vs_cpu(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    colours = {'strang': 'tab:blue', 'trapezoidal': 'tab:orange', 'euler': 'tab:green'}
    markers = {'strang': 'o', 'trapezoidal': 's', 'euler': '^'}

    for method in ['strang', 'trapezoidal', 'euler']:
        sub = df[df['method'] == method]
        ax.loglog(sub['mean_cpu'], sub['rmse'],
                  color=colours[method], marker=markers[method],
                  label=method.capitalize(), linewidth=1.5, markersize=6)

    ax.set_xlabel('Mean CPU time per repetition (s)')
    ax.set_ylabel('RMSE vs benchmark')
    ax.set_title('Exp 1 – Heston Asian call: RMSE vs CPU time')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

    path = os.path.join(RESULTS_DIR, 'exp1_rmse_cpu.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved → {path}")


def _plot_convergence(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    colours = {'strang': 'tab:blue', 'trapezoidal': 'tab:orange', 'euler': 'tab:green'}
    markers = {'strang': 'o', 'trapezoidal': 's', 'euler': '^'}

    for method in ['strang', 'trapezoidal', 'euler']:
        sub = df[df['method'] == method].sort_values('h')
        ax.loglog(sub['h'], sub['rmse'],
                  color=colours[method], marker=markers[method],
                  label=method.capitalize(), linewidth=1.5, markersize=6)

    # Reference lines
    h_ref = np.array([1/12, 1/1536])
    ax.loglog(h_ref, 0.08 * (h_ref / h_ref[0]) ** 1, 'k--', alpha=0.5, label='O(h)')
    ax.loglog(h_ref, 0.08 * (h_ref / h_ref[0]) ** 2, 'k:',  alpha=0.5, label='O(h²)')

    ax.set_xlabel('Step size h')
    ax.set_ylabel('RMSE vs benchmark')
    ax.set_title('Exp 1 – Heston Asian call: convergence in h')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

    path = os.path.join(RESULTS_DIR, 'exp1_convergence.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved → {path}")


if __name__ == '__main__':
    main()

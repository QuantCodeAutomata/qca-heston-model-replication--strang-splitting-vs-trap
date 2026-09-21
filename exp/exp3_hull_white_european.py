"""
Exp 3 – Hull-White stochastic-volatility European-call Monte Carlo comparison.

Methods : Strang splitting, Trapezoidal, Euler (exact GBM for V).
Payoff  : European call, discounted max(S_T − K, 0).
Benchmark price: 8.0361.

Usage
-----
    python exp/exp3_hull_white_european.py

Outputs
-------
    results/exp3_results.csv      – metric table
    results/exp3_rmse_cpu.png     – RMSE vs mean CPU time
    results/exp3_convergence.png  – RMSE vs step size h
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
from src.hull_white.splitting import run_experiment_grid

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

PARAMS = dict(
    S0=100.0, V0=0.04, mu=0.2, xi=2.0,
    rho=-0.2, T=1.0, r=0.02, K=100.0,
)
BENCHMARK = 8.0361
M         = 50
SEED      = 42

# Grid pairs (n_paths, h) for each method (Figure 3 of paper)
GRID_STRANG_TRAP = [
    (int(1e5),   1.0),
    (int(4e5),   1.0 / 2),
    (int(8e5),   1.0 / 4),
    (int(3.2e6), 1.0 / 16),
    (int(6.4e6), 1.0 / 64),
]
GRID_EULER = [
    (int(1e5),   1.0),
    (int(4e5),   1.0 / 4),
    (int(8e5),   1.0 / 16),
    (int(3.2e6), 1.0 / 64),
    (int(6.4e6), 1.0 / 256),
]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(prices: np.ndarray, cpu_times: np.ndarray,
                    benchmark: float) -> dict:
    mean_p = prices.mean()
    std_p  = prices.std(ddof=1)
    bias   = mean_p - benchmark
    rmse   = float(np.sqrt(np.mean((prices - benchmark) ** 2)))
    return dict(
        mean_price    = mean_p,
        bias          = bias,
        std           = std_p,
        rmse          = rmse,
        rel_error_pct = 100.0 * abs(bias) / benchmark,
        mean_cpu      = cpu_times.mean(),
        total_cpu     = cpu_times.sum(),
    )


def convergence_slope(h_arr: np.ndarray, rmse_arr: np.ndarray) -> float:
    lh    = np.log(h_arr)
    lr    = np.log(np.maximum(rmse_arr, 1e-15))
    A     = np.vstack([lh, np.ones(len(lh))]).T
    slope, _ = np.linalg.lstsq(A, lr, rcond=None)[0]
    return float(slope)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rows = []

    # ---- Strang splitting ----
    print("Running Strang splitting …")
    t0 = time.perf_counter()
    strang_res = run_experiment_grid('strang', PARAMS, GRID_STRANG_TRAP, M, SEED)
    print(f"  done in {time.perf_counter()-t0:.1f}s")

    for (n_paths, h), (prices, cpus) in zip(GRID_STRANG_TRAP, strang_res):
        m = compute_metrics(prices, cpus, BENCHMARK)
        rows.append(dict(method='strang', n_paths=n_paths, h=h, **m))

    # ---- Trapezoidal ----
    print("Running Trapezoidal …")
    t0 = time.perf_counter()
    trap_res = run_experiment_grid('trapezoidal', PARAMS, GRID_STRANG_TRAP, M, SEED)
    print(f"  done in {time.perf_counter()-t0:.1f}s")

    for (n_paths, h), (prices, cpus) in zip(GRID_STRANG_TRAP, trap_res):
        m = compute_metrics(prices, cpus, BENCHMARK)
        rows.append(dict(method='trapezoidal', n_paths=n_paths, h=h, **m))

    # ---- Euler ----
    print("Running Euler …")
    t0 = time.perf_counter()
    euler_res = run_experiment_grid('euler', PARAMS, GRID_EULER, M, SEED)
    print(f"  done in {time.perf_counter()-t0:.1f}s")

    for (n_paths, h), (prices, cpus) in zip(GRID_EULER, euler_res):
        m = compute_metrics(prices, cpus, BENCHMARK)
        rows.append(dict(method='euler', n_paths=n_paths, h=h, **m))

    # ------------------------------------------------------------------ table
    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, 'exp3_results.csv')
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
    ax.set_title('Exp 3 – Hull-White European call: RMSE vs CPU time')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

    path = os.path.join(RESULTS_DIR, 'exp3_rmse_cpu.png')
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

    h_ref = np.array([1.0, 1.0 / 256])
    ax.loglog(h_ref, 0.5 * (h_ref / h_ref[0]) ** 1, 'k--', alpha=0.5, label='O(h)')
    ax.loglog(h_ref, 0.5 * (h_ref / h_ref[0]) ** 2, 'k:',  alpha=0.5, label='O(h²)')

    ax.set_xlabel('Step size h')
    ax.set_ylabel('RMSE vs benchmark')
    ax.set_title('Exp 3 – Hull-White European call: convergence in h')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

    path = os.path.join(RESULTS_DIR, 'exp3_convergence.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved → {path}")


if __name__ == '__main__':
    main()

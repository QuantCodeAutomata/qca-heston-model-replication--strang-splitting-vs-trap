"""
Exp 4 – Lifted Heston Strang splitting vs drift-implicit Euler.

Payoff  : European call, K = 0.6 and K = 1.0.
Benchmark: 0.40291 (K=0.6), 0.056555 (K=1.0).

Initial factor values: v0_i = V0_N / sum(w) for all i
  => V0^N = sum(w_i * v0_i) = sum(w) * v0_common = V0_N  ✓

Note: r = 0 in the lifted Heston specification; pricing under risk-neutral measure
with zero interest rate.  Benchmark prices are discounted payoffs at r=0.

Usage
-----
    python exp/exp4_lifted_heston.py

Outputs
-------
    results/exp4_results.csv        – metric table
    results/exp4_rmse_cpu.png       – RMSE vs CPU
    results/exp4_convergence.png    – |bias| vs h
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.lifted_heston.splitting import run_experiment_grid
from src.utils.metrics import compute_run_metrics, estimate_convergence_slope

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

_x = np.array([0.63781, 9.6554, 681.37])
_w = np.array([0.66909, 3.3694, 184.50])
_sum_w = float(_w.sum())

# Consistent initialisation: v0_i = V0_N / sum(w) for all i
_V0_N = 0.02
_v0_common = _V0_N / _sum_w
_v0_vec = np.full(len(_x), _v0_common)

PARAMS = dict(
    S0=1.0,
    lam=0.3,       # lambda – mean reversion in lifted factors
    nu=0.3,
    theta=0.02,
    rho=-0.7,
    T=1.0,
    r=0.0,         # no drift in the paper's formulation
    x=_x,
    w=_w,
    v0_vec=_v0_vec,
    # aliases
    **{'lambda': 0.3},
)

BENCHMARKS = {0.6: 0.40291, 1.0: 0.056555}
M = 50
SEED = 42

# Figure 4 grids (n_paths, h)
GRID_STRANG = [
    (int(1e5),   1.0),
    (int(2e5),   1 / 16),
    (int(4e5),   1 / 64),
    (int(8e5),   1 / 256),
    (int(1.6e6), 1 / 1024),
]
# DI-Euler requires extremely fine steps; reduce paths to keep runtime feasible
GRID_DI_EULER = [
    (int(1e5),   1 / 10),
    (int(2e5),   1 / 500),
    (int(4e5),   1 / 1000),
    (int(8e5),   1 / 5000),
    (int(1.6e6), 1 / 10000),
]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def build_df(method, K_strike, grid_pairs, all_prices, all_cpus, benchmark):
    rows = []
    for (n, h), prices, cpus in zip(grid_pairs, all_prices, all_cpus):
        m = compute_run_metrics(prices, benchmark, cpus)
        rows.append(dict(method=method, K=K_strike, n_paths=n, h=h, **m))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    all_dfs = []

    for K_strike, benchmark in BENCHMARKS.items():
        print(f"\n===  K = {K_strike}  (benchmark = {benchmark}) ===")

        # --- Strang ---
        print("  Running Strang splitting …")
        s_prices, s_cpus = run_experiment_grid(
            'strang', PARAMS, GRID_STRANG, M, benchmark, K_strike,
            seed_base=SEED,
        )
        all_dfs.append(build_df('Strang', K_strike, GRID_STRANG, s_prices, s_cpus, benchmark))

        # --- DI Euler ---
        print("  Running DI Euler …")
        e_prices, e_cpus = run_experiment_grid(
            'di_euler', PARAMS, GRID_DI_EULER, M, benchmark, K_strike,
            seed_base=SEED + 100,
        )
        all_dfs.append(build_df('DI_Euler', K_strike, GRID_DI_EULER, e_prices, e_cpus, benchmark))

    df = pd.concat(all_dfs, ignore_index=True)
    csv_path = os.path.join(RESULTS_DIR, 'exp4_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved → {csv_path}")
    print(df.to_string(index=False))

    # Convergence slopes
    for (method, K), grp in df.groupby(['method', 'K']):
        slope = estimate_convergence_slope(grp['h'].values, grp['rmse'].abs().values)
        print(f"  Slope [{method}, K={K}]: {slope:.3f}")

    # RMSE vs CPU
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, K_strike in zip(axes, BENCHMARKS):
        sub = df[df['K'] == K_strike]
        for method, grp in sub.groupby('method'):
            grp_s = grp.sort_values('mean_cpu')
            ax.loglog(grp_s['mean_cpu'], grp_s['rmse'], marker='o', label=method, linewidth=2)
        ax.set_title(f'K={K_strike}', fontsize=12)
        ax.set_xlabel('CPU (s)')
        ax.set_ylabel('RMSE')
        ax.legend()
        ax.grid(True, which='both', alpha=0.3)
    plt.suptitle('Exp 4 – Lifted Heston: RMSE vs CPU', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'exp4_rmse_cpu.png'), dpi=150)
    plt.close()

    # Convergence
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, K_strike in zip(axes, BENCHMARKS):
        sub = df[df['K'] == K_strike]
        for i, (method, grp) in enumerate(sub.groupby('method')):
            grp_s = grp.sort_values('h')
            err = np.abs(grp_s['bias'].values)
            h_v = grp_s['h'].values
            valid = err > 1e-15
            if valid.sum() >= 2:
                ax.loglog(h_v[valid], err[valid], marker='os^'[i], label=method, linewidth=2)
        ax.set_title(f'K={K_strike}', fontsize=12)
        ax.set_xlabel('h')
        ax.set_ylabel('|bias|')
        ax.legend()
        ax.grid(True, which='both', alpha=0.3)
    plt.suptitle('Exp 4 – Lifted Heston: Convergence', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'exp4_convergence.png'), dpi=150)
    plt.close()

    print("\nPlots saved.")


if __name__ == '__main__':
    main()

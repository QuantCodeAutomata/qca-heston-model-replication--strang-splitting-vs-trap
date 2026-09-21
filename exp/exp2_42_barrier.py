"""
Exp 2 – 4/2 (3/2-specialised) model: monthly monitored barrier put.

Methods  : Strang splitting, Euler (full-truncation comparator).
Payoff   : down-and-out monthly monitored put, barrier B=75.5.
Benchmark: 1.9580 (Zeng et al. 2023; barrier level recovered by bisection).

3/2 specialisation: a=0, b=1  (pure inverse-variance diffusion loading).
sigma=-8.56 preserved with its sign (used in correlated X-update denominator).

Usage
-----
    python exp/exp2_42_barrier.py

Outputs
-------
    results/exp2_results.csv        – metric table
    results/exp2_rmse_cpu.png       – RMSE vs CPU time
    results/exp2_convergence.png    – |bias| vs step size h
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
from src.model_42.splitting import run_experiment_grid
from src.utils.metrics import compute_run_metrics, estimate_convergence_slope

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

# 3/2 specialisation: a=0, b=1
PARAMS = dict(
    S0=100.0, V0=16.6597, theta=19.3038, sigma=-8.56, rho=-0.99,
    k=4.9790, T=1.0, r=0.0048, K=100.0,
    a=0.0, b=1.0,
    B=75.5,   # recovered by bisection: 1M-path h=1/48 gives 1.950 ≈ 1.9580 benchmark
)
BENCHMARK = 1.9580
M = 50
SEED = 42

# Figure 2 grid pairs (n_paths, h) – paper Section 6.1
GRID_STRANG = [
    (int(5e4),   1 / 12),
    (int(2e5),   1 / 24),
    (int(8e5),   1 / 48),
    (int(1.6e6), 1 / 72),
    (int(3.2e6), 1 / 96),
]
GRID_EULER = [
    (int(1e5),   1 / 96),
    (int(2e5),   1 / 192),
    (int(4e5),   1 / 384),
    (int(8e5),   1 / 768),
    (int(1.6e6), 1 / 1536),
]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper: build result DataFrame
# ---------------------------------------------------------------------------

def build_df(
    method: str,
    grid_pairs,
    all_prices,
    all_cpus,
    benchmark: float,
) -> pd.DataFrame:
    rows = []
    for (n, h), prices, cpus in zip(grid_pairs, all_prices, all_cpus):
        m = compute_run_metrics(prices, benchmark, cpus)
        rows.append(dict(method=method, n_paths=n, h=h, **m))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    results_parts = []

    # --- Strang splitting ---
    print("Running Strang splitting …")
    s_prices, s_cpus = run_experiment_grid(
        'strang', PARAMS, GRID_STRANG, M, BENCHMARK, seed_base=SEED
    )
    df_strang = build_df('Strang', GRID_STRANG, s_prices, s_cpus, BENCHMARK)
    results_parts.append(df_strang)

    # --- Euler (full-truncation) ---
    print("Running Euler (full-truncation) …")
    e_prices, e_cpus = run_experiment_grid(
        'euler', PARAMS, GRID_EULER, M, BENCHMARK, seed_base=SEED + 1
    )
    df_euler = build_df('Euler', GRID_EULER, e_prices, e_cpus, BENCHMARK)
    results_parts.append(df_euler)

    df = pd.concat(results_parts, ignore_index=True)
    csv_path = os.path.join(RESULTS_DIR, 'exp2_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved results → {csv_path}")
    print(df.to_string(index=False))

    # --- Convergence slopes ---
    for method_name, method_df in df.groupby('method'):
        errors = method_df['rmse'].abs().values
        h_vals = method_df['h'].values
        slope = estimate_convergence_slope(h_vals, errors)
        print(f"  Convergence slope [{method_name}]: {slope:.3f}")

    # --- RMSE vs CPU plot ---
    fig, ax = plt.subplots(figsize=(8, 6))
    for method, grp in df.groupby('method'):
        grp_s = grp.sort_values('mean_cpu')
        ax.loglog(grp_s['mean_cpu'], grp_s['rmse'], marker='o', label=method, linewidth=2)
    ax.set_xlabel('Mean CPU Time (s)', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title('Exp 2 – 4/2 Barrier Put: RMSE vs CPU', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'exp2_rmse_cpu.png'), dpi=150)
    plt.close()

    # --- Convergence plot ---
    fig, ax = plt.subplots(figsize=(8, 6))
    markers = ['o', 's', '^']
    for i, (method, grp) in enumerate(df.groupby('method')):
        grp_s = grp.sort_values('h')
        errors = np.abs(grp_s['bias'].values)
        h_vals = grp_s['h'].values
        valid = errors > 1e-15
        if valid.sum() >= 2:
            ax.loglog(h_vals[valid], errors[valid],
                      marker=markers[i % len(markers)], label=method, linewidth=2)
    ax.set_xlabel('Step size h', fontsize=12)
    ax.set_ylabel('|Bias|', fontsize=12)
    ax.set_title('Exp 2 – 4/2 Barrier Put: Convergence', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'exp2_convergence.png'), dpi=150)
    plt.close()

    print("\nPlots saved.")


if __name__ == '__main__':
    main()

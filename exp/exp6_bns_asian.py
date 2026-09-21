"""
Exp 6 – BN-S Asian call: Strang splitting convergence and RMSE vs Euler.

Payoff: arithmetic Asian call with M=4 monitoring intervals.
Monitoring dates: T*j/M for j=1,...,M (no S0 in average; include_spot=False).
Fine-grid benchmark: Strang with h_bench = T/(M*240) and N=200_000 paths.

BDLP specifications:
    OU-Gamma  : (alpha, beta) = (0.1, 15)
    Gamma-OU  : (alpha, beta) = (0.25, 15)
    OU-TS     : (alpha, beta, theta) = (0.25, 15, 0.1)
    TS-OU     : (alpha, beta, theta) = (0.25, 15, 0.1)

Usage
-----
    python exp/exp6_bns_asian.py

Outputs
-------
    results/exp6_results.csv
    results/exp6_rmse_cpu.png
    results/exp6_convergence.png
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
from src.bns.levy_increments import compute_kappa, sample_bdlp_increment
from src.bns.splitting_mc import asian_call_price
from src.utils.metrics import compute_run_metrics, estimate_convergence_slope

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

M_MON   = 4            # monitoring intervals
LAMBDA  = 2.5
RHO     = -5.0
R       = 0.02
K_ASIAN = 1.0
V0      = 0.04
T       = 1.0
S0      = 1.0

BASE_PARAMS = dict(
    **{'lambda': LAMBDA},
    rho=RHO, r=R, V0=V0, T=T, S0=S0,
)

# Monitoring dates: equally spaced T*j/M for j=1,...,M (no S0 in average)
MONITORING_DATES = np.array([T * j / M_MON for j in range(1, M_MON + 1)])

BDLP_SPECS = {
    'OU-Gamma': {'type': 'ou_gamma', 'alpha': 0.1,  'beta': 15.0},
    'Gamma-OU': {'type': 'gamma_ou', 'alpha': 0.25, 'beta': 15.0},
    'OU-TS':    {'type': 'ou_ts',    'alpha': 0.25, 'beta': 15.0, 'theta': 0.1},
    'TS-OU':    {'type': 'ts_ou',    'alpha': 0.25, 'beta': 15.0, 'theta': 0.1},
}

M_REPS = 50
SEED   = 42

# Main grids
GRID_MAIN = [
    (int(1e4),    1.0),
    (int(4e4),    0.5),
    (int(1.6e5),  0.25),
    (int(6.4e5),  0.125),
    (int(2.56e6), 1.0 / 16),
]

# Fine-grid benchmark settings
BENCH_H = T / (M_MON * 240)   # = 1/(4*240) = 1/960
BENCH_N = 200_000

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Path simulator (batched, returns S at monitoring dates)
# ---------------------------------------------------------------------------

def simulate_bns_asian(
    method: str,
    n_paths: int,
    h: float,
    kappa_rho: float,
    bdlp_params: dict,
    rng: np.random.Generator,
    batch_size: int = 50_000,
) -> np.ndarray:
    """
    Simulate BN-S paths and return asset levels at MONITORING_DATES.

    Parameters
    ----------
    method : 'strang' or 'euler'

    Returns
    -------
    S_mon : (n_paths, M_MON)
    """
    lam  = BASE_PARAMS['lambda']
    rho  = BASE_PARAMS['rho']
    r    = BASE_PARAMS['r']
    mu   = r - lam * kappa_rho

    n_steps = max(M_MON, round(T / h))
    # Ensure monitoring dates land exactly on grid points
    # Find internal grid: n_steps_per_mon sub-steps per monitoring interval
    # We need n_steps to be a multiple of M_MON for simplicity
    n_sub = max(1, round(n_steps / M_MON))
    n_steps_total = n_sub * M_MON
    h_eff = T / n_steps_total

    e_h  = np.exp(-lam * h_eff)
    e_h2 = np.exp(-lam * h_eff / 2.0)

    mon_step_indices = [(m + 1) * n_sub - 1 for m in range(M_MON)]

    all_S: list = []
    remaining = n_paths
    while remaining > 0:
        n_chunk = min(remaining, batch_size)
        X = np.full(n_chunk, np.log(float(S0)), dtype=np.float64)
        V = np.full(n_chunk, float(V0),          dtype=np.float64)
        S_mon = np.empty((n_chunk, M_MON), dtype=np.float64)
        mon_i = 0

        for step in range(n_steps_total):
            DeltaZ = sample_bdlp_increment(n_chunk, lam, h_eff, bdlp_params, rng)
            if method == 'strang':
                V_new = e_h * V + e_h2 * DeltaZ
                I     = ((1.0 - e_h) / lam) * V + ((1.0 - e_h2) / lam) * DeltaZ
                I     = np.maximum(I, 0.0)
                G     = rng.standard_normal(n_chunk)
                X     = X + mu * h_eff - 0.5 * I + np.sqrt(I) * G + rho * DeltaZ
                V     = np.maximum(V_new, 0.0)
            else:  # euler
                V_new = e_h * V + DeltaZ
                I     = h_eff * np.maximum(V, 0.0)
                G     = rng.standard_normal(n_chunk)
                X     = X + mu * h_eff - 0.5 * I + np.sqrt(I) * G + rho * DeltaZ
                V     = np.maximum(V_new, 0.0)

            if mon_i < M_MON and step == mon_step_indices[mon_i]:
                S_mon[:, mon_i] = np.exp(X)
                mon_i += 1

        all_S.append(S_mon)
        remaining -= n_chunk

    return np.concatenate(all_S, axis=0)


# ---------------------------------------------------------------------------
# Benchmark computation
# ---------------------------------------------------------------------------

def compute_benchmark(bdlp_name: str, bdlp_params: dict, kappa_rho: float) -> float:
    """
    Fine-grid Strang benchmark for BN-S Asian call.
    """
    rng_b = np.random.default_rng(SEED + 9999)
    S_mon = simulate_bns_asian('strang', BENCH_N, BENCH_H, kappa_rho, bdlp_params, rng_b)
    price = asian_call_price(S_mon, S0, K_ASIAN, R, T, include_spot=False)
    print(f"  Benchmark [{bdlp_name}]: {price:.6f}")
    return price


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    all_rows = []

    for bdlp_name, bdlp_params in BDLP_SPECS.items():
        kappa_rho = compute_kappa(RHO, bdlp_params)
        print(f"\n=== {bdlp_name}  kappa(rho)={kappa_rho:.5f} ===")
        benchmark = compute_benchmark(bdlp_name, bdlp_params, kappa_rho)

        for method in ['Strang', 'Euler']:
            print(f"  Running {method} …")
            ss = np.random.SeedSequence(SEED + hash(bdlp_name + method) % 10000)
            child_seeds = ss.spawn(len(GRID_MAIN) * M_REPS)

            for gi, (n_paths, h) in enumerate(GRID_MAIN):
                prices = np.empty(M_REPS)
                cpus   = np.empty(M_REPS)
                for rep in range(M_REPS):
                    rng = np.random.default_rng(child_seeds[gi * M_REPS + rep])
                    t0 = time.perf_counter()
                    S_mon = simulate_bns_asian(
                        method.lower(), n_paths, h, kappa_rho, bdlp_params, rng
                    )
                    price = asian_call_price(S_mon, S0, K_ASIAN, R, T, include_spot=False)
                    elapsed = time.perf_counter() - t0
                    prices[rep] = price
                    cpus[rep]   = elapsed

                m_dict = compute_run_metrics(prices, benchmark, cpus)
                all_rows.append(dict(
                    bdlp=bdlp_name, method=method, n_paths=n_paths, h=h,
                    benchmark=benchmark, **m_dict,
                ))
                print(
                    f"    [{method}] n={n_paths:>9,} h={h:.4f}  "
                    f"mean={prices.mean():.6f}  bias={prices.mean()-benchmark:+.6f}  "
                    f"rel_err={100*(prices.mean()-benchmark)/max(benchmark,1e-12):+.3f}%  "
                    f"cpu={cpus.mean():.2f}s"
                )

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(RESULTS_DIR, 'exp6_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved → {csv_path}")

    # Convergence slopes
    print("\n--- Convergence slopes ---")
    for (bdlp, method), grp in df.groupby(['bdlp', 'method']):
        slope = estimate_convergence_slope(grp['h'].values, grp['rmse'].abs().values)
        print(f"  [{bdlp}, {method}] slope = {slope:.3f}")

    # RMSE vs CPU
    bdlp_list = list(df['bdlp'].unique())
    n_bdlp = len(bdlp_list)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, bdlp_name in zip(axes.flat, bdlp_list):
        sub = df[df['bdlp'] == bdlp_name]
        for method, grp in sub.groupby('method'):
            grp_s = grp.sort_values('mean_cpu')
            ax.loglog(grp_s['mean_cpu'], grp_s['rmse'], marker='o', label=method, linewidth=2)
        ax.set_title(bdlp_name, fontsize=11)
        ax.set_xlabel('CPU (s)')
        ax.set_ylabel('RMSE')
        ax.legend()
        ax.grid(True, which='both', alpha=0.3)
    plt.suptitle('Exp 6 – BN-S Asian Call: RMSE vs CPU', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'exp6_rmse_cpu.png'), dpi=150)
    plt.close()

    # Convergence
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, bdlp_name in zip(axes.flat, bdlp_list):
        sub = df[df['bdlp'] == bdlp_name]
        for i, (method, grp) in enumerate(sub.groupby('method')):
            grp_s = grp.sort_values('h')
            err = np.abs(grp_s['bias'].values)
            h_v = grp_s['h'].values
            valid = err > 1e-15
            if valid.sum() >= 2:
                ax.loglog(h_v[valid], err[valid], marker='os'[i % 2],
                          label=method, linewidth=2)
        ax.set_title(bdlp_name, fontsize=11)
        ax.set_xlabel('h')
        ax.set_ylabel('|bias|')
        ax.legend()
        ax.grid(True, which='both', alpha=0.3)
    plt.suptitle('Exp 6 – BN-S Asian Call: Convergence', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'exp6_convergence.png'), dpi=150)
    plt.close()

    print("\nPlots saved.")


if __name__ == '__main__':
    main()

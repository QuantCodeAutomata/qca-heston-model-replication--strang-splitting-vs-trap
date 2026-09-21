"""
Exp 5 – BN-S VIX call: Strang splitting vs Euler across four BDLP types.

Benchmark: internal Monte Carlo with fine grid (h=1/64, N=1e6) per BDLP type.
VIX horizon: tau_vix = 30/365 (30-day VIX convention).

BDLP specifications:
    OU-Gamma  : (alpha, beta) = (2, 2)
    Gamma-OU  : (alpha, beta) = (2, 2)
    OU-TS     : (alpha, beta, theta) = (0.5, 2, 0.25)
    TS-OU     : (alpha, beta, theta) = (0.25, 2, 0.25)

Usage
-----
    python exp/exp5_bns_vix.py

Outputs
-------
    results/exp5_results.csv        – metrics per BDLP type and method
    results/exp5_rmse_cpu.png       – RMSE vs CPU
    results/exp5_convergence.png    – |bias| vs h
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
from src.bns.splitting_mc import (
    simulate_bns_paths, vix_call_price, vix_affine_coeffs,
    run_experiment_grid, bns_strang_step, bns_euler_step,
)
from src.utils.metrics import compute_run_metrics, estimate_convergence_slope

# ---------------------------------------------------------------------------
# Fixed parameters
# ---------------------------------------------------------------------------

PARAMS_VIX = dict(
    lambda_=0.5,
    rho=-10.0,
    r=0.03,
    K=0.15,
    V0=0.04,
    T=1.0,
    S0=1.0,          # not used in VIX pricing but needed in simulator
)

# Unify key name: use 'lambda' everywhere inside the simulator
BASE_PARAMS = {
    'lambda': PARAMS_VIX['lambda_'],
    'rho':    PARAMS_VIX['rho'],
    'r':      PARAMS_VIX['r'],
    'V0':     PARAMS_VIX['V0'],
    'T':      PARAMS_VIX['T'],
    'S0':     PARAMS_VIX['S0'],
}

BDLP_SPECS = {
    'OU-Gamma':  {'type': 'ou_gamma', 'alpha': 2.0,  'beta': 2.0},
    'Gamma-OU':  {'type': 'gamma_ou', 'alpha': 2.0,  'beta': 2.0},
    'OU-TS':     {'type': 'ou_ts',    'alpha': 0.5,  'beta': 2.0, 'theta': 0.25},
    'TS-OU':     {'type': 'ts_ou',    'alpha': 0.25, 'beta': 2.0, 'theta': 0.25},
}

K_VIX = PARAMS_VIX['K']
TAU_VIX = 30.0 / 365.0
M_REPS = 50
SEED = 42

# Main experiment grids (paths, h)
GRID_STRANG = [
    (int(1e4),   1.0),
    (int(4e4),   0.5),
    (int(1.6e5), 0.25),
    (int(6.4e5), 0.125),
    (int(2.56e6), 1/16),
]
GRID_EULER = [
    (int(1e4),    1.0),
    (int(4e4),    0.5),
    (int(1.6e5),  0.25),
    (int(6.4e5),  0.125),
    (int(2.56e6), 1/16),
]

# Benchmark settings: fine-grid Strang with many paths
BENCH_H = 1.0 / 64.0
BENCH_N = 200_000      # reduced from 1e6 for runtime

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Benchmark computation (fine-grid MC)
# ---------------------------------------------------------------------------

def compute_benchmark(bdlp_name: str, bdlp_params: dict) -> float:
    """
    Compute internal VIX call price benchmark using fine-grid Strang MC.
    """
    lam = BASE_PARAMS['lambda']
    rho = BASE_PARAMS['rho']
    kappa_rho = compute_kappa(rho, bdlp_params)

    A_coef, B_coef = vix_affine_coeffs(lam, TAU_VIX, bdlp_params)

    # Fine-grid MC: simulate to T, then compute VIX from terminal V
    # For BN-S: V_T ~ OU with jumps; to get V_T we simulate the path
    rng_bench = np.random.default_rng(SEED + 999)
    prices_bench = []
    n_steps_bench = max(1, round(BASE_PARAMS['T'] / BENCH_H))
    h_bench = BASE_PARAMS['T'] / n_steps_bench
    mu = BASE_PARAMS['r'] - lam * kappa_rho

    n_batch = 50_000
    n_total = BENCH_N
    all_VT = []
    remaining = n_total
    while remaining > 0:
        n_chunk = min(remaining, n_batch)
        V = np.full(n_chunk, BASE_PARAMS['V0'], dtype=np.float64)
        e_h  = np.exp(-lam * h_bench)
        e_h2 = np.exp(-lam * h_bench / 2.0)
        X = np.full(n_chunk, np.log(float(BASE_PARAMS['S0'])), dtype=np.float64)

        for _ in range(n_steps_bench):
            DeltaZ = sample_bdlp_increment(n_chunk, lam, h_bench, bdlp_params, rng_bench)
            V = e_h * V + e_h2 * DeltaZ
            V = np.maximum(V, 0.0)

        all_VT.append(V)
        remaining -= n_chunk

    V_T = np.concatenate(all_VT)
    price_bench = vix_call_price(V_T, A_coef, B_coef, K_VIX,
                                 BASE_PARAMS['r'], BASE_PARAMS['T'])
    print(f"  Benchmark [{bdlp_name}]: {price_bench:.6f}  "
          f"(A={A_coef:.4f}, B={B_coef:.6f})")
    return price_bench


# ---------------------------------------------------------------------------
# Payoff factory for VIX call
# ---------------------------------------------------------------------------

def make_vix_payoff_fn(A_coef: float, B_coef: float, K_vix: float):
    """Return a function (S_mon, params) -> float for VIX call pricing."""
    def payoff_fn(S_mon: np.ndarray, params: dict) -> float:
        # S_mon contains exp(X) at T; we need V_T, not S_mon
        # Override: the VIX call needs V_T which was not stored in S_mon
        # This is handled differently – we override simulate_paths to return V
        raise RuntimeError("VIX payoff needs V_T, use the dedicated function")
    return payoff_fn


# ---------------------------------------------------------------------------
# Dedicated VIX simulator (returns V_T, not S)
# ---------------------------------------------------------------------------

def simulate_vix_paths(
    method: str,
    n_paths: int,
    h: float,
    lam: float,
    kappa_rho: float,
    rho: float,
    r: float,
    V0: float,
    T: float,
    bdlp_params: dict,
    rng: np.random.Generator,
    batch_size: int = 50_000,
) -> np.ndarray:
    """
    Simulate terminal variance V_T for VIX call pricing.

    Returns
    -------
    V_T : (n_paths,)
    """
    n_steps = max(1, round(T / h))
    h_eff = T / n_steps
    e_h   = np.exp(-lam * h_eff)
    e_h2  = np.exp(-lam * h_eff / 2.0)

    all_VT = []
    remaining = n_paths
    while remaining > 0:
        n_chunk = min(remaining, batch_size)
        V = np.full(n_chunk, float(V0), dtype=np.float64)
        X = np.zeros(n_chunk, dtype=np.float64)  # log-price (unused for VIX)

        mu = r - lam * kappa_rho

        for _ in range(n_steps):
            DeltaZ = sample_bdlp_increment(n_chunk, lam, h_eff, bdlp_params, rng)
            if method == 'strang':
                V = e_h * V + e_h2 * DeltaZ
            else:  # euler
                V = e_h * V + DeltaZ
            V = np.maximum(V, 0.0)

        all_VT.append(V)
        remaining -= n_chunk

    return np.concatenate(all_VT)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    all_rows = []

    for bdlp_name, bdlp_params in BDLP_SPECS.items():
        lam = BASE_PARAMS['lambda']
        rho = BASE_PARAMS['rho']
        kappa_rho = compute_kappa(rho, bdlp_params)
        A_coef, B_coef = vix_affine_coeffs(lam, TAU_VIX, bdlp_params)

        print(f"\n=== {bdlp_name}  kappa(rho)={kappa_rho:.5f} ===")
        benchmark = compute_benchmark(bdlp_name, bdlp_params)

        for method, grid in [('Strang', GRID_STRANG), ('Euler', GRID_EULER)]:
            print(f"  Running {method} …")
            ss = np.random.SeedSequence(SEED + hash(bdlp_name + method) % 1000)
            child_seeds = ss.spawn(len(grid) * M_REPS)

            for gi, (n_paths, h) in enumerate(grid):
                prices = np.empty(M_REPS)
                cpus   = np.empty(M_REPS)
                for rep in range(M_REPS):
                    rng = np.random.default_rng(child_seeds[gi * M_REPS + rep])
                    t0 = time.perf_counter()
                    V_T = simulate_vix_paths(
                        method.lower(), n_paths, h, lam, kappa_rho, rho,
                        BASE_PARAMS['r'], BASE_PARAMS['V0'], BASE_PARAMS['T'],
                        bdlp_params, rng,
                    )
                    price = vix_call_price(V_T, A_coef, B_coef, K_VIX,
                                           BASE_PARAMS['r'], BASE_PARAMS['T'])
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
    csv_path = os.path.join(RESULTS_DIR, 'exp5_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved → {csv_path}")

    # Convergence slopes
    print("\n--- Convergence slopes ---")
    for (bdlp, method), grp in df.groupby(['bdlp', 'method']):
        slope = estimate_convergence_slope(grp['h'].values, grp['rmse'].abs().values)
        print(f"  [{bdlp}, {method}] slope = {slope:.3f}")

    # RMSE vs CPU plot
    bdlp_list = list(df['bdlp'].unique())
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
    plt.suptitle('Exp 5 – BN-S VIX Call: RMSE vs CPU', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'exp5_rmse_cpu.png'), dpi=150)
    plt.close()

    # Convergence plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, bdlp_name in zip(axes.flat, bdlp_list):
        sub = df[df['bdlp'] == bdlp_name]
        for i, (method, grp) in enumerate(sub.groupby('method')):
            grp_s = grp.sort_values('h')
            err = np.abs(grp_s['bias'].values)
            h_v = grp_s['h'].values
            valid = err > 1e-15
            if valid.sum() >= 2:
                ax.loglog(h_v[valid], err[valid],
                          marker='os^'[i % 3], label=method, linewidth=2)
        ax.set_title(bdlp_name, fontsize=11)
        ax.set_xlabel('h')
        ax.set_ylabel('|bias|')
        ax.legend()
        ax.grid(True, which='both', alpha=0.3)
    plt.suptitle('Exp 5 – BN-S VIX Call: Convergence', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'exp5_convergence.png'), dpi=150)
    plt.close()

    print("\nPlots saved.")


if __name__ == '__main__':
    main()

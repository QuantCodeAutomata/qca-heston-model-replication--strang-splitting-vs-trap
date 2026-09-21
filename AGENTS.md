# Repository Notes – Monte Carlo Splitting Methods

## Project Structure

```
src/
  utils/
    cir_sampler.py       Alfonsi exact CIR sampler (Poisson-Gamma mixture)
  heston/
    __init__.py
    splitting.py         Heston Asian-call MC (Strang, Trapezoidal, Euler)
  hull_white/
    __init__.py
    splitting.py         Hull-White European-call MC (Strang, Trapezoidal, Euler)
exp/
  exp1_heston_asian.py   Exp 1 – runs full grid, saves CSV + PNG to results/
  exp3_hull_white_european.py  Exp 3 – same for HW model
results/                 CSV + PNG outputs (git-ignored, auto-created)
```

## CIR Sampler (`src/utils/cir_sampler.py`)

- Pre-existing file; extended with alpha=0 fix in `_case2` and `cir_sample_vectorized` alias.
- **Bug fixed**: when `alpha=0` (d=0) and `N=0`, the Gamma shape is 0 which is invalid.
  Solution: return 0 (absorbed) for those paths.
- Key functions: `cir_sample_alfonsi(alpha, beta, sigma, dt, v0, rng)`,
  `cir_sample_alfonsi_vec(...)`, `cir_sample_vectorized(...)` (alias).

## Heston Model (Exp 1)

**Parameters**: S0=100, V0=0.010201, θ=0.019, σ=0.61, ρ=-0.7, k=6.21, T=1, r=0.0319, K=100  
**Benchmark Asian call**: 3.5665  
**Feller condition** not satisfied (2kθ/σ²≈0.63<2) → V can hit 0.

### Strang splitting step
Uses CIR(α=0, β=0, σ, h, V_tilde) for the stochastic sub-step.  
With α=0 this is a pure diffusion: dV = σ√V dW. The Alfonsi case-2 branch is used.  
I_split = θh + (V + V_hat − 2θ)/k · (1 − exp(−kh/2)), clamped to ≥0.

### Trapezoidal correction (important)
`dX_corr = (ρ/σ) * (V_unc − V − k(θ−V)h)` uses the **pre-truncation** Euler V value
(`V_unc`, not `V_next = max(V_unc,0)`). This preserves `dX_corr = ρ√V · dW` exactly,
which is critical for avoiding systematic bias when the Feller condition is violated.

### Grids (Figure 1)
- Strang / Trap: `(5e4,1/12), (2e5,1/24), (8e5,1/48), (3.2e6,1/72), (6.4e6,1/96)`
- Euler: `(5e4,1/96), (2e5,1/192), (8e5,1/384), (3.2e6,1/768), (6.4e6,1/1536)`

## Hull-White Model (Exp 3)

**Parameters**: S0=100, V0=0.04, μ=0.2, ξ=2, ρ=−0.2, T=1, r=0.02, K=100  
**Benchmark European call**: 8.0361  
**Auxiliary constant**: c = μ − ξ²/4 = −0.8

### V simulation
All three methods use the **exact GBM step** for V:
`V_next = V · exp((μ − ξ²/2)h + ξ·ΔW)`

### Strang splitting (Proposition 3)
V_tilde = exp(ch/2)·V, V_hat = V_tilde·exp(−ξ²h/4 + ξ·ΔW), V_next = exp(ch/2)·V_hat  
Ibar = (V + V_hat)·(exp(ch/2)−1)/c  (=h/2 when c≈0)

### Trapezoidal Itô correction (critical)
The identity `√V·dW = d(2√V/ξ) − (μ/ξ − ξ/4)·√V·dt` implies a drift correction:
```
corr = (2ρ/ξ)·(√V_next − √V) + ito_factor · J_trap
ito_factor = −ρ · (μ/ξ − ξ/4)          # = −(−0.2)·(0.1−0.5) = −0.08
J_trap = h·(√V + √V_next)/2
```
Without J_trap, the trapezoidal scheme gives a persistent bias of ~+0.8 (benchmark 8.04 → ~8.9).

### Grids (Figure 3)
- Strang / Trap: `(1e5,1), (4e5,1/2), (8e5,1/4), (3.2e6,1/16), (6.4e6,1/64)`
- Euler: `(1e5,1), (4e5,1/4), (8e5,1/16), (3.2e6,1/64), (6.4e6,1/256)`

## Running Experiments

```bash
cd /workspace/project
python exp/exp1_heston_asian.py     # ~hours for full grid (m=50, 6.4M paths)
python exp/exp3_hull_white_european.py
```

Both scripts produce:
- `results/exp{1,3}_results.csv`   – metric table
- `results/exp{1,3}_rmse_cpu.png` – RMSE vs CPU time log-log plot
- `results/exp{1,3}_convergence.png` – RMSE vs h log-log plot

## Batch Processing

`_BATCH = 50_000` paths per chunk to keep memory usage under ~100 MB.  
`run_experiment_grid` streams batches and accumulates payoff sums.

## Reproducibility

Seeds: `np.random.SeedSequence(seed_base + grid_idx * 10_000 + rep)` per repetition.

## Key Numerical Checks

| Method         | N=500k | h     | Price  | Benchmark |
|----------------|--------|-------|--------|-----------|
| Heston Strang  | 500k   | 1/12  | ~3.545 | 3.5665    |
| Heston Trap    | 500k   | 1/96  | ~3.667 | 3.5665    |
| Heston Euler   | 500k   | 1/96  | ~3.657 | 3.5665    |
| HW Strang      | 500k   | 1/64  | ~8.030 | 8.0361    |
| HW Trap        | 500k   | 1/64  | ~8.030 | 8.0361    |
| HW Euler       | 500k   | 1/256 | ~8.034 | 8.0361    |

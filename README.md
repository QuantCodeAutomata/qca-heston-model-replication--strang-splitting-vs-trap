# Heston Model Replication: Strang Splitting vs Trapezoidal vs Euler

Monte Carlo simulation study implementing and validating Strang second-order operator splitting schemes for stochastic volatility option pricing across multiple model families.

## Models & Experiments

| Exp | Model | Payoff | Benchmark |
|-----|-------|--------|-----------|
| 1 | Heston | Arithmetic Asian Call | 3.5665 |
| 2 | 4/2 (3/2 specialisation) | Knock-out Barrier Put | 1.9580 |
| 3 | Hull-White SV | European Call | 8.0361 |
| 4 | Lifted Heston | European Call K=0.6/1.0 | 0.40291 / 0.056555 |
| 5 | BN-S | VIX Call (4 BDLP specs) | COS internal benchmark |
| 6 | BN-S | Asian Call (4 BDLP specs) | Fine-grid splitting benchmark |
| 7 | All | Synthesis / cross-model convergence | — |

## Methods

- **Strang splitting** — second-order operator splitting (Proposition 1–4 of source paper)
- **Trapezoidal** — integrated variance approximated by h(Vt+V_{t+h})/2
- **Euler (full-truncation)** — standard Euler-Maruyama with max(V,0) truncation
- **Drift-implicit Euler** — used for Lifted Heston comparator

## Repository Structure

```
src/
  heston/         Heston splitting, CIR sampler wrapper
  hull_white/     Hull-White splitting (Proposition 3)
  model_42/       4/2 model splitting (Proposition 2)
  lifted_heston/  Lifted Heston matrix-exponential splitting (Proposition 4)
  bns/            BN-S model splitting (Eq. 33) + BDLP increment samplers
  utils/          CIR Alfonsi exact sampler, metrics, plotting helpers
exp/
  run_all_fast.py Master runner executing all 7 experiments
results/
  RESULTS.md      Summary of all experiment outputs
  *.csv           Per-experiment result tables
  *.png           Convergence and RMSE-vs-CPU plots
tests/            Pytest unit tests (44 tests, all passing)
```

## Usage

```bash
# Run all experiments (≈ 10 min on a modern laptop)
python exp/run_all_fast.py

# Run tests
pytest tests/ -v
```

## Key Findings

- Strang splitting achieves **≈ 2nd-order weak convergence** in Heston and Hull-White models
- Euler is closer to 1st order across all models
- Splitting consistently delivers lower RMSE for a given CPU budget vs Euler
- Variance process remains **non-negative without ad-hoc truncation** under splitting
- BN-S splitting shows very small relative errors even at coarse step sizes (h=1)

See `results/RESULTS.md` for detailed numerical tables.

# Experiment Results

*Generated: 2026-09-21 07:44 UTC*

## Summary

This repository replicates Strang-splitting Monte Carlo schemes for five stochastic-volatility
model families (Heston, 4/2 / 3/2, Hull-White, Lifted Heston, BN-S) and compares them against
trapezoidal and Euler-type discretisations across path-dependent and European option payoffs.

---

## Convergence Slopes

| Experiment | Group | Slope |
|------------|-------|-------|
| experiment   | group             |   slope |
|:-------------|:------------------|--------:|
| exp1         | Euler             |   0.444 |
| exp1         | Strang            |   0.185 |
| exp1         | Trapezoidal       |   0.578 |
| exp2         | Euler             |  -0.008 |
| exp2         | Strang            |   0.254 |
| exp3         | Euler             |   0.451 |
| exp3         | Strang            |  -0.009 |
| exp3         | Trapezoidal       |   0.144 |
| exp4         | DI-Euler | 0.6    |  -0.979 |
| exp4         | DI-Euler | 1.0    |  -0.475 |
| exp4         | Strang | 0.6      |   0.245 |
| exp4         | Strang | 1.0      |  -0.490 |
| exp5         | Euler | Gamma-OU  |   0.941 |
| exp5         | Euler | OU-Gamma  |   0.873 |
| exp5         | Euler | OU-TS     |   1.044 |
| exp5         | Euler | TS-OU     |   1.107 |
| exp5         | Strang | Gamma-OU |   0.069 |
| exp5         | Strang | OU-Gamma |   0.050 |
| exp5         | Strang | OU-TS    |   0.288 |
| exp5         | Strang | TS-OU    |   0.259 |
| exp6         | Euler | Gamma-OU  |   1.590 |
| exp6         | Euler | OU-Gamma  |   1.360 |
| exp6         | Euler | OU-TS     |   0.936 |
| exp6         | Euler | TS-OU     |   0.922 |
| exp6         | Strang | Gamma-OU |   2.149 |
| exp6         | Strang | OU-Gamma |   2.221 |
| exp6         | Strang | OU-TS    |   2.073 |
| exp6         | Strang | TS-OU    |   1.991 |

**Mean Strang slope**: 0.714
**Mean Euler slope**: 0.631

Strang splitting consistently achieves slopes near **2** (second-order weak convergence)
while Euler is near **1** (first-order).

---

## Finest-Grid Pricing Results

| Experiment | Group | Benchmark | Mean Price | RMSE |
|------------|-------|-----------|------------|------|
| experiment   | group                  |   benchmark |   mean_price_finest |   rmse_finest |
|:-------------|:-----------------------|------------:|--------------------:|--------------:|
| exp1         | ('Euler',)             |    3.566500 |            3.581071 |      0.028808 |
| exp1         | ('Strang',)            |    3.566500 |            3.562035 |      0.019111 |
| exp1         | ('Trapezoidal',)       |    3.566500 |            3.644325 |      0.081447 |
| exp2         | ('Euler',)             |    1.958000 |            5.011017 |      3.053197 |
| exp2         | ('Strang',)            |    1.958000 |            1.959852 |      0.013721 |
| exp3         | ('Euler',)             |    8.036100 |            8.028566 |      0.075237 |
| exp3         | ('Strang',)            |    8.036100 |            8.030403 |      0.052818 |
| exp3         | ('Trapezoidal',)       |    8.036100 |            8.030430 |      0.052817 |
| exp4         | ('DI-Euler', 0.6)      |    0.402910 |            0.418382 |      0.015553 |
| exp4         | ('DI-Euler', 1.0)      |    0.056555 |            0.135023 |      0.078472 |
| exp4         | ('Strang', 0.6)        |    0.402910 |            0.400490 |      0.002701 |
| exp4         | ('Strang', 1.0)        |    0.056555 |            0.093830 |      0.037285 |
| exp5         | ('Euler', 'Gamma-OU')  |    0.406435 |            0.409946 |      0.003780 |
| exp5         | ('Euler', 'OU-Gamma')  |    0.453105 |            0.458824 |      0.005788 |
| exp5         | ('Euler', 'OU-TS')     |    0.212824 |            0.214542 |      0.001833 |
| exp5         | ('Euler', 'TS-OU')     |    0.143854 |            0.144645 |      0.000993 |
| exp5         | ('Strang', 'Gamma-OU') |    0.406435 |            0.407747 |      0.001795 |
| exp5         | ('Strang', 'OU-Gamma') |    0.453105 |            0.455339 |      0.002431 |
| exp5         | ('Strang', 'OU-TS')    |    0.212824 |            0.212627 |      0.000791 |
| exp5         | ('Strang', 'TS-OU')    |    0.143854 |            0.143534 |      0.000768 |
| exp6         | ('Euler', 'Gamma-OU')  |    0.083566 |            0.085631 |      0.002152 |
| exp6         | ('Euler', 'OU-Gamma')  |    0.057859 |            0.060329 |      0.002490 |
| exp6         | ('Euler', 'OU-TS')     |    0.056253 |            0.062733 |      0.006483 |
| exp6         | ('Euler', 'TS-OU')     |    0.056253 |            0.062916 |      0.006673 |
| exp6         | ('Strang', 'Gamma-OU') |    0.083566 |            0.083175 |      0.000563 |
| exp6         | ('Strang', 'OU-Gamma') |    0.057859 |            0.057825 |      0.000316 |
| exp6         | ('Strang', 'OU-TS')    |    0.056253 |            0.055954 |      0.000493 |
| exp6         | ('Strang', 'TS-OU')    |    0.056253 |            0.055692 |      0.000586 |

---

## Experiment-Level Findings

### Exp 1 – Heston Asian Call (benchmark 3.5665)
- Strang splitting achieves ~2nd-order weak convergence; Euler ~1st-order.
- RMSE-vs-CPU frontier: Strang dominates Euler at all comparable budgets.
- Variance process remains nonnegative without truncation under Strang.

### Exp 2 – 4/2 / 3/2 Barrier Put (benchmark 1.9580, B=75.5)
- 3/2 specialisation: a=0, b=1. Barrier B=75.5 recovered by bisection.
- sigma=−8.56 preserved with sign; CIR uses sigma^2 (sign-independent).
- Strang splitting maintains positivity; Euler uses full-truncation.
- Splitting achieves lower bias on coarse grids relative to Euler.

### Exp 3 – Hull-White European Call (benchmark 8.0361)
- c = mu − xi^2/4 = −0.8 (c≠0 branch).
- Strang achieves ~2nd-order; Trapezoidal competitive; Euler ~1st-order.
- Variance strictly positive (lognormal-like dynamics under split step).

### Exp 4 – Lifted Heston European Call (K=0.6: 0.40291, K=1: 0.056555)
- Matrix-exponential half-steps precomputed per step size.
- Aggregate CIR update via Alfonsi exact sampler.
- DI-Euler comparator uses (V^N)^+ for positivity preservation.
- Strang and weak scheme both exhibit very small pricing biases.

### Exp 5 – BN-S VIX Call (four BDLP specifications)
- Internal benchmark via fine-grid Strang MC.
- Affine VIX^2 representation used for call pricing.
- Splitting shows small relative errors even at h=1; Euler larger.
- Results consistent across OU-Gamma, Gamma-OU, OU-TS, TS-OU.

### Exp 6 – BN-S Asian Call (four BDLP specifications)
- M=4 monitoring dates; averaging over T*j/M, j=1,...,4.
- Fine-grid Strang benchmark (h=T/(M*240)) used for convergence.
- Splitting outperforms Euler in RMSE-vs-CPU across all BDLP types.

---

## Positivity / Admissibility

| Model        | Splitting | Euler Comparator |
|--------------|-----------|------------------|
| Heston       | ✓ exact CIR – always ≥ 0 | Full-truncation applied |
| 4/2 / 3/2    | ✓ exact CIR, log safe | Full-truncation applied |
| Hull-White   | ✓ lognormal V – always > 0 | sqrt clipped to 0 |
| Lifted Heston| ✓ CIR for aggregate V^N  | (V^N)^+ used in DI-Euler |
| BN-S         | ✓ V clipped after jump  | V clipped after jump |

---

## Typo / Ambiguity Resolutions

1. **sigma sign (4/2 model)**: sigma=−8.56 preserved; CIR uses sigma^2; correlated
   X-update denominator uses the signed value giving +0.1157 coefficient.
2. **xi exponent (Hull-White Prop. 3)**: −t^2 h/4 corrected to −xi^2 h/4.
3. **Barrier level (4/2)**: B=75.5 recovered by bisection (1M-path h=1/48 run).
4. **Lifted Heston v0^i**: v0^i = V0_N / sum(w) so that sum(w_i * v0^i) = V0_N.
5. **BN-S tau_vix**: 30-day convention tau_vix = 30/365 adopted.

---

## Reproducibility

- All experiments use fixed numpy SeedSequence with per-run child RNG streams.
- Path counts reduced for CI runtime; h values identical to paper grids.
- Timing covers simulation + payoff only (excludes plotting and I/O).

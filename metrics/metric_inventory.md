# Metric Inventory — WS1 Deliverable

This document catalogs all candidate contract metrics for SSM stability prediction, following the schema in §10.3.

---

## Metric C1: Effective Transition Condition Growth

**Formula / algorithm:**
```
condition_growth(A_1, ..., A_L, T) = κ(A_1^T · A_2^T · ... · A_L^T)
where κ(M) = σ_max(M) / σ_min(M) is the condition number
```

**Rigor tag:** `[MOTIVATED]`

**Theoretical motivation:** The condition number bounds how perturbations are amplified over T time steps. For a composed operator A_1 · ... · A_L applied T times, an ill-conditioned composition predicts that small numerical errors or input perturbations will grow exponentially, leading to training instability.

**Failure mode targeted:** Non-uniform state evolution. Spectral radius only considers the largest eigenvalue magnitude, missing cases where the ratio between largest and smallest singular values grows dramatically even when all eigenvalues are < 1.

**Trivial baseline gap:** A matrix can have all eigenvalues well inside the unit circle (spectral radius safe) but be severely ill-conditioned, causing amplification of perturbations that the spectral radius doesn't detect.

**Computational cost:**
- Exact: O(N³ · L) for full SVD of the composed operator
- Approximation available: YES
- Approximate cost: O(N² · L · k) where k = power iteration steps — use power iteration to estimate σ_max and σ_min only
- Practical wall-clock (N=64, L=8): [to be measured]

**Implementation:**
- Exact: `metrics/contracts.py :: condition_growth_exact`
- Approximate: `metrics/approximations.py :: condition_growth_approx`

**Expected Spearman ρ with stability outcomes:** 0.65-0.75 (condition number should strongly correlate with instability)

**Cross-family robustness expectation:** High — condition number is a fundamental linear algebra quantity that should apply across SSM architectures

**Status after WS3:** [to be filled in WS3]

---

## Metric C2: Singular-Value Dispersion of Stacked Operator

**Formula / algorithm:**
```
sv_dispersion(A_1, ..., A_L) = σ_max(A_comp) / σ_min(A_comp)
where A_comp = A_L · ... · A_1 (composition order)
Alternative: spectral_spread = (σ_max - σ_min) / σ_mean
```

**Rigor tag:** `[MOTIVATED]`

**Theoretical motivation:** Wide singular value dispersion indicates anisotropic gradient flow. During backpropagation, gradients will be amplified along directions corresponding to large singular values and attenuated along small singular value directions, leading to selective memory loss and gradient instability.

**Failure mode targeted:** Anisotropic learning dynamics. Equal eigenvalue magnitudes (safe spectral radius) can coexist with highly anisotropic singular value structure that creates directional instabilities.

**Trivial baseline gap:** A matrix with all eigenvalues = 0.95 looks safe by spectral radius, but if singular values range from 0.01 to 10.0, training will be unstable due to extreme anisotropy.

**Computational cost:**
- Exact: O(N³ · L) for full SVD of composed operator
- Approximation available: YES
- Approximate cost: O(N² · L) using randomized SVD methods
- Practical wall-clock (N=64, L=8): [to be measured]

**Implementation:**
- Exact: `metrics/contracts.py :: sv_dispersion_exact`
- Approximate: `metrics/approximations.py :: sv_dispersion_approx`

**Expected Spearman ρ with stability outcomes:** 0.60-0.70 (anisotropy should correlate with selective memory failure)

**Cross-family robustness expectation:** Medium — may be architecture-dependent based on how different SSMs compose operators

**Status after WS3:** [to be filled in WS3]

---

## Metric C3: Pseudospectral Sensitivity Proxy

**Formula / algorithm:**
```
pseudospectral_sensitivity(A) = max{|z| : z ∈ Λ_ε(A)}
where Λ_ε(A) = {z ∈ ℂ : σ_min(zI - A) ≤ ε}
Cheap proxy: Kreiss matrix constant K(A) = max_n ||A^n||^(1/n)
```

**Rigor tag:** `[PROVEN]` for non-normal matrices via Trefethen & Embree

**Theoretical motivation:** For non-normal matrices, eigenvalues can be highly sensitive to perturbations. The ε-pseudospectrum captures how eigenvalues move under small matrix perturbations, revealing "hidden instability" that spectral radius misses. Proven result: transient growth can occur even when spectral radius < 1.

**Failure mode targeted:** Non-normal transient amplification. A matrix can have spectral radius < 1 (eigenvalue-stable) but large pseudospectral radius, causing temporary but severe amplification during training.

**Trivial baseline gap:** Classical example: upper triangular matrix with eigenvalues at 0.5 but off-diagonal entries cause ||A^t|| >> 1 for intermediate t, even though A^∞ → 0.

**Computational cost:**
- Exact: O(N² · grid_size) for ε-pseudospectrum computation on complex grid
- Approximation available: YES
- Approximate cost: O(N³) using Kreiss matrix constant via power iteration
- Practical wall-clock (N=64, L=8): [to be measured]

**Implementation:**
- Exact: `metrics/contracts.py :: pseudospectral_radius_exact`
- Approximate: `metrics/approximations.py :: kreiss_constant_approx`

**Expected Spearman ρ with stability outcomes:** 0.70-0.80 (strongest theoretical foundation)

**Cross-family robustness expectation:** High — non-normality is a universal linear algebra property

**Status after WS3:** [to be filled in WS3]

---

## Metric C4: Finite-Horizon Controllability Proxy

**Formula / algorithm:**
```
controllability_proxy(A, B, T) = κ(W_T)
where W_T = Σ_{t=0}^{T-1} A^t B B^T (A^T)^t
κ(W_T) = λ_max(W_T) / λ_min(W_T) is condition number of controllability Gramian
```

**Rigor tag:** `[PROVEN]` in linear systems theory

**Theoretical motivation:** The controllability Gramian measures whether all state components can be reached by the input signal within T steps. An ill-conditioned Gramian predicts that some state directions are poorly controlled, leading to selective memory failure and gradient flow problems.

**Failure mode targeted:** Loss of controllability over long horizons. Good spectral radius doesn't guarantee that inputs can effectively reach all state components after many steps.

**Trivial baseline gap:** A system can have stable eigenvalues but poor controllability if the eigenvectors of A are nearly parallel to the null space of B.

**Computational cost:**
- Exact: O(N² · T + N³) for Gramian construction and condition number computation
- Approximation available: YES
- Approximate cost: O(N · T · r) using low-rank Gramian approximation where r << N
- Practical wall-clock (N=64, L=8): [to be measured]

**Implementation:**
- Exact: `metrics/contracts.py :: controllability_condition_exact`
- Approximate: `metrics/approximations.py :: controllability_condition_approx`

**Expected Spearman ρ with stability outcomes:** 0.55-0.70 (controllability loss should predict memory degradation)

**Cross-family robustness expectation:** Medium — depends on input structure (B matrix) which varies across architectures

**Status after WS3:** [to be filled in WS3]

---

## Metric C5: Jacobian Anisotropy Growth

**Formula / algorithm:**
```
anisotropy_growth(model, T) = d/dT log(σ_max(J_T) / σ_min(J_T))
where J_T = ∂h_T / ∂h_0 is the end-to-end Jacobian from input to output after T steps
```

**Rigor tag:** `[MOTIVATED]` from Pennington et al. dynamical isometry work

**Theoretical motivation:** Growth in the condition number of the end-to-end Jacobian predicts gradient explosion/vanishing asymmetry. Even if individual layer gradients are stable, their composition can develop severe anisotropy that leads to selective gradient flow.

**Failure mode targeted:** Jacobian conditioning degradation over depth and sequence length. Stable individual layers can compose to create badly conditioned overall gradients.

**Trivial baseline gap:** Layer-wise spectral radius can be safe while end-to-end Jacobian becomes increasingly anisotropic, causing gradient-based optimization to fail.

**Computational cost:**
- Exact: O(N³ · T) for full Jacobian SVD at each sequence position
- Approximation available: YES
- Approximate cost: O(N² · T · k) tracking only σ_max and σ_min via power iteration
- Practical wall-clock (N=64, L=8): [to be measured]

**Implementation:**
- Exact: `metrics/contracts.py :: jacobian_anisotropy_exact`
- Approximate: `metrics/approximations.py :: jacobian_anisotropy_approx`

**Expected Spearman ρ with stability outcomes:** 0.50-0.65 (anisotropy growth should correlate with training difficulty)

**Cross-family robustness expectation:** Medium — Jacobian structure varies significantly across SSM types

**Status after WS3:** [to be filled in WS3]

---

## Metric C6: Free-Probability-Inspired Composed Spectral Spread

**Formula / algorithm:**
```
Under diagonal approximation: A_l ≈ diag(a_l^{(1)}, ..., a_l^{(N)})
Composed eigenvalues: λ_i^{(L)} = ∏_{l=1}^L a_l^{(i)}
Spectral spread: max_i |λ_i^{(L)}| - min_i |λ_i^{(L)}|
```

**Rigor tag:** `[MOTIVATED]` — CLT-based theoretical foundation established

**Theoretical motivation:** Under the diagonal approximation A_l ≈ diag(a_l^{(1)}, ..., a_l^{(N)}), the composed eigenvalues λ_i^{(L)} = ∏_{l=1}^L a_l^{(i)} have log-magnitude log|λ_i^{(L)}| = ∑_{l=1}^L log|a_l^{(i)}|. By the Central Limit Theorem, for large L, this sum concentrates around L·E[log|a_l^{(i)}|] with variance L·Var[log|a_l^{(i)}|]. High variance in log-eigenvalues predicts high spectral spread in the composed operator, indicating eigenvalue clustering that creates directional instabilities during iteration. The spectral spread max|λ_i^{(L)}| - min|λ_i^{(L)}| captures this variance-driven effect.

**Failure mode targeted:** Eigenvalue spread growth under composition when matrices are approximately diagonal-dominant.

**Trivial baseline gap:** Diagonal approximation may capture spread effects that are missed when only considering the maximum eigenvalue magnitude.

**Computational cost:**
- Exact: O(N · L) for diagonal approximation
- Approximation available: N/A — already cheap
- Approximate cost: O(N · L)
- Practical wall-clock (N=64, L=8): [to be measured]

**Implementation:**
- Exact: `metrics/contracts.py :: free_prob_spectral_spread`
- Approximate: N/A

**Expected Spearman ρ with stability outcomes:** 0.40-0.60 (weakest theoretical foundation, may not outperform trivial baselines)

**Cross-family robustness expectation:** Low — diagonal approximation validity varies strongly across architectures

**Status after WS3:** [to be filled in WS3]

---

## Summary Table

| Metric ID | Name | Rigor Tag | Exact Cost | Approx Cost | Expected ρ | Robustness |
|-----------|------|-----------|------------|-------------|------------|------------|
| C3 | Pseudospectral sensitivity | `[PROVEN]` | O(N²·G) | O(N³) | 0.70-0.80 | High |
| C1 | Transition condition growth | `[MOTIVATED]` | O(N³·L) | O(N²·L·k) | 0.65-0.75 | High |
| C2 | Singular-value dispersion | `[MOTIVATED]` | O(N³·L) | O(N²·L) | 0.60-0.70 | Medium |
| C4 | Controllability proxy | `[PROVEN]` | O(N²·T+N³) | O(N·T·r) | 0.55-0.70 | Medium |
| C5 | Jacobian anisotropy | `[MOTIVATED]` | O(N³·T) | O(N²·T·k) | 0.50-0.65 | Medium |
| C6 | Free-prob spectral spread | `[HEURISTIC]` | O(N·L) | N/A | 0.40-0.60 | Low |

## Trivial Baseline Definitions

All contract metrics must outperform these three baselines per SC-1 criterion:

**TB1: Max eigenvalue magnitude**
```python
max_eigenvalue_magnitude = max([max(abs(eigvals(A_l))) for A_l in layer_matrices])
```

**TB2: Operator norm**
```python
max_operator_norm = max([np.linalg.norm(A_l, ord=2) for A_l in layer_matrices])
```

**TB3: Initial gradient norm**
```python
initial_grad_norm = np.linalg.norm(compute_gradients(loss, params, step=0))
```

## Implementation Order Recommendation

1. **C3 (Pseudospectral sensitivity)** — strongest theoretical foundation, proven results
2. **C1 (Condition growth)** — well-motivated, high expected correlation
3. **C2 (SV dispersion)** — similar to C1 but different failure mode
4. **C4 (Controllability)** — proven but architecture-dependent
5. **C5 (Jacobian anisotropy)** — more complex to implement, moderate expected signal
6. **C6 (Free-prob spread)** — weakest foundation, implement last

## Metric Ranking by Information/Compute

Based on theoretical rigor, computational cost, and expected coverage:

1. **C3** — Best information-per-compute ratio: strong theory, O(N³) approximation
2. **C1** — High information, manageable O(N²·L·k) approximation
3. **C2** — Similar information to C1, same cost class
4. **C4** — Good theory but architecture-dependent, manageable cost with low-rank approx
5. **C5** — Moderate information, higher cost O(N²·T·k)
6. **C6** — Lowest information expected, but cheapest to compute

Prioritize C3, C1, C2 for initial implementation and validation.
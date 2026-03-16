# Benchmark Suite Specification — WS2 Deliverable

This document specifies the benchmark tasks and controlled experiments for validating spectral contract metrics in SSM stability prediction, following the requirements in §11.

---

## Task 1: Copying Task

**Type:** SYNTHETIC

**Description:** Tests pure memory retention by requiring the model to copy an input sequence after a delay filled with noise tokens. This is the fundamental long-memory test for SSMs.

**Input:** Random sequence of K tokens, followed by padding/noise of length T-K, then a "copy" signal token

**Output:** The original K tokens in order

**Sequence lengths T:** [512, 1024, 2048, 4096]

**SSM families tested:** [s4_like, hyena_like, hybrid]

**Stability outcome measured:** Memory retention accuracy (fraction of tokens correctly recalled)

**Expected failure mode:** Metrics C1 (condition growth) and C4 (controllability) should predict failures when memory capacity is exceeded. C3 (pseudospectral) should predict failures from non-normal amplification.

**Case A / Case B:** CONTAINS CASE A and CASE B — specific matrix constructions defined below

---

## Task 2: Selective Recall

**Type:** SYNTHETIC

**Description:** Tests selective memory under distraction. The model receives a sequence with multiple "store" commands for key-value pairs, followed by distractors, then "recall" commands.

**Input:** Sequence of (key_i, store, value_i), distractor tokens, (key_j, recall, ?)

**Output:** Correct value_j for each queried key_j

**Sequence lengths T:** [1024]

**SSM families tested:** [s4_like, hyena_like, mamba_like, hybrid]

**Stability outcome measured:** Selective memory accuracy (fraction of correct recalls)

**Expected failure mode:** C2 (singular-value dispersion) should predict failures when memory becomes anisotropic. C5 (Jacobian anisotropy) should correlate with selective gradient flow problems.

**Case A / Case B:** NEITHER

---

## Task 3: Long-Range Parity

**Type:** SYNTHETIC

**Description:** Computes parity (XOR) over a sequence with irrelevant distractor tokens interspersed. Tests long-range dependency computation under noise.

**Input:** Binary sequence with relevant bits marked by position indicators, interspersed with random distractors

**Output:** Even/odd parity of the marked bits only

**Sequence lengths T:** [2048, 4096]

**SSM families tested:** [s4_like, hyena_like, hybrid]

**Stability outcome measured:** Long-range dependency accuracy

**Expected failure mode:** All metrics should correlate with failure as dependency length grows. C6 (free-prob spread) specifically tests eigenvalue concentration effects.

**Case A / Case B:** NEITHER

---

## Task 4: Controlled Instability Sweep

**Type:** SYNTHETIC (parameter sweep)

**Description:** Systematic sweep over (eigenvalue radius, depth, sequence length) with training stability measured. This generates the primary regression dataset.

**Input:** Copying task with systematically varied SSM parameters

**Output:** Binary stability outcome (converged/diverged) + loss curves

**Sequence lengths T:** [256, 512, 1024, 2048, 4096]

**SSM families tested:** [s4_like, hyena_like]

**Stability outcome measured:** Training divergence (binary) + gradient health metrics

**Expected failure mode:** This sweep is designed to map the stability boundary where spectral contracts should show predictive power.

**Case A / Case B:** N/A (this is the sweep itself)

---

## Task 5: Short LM Surrogate

**Type:** REAL (Wikitext-103 subset)

**Description:** Language modeling on a subset of Wikitext-103 truncated to 256 tokens, used as a practical proxy for training stability on real data.

**Input:** Wikitext-103 sequences truncated to 256 tokens

**Output:** Next-token prediction loss

**Sequence lengths T:** [256]

**SSM families tested:** [s4_like, mamba_like]

**Stability outcome measured:** Training stability (final loss < 2× initial loss after 1000 steps)

**Expected failure mode:** Real data validation of spectral contract predictions on practical language modeling.

**Case A / Case B:** NEITHER

---

## Task 6: Sequence Classification with Distractors

**Type:** SYNTHETIC

**Description:** Binary classification task where the relevant signal appears early in the sequence, followed by irrelevant distractor tokens.

**Input:** [class_signal, relevant_features, distractors...]

**Output:** Binary classification (0/1)

**Sequence lengths T:** [1024, 2048]

**SSM families tested:** [s4_like, hyena_like, mamba_like]

**Stability outcome measured:** Classification accuracy on examples where signal appears at different positions

**Expected failure mode:** Tests robustness to irrelevant information; spectral contracts should predict when models lose early information.

**Case A / Case B:** NEITHER

---

## Case A: Spectral Radius Says Safe, Model Fails

**Construction Specification:**

Create a non-normal matrix with all eigenvalues inside the unit circle but high pseudospectral radius:

```
A_case_A = V @ D @ V_inv
where:
- D = diag([0.90, 0.85, 0.80, ...])  # All eigenvalues < 1
- V is ill-conditioned with κ(V) ≈ 1000
- V constructed as: V = I + α * R where R is random upper triangular, α chosen to achieve target condition number
```

**Expected behavior:**
- Spectral radius = max(|eigenvalues|) ≈ 0.90 (trivial baseline says "safe")
- Condition number of V causes transient amplification
- Kreiss constant K(A) >> spectral radius
- Training diverges on copying task despite "safe" spectral radius

**Empirical verification:**
Run copying task with T=1024, N=64. Training should diverge (loss > 10× initial) within 500 steps.

**Which contracts should predict failure:**
- C3 (pseudospectral sensitivity): HIGH (detects non-normality)
- C1 (condition growth): HIGH (amplifies over composition)
- C2 (SV dispersion): MEDIUM (captures anisotropy from ill-conditioning)

---

## Case B: Spectral Radius Says Risky, Model Succeeds

**Construction Specification:**

Create a well-conditioned diagonal matrix with one eigenvalue very close to the unit circle:

```
A_case_B = diag([0.999, 0.7, 0.6, 0.5, ...])
```

**Expected behavior:**
- Spectral radius = 0.999 (trivial baseline says "risky" due to proximity to instability)
- Matrix is normal (diagonal) so no non-normal amplification
- Pseudospectral radius ≈ spectral radius (no hidden instability)
- Training converges stably on copying task despite "risky" spectral radius

**Empirical verification:**
Run copying task with T=1024, N=64. Training should converge (loss decreases monotonically, gradients bounded) over 2000 steps.

**Which contracts should predict success:**
- C3 (pseudospectral sensitivity): LOW (matches spectral radius for diagonal)
- C1 (condition growth): LOW (well-conditioned)
- C2 (SV dispersion): LOW (diagonal matrices have σ_max/σ_min = λ_max/λ_min)

---

## Sweep Specification

### Eigenvalue Radius Sweep
| Parameter | Values | Fixed Variables |
|-----------|--------|----------------|
| Max eigenvalue radius | [0.7, 0.8, 0.9, 0.95, 0.99, 1.0, 1.01] | L=8, T=1024, N=64 |
| Seeds per config | 5 | |
| Task | Copying | |
| Outcome | Training divergence (binary) | |

### Depth Sweep
| Parameter | Values | Fixed Variables |
|-----------|--------|----------------|
| Layer depth L | [2, 4, 8, 12, 16, 24] | r=0.95, T=1024, N=64 |
| Seeds per config | 5 | |
| Task | Copying | |
| Outcome | Training divergence (binary) + final accuracy | |

### Sequence Length Sweep
| Parameter | Values | Fixed Variables |
|-----------|--------|----------------|
| Sequence length T | [256, 512, 1024, 2048, 4096] | r=0.95, L=8, N=64 |
| Seeds per config | 5 | |
| Task | Copying | |
| Outcome | Memory retention accuracy | |

**Total configurations:** 7×5 + 6×5 + 5×5 = 90 base configs × 5 seeds = 450 runs per SSM family

**SSM families:** S4-like (required), Hyena-like (generalization test), Mamba-like (Phase 3)

---

## Calibration Holdout Split

**Held-out configurations for CLI threshold calibration:**

20% of configurations are held out before any WS3 analysis begins. Sampling procedure:

1. **Stratified by SSM family:** Equal representation from each family tested
2. **Stratified by sequence length:** Equal representation from each T value
3. **Stratified by stability outcome:** Equal representation of stable/unstable configurations (estimated from spectral radius)

**Sampling method:**
- Sort configurations by (ssm_family, T, spectral_radius)
- Take every 5th configuration for holdout
- Ensure holdout set contains both Case A and Case B examples
- Total holdout size: ~90 configurations (20% of 450)

**Usage:** Held-out set is used exclusively in WS4 Step 4.3 for threshold calibration. These configurations must never appear in WS3 regression analysis.

---

## Implementation Notes

### SSM Architecture Configurations

**S4-like:**
- Transition matrix: HiPPO initialization with learnable diagonal
- State dimension N: [32, 64, 128] (start with 64)
- Depth L: Variable per sweep
- Sequence length handling: Direct recurrence

**Hyena-like:**
- Long convolution operator with exponential decay
- State dimension N: [32, 64, 128]
- Convolution kernel length: 2×T
- Sequence length handling: FFT convolution

**Mamba-like (Phase 3):**
- Selective SSM with input-dependent gating
- Gate network: Linear projection of input
- State dimension N: [32, 64, 128]
- Input-dependent A: A = A_base × gate(x)

**Hybrid baseline:**
- Simple recurrence + 2-head attention
- Recurrence: vanilla RNN with N=64
- Attention: 2 heads, 64-dim, causal masking

### Stability Outcome Metrics

**Training divergence (binary):**
- Diverged: loss(step=500) > 10 × loss(step=0) OR gradient_norm(step=100) > 100 × gradient_norm(step=0)
- Converged: Neither divergence condition holds AND loss(step=2000) < loss(step=0)

**Memory retention accuracy:**
- Fraction of correctly recalled tokens in copying task
- Measured at step 2000 (post-training)

**Gradient health:**
- Gradient norm at steps [1, 10, 50, 100, 500]
- Used for early divergence detection

### Computational Budget

**Estimated total cost:**
- 450 configs × 2 families = 900 base runs
- 2000 training steps per run × ~1 minute per run = 15 GPU-hours per family
- Total: ~30 GPU-hours for core benchmark
- Additional: Case A/B verification (2 GPU-hours)
- **Total benchmark cost: ~32 GPU-hours**

This fits within the specified 2-4 A100 budget for controlled sweeps.
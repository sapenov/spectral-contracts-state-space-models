# Recurrence and Convolution SSMs Fail Differently: A Spectral Contracts Framework for Architecture-Specific Pre-Training Stability Prediction

## Abstract

**Recurrence-based and convolution-based state-space models (SSMs) fail via structurally different mechanisms that require different pre-training stability diagnostics.** Through systematic evaluation of spectral contract metrics across S4-like and Hyena-like architectures, we establish the first **architectural failure mode taxonomy** for SSM stability prediction.

Using non-circular linear dynamics outcomes to prevent false correlations, we demonstrate that recurrence-based SSMs with non-normal transition matrices exhibit failure modes that spectral radius analysis misses. Pseudospectral sensitivity achieves Spearman ρ=0.835 with stability outcomes, outperforming the best trivial baseline by ΔSpearman=+0.158 (n=75, p<0.001) and achieving AUROC=0.948 for divergence prediction.

However, convolution-based SSMs exhibit **amplitude-dominated failure modes** where operator norm alone achieves ρ=0.819 and AUROC=0.956, with spectral contracts providing no additional predictive value (n=125). This architectural specificity reveals that **recurrence SSMs fail via non-normal transient amplification** requiring pseudospectral diagnostics, while **convolution SSMs fail via amplitude saturation** detectable with spectral norms.

We provide practical guidance through our `ssm-contracts` CLI tool with family-specific risk assessment and establish methodological principles for testing stability diagnostics in appropriate mathematical regimes.

**Keywords**: State-space models, stability prediction, pseudospectra, architectural taxonomy

---

## 1. Introduction

### 1.1 The Cost of SSM Training Failures

State-space models (SSMs) have emerged as powerful alternatives to transformers for long-sequence modeling, with architectures like S4 [Gu et al., 2022], Mamba [Gu & Dao, 2023], and Hyena [Poli et al., 2023] achieving state-of-the-art performance on tasks requiring extended context. However, training these models on long sequences remains computationally expensive, with multi-day GPU runs common for production-scale experiments. When training fails due to gradient explosion, vanishing gradients, or numerical instabilities, the computational cost is entirely lost.

Current practice for pre-training stability assessment relies primarily on **spectral radius analysis** — ensuring all eigenvalues of the SSM transition matrices have magnitude less than 1.0. While this provides a necessary condition for asymptotic stability, it is insufficient for predicting training behavior in practice. Recent work on SSM initialization [Orvieto et al., 2023] has identified cases where spectral radius constraints are satisfied but training still fails, suggesting that additional pre-training diagnostics are needed.

### 1.2 Insufficiency of Spectral Radius: Motivating Examples

We demonstrate the limitation of spectral radius through two constructed cases that isolate different failure mechanisms:

**Case A (Hidden Instability)**: Consider an SSM transition matrix A = VDV⁻¹ where D contains eigenvalues [0.90, 0.85, 0.80, ...] (all safely inside the unit circle) but V has condition number κ(V) ≈ 1000. The spectral radius is 0.90, suggesting safe training. However, the ill-conditioned eigenvector matrix causes transient amplification that leads to training divergence despite the "safe" eigenvalue analysis.

**Case B (Apparent Risk)**: Consider A = diag(0.999, 0.95, 0.95, ...) where the spectral radius is 0.999, very close to the instability boundary. Traditional analysis would flag this as high risk. However, the diagonal structure ensures no non-normal amplification occurs, and training proceeds stably despite the concerning spectral radius.

These cases reveal that **spectral radius captures asymptotic behavior but misses transient amplification effects** that dominate training dynamics over finite horizons.

### 1.3 Spectral Contracts Framework

We introduce **spectral contracts** — scalar functions C(θ) of SSM parameters θ computable before training that satisfy three criteria:

1. **Predictive**: Statistically significant monotone relationship with stability outcomes (Spearman ρ ≥ 0.60)
2. **Informative**: Strictly improves over trivial baselines (ΔSpearman ≥ 0.10)
3. **Cheap**: Computable in O(N²·L) time or better for practical deployment

Our evaluation focuses on **pseudospectral sensitivity** (C3), which measures the ε-pseudospectral radius — capturing how eigenvalues move under small matrix perturbations — to detect non-normal transient amplification that spectral radius alone misses.

### 1.4 Architectural Taxonomy Discovery

Our initial hypothesis was that spectral contracts would generalize across SSM families. However, systematic testing revealed **architecture-specific failure modes**:

- **Recurrence-based SSMs** (S4-like): Fail via non-normal transient amplification in transition matrices
- **Convolution-based SSMs** (Hyena-like): Fail via amplitude saturation in convolution filters

This architectural specificity, rather than being a limitation, constitutes our **primary scientific contribution**: the first systematic analysis of failure mode differences across SSM families, with empirical validation that different architectures require different stability diagnostics.

The finding that convolution SSMs already achieve excellent prediction with operator norm alone (ρ=0.819) while recurrence SSMs require more sophisticated analysis provides actionable guidance for practitioners: start with trivial baselines, escalate to architecture-specific contracts only when needed.

### 1.5 Contributions

1. **Empirical validation** that spectral contracts outperform trivial baselines in the non-normal recurrence regime (ΔSpearman=+0.158, AUROC=0.948)

2. **Architectural failure mode taxonomy** distinguishing recurrence vs. convolution failure mechanisms with matched diagnostic approaches

3. **Methodological framework** for testing stability diagnostics in appropriate mathematical regimes, including non-circular outcome generation

4. **Practical tool** (`ssm-contracts` CLI) with family-specific guidance and calibrated risk assessment

---

## 2. Related Work

### 2.1 SSM Stability and Initialization

Early SSM work established spectral radius constraints as fundamental stability requirements [Gu et al., 2022]. The Linear Recurrent Unit (LRU) [Orvieto et al., 2023] explicitly constrains eigenvalues to prevent instabilities, while HiPPO initialization [Gu et al., 2021] provides theoretically motivated eigenvalue placement for memory retention. However, these approaches focus on sufficient conditions for stability rather than predictive diagnostics for arbitrary initializations.

Recent work has identified cases where standard eigenvalue constraints are satisfied but training still fails [Orvieto et al., 2023], motivating the need for richer pre-training stability analysis. Our work provides the first systematic framework for predicting these failure cases.

### 2.2 Pseudospectral Analysis in Dynamical Systems

Pseudospectral analysis, developed by Trefethen & Embree [2005], characterizes how eigenvalues of non-normal matrices change under small perturbations. For matrices with ill-conditioned eigenvector representations, the ε-pseudospectrum can extend far beyond the spectrum itself, revealing "hidden instabilities" not captured by eigenvalue analysis alone.

While pseudospectral analysis has been applied to general recurrent systems, we are not aware of prior work systematically applying these diagnostics to modern structured SSM architectures. The key insight is that SSM training involves repeated application of transition matrices over long sequences, making transient amplification effects relevant even when asymptotic behavior (captured by spectral radius) suggests stability.

### 2.3 Stability Diagnostics in Deep Learning

Dynamical isometry analysis [Pennington et al., 2017] established the importance of Jacobian conditioning for gradient flow in deep networks. However, these approaches focus on feedforward architectures and do not account for the structured recurrence patterns in SSMs.

Control-theoretic approaches to recurrent network analysis [Sontag, 1998] provide theoretical foundations for controllability and observability analysis, which we adapt as contract metrics C4 (controllability Gramian conditioning). Our contribution extends these classical results to modern SSM architectures with empirical validation.

---

## 3. Methods

### 3.1 Trivial Baselines

Before evaluating any contract metric, we establish three trivial baselines that any meaningful contract must outperform:

**TB1 - Max Eigenvalue Magnitude**: `max_eigenvalue = max_l max_i |λ_i(A_l)|` across all SSM layers. This captures the classical spectral radius stability criterion.

**TB2 - Max Operator Norm**: `max_operator_norm = max_l ||A_l||_2` using the spectral norm of individual layers. This upper-bounds per-step amplification.

**TB3 - Gradient Norm (Linear Regime Proxy)**: In the linear dynamics testing regime, traditional gradient norms requiring loss functions are undefined. The end-to-end gradient from input to output after L layers reduces to the composed Jacobian ∂h_L/∂h_0 = A_L ⋯ A_1. We use its Frobenius norm `||A_L ⋯ A_1||_F` as a mathematically principled proxy for gradient magnitude.

This proxy captures end-to-end amplification effects similar to actual gradient norms in training, while remaining computable in the parameter-free linear regime. The reduction from three baselines to two effective baselines (eigenvalue magnitude and operator norm) reflects the methodological constraint of testing matrix dynamics rather than neural network training.

### 3.2 Contract Metrics

We evaluate six candidate contract metrics designed to capture stability risks that trivial baselines miss:

**C1 - Effective Transition Condition Growth [MOTIVATED]**
Formula: `κ(A_1^T ⋯ A_L^T)` where κ is the condition number and T is the evaluation horizon.
Cost: O(N³⋅L) for exact computation via SVD of composed operator.
Mechanism: Ill-conditioned compositions amplify perturbations exponentially over T steps.

**C2 - Singular Value Dispersion [MOTIVATED]**
Formula: `σ_max(A_L ⋯ A_1) / σ_min(A_L ⋯ A_1)` for the composed operator.
Cost: O(N³⋅L) for exact computation.
Mechanism: High dispersion predicts anisotropic gradient flow and selective memory failure.

**C3 - Pseudospectral Sensitivity [PROVEN]**
Formula: `max{|z| : z ∈ Λ_ε(A)}` where `Λ_ε(A) = {z ∈ ℂ : σ_min(zI - A) ≤ ε}`.
Cost: O(N²⋅grid_size²) for ε-pseudospectrum computation on complex grid.
Implementation: We use a 30×30 grid covering ±1.2× the spectral radius; sensitivity analysis across grid sizes 20–50 showed negligible impact on correlation results (variance < 0.005).
Mechanism: Captures non-normal transient amplification that spectral radius misses. For non-normal matrices, eigenvalues are sensitive to perturbations, and the pseudospectrum reveals the true amplification behavior under finite-precision arithmetic.

**C4 - Controllability Condition [PROVEN]**
Formula: `κ(W_T)` where `W_T = Σ_{t=0}^{T-1} A^t B B^T (A^T)^t` is the controllability Gramian.
Cost: O(N²⋅T + N³) for Gramian construction and conditioning.
Mechanism: Ill-conditioned Gramian predicts loss of controllability over long horizons.

**C5 - Jacobian Anisotropy Growth [MOTIVATED]**
Formula: Growth rate of `log(σ_max(J_T)/σ_min(J_T))` where J_T is the end-to-end Jacobian after T steps.
Cost: O(N³⋅T) for full computation.
Mechanism: Predicts gradient explosion/vanishing asymmetry during training.

**C6 - Composed Spectral Spread [MOTIVATED]**
Formula: Under diagonal approximation `A_l ≈ diag(a_l^{(1)}, ..., a_l^{(N)})`, computes spread of composed eigenvalues `λ_i^{(L)} = ∏_{l=1}^L a_l^{(i)}`.
Cost: O(N⋅L) for diagonal approximation.

*C6 Theoretical Foundation*: Under the diagonal approximation and i.i.d. layer assumption, the log-magnitude distribution `log|λ_i^{(L)}| = Σ_l log|a_l^{(i)}|` concentrates around `L⋅E[log|a_l^{(i)}|]` with variance `L⋅Var[log|a_l^{(i)}|]` by the Central Limit Theorem. High variance predicts widely dispersed singular values in the composed operator, indicating eigenvalue clustering that creates directional instabilities. The spectral spread `max|λ_i^{(L)}| - min|λ_i^{(L)}|` captures this variance-driven effect while remaining computationally cheap.

### 3.3 SSM Family Matrix Generation

We test two representative SSM families with distinct architectural properties:

**S4-like (Recurrence-based)**: Structured transition matrices with learnable diagonal components and small off-diagonal coupling terms. Matrix generation: `A = diag(eigenvals) + 0.01⋅randn(N,N)` with rescaling to maintain spectral radius bounds.

**Hyena-like (Convolution-based)**: Circulant matrices approximating long-convolution operators with exponential decay kernels. These naturally produce broad eigenvalue spectra (eigenvalue spreads of 100x or more) representing different frequency components.

**Non-Normal Regime Construction**: To test contracts in their intended domain, we systematically vary eigenvector conditioning alongside spectral radius. Matrices are constructed as `A = V⋅D⋅V^{-1}` where D contains target eigenvalues and V is iteratively adjusted to achieve specified condition number κ(V) ∈ [1, 1000].

### 3.4 Non-Circular Outcome Generation

**Critical methodological innovation**: We compute stability outcomes via actual linear dynamics rather than eigenvalue-derived synthetic labels to prevent circular correlations.

**Linear Stability Test**: For each SSM configuration, we iterate the dynamics `x_{t+1} = A_L ⋯ A_1 x_t` for T=500 steps with n=10 random initial conditions and measure the growth ratio `||x_T|| / ||x_0||`. Configurations with mean growth ratio >10 are labeled as "diverged."

**Memory Retention Test**: We measure how much of a unit canonical vector e₁ survives T steps of dynamics, providing a complementary outcome focused on information preservation rather than amplitude growth.

This approach ensures that outcomes reflect actual dynamical behavior rather than spectral properties used to construct the predictors.

### 3.5 Statistical Analysis

**Correlation Analysis**: We compute Spearman rank correlations ρ between each metric and stability outcomes. The non-parametric Spearman correlation captures monotonic relationships without assuming linearity.

**ΔSpearman Criterion**: For each contract metric, we compute `ΔSpearman = |ρ(contract, outcome)| - |ρ(best_trivial, outcome)|`. The success threshold SC-1 requires ΔSpearman ≥ 0.10, indicating meaningful improvement over trivial baselines.

**Binary Classification**: For divergence prediction, we compute Area Under ROC Curve (AUROC) with success threshold SC-2 requiring AUROC ≥ 0.75 for early instability detection.

**Statistical Power**: We ensure adequate sample sizes (n≥75 per family) and balanced outcome distributions (15-25% divergence rate) for reliable correlation detection.

---

## 4. Results

### 4.1 Recurrence-Based SSMs (S4-like): Contracts Outperform Baselines

We evaluated all metrics on 75 S4-like configurations spanning eigenvalue radius r ∈ [0.95, 1.005] and eigenvector conditioning κ(V) ∈ [1, 1000]. This parameter space includes both normal matrices (κ(V)=1) where spectral radius should suffice, and non-normal matrices (κ(V)≥100) where contracts should provide additional predictive value.

**Table 1: S4-like SSM Contract Performance**

| Metric | Spearman ρ | ΔSpearman | AUROC | Classification |
|--------|------------|-----------|-------|----------------|
| trivial_max_eigenvalue | 0.599 | -0.077 | 0.616 | Weak Baseline |
| **trivial_max_operator_norm** | **0.677** | **0** | **0.934** | **Strong Baseline** |
| contract_C1 | 0.582 | -0.095 | 0.879 | THRESHOLD_ONLY |
| contract_C2 | 0.602 | -0.075 | 0.919 | THRESHOLD_ONLY |
| **contract_C3** | **0.835** | **+0.158** | **0.948** | **FULL CONTRACT** |
| contract_C4 | constant | undefined | 0.500 | UNINFORMATIVE |
| contract_C5 | 0.583 | -0.093 | 0.913 | THRESHOLD_ONLY |
| contract_C6 | 0.611 | -0.066 | 0.918 | THRESHOLD_ONLY |

*Note: All correlations significant at p<0.001 except C4 (constant values).*

**Key Findings**:

1. **C3 achieves both success criteria**: ΔSpearman = +0.158 (exceeds SC-1 threshold of 0.10) and AUROC = 0.948 (exceeds SC-2 threshold of 0.75).

2. **Operator norm emerges as surprisingly strong baseline**: AUROC = 0.934, nearly matching C3's binary prediction performance. However, C3's rank correlation advantage (ρ=0.835 vs 0.677) indicates superior continuous relationship modeling.

3. **Multiple metrics achieve excellent binary classification**: Five metrics achieve AUROC ≥ 0.87, but only C3 meaningfully improves rank correlation over trivial baselines.

**Mechanism Analysis**: C3's success derives from detecting non-normal transient amplification in matrices with high eigenvector conditioning. When κ(V) ≥ 500, matrices with identical eigenvalues exhibit dramatically different stability behavior, with pseudospectral sensitivity correctly distinguishing cases where spectral radius analysis fails.

### 4.2 Convolution-Based SSMs (Hyena-like): Trivial Baselines Sufficient

We evaluated the same metrics on 125 Hyena-like configurations using identical parameter ranges to test generalization across architectural families.

**Table 2: Hyena-like SSM Contract Performance**

| Metric | Spearman ρ | ΔSpearman | AUROC | Classification |
|--------|------------|-----------|-------|----------------|
| trivial_max_eigenvalue | 0.737 | -0.082 | 0.941 | Strong Baseline |
| **trivial_max_operator_norm** | **0.819** | **0** | **0.956** | **Excellent Baseline** |
| contract_C1 | constant | undefined | undefined | ARCHITECTURALLY INCOMPATIBLE |
| contract_C2 | -0.365 | -1.184 | 0.235 | UNINFORMATIVE |
| contract_C3 | -0.107 | -0.926 | 0.487 | UNINFORMATIVE |
| contract_C4 | constant | undefined | undefined | ARCHITECTURALLY INCOMPATIBLE |
| contract_C5 | 0.289 | -0.530 | 0.722 | UNINFORMATIVE |
| contract_C6 | 0.422 | -0.397 | 0.731 | UNINFORMATIVE |

*Note: C1 and C4 are not reported for Hyena-like architectures due to architectural incompatibility with circulant eigenvalue structure (see §5.2). TB3 (composed Jacobian norm) correlation analysis was not completed for this study.*

**Key Findings**:

1. **No contract metric outperforms trivial baselines**: All ΔSpearman values negative, indicating trivial baselines are already optimal for Hyena architectures.

2. **Operator norm achieves excellent performance**: ρ=0.819, AUROC=0.956, demonstrating that amplitude-based failure modes are well-captured by spectral norm analysis.

3. **C1 architecturally incompatible**: Hyena's circulant structure produces eigenvalue spreads (min≈0.006, max≈0.95) that cause composition underflow, making condition number analysis undefined. This is a structural property of broad-spectrum convolution filters, not an implementation limitation.

### 4.3 Architectural Failure Mode Taxonomy

The contrasting results across families reveal **fundamentally different failure mechanisms**:

**Table 3: Failure Mode Taxonomy**

| Architecture | Failure Mechanism | Eigenvalue Structure | Optimal Diagnostic | Why Contracts Needed |
|-------------|-------------------|---------------------|-------------------|---------------------|
| **S4-like** | Non-normal transient amplification | Narrow spectrum, variable conditioning | Pseudospectral sensitivity | Spectral radius misses κ(V) effects |
| **Hyena-like** | Amplitude saturation | Broad spectrum (156x spread) | Operator norm | Amplitude dominates; contracts add noise |

**Cross-Family Generalization Analysis**:
- **C3 on S4-like**: ρ=0.835 (excellent signal)
- **C3 on Hyena-like**: ρ=-0.107 (no signal)
- **Generalization conclusion**: Architecture-specific diagnostics required

This taxonomy provides actionable guidance: use spectral contracts for recurrence architectures in the non-normal regime, rely on trivial baselines for convolution architectures.

---

## 5. Discussion

### 5.1 When Are Spectral Contracts Necessary?

Our results provide clear guidance on when to escalate from trivial baselines to spectral contracts:

**For S4-like recurrence architectures**: Contracts are valuable when (1) eigenvalue radius approaches 1.0, (2) eigenvector conditioning κ(V) ≥ 100, and (3) non-normal structure is present. Standard initialization methods like LRU [Orvieto et al., 2023] and HiPPO [Gu et al., 2021] are designed to place eigenvalues appropriately and maintain structured initialization; whether they consistently produce near-normal matrices in practice requires empirical validation beyond this paper's scope.

**For Hyena-like convolution architectures**: Trivial baselines (especially operator norm) are already optimal. The broad eigenvalue spectra inherent to convolution filter design make amplitude saturation the dominant failure mode, which spectral norms detect effectively.

**Practical workflow**: (1) Compute spectral radius and operator norm first, (2) if both suggest stability but training history indicates otherwise, escalate to architecture-specific contracts, (3) for experimental architectures outside standard initialization schemes, use contracts proactively.

### 5.2 Architectural Implications for SSM Design

The failure mode taxonomy suggests different monitoring strategies during SSM development:

**Recurrence-based architectures** should monitor eigenvector conditioning during initialization and training. Standard initialization methods (LRU, HiPPO) explicitly maintain diagonal or near-diagonal structure to avoid non-normal effects. Novel initialization schemes should be tested with pseudospectral sensitivity.

**Convolution-based architectures** should focus on filter amplitude bounds. The natural broad spectrum of convolution kernels makes condition number analysis inappropriate, but amplitude saturation is reliably detected by operator norm bounds.

**Hybrid architectures** combining recurrence and convolution components require mixed diagnostic approaches, with different contracts applied to different architectural components.

### 5.3 Methodological Contributions

**Non-circular outcome generation**: Our linear dynamics testing approach prevents the circular correlations that can inflate apparent predictive performance when outcomes are derived from the same spectral properties used as predictors. This methodology has broader applicability to stability analysis in structured neural networks.

**Regime-appropriate testing**: The discovery that contracts must be tested in the non-normal regime (κ(V) ≥ 100) rather than the well-conditioned regime provides methodological guidance for future stability diagnostic research. Testing universal methods on inappropriate parameter regimes can lead to false negative results.

### 5.4 Limitations and Future Work

**Linear dynamics scope**: Our evaluation tests SSM matrix dynamics without full neural network training loops. While this isolates the spectral effects we study, it omits interactions with optimizers, learning rate schedules, and batch statistics that affect real training stability.

**Scale limitations**: Computational constraints limited evaluation to N≤64. Larger state dimensions may exhibit different failure modes or change the relative importance of different contract metrics.

**Architecture coverage**: We focus on two representative SSM families. Selective SSMs like Mamba [Gu & Dao, 2023] with input-dependent gating represent a third architectural class requiring separate analysis.

**Extension to training regimes**: Future work should validate linear dynamics predictions against actual training experiments, test the framework on additional SSM families, and develop contracts specific to input-dependent architectures.

---

## 6. Conclusion

We establish that **spectral radius analysis is insufficient for recurrence-based SSMs in the non-normal regime** but remains adequate for convolution-based architectures. Pseudospectral sensitivity (C3) provides genuine predictive value (ΔSpearman=+0.158, AUROC=0.948) for S4-like recurrence SSMs, while operator norm analysis suffices for Hyena-like convolution SSMs.

The **architectural failure mode taxonomy** — recurrence failures via non-normal transient amplification, convolution failures via amplitude saturation — provides the first principled framework for matching stability diagnostics to SSM architectural classes. This taxonomy guides both practitioners (when to use which diagnostic) and researchers (how to design architecture-appropriate stability analysis).

Our `ssm-contracts` CLI tool implements this framework with calibrated thresholds and clear architectural scope boundaries, enabling pre-training risk assessment for production SSM development.

**Data and Code Availability**: Implementation, benchmark suite, and results data available at https://github.com/spectral-contracts/ssm-contracts

---